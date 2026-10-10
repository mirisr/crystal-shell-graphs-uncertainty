"""Longitudinal Spatial Graph Convolution (LSGC).

Continuous-filter graph convolution whose filter W_ij is jointly parameterized
by (a) 3D position difference (delta_pos in mm), (b) visit-time difference
(delta_t in integer visits), and optionally (c) a unit direction vector. It
reduces to a SchNet-style spatial continuous filter when delta_t is constant,
and to a pure temporal filter when spatial coordinates collapse.

Expected per-node inputs:
    x           : (N, C_in)   node features (e.g. PE, SER, kinetic summaries).
    pos         : (N, 3)      supervoxel centroid in mm.
    t           : (N,) long   visit index in {0, 1, 2, 3}.
    edge_index  : (2, E) long standard PyG edge index; row 0 is source, row 1
                               is target under PyG's source_to_target flow.
    batch       : (N,) long   optional graph-id per node for batched graphs.

The graph passed to the layer is a single unified spatio-temporal graph that
contains both intra-visit edges (spatial kNN within a visit) and inter-visit
edges (supervoxels from different visits connected by spatial proximity and
feature similarity). This is what lets the same operator reason across space
and across visits.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class LSGCConv(MessagePassing):
    """Longitudinal Spatial Graph Conv layer.

    Parameters
    ----------
    in_channels, out_channels
        Node feature widths in / out.
    num_rbf
        Number of Gaussian radial basis functions used to encode ||delta_pos||.
    rbf_cutoff
        Maximum distance (mm) covered by the RBF basis. Distances beyond this
        are still valid but saturate in the basis; prefer to cap the edge set
        at this distance when building the graph.
    num_time_freq
        Number of sinusoidal frequencies used to encode delta_t.
    max_dt
        Largest absolute visit offset the frequencies are tuned to span; with
        I-SPY 2 this is 3 (T0 <-> T3).
    hidden_filter
        Width of the MLP that maps (RBF, time, direction) -> per-channel filter.
    use_direction
        If True, the MLP also sees the 3D unit direction d = delta_pos / r. Off
        gives an isotropic (SchNet-like) filter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_rbf: int = 16,
        rbf_cutoff: float = 50.0,
        num_time_freq: int = 4,
        max_dt: int = 3,
        hidden_filter: int = 64,
        use_direction: bool = True,
        edge_attr_dim: int = 0,
        use_edge_gating: bool = False,
        gate_hidden: int = 32,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rbf = num_rbf
        self.rbf_cutoff = float(rbf_cutoff)
        self.num_time_freq = num_time_freq
        self.max_dt = float(max(max_dt, 1))
        self.use_direction = use_direction
        self.edge_attr_dim = edge_attr_dim
        self.use_edge_gating = bool(use_edge_gating)

        centers = torch.linspace(0.0, self.rbf_cutoff, num_rbf)
        self.register_buffer("rbf_centers", centers)
        step = self.rbf_cutoff / max(num_rbf - 1, 1)
        self.rbf_gamma = 1.0 / (step**2 + 1e-6)

        freq_init = torch.linspace(1.0, num_time_freq, num_time_freq) * math.pi / (self.max_dt + 1.0)
        self.time_freqs = nn.Parameter(freq_init)

        filter_in = num_rbf + 2 * num_time_freq + (3 if use_direction else 0) + edge_attr_dim
        self.filter_net = nn.Sequential(
            nn.Linear(filter_in, hidden_filter),
            nn.SiLU(),
            nn.Linear(hidden_filter, in_channels),
        )

        if self.use_edge_gating:
            # Per-edge scalar gate g_ij in (0, 1) computed from
            # [edge_attr, h_i, h_j]. Initialised so g≈1 at the start
            # (final bias = +2) so the layer is identical to non-gated
            # LSGCConv on the first forward pass.
            gate_in = edge_attr_dim + 2 * in_channels
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
        else:
            self.gate_net = None

        self.lin_msg = nn.Linear(in_channels, in_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_out = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.filter_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for lin in (self.lin_msg, self.lin_self, self.lin_out):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if self.gate_net is not None:
            for m in self.gate_net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Bias the final logit to +2 so sigmoid(2)≈0.88 ≈ "open" gate
            # at init; the model can still close gates as it learns.
            with torch.no_grad():
                self.gate_net[-1].bias.fill_(2.0)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be (N, C), got {tuple(x.shape)}")
        if pos.shape != (x.size(0), 3):
            raise ValueError(f"pos must be (N, 3), got {tuple(pos.shape)}")
        if t.shape[0] != x.size(0):
            raise ValueError(f"t must have length N={x.size(0)}, got {t.shape[0]}")
        aggr = self.propagate(edge_index, x=x, pos=pos, t=t, edge_attr=edge_attr)
        return self.lin_self(x) + self.lin_out(aggr)

    # ------------------------------------------------------------------
    # PyG internals
    # ------------------------------------------------------------------
    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        t_i: Tensor,
        t_j: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        delta_pos = pos_j - pos_i  # (E, 3)
        r = delta_pos.norm(dim=-1, keepdim=True)  # (E, 1)
        direction = delta_pos / (r + 1e-6)  # (E, 3)

        rbf = torch.exp(
            -self.rbf_gamma * (r - self.rbf_centers.view(1, -1)) ** 2
        )  # (E, K)

        dt = (t_j.view(-1, 1).float() - t_i.view(-1, 1).float())  # (E, 1)
        arg = dt * self.time_freqs.view(1, -1)  # (E, M)
        t_enc = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (E, 2M)

        feats = [rbf, t_enc]
        if self.use_direction:
            feats.append(direction)
        if self.edge_attr_dim > 0:
            if edge_attr is None:
                edge_attr = torch.zeros((rbf.shape[0], self.edge_attr_dim), device=rbf.device, dtype=rbf.dtype)
            feats.append(edge_attr)
        filter_in = torch.cat(feats, dim=-1)

        w = self.filter_net(filter_in)  # (E, C_in)
        msg = w * self.lin_msg(x_j)
        if self.gate_net is not None:
            if self.edge_attr_dim > 0:
                gate_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
            else:
                gate_input = torch.cat([x_i, x_j], dim=-1)
            gate = torch.sigmoid(self.gate_net(gate_input))  # (E, 1)
            msg = gate * msg
        return msg


class LSGCNet(nn.Module):
    """Stackable LSGC backbone with a simple graph-level head.

    Parameters
    ----------
    in_channels
        Node feature width at input (e.g. PE/SER summaries + one-hot visit).
    hidden
        Hidden width carried through every LSGC layer.
    out_channels
        Output dim of the per-graph prediction (e.g. 1 for pCR logits).
    num_layers
        Number of stacked LSGCConv layers with residual connections.
    readout
        'mean' or 'add' graph-level pooling.
    conv_kwargs
        Forwarded to each LSGCConv (e.g. num_rbf, rbf_cutoff, num_time_freq).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        readout: str = "mean",
        clinical_dim: int = 0,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        if readout not in {"mean", "add"}:
            raise ValueError("readout must be 'mean' or 'add'")
        self.readout = readout
        self.clinical_dim = clinical_dim
        self.dropout = float(dropout)
        self.embed = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList(
            [LSGCConv(hidden, hidden, **conv_kwargs) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()
        self.node_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.head_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden + clinical_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_channels)
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        clinical: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.embed(x)
        for conv in self.convs:
            h = h + self.node_dropout(self.act(conv(h, pos, t, edge_index, edge_attr=edge_attr)))
        if batch is None:
            z = h.mean(dim=0, keepdim=True) if self.readout == "mean" else h.sum(dim=0, keepdim=True)
        else:
            z = scatter(h, batch, dim=0, reduce=self.readout)
        if self.clinical_dim > 0:
            if clinical is None:
                clinical = torch.zeros((z.shape[0], self.clinical_dim), device=z.device, dtype=z.dtype)
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)
            if clinical.shape[0] != z.shape[0]:
                clinical = clinical.view(z.shape[0], -1)
            z = torch.cat([z, clinical.to(device=z.device, dtype=z.dtype)], dim=1)
        z = self.head_dropout(z)
        return self.head(z)
