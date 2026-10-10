"""Relational variant of the LSGC convolution.

This layer keeps the spatial/time continuous filter used by ``LSGCConv`` but
adds a Schlichtkrull-style basis decomposition over discrete biological edge
relations. For each edge, the filter network emits ``B`` basis filters and the
edge relation selects a learned combination of those bases.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing


class RelationalLSGCConv(MessagePassing):
    """LSGCConv with relation-specific basis-composed edge filters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int = 4,
        num_bases: int = 2,
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
        self.num_relations = int(num_relations)
        self.num_bases = int(num_bases)
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

        freq_init = torch.linspace(1.0, num_time_freq, num_time_freq) * math.pi / (
            self.max_dt + 1.0
        )
        self.time_freqs = nn.Parameter(freq_init)

        filter_in = num_rbf + 2 * num_time_freq + (3 if use_direction else 0) + edge_attr_dim
        self.filter_net = nn.Sequential(
            nn.Linear(filter_in, hidden_filter),
            nn.SiLU(),
            nn.Linear(hidden_filter, num_bases * in_channels),
        )
        self.rel_coeff = nn.Parameter(torch.empty(num_relations, num_bases))

        if self.use_edge_gating:
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
        nn.init.xavier_uniform_(self.rel_coeff)
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
            with torch.no_grad():
                self.gate_net[-1].bias.fill_(2.0)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be (N, C), got {tuple(x.shape)}")
        if pos.shape != (x.size(0), 3):
            raise ValueError(f"pos must be (N, 3), got {tuple(pos.shape)}")
        if t.shape[0] != x.size(0):
            raise ValueError(f"t must have length N={x.size(0)}, got {t.shape[0]}")
        aggr = self.propagate(
            edge_index, x=x, pos=pos, t=t, edge_attr=edge_attr, edge_type=edge_type
        )
        return self.lin_self(x) + self.lin_out(aggr)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        t_i: Tensor,
        t_j: Tensor,
        edge_attr: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> Tensor:
        delta_pos = pos_j - pos_i
        r = delta_pos.norm(dim=-1, keepdim=True)
        direction = delta_pos / (r + 1e-6)

        rbf = torch.exp(
            -self.rbf_gamma * (r - self.rbf_centers.view(1, -1)) ** 2
        )
        dt = t_j.view(-1, 1).float() - t_i.view(-1, 1).float()
        arg = dt * self.time_freqs.view(1, -1)
        t_enc = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

        feats = [rbf, t_enc]
        if self.use_direction:
            feats.append(direction)
        if self.edge_attr_dim > 0:
            if edge_attr is None:
                edge_attr = torch.zeros(
                    (rbf.shape[0], self.edge_attr_dim),
                    device=rbf.device,
                    dtype=rbf.dtype,
                )
            feats.append(edge_attr)
        filter_in = torch.cat(feats, dim=-1)

        basis = self.filter_net(filter_in).view(-1, self.num_bases, self.in_channels)
        if edge_type is None:
            edge_type = torch.zeros(rbf.shape[0], device=rbf.device, dtype=torch.long)
        edge_type = edge_type.to(device=rbf.device, dtype=torch.long).clamp(
            0, self.num_relations - 1
        )
        coeff = self.rel_coeff[edge_type]
        w = torch.einsum("eb,ebc->ec", coeff, basis)

        msg = w * self.lin_msg(x_j)
        if self.gate_net is not None:
            if self.edge_attr_dim > 0:
                gate_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
            else:
                gate_input = torch.cat([x_i, x_j], dim=-1)
            msg = torch.sigmoid(self.gate_net(gate_input)) * msg
        return msg
