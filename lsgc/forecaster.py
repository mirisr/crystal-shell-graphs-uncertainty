"""LSGC-Forecaster: one-step supervoxel-level digital twin.

Same LSGC backbone as :class:`~lsgc.lsgc_layer.LSGCNet`, but the graph-level
mean-pool + classification head is replaced by three *per-node* MLP heads
predicting the supervoxel's state at the next visit:

* ``delta_pos`` (N, 3)   -- predicted centroid drift in millimeters,
                            in the centroid-subtracted frame used by the
                            Stage-1 matching.
* ``delta_feat`` (N, C)  -- predicted feature change (same dims as ``x``).
* ``alive_logit`` (N,)   -- BCE logit for the probability the supervoxel
                            survives to the next visit.

An explicit ``delta_t`` conditioning token is concatenated to the final
hidden state before the heads, per the proposal
(Sec.~5 "Architecture delta"). Setting ``use_delta_t=False`` reproduces
the no-conditioning ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .lsgc_layer import LSGCConv
from .rgcn_layer import RelationalLSGCConv


class LSGCForecaster(nn.Module):
    """LSGC backbone with three per-node forecasting heads.

    Parameters
    ----------
    in_channels, hidden, num_layers, conv_kwargs
        Same semantics as :class:`LSGCNet`. The winning encoder-validation
        configuration was ``num_layers=2`` with temporal skip edges
        ``(1, 2, 3)`` in the graph; that setup is carried forward as the
        default Stage-1 config.
    feat_out_dim
        Dimensionality of ``delta_feat``. Defaults to ``in_channels`` so
        the feature head predicts changes on the same channels the model
        sees at input.
    use_delta_t
        If True, concatenate a learned embedding of the scalar ``delta_t``
        (next visit offset in visit-index units) to every node's final
        hidden state before the heads.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 2,
        feat_out_dim: Optional[int] = None,
        use_delta_t: bool = True,
        conv_type: str = "lsgc",
        graph_ftv_readout: bool = False,
        graph_pcr_readout: bool = False,
        graph_ftv_affine: bool = False,
        graph_ftv_embed_affine: bool = False,
        graph_ftv_residual: bool = False,
        reaction_diffusion_head: bool = False,
        reaction_diffusion_context_dim: int = 0,
        reaction_diffusion_context_hidden: int = 32,
        core_in_channels: int | None = None,
        ssl_adapter_dim: int = 0,
        ssl_adapter_dropout: float = 0.10,
        ssl_adapter_zero_init: bool = True,
        burden_context_dim: int = 0,
        burden_context_hidden: int = 32,
        burden_num_experts: int = 0,
        burden_adapter_scale: float = 0.5,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.core_in_channels = int(core_in_channels) if core_in_channels is not None else in_channels
        if self.core_in_channels <= 0 or self.core_in_channels > in_channels:
            raise ValueError("core_in_channels must be in [1, in_channels]")
        self.ssl_in_channels = int(in_channels - self.core_in_channels)
        self.feat_out_dim = feat_out_dim if feat_out_dim is not None else in_channels
        self.use_delta_t = use_delta_t
        self.graph_ftv_readout = bool(graph_ftv_readout)
        self.graph_pcr_readout = bool(graph_pcr_readout)
        self.graph_ftv_affine = bool(graph_ftv_affine)
        self.graph_ftv_embed_affine = bool(graph_ftv_embed_affine)
        self.graph_ftv_residual = bool(graph_ftv_residual)
        self.reaction_diffusion_head_enabled = bool(reaction_diffusion_head)
        self.reaction_diffusion_context_dim = int(reaction_diffusion_context_dim)
        self.burden_context_dim = int(burden_context_dim)
        self.burden_num_experts = int(burden_num_experts)
        self.burden_adapter_scale = float(burden_adapter_scale)
        if conv_type not in {"lsgc", "relational"}:
            raise ValueError("conv_type must be 'lsgc' or 'relational'")
        self.conv_type = conv_type

        self.embed = nn.Linear(self.core_in_channels, hidden)
        self.ssl_adapter = None
        if ssl_adapter_dim > 0 and self.ssl_in_channels > 0:
            self.ssl_adapter = nn.Sequential(
                nn.LayerNorm(self.ssl_in_channels),
                nn.Linear(self.ssl_in_channels, int(ssl_adapter_dim)),
                nn.SiLU(),
                nn.Dropout(float(ssl_adapter_dropout)),
                nn.Linear(int(ssl_adapter_dim), hidden),
            )
            if ssl_adapter_zero_init:
                nn.init.zeros_(self.ssl_adapter[-1].weight)
                nn.init.zeros_(self.ssl_adapter[-1].bias)
        conv_cls = RelationalLSGCConv if conv_type == "relational" else LSGCConv
        self.convs = nn.ModuleList(
            [conv_cls(hidden, hidden, **conv_kwargs) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()

        if self.burden_context_dim > 0:
            ctx_hidden = int(max(1, burden_context_hidden))
            self.burden_context_mlp = nn.Sequential(
                nn.LayerNorm(self.burden_context_dim),
                nn.Linear(self.burden_context_dim, ctx_hidden),
                nn.SiLU(),
                nn.Linear(ctx_hidden, ctx_hidden),
                nn.SiLU(),
            )
            self.burden_film = nn.Linear(ctx_hidden, 2 * hidden)
            nn.init.zeros_(self.burden_film.weight)
            nn.init.zeros_(self.burden_film.bias)
            if self.burden_num_experts > 0:
                self.burden_gate = nn.Linear(ctx_hidden, self.burden_num_experts)
                self.burden_adapters = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(hidden, hidden),
                            nn.SiLU(),
                            nn.Linear(hidden, hidden),
                        )
                        for _ in range(self.burden_num_experts)
                    ]
                )
                for adapter in self.burden_adapters:
                    nn.init.zeros_(adapter[-1].weight)
                    nn.init.zeros_(adapter[-1].bias)
            else:
                self.burden_gate = None
                self.burden_adapters = None
        else:
            self.burden_context_mlp = None
            self.burden_film = None
            self.burden_gate = None
            self.burden_adapters = None

        # delta_t is a scalar in {1, 2, ...} of visit-index units; we embed
        # it through a tiny MLP and concatenate to every node's hidden.
        self.dt_dim = 8 if use_delta_t else 0
        if use_delta_t:
            self.dt_mlp = nn.Sequential(
                nn.Linear(1, self.dt_dim), nn.SiLU(), nn.Linear(self.dt_dim, self.dt_dim)
            )

        head_in = hidden + self.dt_dim

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
            )

        self.pos_head = _head(3)
        self.feat_head = _head(self.feat_out_dim)
        self.alive_head = _head(1)
        if self.graph_ftv_readout:
            self.graph_ftv_head = nn.Sequential(
                nn.Linear(2 * head_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.graph_ftv_head = None
        if self.graph_pcr_readout:
            self.graph_pcr_head = nn.Sequential(
                nn.Linear(2 * head_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.graph_pcr_head = None
        if self.graph_ftv_affine:
            self.graph_ftv_affine_head = nn.Sequential(
                nn.Linear(2 * head_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )
            with torch.no_grad():
                self.graph_ftv_affine_head[-1].weight.zero_()
                self.graph_ftv_affine_head[-1].bias[0] = 0.541324854612918
                self.graph_ftv_affine_head[-1].bias[1] = 0.0
        else:
            self.graph_ftv_affine_head = None
        if self.graph_ftv_embed_affine:
            if self.ssl_in_channels <= 0:
                raise ValueError("graph_ftv_embed_affine requires SSL/input channels beyond core_in_channels")
            self.graph_ftv_embed_affine_head = nn.Sequential(
                nn.LayerNorm(2 * self.ssl_in_channels),
                nn.Linear(2 * self.ssl_in_channels, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )
            with torch.no_grad():
                self.graph_ftv_embed_affine_head[-1].weight.zero_()
                self.graph_ftv_embed_affine_head[-1].bias.zero_()
        else:
            self.graph_ftv_embed_affine_head = None
        if self.graph_ftv_residual:
            self.graph_ftv_residual_head = nn.Sequential(
                nn.Linear(2 * head_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            with torch.no_grad():
                self.graph_ftv_residual_head[-1].weight.zero_()
                self.graph_ftv_residual_head[-1].bias.zero_()
        else:
            self.graph_ftv_residual_head = None
        if self.reaction_diffusion_head_enabled:
            rd_context_width = 0
            if self.reaction_diffusion_context_dim > 0:
                rd_context_width = int(max(1, reaction_diffusion_context_hidden))
                self.reaction_diffusion_context_mlp = nn.Sequential(
                    nn.LayerNorm(self.reaction_diffusion_context_dim),
                    nn.Linear(self.reaction_diffusion_context_dim, rd_context_width),
                    nn.SiLU(),
                    nn.Linear(rd_context_width, rd_context_width),
                    nn.SiLU(),
                )
            else:
                self.reaction_diffusion_context_mlp = None
            self.reaction_diffusion_head = nn.Sequential(
                nn.Linear(2 * head_in + rd_context_width, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 3),
            )
            with torch.no_grad():
                self.reaction_diffusion_head[-1].weight.zero_()
                self.reaction_diffusion_head[-1].bias[:] = torch.tensor([-4.0, -4.0, -2.0])
        else:
            self.reaction_diffusion_context_mlp = None
            self.reaction_diffusion_head = None

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        edge_type: Tensor | None = None,
        delta_t: float | Tensor = 1.0,
        burden_context: Tensor | None = None,
        reaction_diffusion_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the LSGC backbone and return per-node head outputs.

        ``delta_t`` is broadcast to every node. Scalars and tensors of
        shape ``(N,)`` are both accepted.
        """
        x_core = x[:, : self.core_in_channels]
        h = self.embed(x_core)
        if self.ssl_adapter is not None:
            h = h + self.ssl_adapter(x[:, self.core_in_channels:])
        for conv in self.convs:
            if self.conv_type == "relational":
                h = h + self.act(
                    conv(h, pos, t, edge_index, edge_attr=edge_attr, edge_type=edge_type)
                )
            else:
                h = h + self.act(conv(h, pos, t, edge_index, edge_attr=edge_attr))

        burden_gate = None
        if self.burden_context_mlp is not None and burden_context is not None:
            ctx = burden_context.to(device=x.device, dtype=x.dtype).reshape(1, -1)
            if ctx.shape[-1] != self.burden_context_dim:
                raise ValueError(
                    f"burden_context has width {ctx.shape[-1]}, expected {self.burden_context_dim}"
                )
            ctx_h = self.burden_context_mlp(ctx)
            gamma_beta = self.burden_film(ctx_h).reshape(2, -1)
            gamma, beta = gamma_beta[0], gamma_beta[1]
            h = h * (1.0 + gamma.reshape(1, -1)) + beta.reshape(1, -1)
            if self.burden_gate is not None and self.burden_adapters is not None:
                burden_gate = torch.softmax(self.burden_gate(ctx_h).reshape(-1), dim=0)
                adapted = torch.zeros_like(h)
                for weight, adapter in zip(burden_gate, self.burden_adapters):
                    adapted = adapted + weight * adapter(h)
                h = h + (self.burden_adapter_scale * adapted)

        if self.use_delta_t:
            if not torch.is_tensor(delta_t):
                dt = torch.full((x.size(0), 1), float(delta_t), device=x.device, dtype=x.dtype)
            else:
                dt = delta_t.view(-1, 1).to(x.dtype).to(x.device)
                if dt.shape[0] == 1:
                    dt = dt.expand(x.size(0), 1)
            dt_emb = self.dt_mlp(dt)
            hdt = torch.cat([h, dt_emb], dim=-1)
        else:
            hdt = h

        out = {
            "delta_pos": self.pos_head(hdt),
            "delta_feat": self.feat_head(hdt),
            "alive_logit": self.alive_head(hdt).squeeze(-1),
            "hidden": h,
        }
        if burden_gate is not None:
            out["burden_gate"] = burden_gate
        if (
            self.graph_ftv_head is not None
            or self.graph_pcr_head is not None
            or self.graph_ftv_affine_head is not None
            or self.graph_ftv_embed_affine_head is not None
            or self.graph_ftv_residual_head is not None
            or self.reaction_diffusion_head is not None
        ):
            pooled = torch.cat([hdt.mean(dim=0), hdt.max(dim=0).values], dim=0)
        if self.graph_ftv_head is not None:
            out["graph_ftv_log"] = self.graph_ftv_head(pooled).squeeze(-1)
        if self.graph_pcr_head is not None:
            out["graph_pcr_logit"] = self.graph_pcr_head(pooled).squeeze(-1)
        affine = None
        if self.graph_ftv_affine_head is not None:
            affine = self.graph_ftv_affine_head(pooled)
        if self.graph_ftv_embed_affine_head is not None:
            ssl_x = x[:, self.core_in_channels:]
            ssl_pooled = torch.cat([ssl_x.mean(dim=0), ssl_x.max(dim=0).values], dim=0)
            embed_delta = self.graph_ftv_embed_affine_head(ssl_pooled)
            if affine is None:
                affine = embed_delta.clone()
                affine[0] = affine[0] + 0.541324854612918
            else:
                affine = affine + embed_delta
        if affine is not None:
            out["graph_ftv_affine"] = affine
        if self.graph_ftv_residual_head is not None:
            out["graph_ftv_residual"] = self.graph_ftv_residual_head(pooled).squeeze(-1)
        if self.reaction_diffusion_head is not None:
            rd_pooled = pooled
            if self.reaction_diffusion_context_mlp is not None:
                if reaction_diffusion_context is None:
                    rd_ctx = torch.zeros(
                        (1, self.reaction_diffusion_context_dim),
                        device=x.device,
                        dtype=x.dtype,
                    )
                else:
                    rd_ctx = reaction_diffusion_context.to(device=x.device, dtype=x.dtype).reshape(1, -1)
                    if rd_ctx.shape[-1] != self.reaction_diffusion_context_dim:
                        raise ValueError(
                            "reaction_diffusion_context has width "
                            f"{rd_ctx.shape[-1]}, expected {self.reaction_diffusion_context_dim}"
                        )
                rd_ctx_h = self.reaction_diffusion_context_mlp(rd_ctx).reshape(-1)
                rd_pooled = torch.cat([pooled, rd_ctx_h], dim=0)
            out["reaction_diffusion_raw"] = self.reaction_diffusion_head(rd_pooled)
        return out
