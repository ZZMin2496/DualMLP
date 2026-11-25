from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import AUX_DIM
from .structure_features import EDGE_ATTR_DIM, EDGE_QUALITY_DIM, NODE_QUALITY_DIM
from .protein_features import ESMC_DIM
from .chemical_features import CHEM_TOTAL_DIM
from .reaction_features import RXNFP_DIM

PROJ_DIM = 256
FUSION_DIM = 1344


def _mlp_block(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden)),
                nn.LayerNorm(int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = int(hidden)
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))


class _GraphLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_input: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        message_input = torch.cat([h[src], edge_input], dim=-1)
        messages = self.message(message_input)
        if edge_weight is not None:
            messages = messages * edge_weight.to(messages.dtype)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        updated = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + updated)


class _StructureUpdate(nn.Module):
    def __init__(self, node_dim: int, edge_attr_dim: int, edge_quality_dim: int, hidden_dim: int, n_layers: int, dropout: float, quality_control: bool) -> None:
        super().__init__()
        self.quality_control = bool(quality_control)
        self.edge_dim = edge_attr_dim + (edge_quality_dim if self.quality_control else 0)
        self.layers = nn.ModuleList([
            _GraphLayer(node_dim, self.edge_dim, hidden_dim, dropout) for _ in range(int(n_layers))
        ])
        if self.quality_control:
            self.gate = nn.Sequential(
                nn.Linear(node_dim * 2 + NODE_QUALITY_DIM, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim),
            )

    def forward(
        self,
        h_padded: torch.Tensor,
        residue_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_quality: torch.Tensor,
        node_quality_padded: torch.Tensor,
        coordinate_mask_padded: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h_padded
        h_flat = h_padded[residue_mask]
        if h_flat.numel() == 0:
            return h_padded
        node_quality_flat = node_quality_padded[residue_mask]
        coordinate_mask_flat = coordinate_mask_padded[residue_mask].float().unsqueeze(-1)
        if self.quality_control and edge_quality.numel():
            edge_input = torch.cat([edge_attr, edge_quality], dim=-1)
            endpoint_quality = edge_quality[:, 0]
            endpoint_available = edge_quality[:, 1]
            pae_quality = edge_quality[:, 2]
            pae_available = edge_quality[:, 3]
            # Missing endpoint/PAE confidence is neutral rather than fabricated as low quality.
            endpoint_weight = torch.where(endpoint_available > 0, endpoint_quality, torch.ones_like(endpoint_quality))
            pae_weight = torch.where(pae_available > 0, pae_quality, torch.ones_like(pae_quality))
            edge_weight = (endpoint_weight * pae_weight).clamp(0.0, 1.0).unsqueeze(-1)
        else:
            edge_input = edge_attr
            edge_weight = torch.ones(edge_attr.shape[0], 1, device=edge_attr.device, dtype=edge_attr.dtype)
        h = h_flat
        for layer in self.layers:
            h = layer(h, edge_index, edge_input, edge_weight=edge_weight)
        delta = h - h_flat
        if self.quality_control:
            gate_input = torch.cat([h_flat, delta, node_quality_flat], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            updated_flat = h_flat + coordinate_mask_flat * gate * delta
        else:
            updated_flat = h_flat + coordinate_mask_flat * delta
        h_g = h_padded.clone()
        h_g[residue_mask] = updated_flat
        return h_g


class _CrossAttentionReadout(nn.Module):
    def __init__(self, dim: int, n_queries: int, num_heads: int, dropout: float, conditioned: bool) -> None:
        super().__init__()
        self.n_queries = int(n_queries)
        self.conditioned = bool(conditioned)
        self.query_seed = nn.Parameter(torch.randn(self.n_queries, dim) * 0.02)
        cond_dim = PROJ_DIM * 2 + 1
        self.condition = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        ) if self.conditioned else None
        self.query_projections = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.n_queries)])
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.cross_ffn_norm = nn.LayerNorm(dim)
        self.query_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.query_ffn_norm = nn.LayerNorm(dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(self.n_queries * dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, s: torch.Tensor, r: torch.Tensor, reaction_available: torch.Tensor, h_g: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        batch_size = s.shape[0]
        if self.conditioned and self.condition is not None:
            cond = torch.cat([s, r, reaction_available.float().unsqueeze(-1)], dim=-1)
            u = self.condition(cond)
            queries = self.query_seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
            for k, projection in enumerate(self.query_projections):
                queries[:, k, :] = queries[:, k, :] + projection(u)
        else:
            queries = self.query_seed.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = ~residue_mask
        attn_out, _ = self.cross_attn(queries, h_g, h_g, key_padding_mask=key_padding_mask, need_weights=False)
        z = self.cross_norm(queries + attn_out)
        z = self.cross_ffn_norm(z + self.cross_ffn(z))
        sa_out, _ = self.query_attn(z, z, z, need_weights=False)
        z = self.query_norm(z + sa_out)
        z = self.query_ffn_norm(z + self.query_ffn(z))
        return self.local_mlp(z.flatten(1))


class KineticsModel(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.config = model_config
        self.proj_dim = int(model_config.get("proj_dim", PROJ_DIM))
        self.n_queries = int(model_config.get("n_queries", 4))
        self.num_heads = int(model_config.get("num_heads", 4))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.graph_mode = str(model_config.get("graph_mode", "none"))  # none | uncontrolled | quality
        self.n_graph_layers = int(model_config.get("graph_layers", 2))
        self.conditioned_queries = bool(model_config.get("conditioned_queries", True))
        self.use_global_branch = bool(model_config.get("use_global_branch", True))
        self.use_reaction = bool(model_config.get("use_reaction", True))
        self.use_structure = self.graph_mode != "none"

        self.protein_proj = nn.Linear(ESMC_DIM, self.proj_dim)
        self.protein_norm = nn.LayerNorm(self.proj_dim)
        self.chemical_proj = nn.Linear(CHEM_TOTAL_DIM, self.proj_dim)
        self.chemical_norm = nn.LayerNorm(self.proj_dim)
        self.reaction_proj = nn.Linear(RXNFP_DIM, self.proj_dim)
        self.reaction_norm = nn.LayerNorm(self.proj_dim)

        if self.use_structure:
            quality_control = self.graph_mode == "quality"
            self.structure_update = _StructureUpdate(
                node_dim=self.proj_dim,
                edge_attr_dim=EDGE_ATTR_DIM,
                edge_quality_dim=EDGE_QUALITY_DIM,
                hidden_dim=int(model_config.get("graph_hidden_dim", self.proj_dim)),
                n_layers=self.n_graph_layers,
                dropout=self.dropout,
                quality_control=quality_control,
            )
        self.readout = _CrossAttentionReadout(
            dim=self.proj_dim,
            n_queries=self.n_queries,
            num_heads=self.num_heads,
            dropout=self.dropout,
            conditioned=self.conditioned_queries,
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(AUX_DIM, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
        )
        fusion_in = self.proj_dim * 5 + 64
        if fusion_in != FUSION_DIM:
            raise ValueError(f"Fusion input dim {fusion_in} != {FUSION_DIM}")
        hidden_dims = model_config.get("head_hidden_dims", [512, 256, 128])
        layers = []
        in_dim = fusion_in
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden)),
                nn.LayerNorm(int(hidden)),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = int(hidden)
        self.fusion = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        residue_embeddings = batch["residue_embeddings"].float()
        residue_mask = batch["residue_mask"].bool()
        chemical = batch["chemical"].float()
        reaction = batch["reaction"].float()
        chemical_available = batch["chemical_available"].float().unsqueeze(-1)
        reaction_available = batch["reaction_available"].float().unsqueeze(-1)
        if not self.use_reaction:
            reaction = torch.zeros_like(reaction)
            reaction_available = torch.zeros_like(reaction_available)

        h = self.protein_norm(self.protein_proj(residue_embeddings))
        # Masked mean over real residues only.
        mask_float = residue_mask.unsqueeze(-1).to(h.dtype)
        denom = mask_float.sum(dim=1).clamp_min(1.0)
        p_global = (h * mask_float).sum(dim=1) / denom
        if not self.use_global_branch:
            p_global = torch.zeros_like(p_global)

        s = self.chemical_norm(self.chemical_proj(chemical))
        s = s * chemical_available
        r = self.reaction_norm(self.reaction_proj(reaction))
        r = r * reaction_available

        if self.use_structure:
            h_g = self.structure_update(
                h,
                residue_mask,
                batch["graph_edge_index"],
                batch["graph_edge_attr"].float(),
                batch["graph_edge_quality"].float(),
                batch["node_quality"].float(),
                batch["coordinate_mask"].bool(),
            )
        else:
            h_g = h
        p_local = self.readout(s, r, reaction_available.squeeze(-1), h_g, residue_mask)
        c = self.aux_mlp(batch["aux"].float())
        z = torch.cat([p_global, p_local, s, r, p_local * s, c], dim=-1)
        return self.output(self.fusion(z))


def baseline_input(batch: dict[str, torch.Tensor], variant: str) -> torch.Tensor:
    protein = batch["protein_global"].float()
    chemical = batch["chemical"].float()
    if variant == "b0":
        return torch.cat([protein, chemical], dim=-1)
    if variant == "b1":
        reaction = batch["reaction"].float()
        flags = torch.cat([
            batch["chemical_available"].float().unsqueeze(-1),
            batch["reaction_available"].float().unsqueeze(-1),
        ], dim=-1)
        return torch.cat([protein, chemical, reaction, flags], dim=-1)
    raise ValueError(f"Unsupported baseline variant: {variant}")


def build_model(model_config: dict[str, Any]) -> nn.Module:
    architecture = str(model_config.get("architecture", "fusion"))
    if architecture == "baseline":
        variant = str(model_config.get("variant", "b0"))
        if variant == "b0":
            input_dim = ESMC_DIM + CHEM_TOTAL_DIM
        elif variant == "b1":
            input_dim = ESMC_DIM + CHEM_TOTAL_DIM + RXNFP_DIM + 2
        else:
            raise ValueError(f"Unsupported baseline variant: {variant}")
        return BaselineMLP(input_dim=input_dim, hidden_dims=model_config.get("hidden_dims", [512, 256, 128]), dropout=float(model_config.get("dropout", 0.1)))
    if architecture == "fusion":
        return KineticsModel(model_config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
