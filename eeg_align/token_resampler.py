"""Trainable fixed-query token compression, independent of the EEG encoder.

These are randomly initialized modules, not pretrained BLIP Q-Former weights.
The encoder must supply position-aware, pre-pooling tokens. No text or labels
are used to choose the queries. True in ``valid`` denotes an observed token.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ResamplerConfig:
    kind: str = "cross_attention"
    input_dim: int = 768
    hidden_dim: int = 768
    num_queries: int = 32
    num_heads: int = 12
    depth: int | None = None
    ff_multiplier: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if self.kind not in ("cross_attention", "qformer"):
            raise ValueError(f"Unknown resampler kind: {self.kind}")
        if self.depth is None:
            object.__setattr__(self, "depth", 1 if self.kind == "cross_attention" else 4)
        for name in ("input_dim", "hidden_dim", "num_queries", "num_heads", "depth", "ff_multiplier"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.kind == "cross_attention" and self.depth != 1:
            raise ValueError("The cross_attention control has exactly one layer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_json(cls, path: str | Path, kind: str):
        recipe = json.loads(Path(path).read_text())
        if kind not in recipe["variants"]:
            raise ValueError(f"No recipe for resampler: {kind}")
        return cls(**(recipe["common"] | recipe["variants"][kind] | {"kind": kind}))

    def to_dict(self):
        return asdict(self)


class _QueryBlock(nn.Module):
    def __init__(self, config: ResamplerConfig):
        super().__init__()
        d = config.hidden_dim
        if config.kind == "qformer":
            self.self_norm = nn.LayerNorm(d)
            self.self_attention = nn.MultiheadAttention(
                d, config.num_heads, dropout=config.dropout, batch_first=True
            )
        else:
            self.self_attention = None
        self.cross_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * config.ff_multiplier, d),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, queries, memory, padding):
        if self.self_attention is not None:
            q = self.self_norm(queries)
            update = self.self_attention(q, q, q, need_weights=False)[0]
            queries = queries + self.residual_dropout(update)
        update = self.cross_attention(
            self.cross_norm(queries), memory, memory,
            key_padding_mask=padding, need_weights=False,
        )[0]
        queries = queries + self.residual_dropout(update)
        return queries + self.residual_dropout(self.ff(self.ff_norm(queries)))


class TokenResampler(nn.Module):
    """[B, N, input_dim] -> ([B, Q, hidden_dim], bool [B, Q]).

    ``cross_attention``: one query-to-memory attention + FFN block.
    ``qformer``: query self-attention + cross-attention + FFN in each block.
    The latter is a Q-Former-style resampler, not the full BLIP-2 architecture
    or its pretraining objectives. Both variants use all unmasked input tokens.
    With N < Q this is resampling/expansion, not token compression.
    """

    def __init__(self, config: ResamplerConfig | None = None):
        super().__init__()
        self.config = config or ResamplerConfig()
        c = self.config
        self.queries = nn.Parameter(torch.empty(1, c.num_queries, c.hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.input_norm = nn.LayerNorm(c.input_dim)
        self.input_projection = (
            nn.Identity() if c.input_dim == c.hidden_dim
            else nn.Linear(c.input_dim, c.hidden_dim)
        )
        self.blocks = nn.ModuleList([_QueryBlock(c) for _ in range(c.depth)])
        self.output_norm = nn.LayerNorm(c.hidden_dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor | None = None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected [B, N, {self.config.input_dim}] tokens")
        if not tokens.is_floating_point() or min(tokens.shape[:2]) < 1:
            raise ValueError("Tokens must be floating point with nonempty batch and sequence")
        if tokens.device != self.queries.device:
            raise ValueError("Tokens and resampler must be on the same device")
        if valid is None:
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
            raise ValueError("valid must be a bool [B, N] mask; True means observed")
        if valid.device != tokens.device:
            raise ValueError("Tokens and mask must be on the same device")
        if not valid.any(dim=1).all():
            raise ValueError("Each sample needs at least one valid input token")
        # Neutralize even NaN/Inf padding *before* normalization and projection.
        memory = tokens.masked_fill(~valid[..., None], 0).to(self.queries.dtype)
        if not torch.isfinite(memory).all():
            raise ValueError("Valid input tokens must be finite")
        memory = self.input_projection(self.input_norm(memory))
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, ~valid)
        compressed = self.output_norm(queries)
        compressed_valid = torch.ones(
            compressed.shape[:2], dtype=torch.bool, device=compressed.device
        )
        return compressed, compressed_valid


class ResampledDecisionModel(nn.Module):
    """Register both modules so optimizers/checkpoints can include the resampler.

    ``decision_model`` can be JEVBridge with input_dim=config.hidden_dim.
    The current trainer registers this module directly on its JEVBridge.
    """

    def __init__(self, resampler: TokenResampler, decision_model: nn.Module):
        super().__init__()
        self.resampler = resampler
        self.decision_model = decision_model

    def forward(self, tokens, valid=None, **decision_kwargs):
        compressed, compressed_valid = self.resampler(tokens, valid)
        return self.decision_model(compressed, compressed_valid, **decision_kwargs)
