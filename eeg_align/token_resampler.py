"""Four text-free readouts. Q-Former-style, not BLIP-2 pretrained weights."""
import torch
from torch import nn

KINDS = ('mean_pool', 'cls', 'cross_attention', 'qformer')


class QueryBlock(nn.Module):
    def __init__(self, width, heads, self_attention):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True) if self_attention else None
        self.self_norm = nn.LayerNorm(width) if self_attention else None
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, q, memory, valid):
        if self.self_attn is not None:
            z = self.self_norm(q)
            q = q + self.self_attn(z, z, z, need_weights=False)[0]
        q = q + self.cross_attn(self.cross_norm(q), memory, memory,
                              key_padding_mask=~valid, need_weights=False)[0]
        return q + self.ff(self.ff_norm(q))


class TokenReadout(nn.Module):
    def __init__(self, kind, width=768, heads=12, queries=32, depth=4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.width = kind, width
        self.output_tokens = queries if kind in {'cross_attention', 'qformer'} else 1
        if kind != 'mean_pool':
            self.query = nn.Parameter(torch.randn(1, self.output_tokens, width) * .02)
            self.input_norm = nn.LayerNorm(width)
            self.output_norm = nn.LayerNorm(width)
            if kind == 'cls':
                self.cls_block = nn.TransformerEncoderLayer(width, heads, 4 * width,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True)
            else:
                self.blocks = nn.ModuleList([QueryBlock(width, heads, kind == 'qformer')
                    for _ in range(depth if kind == 'qformer' else 1)])

    def forward(self, tokens, valid):
        if tokens.ndim != 3 or tokens.shape[-1] != self.width or valid.shape != tokens.shape[:2]:
            raise ValueError('Invalid token shape')
        if valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError('Invalid token mask')
        memory = tokens.masked_fill(~valid[..., None], 0).float()
        if not torch.isfinite(memory).all():
            raise ValueError('Nonfinite valid tokens')
        if self.kind == 'mean_pool':
            output = memory.sum(1, keepdim=True) / valid.sum(1)[:, None, None]
        else:
            memory = self.input_norm(memory)
            q = self.query.expand(len(tokens), -1, -1)
            if self.kind == 'cls':
                seq = torch.cat((q, memory), dim=1)
                mask = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
                # Newly learned global CLS; patch zero is never mistaken for CLS.
                output = self.cls_block(seq, src_key_padding_mask=~mask)[:, :1]
            else:
                for block in self.blocks:
                    q = block(q, memory, valid)
                output = q
            output = self.output_norm(output)
        return output, torch.ones(output.shape[:2], device=tokens.device, dtype=torch.bool)
