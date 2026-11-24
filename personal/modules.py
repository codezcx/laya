"""Small, independent user parameters; shared model weights are never optimized."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dim=256, width=64):
        super().__init__()
        self.down = nn.Linear(dim, width)
        self.up = nn.Linear(width, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class ConnectorDelta(nn.Module):
    def __init__(self, dim=5120, rank=4, scale=1.):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.a), self.b) * self.scale


class KVPrefix(nn.Module):
    def __init__(self, layers, tokens=5, heads=4, dim=256):
        super().__init__()
        self.keys = nn.ParameterDict({str(i): nn.Parameter(torch.randn(heads, tokens, dim) * .02) for i in layers})
        self.values = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(heads, tokens, dim)) for i in layers})


class PrefixAttention(nn.Module):
    """Explicit KV-prefix for full attention only; original path is exact when off.

    Prefix keys are learned directly in post-RoPE attention coordinates. Existing
    sequence positions do not shift. All real tokens may attend these virtual
    entries. No decoding cache is supported or used in the one-forward classifier.
    """
    def __init__(self, base, layer):
        super().__init__()
        self.base = base
        self.layer = str(layer)
        object.__setattr__(self, 'active_prefix', None)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        if self.active_prefix is None:
            return self.base(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        if past_key_values is not None:
            raise ValueError('Personal prefix supports uncached full-sequence classification only')
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
        base = self.base
        shape = hidden_states.shape[:-1]
        hshape = (*shape, -1, base.head_dim)
        q, gate = torch.chunk(base.q_proj(hidden_states).view(*shape, -1, base.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        q = base.q_norm(q.view(hshape)).transpose(1, 2)
        k = base.k_norm(base.k_proj(hidden_states).view(hshape)).transpose(1, 2)
        v = base.v_proj(hidden_states).view(hshape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        pk = self.active_prefix.keys[self.layer].to(k.dtype)[None].expand(shape[0], -1, -1, -1)
        pv = self.active_prefix.values[self.layer].to(v.dtype)[None].expand(shape[0], -1, -1, -1)
        length = pk.shape[-2]
        k = repeat_kv(torch.cat([pk, k], dim=-2), base.num_key_value_groups)
        v = repeat_kv(torch.cat([pv, v], dim=-2), base.num_key_value_groups)
        if attention_mask is None:
            real_mask = torch.ones(shape[1], shape[1], dtype=torch.bool, device=q.device).tril()[None, None]
            mask = torch.cat([torch.ones(1, 1, shape[1], length, dtype=torch.bool, device=q.device), real_mask], dim=-1)
        else:
            prefix_shape = (*attention_mask.shape[:-1], length)
            pm = torch.ones(prefix_shape, dtype=torch.bool, device=q.device) if attention_mask.dtype == torch.bool else torch.zeros(prefix_shape, dtype=attention_mask.dtype, device=q.device)
            mask = torch.cat([pm, attention_mask], dim=-1)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=0., is_causal=False, scale=base.scaling)
        output = output.transpose(1, 2).reshape(*shape, -1).contiguous() * torch.sigmoid(gate)
        return base.o_proj(output), None


class UserParameters(nn.Module):
    def __init__(self, arm, prompt_init, config, prefix_heads=4, prefix_dim=256):
        super().__init__()
        self.arm = arm
        if arm in ('residual', 'residual_kd', 'soft_prompt_residual'):
            self.residual = ResidualAdapter(width=config['adapter_width'])
        if arm in ('soft_prompt', 'soft_prompt_kd', 'soft_prompt_residual'):
            self.prompt = nn.Parameter(prompt_init.detach().float().clone())
        if arm == 'connector_lora':
            self.delta = ConnectorDelta(prompt_init.shape[-1], config['connector_lora_rank'], config['connector_lora_scale'])
        if arm == 'kv_prefix':
            self.prefix = KVPrefix(config['prefix_layers'], config['prefix_tokens'], prefix_heads, prefix_dim)


def canonical_logits(logits, option_orders):
    """Different textual option orders must be compared in semantic class order."""
    out = torch.empty_like(logits)
    order = torch.tensor(option_orders, dtype=torch.long, device=logits.device)
    return out.scatter(1, order, logits)


def support_loss(logits, labels, teacher=None, teacher_correct=None, kd_weight=.1, temperature=1.):
    ce = F.cross_entropy(logits, labels)
    kd = logits.new_zeros(())
    if teacher is not None:
        if teacher.requires_grad:
            raise ValueError('Teacher must be detached')
        kl = F.kl_div(F.log_softmax(logits / temperature, -1), teacher,
                      reduction='none').sum(-1) * temperature ** 2
        # Mean over the full microbatch preserves accumulation scaling and avoids
        # amplifying a single teacher-approved example to represent all examples.
        kd = (kl * teacher_correct.to(kl.dtype)).mean()
    return ce + kd_weight * kd, ce, kd
