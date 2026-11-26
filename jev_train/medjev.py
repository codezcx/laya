"""LEAF-token interface to the pinned MedJev depth reduction and decision readout.

No image encoder, vocabulary head, autoregressive generation, or medical checkpoint.
The complete language blocks are 1..23; matching Q/K and norms come from block 32.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, apply_rotary_pos_emb
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

from vendor.MedJev.medjev.model import MARKERS, MARKER_INIT_TOKEN, MarkerEmbedding
from vendor.MedJev.medjev.readout import DecisionReadout, ResidualReadout
from .runtime import CONFIG, backbone_files


class LoRALinear(nn.Module):
    """Ordinary LoRA: identical scaling/initialization to PEFT's default Linear."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
                               * self.scale).to(x.dtype)


def inject_lora(root, rank, alpha):
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    selected = [(n, m) for n, m in root.named_modules()
                if isinstance(m, nn.Linear) and n.split('.')[-1] in targets]
    for name, module in selected:
        parent, _, key = name.rpartition('.')
        setattr(root.get_submodule(parent), key, LoRALinear(module, rank, alpha))
    return [n for n, _ in selected]


class BatchedResidualReadout(ResidualReadout):
    """Same upstream equations, with explicit per-row marker positions and masks."""
    def forward(self, hidden, position_embeddings, options, decisions, option_mask=None):
        batch, length, _ = hidden.shape
        if options.ndim != 2 or decisions.shape != (batch,) or options.shape[0] != batch:
            raise ValueError('Invalid batched marker positions')
        if option_mask is None:
            option_mask = torch.ones_like(options, dtype=torch.bool)
        if option_mask.shape != options.shape or not (option_mask.sum(-1) >= 2).all():
            raise ValueError('Each row needs at least two candidates')
        if ((decisions < 0) | (decisions >= length)).any():
            raise ValueError('Decision marker outside sequence')
        if (((options < 0) | (options >= decisions[:, None])) & option_mask).any():
            raise ValueError('Option markers must precede decision')
        safe = options.masked_fill(~option_mask, 0)
        index = torch.arange(batch, device=hidden.device)[:, None]
        hd = self.input_layernorm(hidden[index, decisions[:, None]])
        ho = self.input_layernorm(hidden[index, safe])
        q = self.q_norm(self.q_proj(hd).reshape(batch, 1, self.H, self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(ho).reshape(batch, options.shape[1], self.G, self.d)).transpose(1, 2)
        cos, sin = position_embeddings
        qcos, qsin = cos[index, decisions[:, None]], sin[index, decisions[:, None]]
        kcos, ksin = cos[index, safe], sin[index, safe]
        q = apply_rotary_pos_emb(q, q, qcos, qsin)[0]
        k = apply_rotary_pos_emb(k, k, kcos, ksin)[0].repeat_interleave(self.H // self.G, dim=1)
        qk = (q.float() * k.float()).sum(-1).mean(1) / math.sqrt(self.d)
        hd = hd.expand_as(ho)
        features = torch.cat((hd, ho, hd * ho), -1).float()
        residual = self.head_mlp(features).squeeze(-1).float()
        return (qk + residual).masked_fill(~option_mask, -torch.inf)


class MedJevBridge(nn.Module):
    def __init__(self, text, tokenizer, readout, marker_ids, audit):
        super().__init__()
        self.text, self.tokenizer, self.readout = text, tokenizer, readout
        self.marker_ids = tuple(marker_ids)
        device = next(text.parameters()).device
        dim = text.config.hidden_size
        self.connector = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, dim),
                                       nn.GELU(), nn.Linear(dim, dim)).to(device)
        # Apply the author's chat envelope, replacing only the image content by EEG tokens.
        placeholder = 'EEG_SIGNAL_SLOT'
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': placeholder}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if chat.count(placeholder) != 1:
            raise ValueError('Chat template placeholder is not unique')
        prefix, self.chat_tail = chat.split(placeholder)
        prefix += 'EEG recording starts.\n'
        self.register_buffer('prefix_ids', torch.tensor(tokenizer.encode(prefix, add_special_tokens=False),
                                                        device=device, dtype=torch.long))
        self.prompt_cache = {}
        self.requires_grad_(False)
        self.connector.requires_grad_(True)
        targets = inject_lora(self.text, CONFIG['lora_rank'], CONFIG['lora_alpha'])
        targets += ['readout.' + n for n in inject_lora(self.readout, CONFIG['lora_rank'], CONFIG['lora_alpha'])]
        self.text.embed_tokens.markers.requires_grad_(True)
        for module in (self.readout.head_mlp, self.readout.input_layernorm,
                       self.readout.q_norm, self.readout.k_norm):
            module.requires_grad_(True).float()
        self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.architecture_audit = {**audit, 'lora_targets': targets, 'lora_rank': CONFIG['lora_rank'],
            'lora_alpha': CONFIG['lora_alpha'], 'vision_path_loaded': False,
            'medical_checkpoint_loaded': False, 'generation_tokens': 0,
            'parameters_without_leaf': sum(p.numel() for p in self.parameters())}

    def parameter_groups(self):
        lora, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad or name.startswith(('connector.', 'eeg_encoder.')):
                continue
            (lora if name.endswith(('lora_A', 'lora_B')) else head).append(p)
        return [dict(name='connector', params=list(self.connector.parameters())),
                dict(name='jev_lora', params=lora), dict(name='decision_head', params=head)]

    def forward(self, tokens, valid, prompts):
        if valid.dtype != torch.bool or valid.shape != tokens.shape[:2] or not valid.any(-1).all():
            raise ValueError('Invalid EEG token mask')
        if len(prompts) != len(tokens):
            raise ValueError('Prompt and signal batch lengths differ')
        embed = self.text.get_input_embeddings()
        z = self.connector(tokens.masked_fill(~valid[..., None], 0).float())
        prefix = embed(self.prefix_ids)
        sequences, options, decisions = [], [], []
        for i, prompt in enumerate(prompts):
            suffix = prompt['suffix'] + self.chat_tail
            if suffix not in self.prompt_cache:
                self.prompt_cache[suffix] = torch.tensor(self.tokenizer.encode(suffix, add_special_tokens=False),
                                                        device=z.device, dtype=torch.long)
            ids = self.prompt_cache[suffix]
            op = (ids == self.marker_ids[0]).nonzero().flatten()
            dec = (ids == self.marker_ids[1]).nonzero().flatten()
            question = (ids == self.marker_ids[2]).nonzero().flatten()
            if (len(op) != prompt['num_choices'] or len(dec) != 1 or len(question) != 1
                    or not (question[0] < op.min() <= op.max() < dec[0])):
                raise ValueError('Invalid question/option/decision markers')
            eeg = z[i, valid[i]].to(prefix.dtype)
            offset = len(prefix) + len(eeg)
            seq = torch.cat((prefix, eeg, embed(ids)))
            if len(seq) > CONFIG['max_input_tokens']:
                raise ValueError('Refusing to truncate EEG or candidate text')
            sequences.append(seq)
            options.append(op + offset)
            decisions.append(dec[0] + offset)
        inputs = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(s) for s in sequences], device=z.device)
        attention = torch.arange(inputs.shape[1], device=z.device)[None] < lengths[:, None]
        positions = (attention.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.text(inputs_embeds=inputs, attention_mask=attention, position_ids=positions,
                           use_cache=False, return_dict=True).last_hidden_state
        rotary = self.text.rotary_emb(hidden, positions[None].expand(3, -1, -1))
        option_positions = pad_sequence(options, batch_first=True, padding_value=-1)
        return self.readout(hidden, rotary, option_positions, torch.stack(decisions), option_positions >= 0)


def load_medjev(device='cuda:0'):
    root = Path(CONFIG['backbone_path'])
    backbone_files()  # Includes download-completeness checks; never fetch during training.
    cfg = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    if cfg.hidden_size != 2560 or cfg.num_hidden_layers != 32:
        raise ValueError('Expected the original Qwen3.5-4B 32-layer language backbone')
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'
    with init_empty_weights():
        text = Qwen3_5TextModel(cfg)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    expected = set(dict(text.named_parameters()))
    loaded = set()
    for shard in sorted(set(index.values())):
        with safe_open(root / shard, framework='pt', device='cpu') as f:
            for full_name in f.keys():
                if not full_name.startswith('model.language_model.'):
                    continue
                name = full_name.removeprefix('model.language_model.')
                if name not in expected:
                    raise ValueError(f'Unexpected language weight: {name}')
                set_module_tensor_to_device(text, name, device, value=f.get_tensor(full_name), dtype=torch.bfloat16)
                loaded.add(name)
    if loaded != expected or any(p.is_meta for p in text.parameters()):
        raise ValueError(f'Incomplete language weights: {expected - loaded}')
    # Rotary buffers are nonpersistent and are constructed from the original config.
    for name, buffer in text.named_buffers():
        set_module_tensor_to_device(text, name, device, value=buffer)
    original_parameters = sum(p.numel() for p in text.parameters())
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if tokenizer.add_special_tokens({'additional_special_tokens': list(MARKERS)}) != len(MARKERS):
        raise ValueError('Unexpected preexisting MedJev markers')
    marker_ids = [tokenizer.convert_tokens_to_ids(m) for m in MARKERS]
    for marker, token_id in zip(MARKERS, marker_ids):
        if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
            raise ValueError('Decision markers must be single tokens')
    text.embed_tokens = MarkerEmbedding(text.embed_tokens, marker_ids,
                                        tokenizer.convert_tokens_to_ids(MARKER_INIT_TOKEN))
    matching = DecisionReadout(text.layers[31], cfg)
    readout = BatchedResidualReadout(matching, CONFIG['head_hidden_width'], CONFIG['seed'])
    text.layers = nn.ModuleList(list(text.layers)[:CONFIG['retained_layers']])
    text.config.num_hidden_layers = CONFIG['retained_layers']
    text.config.layer_types = list(text.config.layer_types)[:CONFIG['retained_layers']]
    # The author's matching layer reads pre-final-normalization hidden states.
    text.norm = nn.Identity()
    audit = dict(backbone='Qwen/Qwen3.5-4B', original_text_parameters=original_parameters,
        retained_complete_layers=CONFIG['retained_layers'], matching_layer_one_based=32,
        hidden_size=cfg.hidden_size, markers=list(MARKERS), marker_ids=marker_ids,
        upstream_commit=CONFIG['medjev_commit'], attention_implementation='sdpa',
        removed_final_language_norm=True)
    return MedJevBridge(text, tokenizer, readout, marker_ids, audit), tokenizer
