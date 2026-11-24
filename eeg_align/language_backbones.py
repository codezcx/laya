"""Native small-Qwen matched control for the existing EEG decision trainer.

No visual model or remote Python is loaded. JEV remains the default backbone;
its trained LoRA/decision head are never transplanted into a different model.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoTokenizer

from .jev import CHECKPOINT, JEVBridge, LoadedLoRA, load_jev
from .runtime import digest

CONFIG_PATH = Path('configs/language_backbones.json')


def build_decision_bridge(text, weight, bias, tokenizer, **kwargs):
    """Keep the hash-pinned JEV implementation; resize only the native control.

    Reuse its exact forward, checkpointing and trainability rules. Discard the
    temporary 5120-wide bridge without consuming the intended initialization RNG.
    """
    width = text.config.hidden_size
    if weight.ndim != 2 or weight.shape[1] != width or bias.shape != (len(weight),):
        raise ValueError('Decision head does not match language hidden size')
    if width == 5120:
        return JEVBridge(text, weight, bias, tokenizer, **kwargs)
    with torch.random.fork_rng(devices=[]):
        bridge = JEVBridge(text, weight, bias, tokenizer, **kwargs)
    dim = kwargs.get('input_dim', 512)
    bridge.connector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width),
        nn.GELU(), nn.Linear(width, width)).to(weight.device)
    return bridge


def backbone_spec(name='jev'):
    if name == 'jev':
        return dict(kind='jev', path=str(CHECKPOINT))
    profiles = json.loads(CONFIG_PATH.read_text())
    if name not in profiles:
        raise ValueError(f'Unknown language backbone: {name}')
    return profiles[name]


def backbone_files(spec):
    root = Path(spec['path'])
    if spec['kind'] == 'jev':
        names = ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'adapter_vllm/adapter_config.json',
                 'adapter_vllm/decision_head.json', 'adapter_vllm/adapter_model.safetensors']
        names += sorted({*json.loads((root/'model.safetensors.index.json').read_text())['weight_map'].values()})
    else:
        names = sorted(set(spec['sha256']) | set(spec['git_blob_sha1']))
    return [root/n for n in names]


def verify_native_checkpoint(spec, weights=True):
    """Verify official Hub content, not just an inherited model name in config."""
    if spec['kind'] != 'qwen3_vl_native':
        raise ValueError('Native Qwen profile required')
    root = Path(spec['path'])
    checked = {}
    for name, expected in spec['git_blob_sha1'].items():
        data = (root/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise ValueError(f'Official Qwen metadata hash mismatch: {name}')
        checked[name] = actual
    mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    if set(mapping.values()) != set(spec['sha256']):
        raise ValueError('Unpinned Qwen weight shard')
    if weights:
        for name, expected in spec['sha256'].items():
            actual = digest(root/name)
            if actual != expected:
                raise ValueError(f'Official Qwen weight hash mismatch: {name}')
            checked[name] = actual
    return checked


def add_fresh_lora(text, rank, alpha, targets):
    """Same zero-dropout r/alpha convention as JEV, but no pretrained delta."""
    if rank <= 0 or alpha <= 0 or not targets or len(set(targets)) != len(targets):
        raise ValueError('Invalid LoRA configuration')
    layers = [(name, layer) for name, layer in text.named_modules()
              if name.rsplit('.', 1)[-1] in targets and isinstance(layer, nn.Linear)]
    if {name.rsplit('.', 1)[-1] for name, _ in layers} != set(targets):
        raise ValueError('Some requested LoRA targets do not exist')
    text.requires_grad_(False)
    for name, layer in layers:
        a = torch.empty(rank, layer.in_features, device=layer.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = torch.zeros(layer.out_features, rank, device=layer.weight.device, dtype=torch.float32)
        parent, _, child = name.rpartition('.')
        setattr(text.get_submodule(parent), child, LoadedLoRA(layer, a, b, alpha/rank))
    return [name for name, _ in layers]


def choice_token_ids(tokenizer):
    encoded = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(16)]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 16:
        raise ValueError('A-P verbalizers must be distinct single tokens')
    return [ids[0] for ids in encoded]


def load_backbone(name='jev', device='cuda'):
    spec = backbone_spec(name)
    if spec['kind'] == 'jev':
        return load_jev(device)
    # Full shard hashing is done once by the launcher; each rank verifies metadata
    # and tensor names/shapes. Live jobs also guard source file stats.
    verify_native_checkpoint(spec, weights=False)
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    root = Path(spec['path'])
    cfg = Qwen3VLTextConfig(**json.loads((root/'config.json').read_text())['text_config'])
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'
    with init_empty_weights():
        text = Qwen3VLTextModel(cfg)
    mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.'
    selected = {k.removeprefix(prefix): (k, v) for k, v in mapping.items() if k.startswith(prefix)}
    expected = text.state_dict()
    if set(selected) != set(expected):
        raise ValueError('Native checkpoint does not match built-in Qwen3VLTextModel')
    shards = defaultdict(list)
    for key, (source, shard) in selected.items():
        shards[shard].append((key, source))
    for shard, keys in sorted(shards.items()):
        with safe_open(root/shard, framework='pt', device='cpu') as f:
            for key, source in keys:
                value = f.get_tensor(source)
                if value.shape != expected[key].shape:
                    raise ValueError(f'Native Qwen shape mismatch: {key}')
                dtype = (torch.float32 if any(part in key for part in (text._keep_in_fp32_modules or []))
                         else torch.bfloat16)
                set_module_tensor_to_device(text, key, device, value=value, dtype=dtype)
        print('Loaded native Qwen language shard', shard, flush=True)
    if any(p.is_meta for p in text.parameters()):
        raise ValueError('Unloaded native Qwen parameter')
    text.to(device=device).requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    ids = choice_token_ids(tokenizer)
    head_key = 'lm_head.weight'
    if head_key not in mapping:
        if not cfg.tie_word_embeddings:
            raise ValueError('Missing untied LM head')
        head_key = prefix+'embed_tokens.weight'
    with safe_open(root/mapping[head_key], framework='pt', device='cpu') as f:
        view = f.get_slice(head_key)
        weight = torch.cat([view[i:i+1] for i in ids]).float().to(device)
    base_parameters = sum(p.numel() for p in text.parameters())
    modules = add_fresh_lora(text, spec['lora_rank'], spec['lora_alpha'], spec['lora_targets'])
    text.backbone_audit = dict(base_parameters=base_parameters, hidden_size=cfg.hidden_size,
        choice_token_ids=ids, head_source=head_key, lora_modules=modules,
        lora_parameters=sum(p.numel() for p in text.parameters() if p.requires_grad),
        vision_loaded=False, pretrained_jev_adapter_loaded=False)
    return text, weight, torch.zeros(16, device=device), tokenizer
