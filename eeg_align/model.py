"""Current bridge helpers: strict text loading and contiguous EEG insertion."""
import json
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

def load_text_only(path, token_ids, device, dtype):
    """Never instantiate/load ViT, merger, DeepStack features or full vocabulary head."""
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    config.use_cache = False
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3VLTextModel(config)
    mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    selected = {k[len(prefix):]: (k, v) for k, v in mapping.items() if k.startswith(prefix)}
    expected = model.state_dict()
    if set(selected) != set(expected):
        raise ValueError(f"Text weight mismatch: missing={set(expected)-set(selected)}, "
                         f"unexpected={set(selected)-set(expected)}")
    shards = defaultdict(list)
    for key, (source, shard) in selected.items():
        shards[shard].append((key, source))
    for shard, keys in sorted(shards.items()):
        with safe_open(path / shard, framework="pt", device="cpu") as f:
            for key, source in keys:
                tensor = f.get_tensor(source)
                if tensor.shape != expected[key].shape:
                    raise ValueError(f"Shape mismatch: {source}")
                set_module_tensor_to_device(model, key, device, value=tensor, dtype=dtype)
        print(f"Loaded text tensors from {shard}", flush=True)
    model.to(device=device)  # Moves nonpersistent rotary buffers too.
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unloaded meta parameters")
    with safe_open(path / mapping["lm_head.weight"], framework="pt", device="cpu") as f:
        # Read only two rows, not the 151,936-row vocabulary projection.
        view = f.get_slice("lm_head.weight")
        head = torch.cat([view[i:i+1] for i in token_ids]).float().to(device)
    return model, head, {"loaded_text_tensors": len(selected), "prefix": prefix,
                         "vision_loaded": False, "lm_head_rows_only": token_ids}


def assemble_embeddings(prefix, eeg, suffix, eeg_mask):
    """Keep each example contiguous; pad after its answer position, never inside EEG."""
    seqs = [torch.cat((prefix, x[valid], suffix), dim=0)
            for x, valid in zip(eeg, eeg_mask.bool())]
    lengths = torch.tensor([len(s) for s in seqs], device=eeg.device)
    embeds = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    mask = torch.arange(embeds.shape[1], device=eeg.device)[None] < lengths[:, None]
    position_ids = (mask.long().cumsum(-1) - 1).clamp_min(0)
    return embeds, mask, position_ids, lengths - 1
