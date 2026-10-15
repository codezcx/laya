"""Qwen3-VL EEG backbone + fresh decoder LoRA + a small decision head.

The original vocabulary head is used only to initialize A/B, then discarded.
Checkpoints contain the adapter and decision head, never a copy of the base.
"""
import json
from pathlib import Path
import re

import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .decision_data import LABELS, PROMPT, file_sha256

DEFAULT_BASE = "/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss/DataEngine/output/eeg_rl_hf_8b_v3render/step_60"


def source_signature(path):
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("This trainer requires Qwen3-VL (the local 8B/32B RL export)")
    index = path / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return {"path": str(path), "config_sha256": file_sha256(path / "config.json"),
            "index_sha256": file_sha256(index),
            "shards": {name: {"bytes": (path / name).stat().st_size,
                              "mtime_ns": (path / name).stat().st_mtime_ns} for name in shards}}


def last_prompt_state(hidden, attention_mask):
    """Handle both left and right padding, without selecting a padded position."""
    valid = attention_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Empty prompt")
    indices = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    last = indices.masked_fill(~valid, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class EEGDecisionModel(nn.Module):
    def __init__(self, backbone, initial_weight, metadata):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(initial_weight.shape[1], initial_weight.shape[0], bias=False,
                              device=initial_weight.device, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.copy_(initial_weight.float())
        self.metadata = metadata

    def forward(self, **inputs):
        outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
        state = last_prompt_state(outputs.last_hidden_state, inputs["attention_mask"])
        # Keep the tiny head and its optimizer statistics in float32.
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.head(state.float())

    def save(self, directory, processor):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if isinstance(self.backbone, PeftModel):
            self.backbone.save_pretrained(directory / "adapter", safe_serialization=True,
                                          save_embedding_layers=False)
        save_file({"weight": self.head.weight.detach().cpu().contiguous()},
                  str(directory / "decision_head.safetensors"))
        (directory / "decision_config.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        processor.save_pretrained(directory / "processor")


def load_model(base=DEFAULT_BASE, *, device="cuda:0", rank=16, alpha=32,
               dropout=0., attention="sdpa", gradient_checkpointing=True,
               checkpoint=None, trainable=True, prompt=PROMPT, max_pixels=1500000,
               max_length=4096, head_init="tokens", input_mode="waveform"):
    saved = None
    if checkpoint:
        checkpoint = Path(checkpoint)
        saved = json.loads((checkpoint / "decision_config.json").read_text())
        base = saved["source"]["path"]
        rank = saved["lora_rank"]
    signature = source_signature(base)
    if saved and signature != saved["source"]:
        raise ValueError("Base checkpoint changed since the adapter was saved")
    processor = AutoProcessor.from_pretrained(
        checkpoint / "processor" if checkpoint else base, local_files_only=True)
    token_ids = [processor.tokenizer.encode(t, add_special_tokens=False) for t in ("A", "B")]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("A and B must each be a single token")
    token_ids = [ids[0] for ids in token_ids]
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        base, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": str(device)}, attn_implementation=attention)
    initial = full.lm_head.weight[token_ids].detach().float().clone()
    backbone = full.model
    # No vocabulary projection is registered in the resulting model.
    del full
    backbone.config.use_cache = False
    backbone.config.text_config.use_cache = False
    backbone.requires_grad_(False)
    targets = [name for name, module in backbone.named_modules()
               if isinstance(module, nn.Linear) and re.fullmatch(
                   r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)", name)]
    if not targets:
        raise ValueError("No decoder LoRA targets found; architecture changed")
    if saved and rank:
        backbone = PeftModel.from_pretrained(backbone, checkpoint / "adapter", is_trainable=trainable)
    elif rank:
        backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets, bias="none"))
    if gradient_checkpointing and trainable and rank:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    metadata = saved or {"format_version": 1, "architecture": "qwen3_vl_decision",
                        "source": signature, "hidden_size": initial.shape[1],
                        "labels": LABELS, "verbalizer_tokens": ["A", "B"], "token_ids": token_ids,
                        "lora_rank": rank, "lora_alpha": alpha, "lora_dropout": dropout,
                        "target_modules": targets if rank else [], "head_init": head_init,
                        "prompt": prompt, "max_pixels": max_pixels, "max_length": max_length,
                        "input_mode": input_mode}
    if not saved and head_init == "random":
        nn.init.normal_(initial, mean=0., std=.02)
    elif not saved and head_init != "tokens":
        raise ValueError("head_init must be tokens or random")
    model = EEGDecisionModel(backbone, initial, metadata)
    if saved:
        model.head.load_state_dict(load_file(str(checkpoint / "decision_head.safetensors"), device=str(device)))
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    if any("lm_head" in name for name, _ in model.named_parameters()):
        raise AssertionError("Original lm_head was not removed")
    if any(p.requires_grad for name, p in model.named_parameters() if ".visual." in name):
        raise AssertionError("Vision weights must remain frozen")
    return model, processor
