"""Strict, selective loading of Qwen text weights; frozen REVE and trainable bridge."""
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
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


class EEGQwen(nn.Module):
    def __init__(self, encoder, text, positions, prefix_ids, suffix_ids, head_weight, config):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.text = text.requires_grad_(False)
        self.config = config
        self.register_buffer("positions", positions.float())
        self.register_buffer("prefix_ids", prefix_ids)
        self.register_buffer("suffix_ids", suffix_ids)
        dim = encoder.config.embed_dim
        hidden = head_weight.shape[1]
        self.connector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config["connector_hidden_dim"]),
                                       nn.GELU(), nn.Linear(config["connector_hidden_dim"], hidden))
        self.head = nn.Linear(hidden, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(head_weight.float().cpu())
        self.stage = "warmup"

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # REVE positional noise/dropout must stay disabled.
        if self.stage == "warmup":
            self.text.eval()
        return self

    @torch.no_grad()
    def encode(self, eeg, lengths):
        # Encode separately by true length: REVE attention must never see batch padding.
        features = []
        patch = self.encoder.config.patch_size
        stride = patch - self.encoder.config.patch_overlap
        for n in lengths.unique(sorted=True).tolist():
            if n < patch:
                raise ValueError(f"EEG must have at least {patch} samples")
            indices = (lengths == n).nonzero().flatten()
            x = eeg[indices, :, :n].float()
            tail = (-(n - patch)) % stride
            x = F.pad(x, (0, tail), mode="replicate") if tail else x
            # Use fp32 REVE to match the inspected checkpoint's numerical path.
            with torch.autocast(device_type=x.device.type, enabled=False):
                encoded = self.encoder(x, self.positions[None].expand(len(indices), -1, -1))
            if encoded.ndim != 4:
                raise ValueError("Expected REVE B,C,P,D output")
            for idx, value in zip(indices.tolist(), encoded.flatten(1, 2)):
                features.append((idx, value))
        ordered = [v for _, v in sorted(features)]
        padded = nn.utils.rnn.pad_sequence(ordered, batch_first=True)
        token_lengths = torch.tensor([len(v) for v in ordered], device=eeg.device)
        mask = torch.arange(padded.shape[1], device=eeg.device)[None] < token_lengths[:, None]
        return padded, mask

    def forward(self, eeg, lengths, labels=None):
        features, eeg_mask = self.encode(eeg, lengths)
        projected = self.connector(features)
        embed = self.text.get_input_embeddings()
        prefix, suffix = embed(self.prefix_ids), embed(self.suffix_ids)
        inputs, mask, positions, last = assemble_embeddings(prefix, projected.to(prefix.dtype), suffix, eeg_mask)
        # Frozen text weights still propagate gradients to the connector.
        hidden = self.text(inputs_embeds=inputs, attention_mask=mask,
                           position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
        state = hidden[torch.arange(len(hidden), device=hidden.device), last]
        with torch.autocast(device_type=state.device.type, enabled=False):
            logits = self.head(state.float())
            loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def enable_lora(self):
        if isinstance(self.text, PeftModel):
            raise ValueError("LoRA already installed")
        self.text = get_peft_model(self.text, LoraConfig(
            r=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"], bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.stage = "lora"

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
                 if k.startswith(("connector.", "head."))}
        save_file(state, str(path / "bridge.safetensors"))
        if isinstance(self.text, PeftModel):
            self.text.save_pretrained(path / "adapter", save_embedding_layers=False)
        (path / "config.json").write_text(json.dumps({**self.config, "saved_stage": self.stage}, indent=2)+"\n")


def load_model(config, device="cuda:0", stage="warmup", checkpoint=None, trainable=True):
    if stage not in ("warmup", "lora"):
        raise ValueError(stage)
    if checkpoint:
        saved = json.loads((Path(checkpoint) / "config.json").read_text())
        for key in ("qwen_checkpoint", "reve_checkpoint", "positions_checkpoint", "channels",
                    "position_names", "labels", "normalization", "tail_padding", "sample_rate",
                    "connector_hidden_dim", "prompt_before", "prompt_after", "lora_rank", "lora_alpha"):
            if config[key] != saved[key]:
                raise ValueError(f"Checkpoint/config mismatch for {key}")
        if saved["saved_stage"] == "lora" and stage != "lora":
            raise ValueError("A LoRA checkpoint requires --stage lora")
    tokenizer = AutoTokenizer.from_pretrained(config["qwen_checkpoint"], local_files_only=True)
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in ("A", "B")]
    if any(len(x) != 1 for x in ids):
        raise ValueError("A/B must each be a single token")
    text, head, audit = load_text_only(config["qwen_checkpoint"], [x[0] for x in ids], device,
                                      torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    encoder, info = AutoModel.from_pretrained(config["reve_checkpoint"], local_files_only=True,
        trust_remote_code=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"REVE loading mismatch: {info}")
    bank = AutoModel.from_pretrained(config["positions_checkpoint"], local_files_only=True, trust_remote_code=True)
    positions = bank(config["position_names"]).detach()
    if positions.shape != (len(config["channels"]), 3):
        raise ValueError("Missing electrode coordinates")
    # Use the existing chat template, with a non-token marker replaced by EEG embeddings.
    marker = "__RAW_EEG_EMBEDDINGS__"
    rendered = tokenizer.apply_chat_template([
        {"role": "system", "content": "You classify EEG motor imagery. Select A or B."},
        {"role": "user", "content": config["prompt_before"] + marker + config["prompt_after"]}],
        tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError("Invalid EEG placeholder")
    before, after = rendered.split(marker)
    model = EEGQwen(encoder, text, positions,
        torch.tensor(tokenizer.encode(before, add_special_tokens=False)),
        torch.tensor(tokenizer.encode(after, add_special_tokens=False)), head, config).to(device)
    if checkpoint:
        state = load_file(str(Path(checkpoint) / "bridge.safetensors"), device=str(device))
        expected = {k for k in model.state_dict() if k.startswith(("connector.", "head."))}
        if set(state) != expected:
            raise ValueError("Incomplete bridge checkpoint")
        model.connector.load_state_dict({k[len("connector."):]: v for k,v in state.items() if k.startswith("connector.")})
        model.head.load_state_dict({k[len("head."):]: v for k,v in state.items() if k.startswith("head.")})
    if stage == "lora":
        if checkpoint and saved["saved_stage"] == "lora":
            model.text = PeftModel.from_pretrained(model.text, Path(checkpoint)/"adapter", is_trainable=trainable)
            model.stage = "lora"
        else:
            model.enable_lora()
    if trainable:
        model.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False).eval()
    model.loading_audit = audit
    return model
