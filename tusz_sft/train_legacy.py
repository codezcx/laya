"""Historical trainer: JEV-27B decisions or Qwen assistant-only SFT.

This is a new TUSZ trainer, not CerebraGloss's unpublished SFT implementation.
Microbatch size is one; use gradient accumulation for a larger effective batch.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import time

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, set_seed

from .data import read_jsonl


def jev_prompt(row):
    text = f'[kind] choice\n[state] {row["state"]}\n'
    text += '<|vision_start|><|image_pad|><|vision_end|>\n'
    text += f'[question] {row["question"]}\n[options]\n'
    text += "\n".join(f'{chr(65 + i)}) {option}' for i, option in enumerate(row["options"]))
    return text + "\n[decision]:"


def encode_example(processor, row, backend, max_pixels):
    if backend == "jev":
        image_path = row["image"]
        text = jev_prompt(row)
    else:
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Expected one user image/question and one assistant answer")
        image_path = next(c["image"] for c in messages[0]["content"] if c["type"] == "image")
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    batch = processor(text=[text], images=[image], max_pixels=max_pixels,
                      return_tensors="pt", add_special_tokens=False)
    if backend == "jev":
        target = torch.tensor([row["target_index"]], dtype=torch.long)
    else:
        prefix = processor(text=[prompt], images=[image], max_pixels=max_pixels,
                           return_tensors="pt", add_special_tokens=False)
        length = prefix.input_ids.shape[1]
        if not torch.equal(batch.input_ids[:, :length], prefix.input_ids):
            raise ValueError("Chat template prefix is not token-aligned; cannot mask prompt safely")
        # Compute loss only for the assistant continuation, including its end token.
        target = batch.input_ids[:, length:].clone()
        if target.numel() == 0:
            raise ValueError("No assistant tokens to supervise")
    return batch, target


def load_model(backend, model_dir):
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, Qwen3_5ForConditionalGeneration
    cls = Qwen3_5ForConditionalGeneration if backend == "jev" else AutoModelForImageTextToText
    base = cls.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                               device_map={"": "cuda:0"}, attn_implementation="sdpa")
    if backend == "jev" and base.get_input_embeddings().weight.data_ptr() == base.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("JEV input/output weights unexpectedly tied; refusing to train")
    if backend == "jev":
        path = model_dir / "adapter_vllm"
        config = json.loads((path / "adapter_config.json").read_text())
        fields = {f.name for f in dataclasses.fields(LoraConfig)}
        unknown = {k: v for k, v in config.items() if k not in fields and v is not None}
        if unknown:
            raise ValueError(f"Unsupported LoRA settings: {unknown}")
        config = {k: v for k, v in config.items() if k in fields}
        config["inference_mode"] = False
        model = get_peft_model(base, LoraConfig(**config))
        weights = load_file(path / "adapter_model.safetensors")
        expected = get_peft_model_state_dict(model, save_embedding_layers=False)
        if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in expected):
            raise ValueError("Source JEV adapter tensors do not match the model")
        result = set_peft_model_state_dict(model, weights)
        if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
            raise ValueError(f"Adapter failed to load: {result}")
        del weights, expected
        head = json.loads((path / "decision_head.json").read_text())
    else:
        # Select only decoder attention/MLP projections, retaining the visual tower.
        suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        modules = [name for name, module in base.named_modules()
                   if isinstance(module, torch.nn.Linear) and ".visual." not in f".{name}."
                   and name.rsplit(".", 1)[-1] in suffixes]
        if not modules:
            raise ValueError("No language decoder projection layers found")
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                               target_modules=modules, bias="none"))
        head = None
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and ".visual." not in name)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["jev", "qwen"], default="jev")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-pixels", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--check-input-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.grad_accum, args.max_pixels) < 1 or args.learning_rate <= 0:
        parser.error("Steps, accumulation, pixels and learning rate must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new output directory; source models are never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    args.model = args.model.resolve()
    set_seed(args.seed)
    torch.set_num_threads(8)
    rows = list(read_jsonl(args.train_data))
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Training accepts only records marked split=train")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    example, target = encode_example(processor, rows[0], args.backend, args.max_pixels)
    audit = {"backend": args.backend, "model": str(args.model), "rows": len(rows),
             "train_data": str(args.train_data.resolve()),
             "train_data_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
             "seed": args.seed, "max_pixels": args.max_pixels,
             "sequence_tokens": example.input_ids.shape[1],
             "image_grid_thw": example.image_grid_thw.tolist(),
             "supervised_targets": target.tolist(),
             "target_text": (processor.tokenizer.decode(target[0]) if args.backend == "qwen" else ["A", "B"][target.item()]),
             "input_shapes": {k: list(v.shape) for k, v in example.items() if hasattr(v, "shape")},
             "objective": "restricted A/B cross-entropy" if args.backend == "jev" else "assistant-only causal language modeling",
             "steps": args.steps, "gradient_accumulation": args.grad_accum,
             "learning_rate": args.learning_rate, "check_input_only": args.check_input_only}
    for name in ("config.json", "recipe.json"):
        recipe_path = args.train_data.parent / name
        if recipe_path.exists():
            recipe = json.loads(recipe_path.read_text())
            audit["data_recipe"] = recipe.get("recipe", recipe)
            break
    manifest = args.train_data.parent / "train.manifest.jsonl"
    if manifest.exists():
        audit["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (args.out / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_input_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A usable CUDA GPU is required for training")
    del example, target
    model, head = load_model(args.backend, args.model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in parameters)
    print(f"Trainable LoRA parameters: {count:,}", flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01, foreach=False)
    if head:
        lo = head["slots"]["ranges"]["choice"][0]
        answer_ids = head["verbalizer_ids"][lo:lo + 2]
        if [processor.tokenizer.decode([x]) for x in answer_ids] != ["A", "B"]:
            raise ValueError("Decision slots do not map to A/B")
        bias = torch.tensor(head["bias"][lo:lo + 2], device="cuda")
    order = list(range(len(rows)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    cursor = 0
    started = time.monotonic()
    first_trainable = next((n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    before = first_trainable[1].detach().cpu().clone()
    with (args.out / "train_log.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            batch_ids = []
            for _ in range(args.grad_accum):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                row = rows[order[cursor]]
                cursor += 1
                batch_ids.append(row["id"])
                batch, target = encode_example(processor, row, args.backend, args.max_pixels)
                batch, target = batch.to("cuda"), target.to("cuda")
                keep = 1 if args.backend == "jev" else target.shape[1] + 1
                logits = model(**batch, use_cache=False, logits_to_keep=keep).logits
                if args.backend == "jev":
                    loss = F.cross_entropy(logits[:, -1, answer_ids].float() + bias, target)
                else:
                    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), target.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss_sum += loss.item()
                (loss / args.grad_accum).backward()
                del batch, target, logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item()
            optimizer.step()
            item = {"step": step, "loss": loss_sum / args.grad_accum, "grad_norm": grad_norm,
                    "sample_ids": batch_ids, "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(item) + "\n")
            log.flush()
            print(json.dumps(item), flush=True)
    change = (first_trainable[1].detach().cpu() - before).abs().max().item()
    if not change > 0:
        raise RuntimeError("Optimizer did not change the inspected LoRA parameter")
    model.save_pretrained(args.out / "adapter", safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(args.out / "processor")
    if head:
        (args.out / "decision_head.json").write_text(json.dumps(head, indent=2) + "\n")
    summary = {**audit, "trainable_parameters": count, "updated_parameter": first_trainable[0],
               "max_parameter_change": change, "completed_steps": args.steps,
               "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
               "adapter": str((args.out / "adapter").resolve()),
               "decision_temperature": 1.0,
               "note": "Training-chain validation; not a validated seizure detector. No evaluation metrics computed."}
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
