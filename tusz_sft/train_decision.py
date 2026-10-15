"""Single-GPU / torchrun DDP training of the local EEG RL decision model."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from .decision_data import DecisionCollator, LABELS, PROMPTS, file_sha256, load_datasets
from .decision_model import DEFAULT_BASE, load_model, source_signature


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--train-data")
    p.add_argument("--validation-data")
    p.add_argument("--out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-ratio", type=float, default=.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.)
    p.add_argument("--head-init", choices=["tokens", "random"], default="tokens")
    p.add_argument("--input-mode", choices=list(PROMPTS), default="waveform")
    p.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    p.add_argument("--max-pixels", type=int, default=1500000)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0, help="0 = complete configured epochs")
    p.add_argument("--resume", type=Path, help="A checkpoint created by this trainer")
    p.add_argument("--check-input-only", action="store_true")
    p.add_argument("--skip-initial-eval", action="store_true")
    pre, _ = p.parse_known_args()
    if pre.config:
        config = json.loads(pre.config.read_text())
        unknown = set(config) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"Unknown config keys: {unknown}")
        p.set_defaults(**config)
    a = p.parse_args()
    if not a.train_data or not a.out:
        p.error("train-data and out are required (CLI or config)")
    if a.resume and a.check_input_only:
        p.error("Use a fresh output directory for check-input-only; do not combine it with resume")
    if min(a.epochs, a.batch_size, a.grad_accum, a.max_pixels, a.max_length) < 1:
        p.error("epochs/batch-size/grad-accum/max-pixels/max-length must be positive")
    if a.workers < 0 or a.lora_rank < 0 or min(a.learning_rate, a.head_learning_rate) <= 0:
        p.error("Invalid workers, rank or learning rate")
    if min(a.max_steps, a.save_every, a.eval_every) < 0:
        p.error("max-steps, save-every and eval-every cannot be negative")
    if not 0 <= a.warmup_ratio < 1 or not 0 <= a.lora_dropout < 1:
        p.error("Invalid warmup ratio or dropout")
    return a


def json_write(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluate(model, rows, collator, device, rank, world, batch_size):
    from sklearn.metrics import average_precision_score, roc_auc_score
    was_training = model.training
    model.eval()
    predictions = []
    local = rows[rank::world]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(local), batch_size):
            batch = local[i:i + batch_size]
            inputs, labels = collator(batch)
            logits = model(**{k: v.to(device) for k, v in inputs.items()})
            probs = logits.softmax(-1)[:, 1].cpu().tolist()
            predictions.extend({"id": r["id"], "target": int(y), "p_seiz": p}
                               for r, y, p in zip(batch, labels.tolist(), probs))
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, predictions)
        predictions = [p for group in gathered for p in group]
    predictions.sort(key=lambda p: p["id"])
    y = np.array([p["target"] for p in predictions])
    scores = np.array([p["p_seiz"] for p in predictions])
    pred = scores >= .5
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    metrics = {"n": len(y), "threshold": .5, "auroc": float(roc_auc_score(y, scores)),
               "average_precision": float(average_precision_score(y, scores)),
               "accuracy": float((pred == y).mean()), "balanced_accuracy": (sensitivity + specificity) / 2,
               "sensitivity": sensitivity, "specificity": specificity,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    model.train(was_training)
    return metrics, predictions


def rng_state(device):
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def load_resume_checkpoint(checkpoint, out, identity, epochs, max_steps):
    """Validate ownership and compatibility before writing any run metadata."""
    checkpoint, out = Path(checkpoint).resolve(), Path(out).resolve()
    if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent != out:
        raise ValueError("Resume output must be the original checkpoint's run directory")
    latest = json.loads((out / "latest.json").read_text())
    if Path(latest["checkpoint"]).resolve() != checkpoint:
        raise ValueError("Resume only the latest checkpoint; older checkpoints would duplicate run history")
    restored = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    restored["identity"]["hyperparameters"].setdefault("input_mode", "waveform")
    if restored["identity"] != identity:
        raise ValueError("Resume changes data, world size, training schedule, or hyperparameters")
    if restored["epoch"] >= epochs or (max_steps and restored["step"] >= max_steps):
        raise ValueError("Requested training is already finished; existing results were not changed")
    return restored


def save_checkpoint(out, model, processor, optimizer, scheduler, state, identity, config, device, rank, world):
    rngs = [None] * world
    if world > 1:
        dist.all_gather_object(rngs, rng_state(device))
    else:
        rngs[0] = rng_state(device)
    destination = out / "checkpoints" / f"step-{state['step']:06d}"
    if rank == 0:
        temporary = destination.with_name(destination.name + ".incomplete")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        model.save(temporary, processor)
        torch.save({**state, "identity": identity, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rngs": rngs}, temporary / "training_state.pt")
        json_write(temporary / "run_config.json", config)
        temporary.rename(destination)
        json_write(out / "latest.json", {"checkpoint": str(destination.resolve()), "step": state["step"]})
    if world > 1:
        dist.barrier()
    return destination


def main():
    a = parse_args()
    rank, local_rank, world = (int(os.environ.get(k, v)) for k, v in
                               (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    if not a.check_input_only:
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("A visible CUDA GPU is required for each local rank")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        run(a, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(a, rank, world, device):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    out = Path(a.out).resolve()
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory, or --resume a saved checkpoint")
    train, validation = load_datasets(a.train_data, a.validation_data, mode=a.input_mode)
    signature = source_signature(a.base)
    processor = AutoProcessor.from_pretrained(a.base, local_files_only=True)
    collator = DecisionCollator(processor, a.max_pixels, a.max_length, PROMPTS[a.input_mode])
    preview, _ = collator(train[:a.batch_size])
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                 seed=a.seed, drop_last=True)
    loader = DataLoader(train, batch_size=a.batch_size, sampler=sampler, drop_last=True,
                        num_workers=a.workers, collate_fn=collator, pin_memory=True,
                        multiprocessing_context="spawn" if a.workers else None)
    batches = len(loader)
    if batches < 1:
        raise ValueError("Not enough examples for the global microbatch")
    steps_per_epoch = math.ceil(batches / a.grad_accum)
    total_steps = steps_per_epoch * a.epochs
    identity = {"source": signature, "train_sha256": file_sha256(a.train_data),
                "validation_sha256": file_sha256(a.validation_data) if a.validation_data else None,
                "world_size": world, "total_steps": total_steps,
                "hyperparameters": {k: config[k] for k in (
                    "batch_size", "grad_accum", "learning_rate", "head_learning_rate", "weight_decay",
                    "warmup_ratio", "lora_rank", "lora_alpha", "lora_dropout", "head_init", "seed",
                    "class_weights", "max_pixels", "max_length", "input_mode")}}
    restored = load_resume_checkpoint(a.resume, out, identity, a.epochs, a.max_steps) if a.resume else None
    audit = {**identity, "train_windows": len(train), "validation_windows": len(validation),
             "train_classes": dict(Counter(r["label"] for r in train)),
             "effective_batch_size": world * a.batch_size * a.grad_accum,
             "dropped_training_windows_per_epoch": len(train) - batches * a.batch_size * world,
             "input_shapes": {k: list(v.shape) for k, v in preview.items()}}
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        json_write(out / "run_config.json", config)
        json_write(out / "preflight.json", audit)
        print(json.dumps({"event": "preflight", **audit}), flush=True)
    if a.check_input_only:
        return
    del preview, processor
    model, processor = load_model(a.base, device=device, rank=a.lora_rank, alpha=a.lora_alpha,
                                  dropout=a.lora_dropout, checkpoint=a.resume, head_init=a.head_init,
                                  max_pixels=a.max_pixels, max_length=a.max_length,
                                  prompt=PROMPTS[a.input_mode], input_mode=a.input_mode)
    if model.metadata["source"] != signature:
        raise ValueError("Resume base does not match configured base")
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n != "head.weight"]
    groups = [{"params": list(model.head.parameters()), "lr": a.head_learning_rate}]
    if lora:
        groups.append({"params": lora, "lr": a.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    warmup = max(1, round(total_steps * a.warmup_ratio))
    def lr_schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    state = {"step": 0, "epoch": 0, "next_batch": 0, "best_auroc": -1.}
    resume_rng = None
    if a.resume:
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        state.update({k: restored[k] for k in state})
        resume_rng = restored["rngs"][rank]
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                       find_unused_parameters=False) if world > 1 else model
    counts = Counter(r["label"] for r in train)
    weights = torch.tensor([len(train) / (2 * counts[label]) for label in LABELS], device=device)
    if a.class_weights == "none":
        weights.fill_(1)
    initial_head = model.head.weight.detach().clone()
    initial_lora = next((p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n), None)
    if rank == 0:
        json_write(out / "model_audit.json", {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "head_shape": list(model.head.weight.shape), "has_lm_head": False,
            "class_weights": weights.tolist(), "metadata": model.metadata})
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    last_saved = state["step"] if a.resume else -1
    start = time.monotonic()
    stop = bool(a.max_steps and state["step"] >= a.max_steps)
    last_metrics = None
    if validation and not a.resume and not a.skip_initial_eval:
        metrics, predictions = evaluate(model, validation, collator, device, rank, world, a.batch_size)
        state["best_auroc"] = metrics["auroc"]
        checkpoint = save_checkpoint(out, model, processor, optimizer, scheduler, state,
                                     identity, config, device, rank, world)
        last_saved = 0
        if rank == 0:
            json_write(out / "initial_validation.json", {"step": 0, **metrics})
            json_write(out / "validation_step_000000.json", predictions)
            json_write(out / "best.json", {"checkpoint": str(checkpoint), "step": 0,
                                           "selection": "internal validation AUROC", **metrics})
            print(json.dumps({"event": "initial_validation", **metrics}), flush=True)
    for epoch in range(state["epoch"], a.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        skip = state["next_batch"] if epoch == state["epoch"] else 0
        iterator = iter(loader)
        if resume_rng is not None:
            restore_rng(resume_rng, device)
            resume_rng = None
        accumulated_loss = 0.
        for batch_index, (inputs, labels) in enumerate(iterator):
            if batch_index < skip:
                continue
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=True)
            group_start = batch_index // a.grad_accum * a.grad_accum
            group_size = min(a.grad_accum, batches - group_start)
            update = (batch_index + 1) % a.grad_accum == 0 or batch_index + 1 == batches
            sync = nullcontext() if update or world == 1 else wrapped.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(**inputs)
                # Explicit per-example weights: default CE(mean) cancels weights at batch size 1.
                loss = (F.cross_entropy(logits.float(), labels, reduction="none") * weights[labels]).mean()
                (loss / group_size).backward()
            accumulated_loss += loss.detach().item() / group_size
            if not update:
                continue
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            state["epoch"] = epoch + int(batch_index + 1 == batches)
            state["next_batch"] = 0 if batch_index + 1 == batches else batch_index + 1
            loss_tensor = torch.tensor(accumulated_loss, device=device)
            if world > 1:
                dist.all_reduce(loss_tensor)
                loss_tensor /= world
            record = {"event": "train", "step": state["step"], "epoch": epoch,
                      "loss": loss_tensor.item(), "grad_norm": norm.item(),
                      "lr": [g["lr"] for g in optimizer.param_groups],
                      "elapsed_seconds": time.monotonic() - start}
            if rank == 0:
                with (out / "train.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            accumulated_loss = 0.
            stop = bool(a.max_steps and state["step"] >= a.max_steps)
            epoch_end = batch_index + 1 == batches
            do_eval = bool(validation and (epoch_end or stop or (a.eval_every and state["step"] % a.eval_every == 0)))
            improved = False
            if do_eval:
                last_metrics, predictions = evaluate(model, validation, collator, device, rank, world, a.batch_size)
                improved = last_metrics["auroc"] > state["best_auroc"]
                state["best_auroc"] = max(state["best_auroc"], last_metrics["auroc"])
                if rank == 0:
                    val_record = {"step": state["step"], **last_metrics}
                    with (out / "validation.jsonl").open("a") as f:
                        f.write(json.dumps(val_record) + "\n")
                    json_write(out / f"validation_step_{state['step']:06d}.json", predictions)
                    print(json.dumps({"event": "validation", **val_record}), flush=True)
            do_save = improved or stop or epoch_end or (a.save_every and state["step"] % a.save_every == 0)
            if do_save and state["step"] != last_saved:
                checkpoint = save_checkpoint(out, model, processor, optimizer, scheduler, state,
                                             identity, config, device, rank, world)
                last_saved = state["step"]
                if improved and rank == 0:
                    json_write(out / "best.json", {"checkpoint": str(checkpoint), "step": state["step"],
                                                   "selection": "internal validation AUROC", **last_metrics})
            if stop:
                break
    if rank == 0:
        final_lora = next((p for n, p in model.named_parameters() if "lora_B" in n), None)
        json_write(out / "result.json", {"status": "complete" if state["epoch"] == a.epochs else "step_limit",
                   **state, "validation": last_metrics, "world_size": world,
                   "elapsed_seconds": time.monotonic() - start,
                   "head_max_update": (model.head.weight.detach() - initial_head).abs().max().item(),
                   "first_lora_B_max_update": (final_lora.detach() - initial_lora).abs().max().item()
                   if final_lora is not None else None,
                   "peak_cuda_memory_gib_rank0": torch.cuda.max_memory_allocated(device) / 2**30})
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
