"""Single GPU / torchrun DDP trainer. Cross-entropy only; checkpoints exclude frozen bases."""
import argparse
import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .data import MIDataset, BalancedDistributedSampler, collate
from .model import load_model


def evaluate(model, loader, device, dataset_names=None):
    model.eval()
    names = dataset_names or ["default"]
    dataset_confusion = torch.zeros(len(names), 2, 2, device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device)
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            dataset_ids = batch.pop("dataset_index")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(**batch)
            y = batch["labels"]
            dataset_confusion += torch.bincount(dataset_ids*4 + y*2 + out["logits"].argmax(-1),
                                                minlength=len(names)*4).reshape(len(names), 2, 2)
            loss_sum += out["loss"] * len(y)
    if dist.is_initialized():
        dist.all_reduce(dataset_confusion)
        dist.all_reduce(loss_sum)
    confusion = dataset_confusion.sum(0)
    count = int(confusion.sum())
    result = {"accuracy": float(confusion.diag().sum() / max(count, 1)),
            "balanced_accuracy": float((confusion.diag() / confusion.sum(1).clamp_min(1)).mean()),
            "loss": float(loss_sum / max(count, 1)), "n": count, "confusion_matrix": confusion.cpu().tolist()}
    result["per_dataset"] = {name: {"n": int(cm.sum()), "accuracy": float(cm.diag().sum()/cm.sum()),
        "balanced_accuracy": float((cm.diag()/cm.sum(1).clamp_min(1)).mean()), "confusion_matrix": cm.cpu().tolist()}
        for name, cm in zip(names, dataset_confusion) if int(cm.sum())}
    result["macro_dataset_accuracy"] = sum(v["accuracy"] for v in result["per_dataset"].values()) / max(1, len(result["per_dataset"]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reve_qwen_mi.json")
    p.add_argument("--stage", choices=["warmup", "lora"], default="warmup")
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", help="Warm start weights; optimizer and epoch restart intentionally")
    p.add_argument("--epochs", type=int)
    p.add_argument("--smoke", action="store_true", help="One synthetic backward step; no dataset training")
    p.add_argument("--evaluate", choices=["validation", "test", "external_2a", "external_2b"])
    p.add_argument("--allow-unverified-provenance", action="store_true", help="Exploratory runs only; record source audit limitation")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.stage == "lora" and not args.init_checkpoint and not args.smoke:
        p.error("LoRA stage requires --init-checkpoint from warmup")
    if args.evaluate and not args.init_checkpoint:
        p.error("Evaluation requires --init-checkpoint")
    if args.smoke and args.evaluate:
        p.error("--smoke and --evaluate are mutually exclusive")
    if cfg["epochs"] < 1 or cfg["gradient_accumulation"] < 1:
        p.error("epochs and gradient_accumulation must be positive")
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "run.json").exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        (output / "run.json").write_text(json.dumps({"config": cfg, "args": vars(args), "world_size": world,
            "torch": torch.__version__}, indent=2)+"\n")
    if world > 1:
        dist.barrier()
    train_loader = val_loader = None
    if not args.smoke:
        manifests = Path(cfg["manifest_dir"])
        audit = json.loads((manifests / "audit.json").read_text())
        if audit.get("ready_for_exploratory_training") is False:
            raise ValueError("Unresolved dataset issues; see audit.json and issues.json")
        if not audit["provenance_verified"] and not args.allow_unverified_provenance:
            raise ValueError("Dataset source audit is unresolved. For exploratory runs use --allow-unverified-provenance; see audit.json")
        train_ds = MIDataset(manifests / "train.jsonl", cfg)
        val_ds = MIDataset(manifests / f"{args.evaluate or 'validation'}.jsonl", cfg)
        if {r["path"] for r in train_ds.rows} & {r["path"] for r in val_ds.rows}:
            raise ValueError("Train/evaluation overlap")
        if cfg.get("datasets"):
            train_people = {(r["dataset"], r["subject"]) for r in train_ds.rows}
            eval_people = {(r["dataset"], r["subject"]) for r in val_ds.rows}
            if train_people & eval_people:
                raise ValueError("Subject overlap across training/evaluation")
        if cfg.get("sampler") == "dataset_class_balanced":
            sampler = BalancedDistributedSampler(train_ds.rows, rank, world, cfg["seed"])
        else:
            sampler = DistributedSampler(train_ds, shuffle=True, seed=cfg["seed"]) if world > 1 else None
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=sampler is None,
                                  sampler=sampler, collate_fn=collate, num_workers=0)
        # Strided evaluation avoids DistributedSampler duplicating the last examples.
        val_loader = DataLoader(Subset(val_ds, list(range(rank, len(val_ds), world))),
                               batch_size=cfg["batch_size"], collate_fn=collate, num_workers=0)
    model = load_model(cfg, device=str(device), stage=args.stage, checkpoint=args.init_checkpoint,
                       trainable=not args.evaluate)
    if rank == 0:
        (output / "loading_audit.json").write_text(json.dumps(model.loading_audit, indent=2)+"\n")
    if args.evaluate:
        result = evaluate(model, val_loader, device, cfg.get("dataset_names"))
        if rank == 0:
            (output / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
            print(result, flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    groups = [{"params": model.connector.parameters(), "lr": cfg["connector_lr"]},
              {"params": model.head.parameters(), "lr": cfg["head_lr"]}]
    adapter = [v for v in model.text.parameters() if v.requires_grad]
    if adapter:
        groups.append({"params": adapter, "lr": cfg["lora_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    if args.smoke:
        model.train()
        batch = {"eeg": torch.randn(2, 3, 800, device=device), "lengths": torch.tensor([800, 600], device=device),
                 "labels": torch.tensor([0, 1], device=device)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = wrapped(**batch)
        out["loss"].backward()
        connector_grad = sum(float(v.grad.float().norm()) for v in model.connector.parameters() if v.grad is not None)
        lora_grad = sum(float(v.grad.float().norm()) for v in adapter if v.grad is not None)
        assert np.isfinite(connector_grad) and connector_grad > 0
        assert not any(v.grad is not None for v in model.encoder.parameters())
        assert not any(v.grad is not None for v in model.text.parameters() if not v.requires_grad)
        if args.stage == "lora":
            assert np.isfinite(lora_grad) and lora_grad > 0
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model(**batch)["logits"].float().cpu()
        if rank == 0:
            model.save(output / "checkpoint")
            # A second invocation with --init-checkpoint exercises the strict reload path.
            torch.save({"batch": {k: v.cpu() for k,v in batch.items()}, "logits": reference}, output / "roundtrip.pt")
            result = {"loss": float(out["loss"].detach()), "logits_shape": list(out["logits"].shape),
                      "connector_grad_norm_sum": connector_grad, "lora_grad_norm_sum": lora_grad,
                      "frozen_encoder_no_grad": True, "stage": args.stage,
                      "peak_cuda_gb": torch.cuda.max_memory_allocated()/1e9, "synthetic_only": True}
            (output / "smoke.json").write_text(json.dumps(result, indent=2)+"\n")
            print(result, flush=True)
    else:
        best, step = -1., 0
        accumulation = cfg["gradient_accumulation"]
        for epoch in range(cfg["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train(); optimizer.zero_grad(set_to_none=True)
            total_loss, seen = 0., 0
            for i, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k,v in batch.items()}
                batch.pop("dataset_index")
                end = (i+1) % accumulation == 0 or i+1 == len(train_loader)
                group_start = (i // accumulation) * accumulation
                divisor = min(accumulation, len(train_loader)-group_start)
                sync = contextlib.nullcontext() if end or world == 1 else wrapped.no_sync()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    out = wrapped(**batch)
                    loss = out["loss"] / divisor
                    loss.backward()
                total_loss += float(out["loss"].detach()) * len(batch["labels"])
                seen += len(batch["labels"])
                if end:
                    torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                    if rank == 0 and step % 10 == 0:
                        print(json.dumps({"epoch": epoch+1, "step": step, "rank0_train_loss": total_loss/seen}), flush=True)
            metrics = evaluate(model, val_loader, device, cfg.get("dataset_names"))
            if rank == 0:
                record = {"epoch": epoch+1, "step": step, "validation": metrics}
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")
                model.save(output / "last")
                score = metrics[cfg.get("selection_metric", "accuracy")]
                if score > best:
                    best = score
                    model.save(output / "best")
                print(record, flush=True)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
