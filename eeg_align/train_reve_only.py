"""Reproducible REVE-only linear probe / full fine-tuning on the locked MI splits."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from .data import BalancedDistributedSampler, MIDataset, collate
from .reve_baseline import (SPLITS, audit_manifests, classification_metrics, file_sha256,
                            load_reve_classifier)


def write_json(path, value):
    def encode_special(item):
        # Recent Transformers returns sets in its pretrained loading audit.
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Unsupported JSON audit value: {type(item).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=encode_special) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_config(source, settings, mode):
    keep = ("reve_checkpoint", "positions_checkpoint", "manifest_dir", "channels", "position_names",
            "labels", "sample_rate", "normalization", "tail_padding", "dataset_names", "seed")
    config = {key: source[key] for key in keep}
    config.update(settings["common"])
    config.update(settings[mode])
    config["mode"] = mode
    config["pooling"] = "time_mean_per_channel_then_concatenate"
    config["selection_metric"] = "macro_dataset_accuracy"
    return config


def raw_loader(dataset, batch_size, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=0, collate_fn=collate)


@torch.inference_mode()
def extract_features(model, dataset, config, device, path):
    model.eval()
    values = []
    for i, batch in enumerate(raw_loader(dataset, config["extract_batch_size"])):
        values.append(model.encode(batch["eeg"].to(device), batch["lengths"].to(device)).cpu())
        if i % 50 == 0:
            print(json.dumps({"feature_cache": str(path.name), "batches_done": i+1}), flush=True)
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite REVE features")
    np.save(path, result.numpy(), allow_pickle=False)
    return result


@torch.inference_mode()
def evaluate(model, dataset, config, device, cached=None, output=None):
    model.eval()
    if cached is None:
        loader = raw_loader(dataset, config["eval_batch_size"])
    else:
        loader = DataLoader(TensorDataset(cached, torch.tensor([r["label"] for r in dataset.rows]),
                                         torch.tensor([r["dataset_index"] for r in dataset.rows])),
                            batch_size=config["eval_batch_size"])
    ys, ps, ds, probs = [], [], [], []
    losses = 0.
    for batch in loader:
        if cached is None:
            batch = {k: v.to(device) for k, v in batch.items()}
            dataset_ids = batch.pop("dataset_index")
            labels = batch["labels"]
            out = model(**batch)
        else:
            features, labels, dataset_ids = [v.to(device) for v in batch]
            out = model(features=features, labels=labels)
        if not torch.isfinite(out["logits"]).all():
            raise ValueError("Nonfinite evaluation logits")
        ys.extend(labels.cpu().tolist())
        ps.extend(out["logits"].argmax(-1).cpu().tolist())
        ds.extend(dataset_ids.cpu().tolist())
        probs.extend(out["logits"].softmax(-1).cpu().tolist())
        losses += float(out["loss"]) * len(labels)
    result = classification_metrics(ys, ps, ds, config["dataset_names"])
    result.update(loss=losses/len(ys), coverage=1., label_order=config["labels"])
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "metrics.json", result)
        with (output / "predictions.jsonl").open("w") as handle:
            for row, y, p, probability in zip(dataset.rows, ys, ps, probs):
                handle.write(json.dumps({"path": row["path"], "dataset": row["dataset"],
                    "subject": str(row["subject"]), "label": y, "prediction": p,
                    "probabilities": probability}) + "\n")
    return result


def checkpoint_state(model, mode):
    return {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()
            if mode == "full_sft" or k.startswith("head.") or k == "positions"}


def restore_checkpoint(model, state, mode):
    if mode == "full_sft":
        model.load_state_dict(state, strict=True)
    else:
        expected = {k for k in model.state_dict() if k.startswith("head.") or k == "positions"}
        if set(state) != expected:
            raise ValueError("Incomplete linear probe checkpoint")
        model.head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")}, strict=True)
        if not torch.equal(model.positions.cpu(), state["positions"].cpu()):
            raise ValueError("Changed electrode positions")


def verify_gradients(model):
    encoder_grads = {k: p.grad for k, p in model.encoder.named_parameters() if p.requires_grad}
    if model.mode == "linear_probe":
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Frozen encoder received a gradient")
    else:
        missing = [k for k, v in encoder_grads.items() if v is None]
        if missing:
            raise AssertionError(f"Full fine-tuning has disconnected parameters: {missing}")
        if not any(float(v.norm()) > 0 for v in encoder_grads.values()):
            raise AssertionError("Encoder gradients are all zero")
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(v).all() for v in grads):
        raise AssertionError("Missing/nonfinite gradients")
    return {"encoder_gradient_tensors": len(encoder_grads),
            "head_gradient_norm": float(model.head.weight.grad.norm()),
            "frozen_encoder_has_no_grad": model.mode == "linear_probe"}


def smoke(model, config, device, output):
    # Includes mixed lengths and an intentionally unused right-padded batch area.
    eeg = torch.randn(3, 3, 1000, device=device)
    lengths = torch.tensor([600, 1000, 800], device=device)
    labels = torch.tensor([0, 1, 0], device=device)
    before = model.encoder.to_patch_embedding[0].weight.detach().clone()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    out = model(eeg=eeg, lengths=lengths, labels=labels)
    out["loss"].backward()
    record = verify_gradients(model)
    optimizer.step()
    changed = not torch.equal(before, model.encoder.to_patch_embedding[0].weight)
    if changed != (model.mode == "full_sft"):
        raise AssertionError("Unexpected encoder parameter update")
    model.eval()
    with torch.inference_mode():
        reference = model(eeg=eeg, lengths=lengths)["logits"].cpu()
        contaminated = eeg.clone()
        for i, n in enumerate(lengths.tolist()):
            contaminated[i, :, n:] = 1e6
        torch.testing.assert_close(reference, model(eeg=contaminated, lengths=lengths)["logits"].cpu())
        cache = model.encode(eeg, lengths)
        if model.mode == "linear_probe":
            torch.testing.assert_close(reference, model(features=cache)["logits"].cpu())
    state = checkpoint_state(model, model.mode)
    save_file(state, str(output / "smoke.safetensors"))
    with torch.no_grad():
        model.head.weight.add_(1.)
    restore_checkpoint(model, load_file(str(output / "smoke.safetensors")), model.mode)
    with torch.inference_mode():
        torch.testing.assert_close(reference, model(eeg=eeg, lengths=lengths)["logits"].cpu())
    record.update(synthetic_only=True, mode=model.mode, encoder_updated=changed,
                  checkpoint_roundtrip=True, padding_invariant=True, loss=float(out["loss"].detach()))
    write_json(output / "smoke.json", record)
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", default="configs/reve_only_mi_multi.json")
    p.add_argument("--mode", choices=["linear_probe", "full_sft"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-local-reve-code", action="store_true")
    p.add_argument("--allow-unverified-provenance", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    source = json.loads(Path(settings["source_config"]).read_text())
    config = make_config(source, settings, args.mode)
    audit = audit_manifests(config, settings["reference_hashes"])
    if args.audit_only:
        print(json.dumps(audit, indent=2), flush=True)
        return
    if not args.allow_local_reve_code:
        p.error("Need explicit permission and --allow-local-reve-code before executing local custom model code")
    if not audit["provenance_verified"] and not args.allow_unverified_provenance:
        p.error("Inherited source limitations require --allow-unverified-provenance for exploratory runs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "split_audit.json", audit)
    source_files = [Path(__file__), Path(__file__).with_name("reve_baseline.py"),
                    Path(__file__).with_name("data.py"), Path(args.settings), Path(settings["source_config"])]
    for key in ("reve_checkpoint", "positions_checkpoint"):
        base = Path(config[key])
        source_files.extend(sorted(base.glob("*.py")))
        source_files.extend(base / name for name in ("config.json", "model.safetensors", "README.md"))
    hashes = {str(path): file_sha256(path) for path in source_files}
    write_json(output / "provenance.json", {"files": hashes, "split_hashes": audit["hashes"],
        "torch": torch.__version__, "numpy": np.__version__, "custom_code_authorized": True,
        "pretraining_overlap_verified": False})
    (output / "status.txt").write_text("loading\n")
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.monotonic()
    try:
        model = load_reve_classifier(config, args.mode, device, allow_local_reve_code=True)
        write_json(output / "loading_audit.json", model.loading_audit)
        if args.smoke:
            smoke(model, config, device, output)
            (output / "status.txt").write_text("smoke_complete\n")
            return
        root = Path(config["manifest_dir"])
        train_ds, val_ds = [MIDataset(root / f"{s}.jsonl", config) for s in ("train", "validation")]
        sampler = BalancedDistributedSampler(train_ds.rows, seed=config["seed"])
        train_cache = val_cache = None
        if args.mode == "linear_probe":
            (output / "status.txt").write_text("extracting_train_validation\n")
            cache = output / "features"
            cache.mkdir()
            train_cache = extract_features(model, train_ds, config, device, cache / "train.npy")
            val_cache = extract_features(model, val_ds, config, device, cache / "validation.npy")
            write_json(cache / "metadata.json", {"dtype": "float32", "shape": list(train_cache.shape),
                "encoder_sha256": hashes[str(Path(config["reve_checkpoint"]) / "model.safetensors")],
                "manifest_hashes": audit["hashes"], "pooling": config["pooling"]})
            labels = torch.tensor([r["label"] for r in train_ds.rows])
            train_loader = DataLoader(TensorDataset(train_cache, labels), batch_size=config["batch_size"],
                                      sampler=sampler, num_workers=0)
        else:
            train_loader = raw_loader(train_ds, config["batch_size"], sampler)
        # Reset after model loading/cache extraction so both initial heads and sampled
        # example order use the same declared seed, independent of I/O timing.
        seed_everything(config["seed"])
        model.head.reset_parameters()
        groups = [{"params": model.head.parameters(), "lr": config["head_lr"]}]
        if args.mode == "full_sft":
            groups.append({"params": [p for p in model.encoder.parameters() if p.requires_grad],
                           "lr": config["encoder_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        best_score, best_epoch, stale = -1., 0, 0
        gradient_check = None
        for epoch in range(1, config["epochs"] + 1):
            (output / "status.txt").write_text(f"training_epoch_{epoch}\n")
            epoch_start = time.monotonic()
            sampler.set_epoch(epoch-1)
            model.train()
            total_loss, seen = 0., 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                if train_cache is None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch.pop("dataset_index")
                    out = model(**batch)
                    count = len(batch["labels"])
                else:
                    features, labels = [v.to(device) for v in batch]
                    out = model(features=features, labels=labels)
                    count = len(labels)
                if not torch.isfinite(out["loss"]):
                    raise ValueError("Nonfinite training loss")
                out["loss"].backward()
                if gradient_check is None:
                    gradient_check = verify_gradients(model)
                    write_json(output / "gradient_check.json", gradient_check)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(out["loss"].detach()) * count
                seen += count
                if (i+1) % 100 == 0:
                    print(json.dumps({"mode": args.mode, "epoch": epoch, "step": i+1,
                                      "train_loss": total_loss/seen}), flush=True)
            metrics = evaluate(model, val_ds, config, device, cached=val_cache)
            score = metrics[config["selection_metric"]]
            record = {"epoch": epoch, "train_loss": total_loss/seen, "validation": metrics,
                      "seconds": time.monotonic()-epoch_start}
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                save_file(checkpoint_state(model, args.mode), str(output / "best.safetensors"))
                write_json(output / "best.json", record)
            else:
                stale += 1
            print(json.dumps(record), flush=True)
            if stale >= config["patience"]:
                break
        # Last checkpoint is useful for audits, but never replaces the val-selected best.
        save_file(checkpoint_state(model, args.mode), str(output / "last.safetensors"))
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        restore_checkpoint(model, load_file(str(output / "best.safetensors")), args.mode)
        selected = evaluate(model, val_ds, config, device, cached=val_cache)
        if abs(selected[config["selection_metric"]] - best_score) > 1e-7:
            raise AssertionError("Best checkpoint reload changed validation accuracy")
        write_json(output / "selected_validation.json", selected)
        results = {}
        # Access test/external EEG only after all training and model selection finish.
        for split in ("test", "external_2a", "external_2b"):
            (output / "status.txt").write_text(f"evaluating_{split}\n")
            dataset = MIDataset(root / f"{split}.jsonl", config)
            results[split] = evaluate(model, dataset, config, device, output=output / f"eval_{split}")
            print(json.dumps({"split": split, **results[split]}), flush=True)
        # Recheck manifests to detect changes made by another process during training.
        audit_manifests(config, settings["reference_hashes"])
        write_json(output / "result.json", {"mode": args.mode, "best_epoch": best_epoch,
            "epochs_run": epoch, "validation_macro_accuracy": best_score, "results": results,
            "seconds": time.monotonic()-started, "peak_cuda_gb": torch.cuda.max_memory_allocated(device)/1e9,
            "status": "complete"})
        (output / "status.txt").write_text("complete\n")
    except BaseException as exc:
        (output / "status.txt").write_text(f"failed: {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
