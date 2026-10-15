"""Identical weighted, regularized logistic probes on frozen intermediate states."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .extract_stage_features import SOURCE, STAGES
from .reve_baseline import SPLITS, audit_manifests, classification_metrics, file_sha256
from .train_reve_only import write_json


LAMBDAS = (1e-4, 1e-3, 1e-2, .1, 1.)


def group_weights(rows):
    groups = [(r["dataset"], r["label"]) for r in rows]
    counts = Counter(groups)
    w = np.array([1 / counts[g] for g in groups], dtype=np.float64)
    return w / w.mean()


def load_features(root, split, stage, n, world):
    values = np.empty((n, STAGES[stage]), dtype=np.float32)
    original = np.empty((n, 2), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    for rank in range(world):
        folder = root / f"rank_{rank}"
        if (folder / "status.txt").read_text().strip() != "complete":
            raise ValueError(f"Incomplete extraction {rank}")
        indices = np.load(folder / split / "indices.npy", allow_pickle=False)
        if len(set(indices.tolist())) != len(indices) or seen[indices].any():
            raise ValueError("Overlapping feature shards")
        feature = np.load(folder / split / f"{stage}.npy", mmap_mode="r", allow_pickle=False)
        if feature.shape != (len(indices), STAGES[stage]) or not np.isfinite(feature).all():
            raise ValueError("Invalid feature cache")
        values[indices] = feature
        original[indices] = np.load(folder / split / "original_logits.npy", allow_pickle=False)
        seen[indices] = True
    if not seen.all():
        raise ValueError("Incomplete feature coverage")
    return values, original


def fit_logistic(x, y, sample_weight, regularization, max_iter=300):
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1., max_iter=max_iter, max_eval=max_iter*2,
        tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    calls = 0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none") * sample_weight).mean()
        loss = loss + .5 * regularization * w.square().sum()
        loss.backward()
        calls += 1
        return loss
    optimizer.step(closure)
    objective = float(closure().detach())
    grad_max = max(float(w.grad.abs().max()), float(b.grad.abs()))
    if not np.isfinite(objective) or not np.isfinite(grad_max):
        raise ValueError("Nonfinite optimizer result")
    return w.detach(), b.detach(), {"objective": objective, "gradient_max_abs": grad_max,
        "iterations": optimizer.state[w]["n_iter"], "function_evaluations": calls,
        "iteration_limit_reached": optimizer.state[w]["n_iter"] >= max_iter}


def metrics_from_scores(scores, rows, names):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.array([r["label"] for r in rows])
    pred = (scores > 0).astype(np.int64)
    result = classification_metrics(labels, pred, [r["dataset_index"] for r in rows], names)
    result["log_loss"] = float(np.mean(np.logaddexp(0, scores) - labels*scores))
    # Rank AUC with average ranks for ties; no extra classifier/threshold fitting.
    from scipy.stats import rankdata
    positive = labels == 1
    ranks = rankdata(scores)
    result["roc_auc"] = float((ranks[positive].sum() - positive.sum()*(positive.sum()+1)/2) /
                              (positive.sum()*(~positive).sum()))
    result["mean_confidence"] = float(np.mean(1 / (1 + np.exp(-np.abs(scores)))))
    result["predicted_right_fraction"] = float(pred.mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--probe-dir", default="probes")
    args = parser.parse_args()
    root = Path(args.root)
    if args.max_iter < 1 or Path(args.probe_dir).name != args.probe_dir:
        parser.error("Invalid iteration budget/output subdirectory")
    output = root / args.probe_dir / args.stage
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((SOURCE / "config.json").read_text())
    audit = audit_manifests(cfg, SOURCE / "source_and_manifest_hashes.json")
    write_json(output / "protocol.json", {"stage": args.stage, "lambda_grid": LAMBDAS,
        "max_iter": args.max_iter, "solver": "full_batch_LBFGS_strong_wolfe_float64", "weighting": "dataset_class_balanced",
        "normalization": "training_only_featurewise_mean_std", "selection": "validation_macro_accuracy_then_log_loss",
        "classification_bias": True, "seed": "zero_initialization_convex_objective",
        "dimension": STAGES[args.stage], "source_hash": file_sha256(__file__), "split_hashes": audit["hashes"]})
    rows = {split: [json.loads(s) for s in (Path(cfg["manifest_dir"]) / f"{split}.jsonl").read_text().splitlines()]
            for split in SPLITS}
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    try:
        (output / "status.txt").write_text("fitting\n")
        x0, _ = load_features(root, "train", args.stage, len(rows["train"]), args.world)
        v0, _ = load_features(root, "validation", args.stage, len(rows["validation"]), args.world)
        x = torch.tensor(x0, dtype=torch.float64, device=device)
        mean = x.mean(0)
        std = x.std(0, correction=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        x.sub_(mean).div_(std)
        v = (torch.tensor(v0, dtype=torch.float64, device=device)-mean)/std
        del x0, v0
        y = torch.tensor([r["label"] for r in rows["train"]], dtype=torch.float64, device=device)
        sample_weight = torch.tensor(group_weights(rows["train"]), dtype=torch.float64, device=device)
        best_key, selected, candidates = None, None, []
        for regularization in LAMBDAS:
            w, b, optimizer_audit = fit_logistic(x, y, sample_weight, regularization, max_iter=args.max_iter)
            val = metrics_from_scores((v @ w + b).cpu().numpy(), rows["validation"], cfg["dataset_names"])
            candidate = {"lambda": regularization, "validation": val, "optimizer": optimizer_audit}
            candidates.append(candidate)
            key = (val["macro_dataset_accuracy"], -val["log_loss"])
            if best_key is None or key > best_key:
                best_key, selected = key, {**candidate, "weight": w.clone(), "bias": b.clone()}
            print(json.dumps({"stage": args.stage, "lambda": regularization,
                "val_macro": val["macro_dataset_accuracy"], "grad": optimizer_audit["gradient_max_abs"],
                "iterations": optimizer_audit["iterations"]}), flush=True)
        write_json(output / "selection.json", {"candidates": candidates, "selected_lambda": selected["lambda"],
            "selection_complete_before_test_evaluation": True})
        w, b = selected.pop("weight"), selected.pop("bias")
        np.savez(output / "probe.npz", weight=w.cpu().numpy(), bias=b.cpu().numpy(),
                 mean=mean.cpu().numpy(), std=std.cpu().numpy())
        train_metrics = metrics_from_scores((x @ w + b).cpu().numpy(), rows["train"], cfg["dataset_names"])
        del x, v
        results, originals = {}, {}
        for split in ("test", "external_2a", "external_2b"):
            (output / "status.txt").write_text(f"evaluating_{split}\n")
            data, original = load_features(root, split, args.stage, len(rows[split]), args.world)
            xt = (torch.tensor(data, dtype=torch.float64, device=device)-mean)/std
            scores = (xt @ w + b).cpu().numpy()
            results[split] = metrics_from_scores(scores, rows[split], cfg["dataset_names"])
            originals[split] = metrics_from_scores(original[:, 1]-original[:, 0], rows[split], cfg["dataset_names"])
            with (output / f"{split}_predictions.jsonl").open("w") as handle:
                for row, score in zip(rows[split], scores):
                    handle.write(json.dumps({"path": row["path"], "dataset": row["dataset"], "subject": row["subject"],
                        "label": row["label"], "prediction": int(score > 0), "log_odds_right": float(score)}) + "\n")
            del data, xt
        audit_manifests(cfg, SOURCE / "source_and_manifest_hashes.json")
        write_json(output / "result.json", {"stage": args.stage, "selected": selected, "train": train_metrics,
            "results": results, "original_head": originals, "seconds": time.monotonic()-started,
            "peak_cuda_gb": torch.cuda.max_memory_allocated()/1e9, "status": "complete"})
        (output / "status.txt").write_text("complete\n")
        print(json.dumps({"stage": args.stage, "results": results}), flush=True)
    except BaseException as exc:
        (output / "status.txt").write_text(f"failed: {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
