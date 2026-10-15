"""Independently verify saved models, predictions, metrics and completion gates."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import psutil
from safetensors import safe_open
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from .data import read_jsonl
from .decision_data import LABELS, file_sha256
from .decision_model import source_signature
from .summarize_experiments import ROOT, RUNS
from .train_decision import json_write


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_probabilities(predictions, manifest):
    expected = {row["id"]: LABELS.index(row["label"]) for row in read_jsonl(manifest)}
    require(len(predictions) == len(expected), "Prediction count mismatch")
    require({p["id"] for p in predictions} == set(expected), "Prediction IDs mismatch")
    require(all(p["target"] == expected[p["id"]] for p in predictions), "Prediction targets mismatch")
    y = np.array([p["target"] for p in predictions])
    scores = np.array([p["p_seiz"] for p in predictions])
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), "Invalid probabilities")
    pred = scores >= .5
    tp, tn = int(((y == 1) & pred).sum()), int(((y == 0) & ~pred).sum())
    fp, fn = int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {"auroc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "accuracy": float((y == pred).mean()), "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compare_metrics(actual, expected, description):
    for key, value in actual.items():
        require(abs(value - expected[key]) < 1e-10, f"{description}: {key} differs")


def main():
    require(read(ROOT / "suite_state.json")["status"] == "complete", "Training suite is not complete")
    require(read(ROOT / "evaluation_state.json")["status"] == "complete", "Final evaluations are not complete")
    live = []
    for process in psutil.process_iter(["pid", "cmdline", "status"]):
        command = process.info["cmdline"] or []
        if process.info["status"] != psutil.STATUS_ZOMBIE and any(
                module in command for module in ("tusz_sft.train_decision", "tusz_sft.evaluate_decision")):
            live.append(process.pid)
    require(not live, f"Training/evaluation processes remain: {live}")
    report = {"status": "verified_complete", "training": {}, "evaluations": {}}
    for name in RUNS:
        directory = ROOT / "runs" / name
        config, preflight = read(directory / "run_config.json"), read(directory / "preflight.json")
        result, best = read(directory / "result.json"), read(directory / "best.json")
        configured = read(ROOT.parents[1] / "configs" / (name + ".json"))
        require(all(config[key] == value for key, value in configured.items()), f"{name}: launch config changed")
        train_rows = list(read_jsonl(config["train_data"]))
        validation_rows = list(read_jsonl(config["validation_data"]))
        require(not ({r["subject"] for r in train_rows} & {r["subject"] for r in validation_rows}), "Current train/validation subject overlap")
        require(file_sha256(config["train_data"]) == preflight["train_sha256"], "Training manifest changed")
        require(file_sha256(config["validation_data"]) == preflight["validation_sha256"], "Validation manifest changed")
        batches = len(train_rows) // (preflight["world_size"] * config["batch_size"])
        expected_steps = math.ceil(batches / config["grad_accum"]) * config["epochs"]
        require(result["status"] == "complete" and result["epoch"] == config["epochs"], f"{name}: incomplete epochs")
        require(result["step"] == expected_steps == preflight["total_steps"], f"{name}: incomplete optimizer steps")
        logs = list(read_jsonl(directory / "train.jsonl"))
        require([r["step"] for r in logs] == list(range(1, expected_steps + 1)), "Missing/duplicate optimizer steps")
        require(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in logs), "Nonfinite optimization")
        candidates = [read(directory / "initial_validation.json"), *read_jsonl(directory / "validation.jsonl")]
        selected = max(candidates, key=lambda row: row["auroc"])
        require(selected["step"] == best["step"] and selected["auroc"] == best["auroc"], "Checkpoint selection inconsistent")
        checkpoint = Path(best["checkpoint"])
        meta = read(checkpoint / "decision_config.json")
        require(meta["source"] == preflight["source"] == source_signature(config["base"]), "Source model changed")
        with safe_open(checkpoint / "decision_head.safetensors", framework="pt", device="cpu") as f:
            head = f.get_tensor("weight")
            require(list(head.shape) == [2, meta["hidden_size"]] and bool(torch.isfinite(head).all()), "Invalid decision head")
            require(head.dtype == torch.float32, "Decision head must be stored in float32")
        adapter = checkpoint / "adapter/adapter_model.safetensors"
        tensor_count = 0
        if meta["lora_rank"]:
            with safe_open(adapter, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                tensor_count = len(keys)
                require(tensor_count == 2 * len(meta["target_modules"]), "Missing LoRA tensors")
                require(all("language_model.layers." in k and "lora_" in k and "visual" not in k and "lm_head" not in k for k in keys), "Unexpected trainable modules")
                require(all(bool(torch.isfinite(f.get_tensor(k)).all()) for k in keys), "Nonfinite LoRA weights")
            require(result["first_lora_B_max_update"] > 0, "LoRA did not update")
        else:
            require(not adapter.exists(), "Head-only run unexpectedly saved an adapter")
            require(read(directory / "model_audit.json")["trainable_parameters"] == 2 * meta["hidden_size"], "Head-only parameter mismatch")
        require(result["head_max_update"] > 0, "Decision head did not update")
        predictions = read(directory / f"validation_step_{best['step']:06d}.json")
        metrics = verify_probabilities(predictions, config["validation_data"])
        compare_metrics(metrics, best, name)
        report["training"][name] = {"epochs": result["epoch"], "steps": expected_steps,
                                    "best_step": best["step"], "validation_auroc": metrics["auroc"],
                                    "adapter_tensors": tensor_count, "source_unchanged": True}
    comparison = {r["name"]: r for r in read(ROOT / "comparison.json")}
    development_subjects = {r["subject"] for name in ("train.fit.jsonl", "train.validation.jsonl")
                            for r in read_jsonl(ROOT / "waveform" / name)}
    matched = None
    for name in ["rl_initial", *RUNS]:
        directory = ROOT / "evaluations" / name
        metrics, predictions = read(directory / "metrics.json"), read(directory / "predictions.json")
        require(metrics["status"] == "complete", "Incomplete dev evaluation")
        mode = metrics["input_mode"]
        folder = {"waveform": "dev_waveform", "spectrum": "dev_spectrum", "fusion": "dev_spectrum/fusion"}[mode]
        manifest = ROOT / folder / "dev.manifest.jsonl"
        require(metrics["manifest_sha256"] == file_sha256(manifest), "Dev manifest changed")
        expected_checkpoint = (ROOT / "runs/waveform_lr1e5/checkpoints/step-000000" if name == "rl_initial"
                               else Path(read(ROOT / "runs" / name / "best.json")["checkpoint"]))
        require(Path(metrics["checkpoint"]) == expected_checkpoint, "Evaluation used wrong checkpoint")
        recalculated = verify_probabilities(predictions, manifest)
        compare_metrics(recalculated, metrics, name + " dev")
        cases = [(p["id"], p["target"], p["subject"]) for p in predictions]
        require(not (development_subjects & {p["subject"] for p in predictions}), "Current development/dev subject overlap")
        if matched is None:
            matched = cases
        else:
            require(cases == matched, "Dev cases differ across methods")
        require(len(cases) == 1024 and metrics["threshold"] == .5, "Dev sample/threshold changed")
        row = comparison[name]
        require(abs(row["dev_auroc"] - metrics["auroc"]) < 1e-12, "Summary AUROC mismatch")
        require(0 <= row["auroc_ci_lower"] <= row["auroc_ci_upper"] <= 1, "Invalid confidence interval")
        report["evaluations"][name] = {"cases": len(cases), "recomputed_auroc": recalculated["auroc"],
                                       "checkpoint": str(expected_checkpoint)}
    for filename in ("RESULTS.md", "comparison.csv", "comparison.json", "comparison.png"):
        require((ROOT / filename).stat().st_size > 0, f"Missing final artifact: {filename}")
    with (ROOT / "comparison.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    require({r["name"] for r in csv_rows} == set(comparison), "CSV methods differ from JSON report")
    for row in csv_rows:
        for field in ("dev_auroc", "sensitivity", "specificity", "selected_step"):
            require(abs(float(row[field]) - comparison[row["name"]][field]) < 1e-12, "CSV metric mismatch")
    json_write(ROOT / "completion_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
