"""Collect completed runs and fixed dev evaluations without choosing on dev."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from .train_decision import json_write

ROOT = Path(__file__).resolve().parents[1] / "artifacts/eeg_decision_experiments"
RUNS = ["waveform_lr1e5", "waveform_lr5e5", "spectrum_lr1e5", "spectrum_lr5e5",
        "fusion", "head_only", "native_waveform"]


def subject_bootstrap(predictions, repeats=1000, seed=20261014):
    subjects = sorted({r["subject"] for r in predictions})
    y = np.array([r["target"] for r in predictions])
    p = np.array([r["p_seiz"] for r in predictions])
    groups = {s: np.array([i for i, r in enumerate(predictions) if r["subject"] == s]) for s in subjects}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([groups[s] for s in rng.choice(subjects, len(subjects), replace=True)])
        if len(np.unique(y[indices])) == 2:
            values.append(roc_auc_score(y[indices], p[indices]))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    results = []
    expected_ids = None
    for name in ["rl_initial", *RUNS]:
        directory = ROOT / "evaluations" / name
        metrics = json.loads((directory / "metrics.json").read_text())
        pred = json.loads((directory / "predictions.json").read_text())
        ids = [(r["id"], r["target"], r["subject"]) for r in pred]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("Evaluation cases are not matched across methods")
        ci = subject_bootstrap(pred)
        result = {"name": name, "dev_auroc": metrics["auroc"], "auroc_ci_lower": ci[0],
                  "auroc_ci_upper": ci[1], "sensitivity": metrics["sensitivity"],
                  "specificity": metrics["specificity"], "balanced_accuracy": metrics["balanced_accuracy"],
                  "average_precision": metrics["average_precision"], "dev_n": metrics["n"],
                  "selected_step": 0, "best_internal_auroc": None,
                  "checkpoint": metrics["checkpoint"]}
        if name != "rl_initial":
            run = ROOT / "runs" / name
            final = json.loads((run / "result.json").read_text())
            if final["status"] != "complete":
                raise ValueError(f"Incomplete training: {name}")
            best = json.loads((run / "best.json").read_text())
            result.update(selected_step=best["step"], best_internal_auroc=best["auroc"],
                          optimizer_steps=final["step"], peak_cuda_gib=final["peak_cuda_memory_gib_rank0"])
        results.append(result)
    json_write(ROOT / "comparison.json", results)
    columns = ["name", "selected_step", "best_internal_auroc", "dev_auroc", "auroc_ci_lower",
               "auroc_ci_upper", "sensitivity", "specificity", "balanced_accuracy", "average_precision", "dev_n"]
    with (ROOT / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Fixed 1,024-window dev subset; prior EEG-RL patient overlap remains", fontsize=10)
    for name in RUNS:
        run = ROOT / "runs" / name
        rows = [json.loads((run / "initial_validation.json").read_text())]
        rows += [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines()]
        axes[0].plot([r["step"] for r in rows], [r["auroc"] for r in rows], marker=".", label=name)
    axes[0].set(xlabel="Optimizer step", ylabel="Internal validation AUROC", ylim=(0, 1))
    axes[0].legend(fontsize=7)
    x = np.array([r["dev_auroc"] for r in results])
    axes[1].hlines(np.arange(len(results)), [r["auroc_ci_lower"] for r in results],
                   [r["auroc_ci_upper"] for r in results], color="C0")
    axes[1].plot(x, np.arange(len(results)), "o", color="C0")
    axes[1].set_yticks(np.arange(len(results)), [r["name"] for r in results])
    axes[1].set(xlabel="Dev AUROC (95% subject bootstrap interval)", xlim=(0, 1))
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# EEG 决策实验结果", "", "所有训练与评估任务均已正常结束。每个训练模型只根据内部验证 AUROC 选 checkpoint；dev 未用于参数或阈值选择。", "",
             "| 实验 | 最佳 step | 内部 AUROC | dev AUROC（95%患者 bootstrap） | 敏感度 | 特异度 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        internal = "—" if row["best_internal_auroc"] is None else f"{row['best_internal_auroc']:.3f}"
        lines.append(f"| {row['name']} | {row['selected_step']} | {internal} | {row['dev_auroc']:.3f} "
                     f"[{row['auroc_ci_lower']:.3f}, {row['auroc_ci_upper']:.3f}] | "
                     f"{row['sensitivity']:.3f} | {row['specificity']:.3f} |")
    lines += ["", "局限：dev 是固定的 512 发作 + 512 非发作窗口子集，剔除了可用旧训练清单中的同一窗口；"
              "但 47 名患者中的 45 名出现在旧语料清单中。因此 EEG RL 起点的结果不是严格的新患者泛化成绩，"
              "相邻窗口和未完整记录的旧训练语料仍可能造成相关性。原生 8B 对照没有经过这条本地 EEG SFT/RL 链路，"
              "但不能证明公共基础模型所有预训练数据的来源。置信区间按患者重采样，未包含训练随机种子的变异。",
              "", "曲线与对比：[comparison.png](comparison.png)。完整指标：[comparison.csv](comparison.csv)。"
              "底座、数据哈希、配置、各样本概率均保存在各运行和评估目录中。", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("Wrote RESULTS.md, comparison.csv/json/png", flush=True)


if __name__ == "__main__":
    main()
