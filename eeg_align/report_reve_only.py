"""Audit completed REVE-only predictions and generate the matched-split report."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .reve_baseline import classification_metrics, file_sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_predictions(manifest, prediction_path, metric_path, dataset_names):
    truth, predictions = read_jsonl(manifest), read_jsonl(prediction_path)
    if len(truth) != len(predictions):
        raise ValueError("Prediction count differs from locked manifest")
    for expected, actual in zip(truth, predictions):
        for key in ("path", "dataset", "label", "subject"):
            if str(expected[key]) != str(actual[key]):
                raise ValueError(f"Prediction/manifest mismatch: {key}")
        probability = np.asarray(actual["probabilities"])
        if (probability.shape != (2,) or not np.isfinite(probability).all()
                or not np.isclose(probability.sum(), 1., atol=1e-6)
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError("Invalid probability vector")
        if int(probability.argmax()) != actual["prediction"]:
            raise ValueError("Prediction is not probability argmax")
    recomputed = classification_metrics([p["label"] for p in predictions],
        [p["prediction"] for p in predictions], [p["dataset_index"] for p in truth], dataset_names)
    stored = read_json(metric_path)
    for key in recomputed:
        if recomputed[key] != stored[key]:
            raise ValueError(f"Stored/recomputed metric differs: {key}")
    per_subject = defaultdict(list)
    for row in predictions:
        per_subject[(row["dataset"], str(row["subject"]))].append(row["prediction"] == row["label"])
    subjects = [{"dataset": dataset, "subject": subject, "n": len(correct),
                 "correct": sum(correct), "accuracy": float(np.mean(correct))}
                for (dataset, subject), correct in sorted(per_subject.items())]
    return recomputed, subjects, predictions


def exact_counts(metric):
    cm = np.asarray(metric["confusion_matrix"])
    if int(cm.sum()) != metric["n"]:
        raise ValueError("Inconsistent confusion matrix total")
    correct, n = int(cm.trace()), int(cm.sum())
    if not np.isclose(correct / n, metric["accuracy"], atol=1e-7):
        raise ValueError("Inconsistent accuracy")
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--previous", default="artifacts/reve_mi_multi/train_4gpu_20261015_121912")
    args = parser.parse_args()
    root, previous = Path(args.run), Path(args.previous)
    targets = [root / name for name in ("comparison.json", "per_subject.json", "RESULTS.md")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite an existing report")
    modes = ("linear_probe", "full_sft")
    all_metrics, runs, subjects, pred_by_mode, provenance = {}, {}, {}, {}, {}
    for mode in modes:
        folder = root / mode
        if (folder / "status.txt").read_text().strip() != "complete":
            raise ValueError(f"Not complete: {mode}")
        runs[mode] = read_json(folder / "result.json")
        config = read_json(folder / "config.json")
        all_metrics[mode], subjects[mode], pred_by_mode[mode] = {}, {}, {}
        for split in ("test", "external_2a", "external_2b"):
            manifest = Path(config["manifest_dir"]) / f"{split}.jsonl"
            audit = read_json(folder / "split_audit.json")
            if file_sha256(manifest) != audit["hashes"][manifest.name]:
                raise ValueError("Manifest changed after training")
            ppath, mpath = [folder / f"eval_{split}" / name for name in ("predictions.jsonl", "metrics.json")]
            metrics, people, rows = audit_predictions(manifest, ppath, mpath, config["dataset_names"])
            all_metrics[mode][split], subjects[mode][split], pred_by_mode[mode][split] = metrics, people, rows
            for path in (manifest, ppath, mpath):
                provenance[str(path)] = file_sha256(path)
        for name in ("best.safetensors", "result.json", "config.json", "provenance.json"):
            path = folder / name
            provenance[str(path)] = file_sha256(path)
    all_metrics["previous_reve_qwen"] = {}
    previous_hashes = {Path(k).name: v for k, v in read_json(previous / "source_and_manifest_hashes.json").items()}
    for split in ("test", "external_2a", "external_2b"):
        manifest = Path(config["manifest_dir"]) / f"{split}.jsonl"
        if file_sha256(manifest) != previous_hashes[manifest.name]:
            raise ValueError("Previous model used a different split")
        path = previous / f"eval_{split}" / "metrics.json"
        all_metrics["previous_reve_qwen"][split] = read_json(path)
        provenance[str(path)] = file_sha256(path)
    table = []
    for name, split, dataset in (("Cho2017", "test", "Cho2017"),
            ("Lee2019_MI", "test", "Lee2019_MI"), ("Dreyer2023", "test", "Dreyer2023"),
            ("内部测试合计", "test", None), ("BCIC-IV-2a（左右手）", "external_2a", None),
            ("BCIC-IV-2b", "external_2b", None)):
        row = {"dataset": name}
        for method, metrics in all_metrics.items():
            selected = metrics[split] if dataset is None else metrics[split]["per_dataset"][dataset]
            row[method] = exact_counts(selected)
        if len({row[method]["n"] for method in all_metrics}) != 1:
            raise ValueError("Comparison sample counts differ")
        table.append(row)
    paired = {}
    for split in pred_by_mode["linear_probe"]:
        a, b = pred_by_mode["linear_probe"][split], pred_by_mode["full_sft"][split]
        paired[split] = {"full_only_correct": sum(x["prediction"] != x["label"] and y["prediction"] == y["label"] for x, y in zip(a, b)),
                         "probe_only_correct": sum(x["prediction"] == x["label"] and y["prediction"] != y["label"] for x, y in zip(a, b))}
    report = {"table": table, "runs": runs, "paired_descriptive_counts": paired,
              "prediction_audit_passed": True, "source_hashes": provenance,
              "previous_model": "frozen REVE + connector + CerebraGloss Qwen3-VL-8B language tower LoRA; not JEV-27B"}
    targets[0].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    targets[1].write_text(json.dumps(subjects, indent=2, ensure_ascii=False) + "\n")
    lines = ["# REVE-only：固定划分实验结果", "", "日期：2026-10-15。两组训练、最佳 checkpoint 重载和三个测试均已完成。",
        "逐样本预测已重新与锁定 manifest 核对（路径、数据集、被试、标签、概率及 argmax），准确率/混淆矩阵重算一致。", "",
        "## 测试准确率", "", "所有列使用同一测试划分。2a **仅左右手二分类**，不是四分类。全部结果覆盖率 100%。", "",
        "| 数据集 | N | 冻结 REVE + Linear | 全量 REVE + Linear | 上轮 REVE→Qwen |",
        "|---|---:|---:|---:|---:|"]
    for row in table:
        values = [f'{row[m]["accuracy"]*100:.2f}% ({row[m]["correct"]}/{row[m]["n"]})'
                  for m in (*modes, "previous_reve_qwen")]
        lines.append(f'| {row["dataset"]} | {row["linear_probe"]["n"]} | ' + " | ".join(values) + " |")
    lines.extend(["", "上轮 Qwen 指 CerebraGloss Qwen3-VL-8B 的语言塔 + LoRA，不是 JEV-27B。",
        "内部合计是按试次数加权的准确率，不是三个数据集的等权平均。", "", "## 训练与模型选择", "",
        "| 方案 | 实际 epochs | 最佳 epoch | 验证集数据集宏平均 | 总耗时（分钟） | 峰值 CUDA 分配（GB） |",
        "|---|---:|---:|---:|---:|---:|"])
    for mode in modes:
        run = runs[mode]
        lines.append(f'| {mode} | {run["epochs_run"]} | {run["best_epoch"]} | {run["validation_macro_accuracy"]*100:.2f}% | {run["seconds"]/60:.2f} | {run["peak_cuda_gb"]:.2f} |')
    lines.extend(["", "两组独立从同一预训练 REVE 初始化；逐通道时间均值后拼接，Linear(1536,2)。",
        "LP 仅优化 3,074 个头参数；FT 优化 69,192,194 个参数（唯一未使用的 cls_query_token 保持冻结）。",
        "seed=42、FP32、batch=64、dataset×class 均衡采样、AdamW。LP lr=1e-3；FT encoder lr=1e-5、head lr=1e-3。",
        "只按验证集三个数据集准确率的等权平均选模型。训练结束后才评估内部测试和外部 2a/2b；未按测试结果调参。", "",
        "## 解读边界", "", "- 这是单种子、固定首轮超参数的探索性结果，不代表模型上限或显著性检验。",
        "- 仅 C3/Cz/C4，200 Hz、逐试次逐通道 z-score/clip15；与上轮一致。没有使用全部电极，也没有额外 baseline/ERD 特征。",
        "- 时间均值池化损失时序信息；LP、FT、REVE→Qwen 的训练预算和可训练参数不同，不能将差异完全归因于结构。",
        "- 2a/2b 不参与本轮监督微调和模型选择；REVE 预训练数据重叠尚未核验，不能声称基础模型从未见过这些数据集。",
        "- 2b 双极导联与其他来源存在参考差异；上游部分来源元数据尚未从原始记录完整重建。保留 Dreyer 13/14 隔离。",
        "- 同一被试内试次相关，不将试次当成独立被试来宣称统计显著。per_subject.json 保留各被试结果。", "",
        "## 可追溯产物", "", "- [机器可读对比与输入哈希](comparison.json)",
        "- [逐被试结果](per_subject.json)", "- [Linear probe 最佳权重](linear_probe/best.safetensors)",
        "- [全量微调最佳权重](full_sft/best.safetensors)",
        "- 每组 eval_test、eval_external_2a、eval_external_2b 下保存 metrics.json 与 predictions.jsonl。",
        "- [实验协议](../../../eeg_align/REVE_ONLY_EXPERIMENT.md)，含本地加载许可、技能流程、数据限制和复现命令。", ""])
    targets[2].write_text("\n".join(lines))
    print(json.dumps({"report": str(targets[2]), "prediction_audit_passed": True, "table": table}, ensure_ascii=False))


if __name__ == "__main__":
    main()
