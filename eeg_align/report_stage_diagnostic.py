"""Verify probe artifacts and summarize the readout diagnosis without changing models."""
import argparse
import json
from pathlib import Path

import numpy as np

from .extract_stage_features import SOURCE, STAGES
from .fit_stage_probe import load_features, metrics_from_scores
from .reve_baseline import file_sha256
from .train_reve_only import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--probe-dir", default="converged")
    args = p.parse_args()
    root = Path(args.root)
    if (root / "RESULTS.md").exists() or (root / "comparison.json").exists():
        raise FileExistsError("Refuse report overwrite")
    cfg = json.loads((SOURCE / "config.json").read_text())
    results, hashes = {}, {}
    for stage in STAGES:
        folder = root / args.probe_dir / stage
        if (folder / "status.txt").read_text().strip() != "complete":
            raise ValueError(f"Incomplete {stage}")
        result = json.loads((folder / "result.json").read_text())
        for split in ("test", "external_2a", "external_2b"):
            truth = [json.loads(s) for s in (Path(cfg["manifest_dir"]) / f"{split}.jsonl").read_text().splitlines()]
            path = folder / f"{split}_predictions.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            if len(rows) != len(truth):
                raise ValueError("Wrong prediction length")
            for expected, actual in zip(truth, rows):
                if any(str(expected[k]) != str(actual[k]) for k in ("path", "subject", "label", "dataset")):
                    raise ValueError("Prediction identity mismatch")
                if actual["prediction"] != int(actual["log_odds_right"] > 0):
                    raise ValueError("Wrong score sign")
            check = metrics_from_scores([r["log_odds_right"] for r in rows], truth, cfg["dataset_names"])
            if check != result["results"][split]:
                raise ValueError("Prediction/metric mismatch")
            hashes[str(path)] = file_sha256(path)
        for name in ("result.json", "selection.json", "probe.npz", "protocol.json"):
            hashes[str(folder/name)] = file_sha256(folder/name)
        results[stage] = result
    original = results["reve"]["original_head"]
    if any(result["original_head"] != original for result in results.values()):
        raise ValueError("Different original predictions across stages")
    numerical = {}
    for split in ("test", "external_2a", "external_2b"):
        old = json.loads((SOURCE / f"eval_{split}/metrics.json").read_text())
        numerical[split] = {"historical_accuracy": old["accuracy"],
            "current_accuracy": original[split]["accuracy"],
            "delta_percentage_points": 100*(original[split]["accuracy"]-old["accuracy"])}
    feature_check = {}
    for split in ("train", "validation"):
        prior = np.load(Path("artifacts/reve_only_multi/train_20261015_1404/linear_probe/features") / f"{split}.npy", mmap_mode="r")
        current, _ = load_features(root, split, "reve", len(prior), 4)
        delta = np.abs(current-prior)
        relative = np.linalg.norm(delta, axis=1) / np.linalg.norm(prior, axis=1).clip(1e-12)
        feature_check[split] = {"max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
                               "mean_relative_row_norm": float(relative.mean()), "max_relative_row_norm": float(relative.max()),
                               "allclose_rtol_2e5_atol_2e4": bool(np.allclose(current, prior, rtol=2e-5, atol=2e-4))}
        # An initial absolute 0.01 guard triggered. A same-batch direct experiment
        # showed exact equality of EEGQwen.encode vs ReveClassifier.encode, and
        # reproduced both caches exactly with their respective 64/128 batches.
        # Preserve/report the deviations; compare relative feature scale as well.
        from safetensors.numpy import load_file
        lp = load_file("artifacts/reve_only_multi/train_20261015_1404/linear_probe/best.safetensors")
        old_logits = prior @ lp["head.weight"].T + lp["head.bias"]
        new_logits = current @ lp["head.weight"].T + lp["head.bias"]
        feature_check[split]["old_linear_head_prediction_disagreements"] = int(np.sum(old_logits.argmax(1) != new_logits.argmax(1)))
        if float(relative.mean()) > .01:
            raise ValueError("Unexpected relative REVE feature discrepancy")
        del current, delta
    comparison = {"probe_dir": args.probe_dir, "results": results, "original_head": original, "original_rerun_difference": numerical,
                  "reve_cache_check": feature_check, "prediction_audit_passed": True, "hashes": hashes}
    write_json(root / "comparison.json", comparison)
    labels = {"reve": "REVE → 新线性探针", "connector": "Connector → 新线性探针",
              "qwen_eeg": "Qwen EEG token → 新线性探针", "qwen_answer": "Qwen 最后读出 → 新线性探针"}
    lines = ["# REVE→Qwen：阶段可读性诊断", "", "2026-10-15。原模型冻结，只训练诊断线性探针；不是新的端到端模型成绩。",
             "四组逐样本预测已与 manifest 核对并重算指标。所有参数只按源域 train/validation 拟合/选择。", "",
             "## 相同探针训练规则下的测试结果", "",
             "| 读出位置 | 内部测试 | Cho2017 | Lee2019_MI | Dreyer2023 | 外部 2a | 外部 2b | 2b AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in [*STAGES, "original"]:
        rr = original if stage == "original" else results[stage]["results"]
        scores = [rr["test"]["accuracy"], *[rr["test"]["per_dataset"][d]["accuracy"] for d in ("Cho2017", "Lee2019_MI", "Dreyer2023")],
                  rr["external_2a"]["accuracy"], rr["external_2b"]["accuracy"]]
        name = "原已训练 Qwen 分类头（重跑）" if stage == "original" else labels[stage]
        lines.append("| " + name + " | " + " | ".join(f"{v*100:.2f}%" for v in scores) + f' | {rr["external_2b"]["roc_auc"]:.4f} |')
    lines.extend(["", "2a 仅左右手二分类。内部测试 N=4200，2a N=1296，2b N=2840。所有试次均分类，不拒绝。", "",
                  "## 拟合/验证审计", "", "| 位置 | 维度 | λ | 训练准确率 | 验证宏平均 | 最终最大梯度 | 迭代数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for stage, result in results.items():
        selected = result["selected"]
        opt = selected["optimizer"]
        lines.append(f'| {stage} | {STAGES[stage]} | {selected["lambda"]:g} | {result["train"]["accuracy"]*100:.2f}% | {selected["validation"]["macro_dataset_accuracy"]*100:.2f}% | {opt["gradient_max_abs"]:.2e} | {opt["iterations"]} |')
    lines.extend(["", "训练特征标准化、含 bias 的 logistic、dataset×class 均衡加权、FP64 全批量 L-BFGS；固定五档 L2 正则。",
        "首轮上限 300 次迭代，因部分探针未收敛，统一将上限提高到 2000 次复核；没有改变 λ 候选或验证集选择规则。选中探针全部在上限前停止，最大梯度约 1.6e-7–3.0e-7。",
        "各层维度不同，不能将相同训练规则等同于完全相同容量。所有位置来自同一已有最佳 checkpoint，不重新训练 connector/LoRA。", "",
        "## 数值复现核验", ""])
    for split, item in numerical.items():
        lines.append(f'- {split}：原 head 历史 {item["historical_accuracy"]*100:.2f}%，本次 {item["current_accuracy"]*100:.2f}%，差 {item["delta_percentage_points"]:+.3f} 个百分点（batch size/BF16 数值路径差异）。')
    for split, item in feature_check.items():
        lines.append(f'- {split} 的 REVE 特征与上次 LP 缓存：max abs delta={item["max_abs_delta"]:.3g}，mean abs delta={item["mean_abs_delta"]:.3g}，平均相对行范数差={item["mean_relative_row_norm"]:.3g}；原 LP 头预测不一致 {item["old_linear_head_prediction_disagreements"]} 条。')
    lines.extend(["", "首次报告生成因 max abs delta > 0.01 被检查中止，未输出报告；随后直接核验了两条编码路径。",
        "同一 REVE、同一输入批次下 EEGQwen.encode 与 ReveClassifier.encode 的最大差为 0；原连续 128 批次精确重现旧缓存、当前交错 64 批次精确重现新缓存。",
        "当前环境 float32 matmul precision=high、TF32 enabled；浮点计算路径随批量而变。该差异保留记录，不将其当作模型/数据不同。"])
    lines.extend(["", "## 本轮诊断结论", "",
        "1. 在相同探针训练和源域验证规则下，2b 的跨域线性可读性在 connector 后下降，在 Qwen 输出上下降更明显；与此同时内部测试准确率上升，符合源域判别表示更强但跨域迁移较弱的现象。",
        "2. 从 Qwen 最后读出重新拟合线性头，仍未恢复到 REVE 直接探针水平；Qwen EEG token 的读出也低。因此不能只把问题归结为原最后一层 Linear 或最后 token 的选取。",
        "3. 2b AUC 也随阶段下降，不只是概率校准或固定阈值的问题；正温度缩放不改变类别排序。",
        "4. 这是阶段定位，不是因果证明；尚未区分预训练 Qwen、LoRA、连接层训练目标和时序池化的独立影响。未实施结构修复。",
        "5. 下一步可做保持其他条件一致的冻结 Qwen/LoRA 对照，并使用源数据集留一验证来检查跨数据集选择规则，不能按 2b E 的结果反复调参。",
        "", "## 解释边界及产物", "",
        "- 线性可读性不等于全部信息量；差成绩不能证明信息彻底消失。均值池化、特征维度、正则化都影响探针能力。",
        "- 如果新 answer 探针明显优于原 head，只能支持原决策方向/读出训练有问题，不能声称其余架构已完成语义对齐。",
        "- 测试标签未用于选择 λ；但诊断发生在已看到历史 2b 结果之后，属于探索性研究，后续方法需新验证。",
        "- 原始权重/原实验均未修改；大模型权重未重新下载、上传或提交。",
        f"- [机器可读结果和哈希](comparison.json)；{args.probe_dir}/*/ 下保存 selection、probe.npz、预测和完整指标，probes/*/ 保留初始 300 次迭代版本。",
        "- [固定协议及技能引用](../../../eeg_align/STAGE_DIAGNOSTIC.md)。资源/模型技能帮助完成冻结、加载及提取校验，其方法引用已列入协议。", ""])
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(root / "RESULTS.md"), "original_rerun_difference": numerical,
                      "reve_cache_check": feature_check, "prediction_audit_passed": True}))


if __name__ == "__main__":
    main()
