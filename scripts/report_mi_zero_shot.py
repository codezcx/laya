"""Render measured results only; preserve split and preprocessing distinctions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("artifacts/mi_zero_shot_v1")
arms=["legacy_ce","harmonized_ce","grounded_qa","reve_mlp","reve_mlp_legacy"]
names={"legacy_ce":"Qwen + 原预处理 CE","harmonized_ce":"Qwen + 统一预处理 CE",
       "grounded_qa":"Qwen + 统一预处理 + 属性问答","reve_mlp":"REVE MLP + 统一预处理",
       "reve_mlp_legacy":"REVE MLP + 原预处理"}
def read(p): return json.loads(p.read_text())
def percent(v): return f"{100*v:.2f}%"
lines=["# REVE → Qwen 跨数据集迁移实验", "", "所有数字来自本目录 JSON 记录。源域外验证用于选择；2b 不用于选择。",
       "REVE 预训练包含 2b，因此这里指无目标域下游训练的迁移，不是编码器完全未见目标数据。", "",
       "## 两源训练的开发实验", "", "训练 Cho2017 + Dreyer2023；验证 Lee2019_MI 固定 10 名被试、2,000 trials。",
       "每组 400 步；最好 checkpoint 按 Lee 被试平均准确率选择。", "",
       "|方案|选中步数|Lee 验证准确率|2a E 准确率|2b E 准确率|", "|---|---:|---:|---:|---:|"]
for arm in arms:
    folder=ROOT/"pilot"/arm
    if not (folder/"best_metrics.json").exists(): continue
    v=read(folder/"best_metrics.json")
    external=[percent(read(folder/"evaluation"/f"{s}.json")["accuracy"]) if (folder/"evaluation"/f"{s}.json").exists() else "待完成" for s in ("external_2a","external_2b")]
    lines.append(f"|{names[arm]}|{v['step']}|{percent(v['accuracy'])}|{'|'.join(external)}|")
lines += ["", "## 三源重新训练后的冻结外部测试", "", "加入 Lee 原训练 split 后，总计 33,272 个源域 trials；保持开发阶段选定的步数。",
          "这些结果与上表的训练数据不同，不能混在一起做直接消融。", "",
          "|方案|2a E 准确率（n=1296）|2b E 准确率（n=2840）|2b 被试平均准确率|被试 bootstrap 95% 区间|", "|---|---:|---:|---:|---|"]
for folder in sorted((ROOT/"refit").glob("*")):
    arm=folder.name
    if not (folder/"evaluation/external_2b.json").exists(): continue
    a=read(folder/"evaluation/external_2a.json"); b=read(folder/"evaluation/external_2b.json")
    lo,hi=b["subject_bootstrap_95ci"]
    lines.append(f"|{names[arm]}|{percent(a['accuracy'])}|{percent(b['accuracy'])}|{percent(b['subject_macro_accuracy'])}|{percent(lo)}–{percent(hi)}|")
lines += ["", "## 语言/信号依赖诊断", "", "以下均在 Lee 源域外开发集上做，数字为 accuracy。零 token 是分布外扰动，只作为补充证据。", "",
          "|两源模型|正常 EEG|零 EEG token|错配 EEG|交换选项后还原类别|μ 问答|μ 错配|β 问答|β 错配|",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for arm in arms:
    p=ROOT/"pilot"/arm/"evaluation/source_diagnostics.json"
    if not p.exists():continue
    d=read(p)
    keys=["source_proxy_normal","zero","shuffle","option_swap","aux_task_1","aux_task_1_shuffle","aux_task_2","aux_task_2_shuffle"]
    lines.append(f"|{names[arm]}|"+"|".join(percent(d[k]["accuracy"]) if k in d else "—" for k in keys)+"|")
lines += ["", "属性问答只涉及两个可计算的频谱比较，不代表完整 EEG 解释能力。对照中 MI 问题的类别先验为 50%。",
          "辅助问题需同时查看各 JSON 内的多数类基线与 balanced accuracy。", "",
          "## 固定岭回归的源域诊断", "", "只使用 Cho/Dreyer 拟合，训练集统计标准化、alpha=100；它不参与 Qwen checkpoint 选择。",
          "该探针利用全部源训练样本拟合，神经模型是有限步随机采样训练，不能视为优化预算完全相同。", "",
          "|REVE 输入|MI 准确率|μ 属性准确率|β 属性准确率|", "|---|---:|---:|---:|"]
if (ROOT/"source_ridge_diagnostic.json").exists():
    ridge=read(ROOT/"source_ridge_diagnostic.json")
    for mode in ("legacy","harmonized"):
        lines.append(f"|{mode}|"+"|".join(percent(ridge[mode][task]["accuracy"]) for task in ("mi","mu_C3_vs_C4","beta_C3_vs_C4"))+"|")
pair_path=ROOT/"target_paired_comparison.json"
if pair_path.exists():
    pair=read(pair_path)
    lo,hi=pair["paired_subject_bootstrap_95ci"]
    lines += ["", "## 本轮判断", "",
              f"三源 Qwen 的 2b accuracy 为 {percent(pair['qwen_accuracy'])}，匹配的 REVE MLP 为 {percent(pair['mlp_accuracy'])}。",
              "本轮未实现 Qwen 的有效跨数据集迁移，也没有证明语言主干提供额外收益。",
              f"Qwen−MLP 的被试平均差为 {100*pair['subject_macro_difference']:.2f} 个百分点；按被试配对 bootstrap 的 95% 区间为 [{100*lo:.2f}, {100*hi:.2f}] 个百分点。",
              "这是单次探索实验，不包含训练种子和模型选择的不确定性。", "",
              "被选中的属性 QA checkpoint 在 EEG 错配后没有性能下降，两个属性题分别等于多数答案基线；",
              "这次接口训练尚未建立可验证的属性读取能力。它按 MI accuracy 选择，平分时选中第 100 步，",
              "不能据此声称所有充分训练后的 QA 方法均无效。统一预处理方案也没有达到源域任务的可靠拟合。", "",
              "后续优先验证：源数据真实相邻电极构成的局部差分增广及有限编码器适配；",
              "再把属性对齐作为独立训练阶段，以未参与训练的源数据集属性准确率和错配诊断作为验收。",
              "这些是后续假设，本轮没有实施，也不会根据这次 2b 分数继续调参。", "",
              "## 三源模型的 2b 被试结果", "", "|被试|trials|Qwen|匹配 REVE MLP|", "|---|---:|---:|---:|"]
    qa=read(ROOT/"refit/legacy_ce/evaluation/external_2b.json")["per_subject"]
    mlp=read(ROOT/"refit/reve_mlp_legacy/evaluation/external_2b.json")["per_subject"]
    for k in sorted(qa):
        lines.append(f"|{k.split('/')[-1]}|{qa[k]['n']}|{percent(qa[k]['accuracy'])}|{percent(mlp[k]['accuracy'])}|")
lines += ["", "## 复现与结论边界", "",
          "- 实验协议：`PROTOCOL.md`；文献核查与方法判断：`RESEARCH.md`。",
          "- `selection.json` 固定源域选择，`frozen_checkpoints.json` 记录外测前权重哈希。",
          "- 原预处理 MLP 是根据源域开发结果增加的匹配对照，见 `matched_control_amendment.json`。",
          "- `cache_roundtrip_and_early_source_fit.json` 记录缓存一致性及早期源域拟合检查。",
          "- 相同编码 batch 可精确重现缓存；TF32/bf16 下不同 batch 的数值结果不保证逐位一致。",
          "- 单随机种子、有限训练步数、一个开发数据集，不能据此证明方法上限或统计显著优越。",
          "- 频段/长度/归一化是一个组合处理；属性 QA 与选项随机化也是组合处理，不能拆分归因。",
          "- 2b 曾在旧实验中被查看，REVE 预训练也包含它；原始记录 provenance 尚未全部核验。",
          "", "![训练与源域验证曲线](training_curves.png)"]
(ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
fig,axs=plt.subplots(1,2,figsize=(12,4))
for arm in arms:
    folder=ROOT/"pilot"/arm
    if (folder/"train.jsonl").exists():
        logs=[json.loads(s) for s in (folder/"train.jsonl").read_text().splitlines()]
        y=np.array([v["mi_loss"] for v in logs]); smooth=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
        axs[0].plot([v["step"] for v in logs],smooth,label=arm)
    if (folder/"validation.jsonl").exists():
        logs=[json.loads(s) for s in (folder/"validation.jsonl").read_text().splitlines()]
        axs[1].plot([v["step"] for v in logs],[100*v["accuracy"] for v in logs],marker="o",label=arm)
axs[0].set(title="MI loss (5 logged batches moving mean)",xlabel="Optimizer step",ylabel="Cross-entropy")
axs[1].set(title="Held-out dataset: Lee2019_MI (n=2,000)",xlabel="Optimizer step",ylabel="Accuracy (%)")
axs[1].axhline(50,color="gray",linestyle="--")
for ax in axs:
    ax.axvline(80,color="gray",linestyle=":",alpha=.6); ax.grid(alpha=.2); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT/"training_curves.png",dpi=160); plt.close(fig)
print(ROOT/"RESULTS.md")
