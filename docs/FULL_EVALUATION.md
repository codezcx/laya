# 全量测试评估：范围与固定协议

2026-10-27 按要求使用 GPU 2、3，不占用 GPU 0、1，不追加训练。

## 本轮范围

覆盖当前任务目录中 **11 个数据集、第 0 折的全部 60,275 条测试样本**，不使用训练时的 validation cap。两卡对每个数据集交错分片，合并时检查 ID 唯一、无缺失、无重复，且与原始测试清单完全一致。检查三个 split 的样本和被试互斥。

| 数据集 | 测试样本数 |
|---|---:|
| BNCI2014001（2a） | 288 |
| BNCI2014004（2b） | 160 |
| BNCI2015001 | 200 |
| BNCI2014008 | 4,200 |
| BNCI2014009 | 576 |
| CHB_MIT | 1,807 |
| TUAB | 5,508 |
| SleepEDF | 43,932 |
| SEED | 3,394 |
| EEGMAT | 30 |
| Nakanishi2015 | 180 |

这里的单位是原 benchmark 预处理后的 trial/window，不统一解释为完整试次。SEED 是窗口级；没有自动聚合成 trial。

**范围限制：不是 13 数据集、全部 LOSO／10-fold 的完整 benchmark 复现。** SEED-VIG 是连续值回归，ThingsEEG2 涉及视觉解码／检索；当前 choice checkpoint 没有对应接口。其他跨被试折也没有各自独立训练的 checkpoint，不能用 fold-0 模型在它训练过的被试上补测后当作跨被试结果。

## 固定模型与输入

- **LEAF→JEV**：11 数据集联合训练的 BN-frozen 分支，使用先前验证集宏平均 BA 选出的 `cross_subject_fold0_bn_control/frozen/best.pt`，step 600。不是只训练 2a＋SEED 的四组匹配实验模型。测试集不参与选择 checkpoint、prompt、阈值或温度。
- **原生 LEAF 原型对照**：发布的 `leaf-v1.0-instruct-mpnet-base.ckpt`，不追加微调。使用任务描述和类别文本原型的余弦相似度分类；MI、SEED、EEGMAT 使用原生类名，其余固定为现有任务目录的选项文本。全部文本提前写入 `plan.json`，不根据测试结果改写。
- 两者复用 11 数据集训练时的 `SignalStore`：200 Hz、单窗口 robust scaling、65 电极映射；超过 10 秒保留所有片段。JEV 拼接各片段 query tokens；原生 LEAF 对各片段单位 embedding 求均值后归一化。
- LEAF 收到 task-only instruction，不接收真实类别。JEV 同时收到固定问题、全部候选选项。评估标准顺序和完全倒序，输出均映射回类别语义。
- 原生对照是**相同 benchmark 输入管线下的发布模型基线**，不是官方 LEAF 预处理／训练协议的复现。其 `softmax(cosine)` 仅为未校准相似度分数。原生原型匹配的顺序一致性是构造保证，不是实测的语言理解能力。
- 未排除 LEAF 预训练数据重叠；本轮是固定留出被试测试，不据此宣称严格的未见数据集 zero-shot。

## 实现与检查

入口：[evaluate_benchmark.py](../scripts/evaluate_benchmark.py)。历史运行依赖放在隔离 worktree `../eegjev-worktrees/full-eval-bb25191/`，版本为 checkpoint 记录的 `bb25191bad6ce189cb59d2ffba9b438dec70584d`；数据、权重和环境只链接、不复制、不删除。当前训练配置不改动。

运行前检查历史代码、任务目录、instruction embedding 和划分清单哈希。JEV 恢复全部 trainable parameters 和 buffers 后，验证张量指纹与 checkpoint 逐位一致，并重放历史验证集的 38 条检查样本。当前环境中破坏再恢复决策头后的预测必须逐位一致；对历史 BF16 概率采用最大绝对差 `<0.005`、类别预测完全一致的跨运行检查，同时报告实际差异。两个模型均 `eval()` 且不计算梯度，运行结束检查 buffers 和 checkpoint 没有变化。

初次启动采用更严格的 `1e-4` 历史概率容差，两卡均为最大差 `0.0015554`、38/38 类别一致，因而在接触 JEV 测试数据前停止。尚不能仅凭该差异断定具体数值来源。后续增加权重／buffer 逐位验证及当前环境的严格恢复检查，将跨运行概率容差与本地逐位重现分开；保留历史 requires-grad 标志和 3 个 PyTorch 线程，与原验证路径一致。不调整 checkpoint、prompt 或预测规则。首次原生 LEAF 的全部预测保留，经过协议、清单、ID 和文件哈希验证后复用。

当前输出目录：`artifacts/benchmark/full_eval_fold0_20261027_checked/`。初次停止记录和已完成的原生预测位于 `artifacts/benchmark/full_eval_fold0_20261027/`，不覆盖或删除。

- `plan.json`：固定 checkpoint、文本、版本、输入协议与覆盖范围。
- `pipeline.json`、`worker*/progress.json`、`worker*.log`：进度／失败状态。
- `worker*/checkpoint_replay.json`：正式测试前的恢复验证。
- `native_predictions.jsonl`、`jev_predictions.jsonl`：合并后的全部预测；运行期间保存在各 worker 下。
- `results.json`、`RESULTS.md`：完成后生成，各数据集 ACC／BA、混淆矩阵、被试分项以及倒序选项结果。不以大样本 SleepEDF 的 pooled accuracy 代替各任务成绩。

启动器拒绝覆盖已有输出；失败输出保留以供诊断，不会悄悄跳过失败数据集。
