# EEG→JEV 实验运行记录

_2026-10-29；当前为 LEAF 多任务验证实验，以下旧 BSR 结果单独标注协议，不能混为本轮成绩。_

---

## 📋 当前 LEAF 多任务实验

固定 9 个 benchmark + 8 个 train-only 数据集，2b 排除，用户已同意保留 HMC 子集和跨 TUAB/TUEV 候选身份重合。17 库全量转换后原有 train 954,569、valid 49,069、test 49,431 个窗口／缓存样本。首轮在第 42 步失败；全量 QC 排除 1,144 个无效训练窗口后，有效 train 为 953,425，验证／测试数量不变。修复轮重新启动，尚无最终测试成绩。[协议](PROTOCOL.md)记录精确范围与 2040 次更新预算。

| benchmark | train 窗口 | valid 窗口 | test 窗口 |
| --- | ---: | ---: | ---: |
| 2a | 2880 | 1152 | 1152 |
| PhysioMI | 6210 | 1734 | 1893 |
| SEED | 32838 | 2526 | 2526 |
| SEED-VII | 15536 | 1942 | 1942 |
| TUAB | 247728 | 12315 | 12277 |
| TUEV | 87834 | 12473 | 13046 |
| HMC 子集 | 68215 | 15512 | 15140 |
| Workload | 1534 | 300 | 300 |
| ADFTD | 4743 | 1115 | 1155 |

9 库合计训练 467,518 条，8 个额外库原有 487,051 条 train-only 缓存样本，QC 后有效 485,907 条。窗口数含 TUEV 上游保留的重复标注，不等同于独立临床事件数。预处理仅有 PhysioMI 的 `S104R08`（test）第 12 个零起始事件越界未生成窗口，原始记录和被试均保留；其余库未发生事件跳过，详情见逐记录 JSON。不会静默补信号或改测试被试。

已完成 [9 库真实训练样本的组件检查](../artifacts/benchmark/leaf_jev_multitask_preflight_20261029/complete.json)：每库 2 条样本，LEAF→JEV loss 和梯度有限，encoder 梯度非零，峰值显存 52.965 GiB。检查用权重已丢弃，测试访问为 0；这些 loss 不代表任务准确率。

正式配方的 [双卡 smoke](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/smoke/complete.json)完成 2 次更新（CHB_MIT、Nakanishi2015），第二步 encoder 梯度范数 0.11269，峰值显存 54.505 GiB；[磁盘重载检查](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/smoke/reload_check.json) logits 逐元素完全一致，测试访问 0。正式模型另行初始化，不继承 smoke 权重。当前 35 项单元测试通过。

可训练参数：connector 27,535,872，JEV LoRA 217,579,520，decision head 81,936，LEAF 有效路径 24,177,920；主干仍冻结。训练代码提交 `93ed85d`，文档后续提交不改变运行代码哈希。

SEED 原 CNT 的 6 个正数样本长度字段异常，已通过事件表字节偏移重算，不删除被试。第一次失败缓存保留在 `datasets/multitask_20261029_cnt_header_firstpass/`，当前只使用 `datasets/multitask_20261029/`。

### 第 42 步故障与修复重启

首轮在 `2026-10-29T09:38:37Z` 失败，实际完成 41/2040 次更新。第 42 步的 CHB_MIT batch 包含原始行 11504、11880、11333（被试 12），两 rank 的双极适配报 `All-zero EEG epoch`。只有 step 0 的 `best.pt`，尚未生成可恢复的 `last.pt`；不能声称从第 41 步继续。

[源数据核对记录](../artifacts/benchmark/multitask_signal_quality_20261029/CHB_MIT_source_verification.json)确认源 PKL 的 SHA256 与冻结 manifest 一致；源数据已有 1,144 个全零窗口，与转换后的缓存对应全零行一致。1,130 个来自被试 12，其余分布于 02/03/06/13/15/16/18/20；标签 0/1 各 615/529。上游读入缺失导联时保留初始化零值，是可疑机制，但未追溯到每个原始 EDF 的确切生成原因，不能称为已证明蒙太奇不匹配。

全量 [QC 报告](../artifacts/benchmark/multitask_signal_quality_20261029/summary.json)覆盖 17 库 1,053,069 个窗口，仅发现以上全零问题。CHB_MIT 有效训练数从 27,436 变为 26,292。原始数据和索引未修改，排除项按原始 ID 记录；其他数据集、所有被试划分和 held-out 分母不变。修复后第 42 步重新采样的 16 条全部有效；当前 45 项单元测试通过。

修复轮用同一模型、学习率、采样策略和 2040 步预算从发布权重重启。新增启动前全部计划 batch 的适配检查、每 20 步保存，以及 step 0 和解冻边界保存；异常进度和 rank 故障详情单独落盘。首轮失败产物完整保留。

`2026-10-29T11:00:15Z` [重启核验快照](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/restart_verification.json)：代码 `7a37a39`，已完成 60/2040 步。第 42 步 CHB_MIT 正常完成（loss 0.69834，梯度范数 6.9334）；双卡训练峰值分配显存约 54.20 GiB/卡，无测试评估。已读回确认第 40 步 checkpoint 包含 808 个 optimizer 参数状态、scheduler 和两 rank RNG；快照时第 60 步保存正在进行。这是带时间戳的运行检查，不是最终准确率。

[全部计划输入检查](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/signal_preflight.json)验证 2040 步、32,640 次训练样本呈现，耗时 33.77 s。[新 smoke](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/smoke/complete.json)完成两步，解冻 encoder 梯度范数 0.16390，磁盘重载 logits 完全一致。新旧模型初始化参数、buffer 及 step 0 验证结果均完全相同；step 0 九库验证子集 macro balanced accuracy 为 30.2882%，不是训练后或测试集成绩。

新运行日志：[pipeline.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/pipeline.json)、[train/progress.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/progress.json)。文件尚不存在时，不能理解为该阶段已启动。训练完成后以各数据集完整测试输出为准。

## 📊 历史 BSR：已完成 2a 和已取消 2b

原定 2a/2b，seed 0、官方 Cross fold 0；每个数据集分别运行 cross-attention 与 Q-Former-style，输出均为 32 tokens，正式训练固定 20 epochs。2a 两组完成；用户要求不再训练 2b 后，向已核验的控制器 PID 1373666 发送 SIGTERM，由控制器终止其两名 2b worker。控制器已记录 `status=stopped`、`active={}`、`queued=[]`，相关进程已退出，未终止其他任务。所有既有数据、日志和 checkpoint 保留。

| 数据集 | 压缩器 | 正式 updates | 测试 trial | 最终 accuracy |
| --- | --- | ---: | ---: | --- |
| 2a | cross-attention | 2,880 | 288 | 25.6944% |
| 2a | Q-Former-style | 2,880 | 288 | 25.0000% |
| 2b | cross-attention | 最后进度 474 / 1,560 | 未正式测试 | 用户取消 |
| 2b | Q-Former-style | 最后进度 424 / 1,560 | 未正式测试 | 用户取消 |

2a cross-attention 的预测计数为 `[283, 0, 5, 0]`，Q-Former-style 为 `[288, 0, 0, 0]`，均接近四分类随机水平。两者反转候选顺序后的准确率都是 25%，语义类别一致率都是 0；这提示明显的输出偏置，但尚不能仅凭这些现象确定其原因。完整结果读取各组 `complete.json` 和 `test_predictions.npz`；不能把选项诊断当作另一次模型选择。

后续按用户要求切换到 xw1216/EEG-FM-Bench 的多数据集联合 SFT，先 LEAF，再 BSR-VideoMAE，当前状态见本文首节。不得加载本轮 2a 下游权重：旧训练被试 2–9 包含新协议验证被试 6、7 和测试被试 8、9。原始基础模型的预训练重叠仍需另行审计。

## 🔍 已完成的输入与组件检查

官方 source / masks 哈希核验通过。仅从训练行检查真实输入：2a `[2,22,800]`、2b `[2,3,901]`，均为 200 Hz；保留完整 epoch 和原始通道，不使用旧 LEAF 插值/归一化缓存。

先前随机特征检查在 `artifacts/token_resamplers/smoke_20261028/`：两组均为 `[2,1568,768] → [2,32,768]`，输出与梯度有限；参数量分别 7,115,520 和 37,834,752。该产物不是训练权重或真实 EEG 准确率。

## ✅ 历史启动核验：08:57 UTC

当前实现提交 `5835493`；全仓库 65 项测试通过。控制器 PID `1373666`，输出目录为上文路径。GPU 3 启动时被外部任务占用，待其空闲后才启动本轮 worker，没有终止或修改其他任务。

4 组真实 train-only smoke 全部完成，第 2 步解冻后的梯度如下；每组磁盘 checkpoint 重载 logits 逐元素一致，测试访问均为 0：

| 数据集 | 压缩器 | BSR 梯度范数 | 压缩器梯度范数 |
| --- | --- | ---: | ---: |
| 2a | cross-attention | 0.739894 | 4.524652 |
| 2a | Q-Former-style | 0.296867 | 2.367886 |
| 2b | cross-attention | 0.057141 | 0.317723 |
| 2b | Q-Former-style | 0.200537 | 0.784290 |

同 seed/variant 初始化一致，两种 variant 的共享参数指纹也一致；共享 SHA256 为 `ffb60049ecac0b8f3b6c183f69b4a2da9c41dc822e8f17cd5aa9353f647d15e8`。

该历史核验时刻，GPU 2 上的 2a cross-attention、GPU 3 上的 2a Q-Former-style 均完成 16/2,880 updates；loss 有限，压缩器梯度非零，测试访问为 0。正式前 100 步处于 BSR 冻结预热，之后自动解冻。这不是当前状态：2a 已结束，2b 已取消，队列不再自动启动任何训练。

这只是带时间戳的启动快照，最新进度应读取 `pipeline.json` 和每组 `progress.json`，不能将早期训练 batch accuracy 当作最终测试结果。

## ⚠️ 解释边界与历史

BSR 的 Stage-B 包含 2b，预训练被试/trial 重叠未排除；下游被试隔离不等于 foundation pretraining 无泄漏。不将本轮称为干净 zero-shot，也不把两组准确率差异只归因于 self-attention（参数量不匹配）。本轮完整旧协议见 Git `14c60cf:docs/PROTOCOL.md`；[当前协议](PROTOCOL.md)记录正在进行的 xw1216 多数据集迁移。

LEAF 当前轮以前的协议和启动记录在 Git `3033ac9`；旧实际产物保留于 `artifacts/benchmark/official_cross_leaf_jev_20261027*/`。更早结果见[归档索引](ARCHIVE.md)。此次没有删除旧权重、数据、缓存或日志。
