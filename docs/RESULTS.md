# EEG→JEV 实验运行记录

_2026-10-29；当前为 LEAF 多任务验证实验，以下旧 BSR 结果单独标注协议，不能混为本轮成绩。_

---

## 📋 当前 LEAF 多任务实验

固定 9 个 benchmark + 8 个 train-only 数据集，2b 排除，用户已同意保留 HMC 子集和跨 TUAB/TUEV 候选身份重合。新训练入口和预处理已实现，全量转换正在进行，正式训练尚未产出指标。[协议](PROTOCOL.md)记录精确范围与 2040 次更新预算。

已完成 [9 库真实训练样本的组件检查](../artifacts/benchmark/leaf_jev_multitask_preflight_20261029/complete.json)：每库 2 条样本，LEAF→JEV loss 和梯度有限，encoder 梯度非零，峰值显存 52.965 GiB。检查用权重已丢弃，测试访问为 0；这些 loss 不代表任务准确率。独立的双卡 smoke 和 checkpoint 重载检查由正式控制器继续执行。

SEED 原 CNT 的 6 个正数样本长度字段异常，已通过事件表字节偏移重算，不删除被试。第一次失败缓存保留在 `datasets/multitask_20261029_cnt_header_firstpass/`，当前只使用 `datasets/multitask_20261029/`。

运行日志：[pipeline.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/pipeline.json)、[train/progress.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/train/progress.json)。文件尚不存在时，不能理解为训练已启动。训练完成后以各数据集完整测试输出为准。

## 📊 历史 BSR：已完成 2a 和已取消 2b

原定 2a/2b，seed 0、官方 Cross fold 0；每个数据集分别运行 cross-attention 与 Q-Former-style，输出均为 32 tokens，正式训练固定 20 epochs。2a 两组完成；用户要求不再训练 2b 后，向已核验的控制器 PID 1373666 发送 SIGTERM，由控制器终止其两名 2b worker。控制器已记录 `status=stopped`、`active={}`、`queued=[]`，相关进程已退出，未终止其他任务。所有既有数据、日志和 checkpoint 保留。

| 数据集 | 压缩器 | 正式 updates | 测试 trial | 最终 accuracy |
| --- | --- | ---: | ---: | --- |
| 2a | cross-attention | 2,880 | 288 | 25.6944% |
| 2a | Q-Former-style | 2,880 | 288 | 25.0000% |
| 2b | cross-attention | 最后进度 474 / 1,560 | 未正式测试 | 用户取消 |
| 2b | Q-Former-style | 最后进度 424 / 1,560 | 未正式测试 | 用户取消 |

2a cross-attention 的预测计数为 `[283, 0, 5, 0]`，Q-Former-style 为 `[288, 0, 0, 0]`，均接近四分类随机水平。两者反转候选顺序后的准确率都是 25%，语义类别一致率都是 0；这提示明显的输出偏置，但尚不能仅凭这些现象确定其原因。完整结果读取各组 `complete.json` 和 `test_predictions.npz`；不能把选项诊断当作另一次模型选择。

后续按用户要求切换到 xw1216/EEG-FM-Bench 的多数据集联合 SFT，先 LEAF，再 BSR-VideoMAE。新训练尚未启动。不得加载本轮 2a 下游权重：旧训练被试 2–9 包含新协议验证被试 6、7 和测试被试 8、9。原始基础模型的预训练重叠仍需另行审计。

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

BSR 的 Stage-B 包含 2b，预训练被试/trial 重叠未排除；下游被试隔离不等于 foundation pretraining 无泄漏。不将本轮称为干净 zero-shot，也不把两组准确率差异只归因于 self-attention（参数量不匹配）。本轮完整旧协议见 Git `14c60cf:docs/PROTOCOL.md`；[当前协议](PROTOCOL.md)记录尚未启动的 xw1216 多数据集迁移。

LEAF 当前轮以前的协议和启动记录在 Git `3033ac9`；旧实际产物保留于 `artifacts/benchmark/official_cross_leaf_jev_20261027*/`。更早结果见[归档索引](ARCHIVE.md)。此次没有删除旧权重、数据、缓存或日志。
