# BSR → 32-token resampler → JEV：运行记录

_2026-10-28；本文件记录启动核验，不把排队、加载或 smoke 当作准确率结果。_

---

## 📋 当前范围

2a/2b，seed 0、官方 Cross fold 0；每个数据集分别运行 cross-attention 与 Q-Former-style，输出均为 32 tokens，正式训练固定 20 epochs。四组先执行两步 train-only smoke，全部通过后重新加载原始权重进行正式训练。实际状态以 `artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028/pipeline.json` 为准。

| 数据集 | 压缩器 | 正式 updates | 测试 trial | 最终 accuracy |
| --- | --- | ---: | ---: | --- |
| 2a | cross-attention | 2,880 | 288 | 尚未产出 |
| 2a | Q-Former-style | 2,880 | 288 | 尚未产出 |
| 2b | cross-attention | 1,560 | 160 | 尚未产出 |
| 2b | Q-Former-style | 1,560 | 160 | 尚未产出 |

## 🔍 已完成的输入与组件检查

官方 source / masks 哈希核验通过。仅从训练行检查真实输入：2a `[2,22,800]`、2b `[2,3,901]`，均为 200 Hz；保留完整 epoch 和原始通道，不使用旧 LEAF 插值/归一化缓存。

先前随机特征检查在 `artifacts/token_resamplers/smoke_20261028/`：两组均为 `[2,1568,768] → [2,32,768]`，输出与梯度有限；参数量分别 7,115,520 和 37,834,752。该产物不是训练权重或真实 EEG 准确率。

## ✅ 启动核验：08:57 UTC

当前实现提交 `5835493`；全仓库 65 项测试通过。控制器 PID `1373666`，输出目录为上文路径。GPU 3 启动时被外部任务占用，待其空闲后才启动本轮 worker，没有终止或修改其他任务。

4 组真实 train-only smoke 全部完成，第 2 步解冻后的梯度如下；每组磁盘 checkpoint 重载 logits 逐元素一致，测试访问均为 0：

| 数据集 | 压缩器 | BSR 梯度范数 | 压缩器梯度范数 |
| --- | --- | ---: | ---: |
| 2a | cross-attention | 0.739894 | 4.524652 |
| 2a | Q-Former-style | 0.296867 | 2.367886 |
| 2b | cross-attention | 0.057141 | 0.317723 |
| 2b | Q-Former-style | 0.200537 | 0.784290 |

同 seed/variant 初始化一致，两种 variant 的共享参数指纹也一致；共享 SHA256 为 `ffb60049ecac0b8f3b6c183f69b4a2da9c41dc822e8f17cd5aa9353f647d15e8`。

**正式训练已经开始，不只是排队或 smoke**：GPU 2 上的 2a cross-attention、GPU 3 上的 2a Q-Former-style 在核验时均完成 16/2,880 updates；loss 有限，压缩器梯度非零，测试访问为 0。正式前 100 步处于 BSR 冻结预热，之后自动解冻。2b 两组仍在同一队列，待卡释放自动启动；无须再次人工发命令。

这只是带时间戳的启动快照，最新进度应读取 `pipeline.json` 和每组 `progress.json`，不能将早期训练 batch accuracy 当作最终测试结果。

## ⚠️ 解释边界与历史

BSR 的 Stage-B 包含 2b，预训练被试/trial 重叠未排除；下游被试隔离不等于 foundation pretraining 无泄漏。不将本轮称为干净 zero-shot，也不把两组准确率差异只归因于 self-attention（参数量不匹配）。完整规则见[当前协议](PROTOCOL.md)。

LEAF 当前轮以前的协议和启动记录在 Git `3033ac9`；旧实际产物保留于 `artifacts/benchmark/official_cross_leaf_jev_20261027*/`。更早结果见[归档索引](ARCHIVE.md)。此次没有删除旧权重、数据、缓存或日志。
