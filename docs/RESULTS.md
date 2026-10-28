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

## ⚠️ 解释边界与历史

BSR 的 Stage-B 包含 2b，预训练被试/trial 重叠未排除；下游被试隔离不等于 foundation pretraining 无泄漏。不将本轮称为干净 zero-shot，也不把两组准确率差异只归因于 self-attention（参数量不匹配）。完整规则见[当前协议](PROTOCOL.md)。

LEAF 当前轮以前的协议和启动记录在 Git `3033ac9`；旧实际产物保留于 `artifacts/benchmark/official_cross_leaf_jev_20261027*/`。更早结果见[归档索引](ARCHIVE.md)。此次没有删除旧权重、数据、缓存或日志。
