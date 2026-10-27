# LEAF → JEV：官方 Cross SFT 结果状态

_2026-10-27；只报告有产物验证的状态，不把进程启动视为完成。_

## 📋 划分准备完成

官方原始 splitter 已导出 10 个标准 PKL 数据集的所有 Cross folds × seeds 0、1、2，共 426 组。证据：`artifacts/benchmark/official_splits_20261027/complete.json` 及各数据集 manifest。保留原 train/test masks 和 usage，不另减一个验证被试。

完整预算计划：`artifacts/benchmark/official_cross_leaf_jev_20261027_plan/plan.json`。20 epochs / batch 16 合计 18,107,880 optimizer updates；这是计划量，不是已完成量。13 数据集尚未全部就绪，缺失数据列在 pending。

## 🔍 首批验证

31 项测试通过，覆盖被试隔离、重复/越界行、usage、训练行覆盖、标签不进入提示词、测试门控、checkpoint、LR 与 JEV 选项掩码。2a/2b 第 0 折全部训练行通过信号适配检查。

| 数据集 | 训练样本 | 测试样本 | 正式预算 | 最终测试准确率 |
| --- | ---: | ---: | ---: | --- |
| 2a | 2,304 | 288 | 2,880 updates | 尚无结果 |
| 2b | 1,240 | 160 | 1,560 updates | 尚无结果 |

两个真实模型 smoke 均已完成：`artifacts/benchmark/official_cross_leaf_jev_20261027_smoke/pipeline.json`。第 2 步 LEAF 梯度范数为 1.0384（2a）和 0.1888（2b），BN 不变；磁盘 checkpoint 重载 logits 逐元素一致；同 seed 初始参数/buffers 指纹一致；测试访问均为 0。峰值显存约 53.14 / 52.69 GiB。这只证明功能正常，不是准确率实验。

首批正式范围：2a/2b seed 0 / fold 0；输出根目录 `artifacts/benchmark/official_cross_leaf_jev_20261027/`。仅在该目录出现运行记录后才算启动。smoke 不向正式训练传递权重。smoke 后仅优化 epoch 排列的缓存方式（同一序列每 epoch 计算一次），不改变样本顺序或损失。

## 📚 历史结果不可混用

旧 custom fold-0 的 11 数据集、60,275 条测试评估已完成，产物在 `artifacts/benchmark/full_eval_fold0_20261027_checked/`。其 JEV 来自旧多任务 SFT，且有额外验证被试、不同 usage/坏样本处理，不能填入上表。

旧完整 ledger、四组诊断和旧评估协议/脚本已提交在 Git `999380b`，在隔离 worktree 中查看。旧权重、数据、缓存和日志没有删除；用户独立文档和未提交修改保留。

配方与限制见[当前协议](PROTOCOL.md)；历史入口见[归档索引](ARCHIVE.md)。
