# EEG → JEV：文本对齐 VideoMAE 的五组迁移

当前把已完成多正例 InfoNCE／Sigmoid 对齐的两个 VideoMAE best encoder 接入 JEV。四组为“来源×CLS／32-token Q-Former”且冻结 encoder；第五组为 InfoNCE＋Q-Former，预热后解冻 encoder。五组复用同一训练入口、数据划分、采样序列和训练预算，第二阶段均用分类交叉熵。

父对齐 checkpoint 均选中 step 1700，完整验证 BAcc 为 47.49%／46.55%，当时没有 JEV。当前仅迁移前向 renderer、SharedScale、VideoMAE 及 encoder buffers；不迁移父 CLS／投影／优化器。两种读出均重新初始化，同结构初值相同，JEV 及连接器初值在五组一致。[迁移范围与五组设计](docs/PROTOCOL.md)。

2026-11-01：SEED 已改为论文的被试内 trial 顺序 9/3/3，处理窗口 22,455/7,875/7,560；三份均含全部 15 位被试，trial 不重叠。其余 16 库及完整 HMC151 划分不变。新九库完整验证为 61,327 窗。[新划分](artifacts/benchmark/multitask_splits_seedpaper_20261101/summary.json)、[协议](docs/PROTOCOL.md)。旧四读出结果保留为历史对照，不覆盖数据、检查点或分母。

## 📋 实验入口

- [当前协议与五组设计](docs/PROTOCOL.md)：17 库、2040 步、单 seed；SEED 被试内，另八个 benchmark 跨被试，2b 不训练。
- [当前状态与必要历史对照](docs/RESULTS.md)：保留 LEAF 完整训练／测试成绩和论文比较的局限，不冒充新结果。
- [表征分析与可视化](docs/REPRESENTATION_ANALYSIS.md)：四组相同验证窗口，检查任务内类别混叠、文本距离、被试效应和 query 冗余；不访问测试集。
- [配置](configs/benchmark_train.json)、[任务及选项](configs/benchmark_tasks.json)、[Git 归档索引](docs/ARCHIVE.md)。
- 新实验输出根目录：`artifacts/benchmark/aligned_videomae_jev_20261102/`；启动后 `pipeline.json` 为控制器状态，`train/<arm>/progress.json` 为各组实时状态，`transfer_audit.json` 记录父权重加载证据。
- 上游对比学习配置：[contrastive_train.json](configs/contrastive_train.json)；已完成输出 `artifacts/benchmark/contrastive_videomae_20261101/`。原始产物保留，不覆盖或重新训练父实验。

## 🔧 启动与验证

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

不要重复启动已有输出目录。控制器先检查显存余量及全部计划训练输入，再进行五组各 17 步真实 smoke（覆盖所有数据集、冻结／解冻、磁盘重载）。通过后重新加载同一对齐 best 权重，正式训练不继承 smoke 更新。使用 GPU 0、1、4、5、6，每张卡一组；用户允许与测评共卡，每次启动需至少 72,000 MiB 空闲，不停止现有进程。控制器脱离终端运行；`--resume` 仅用于未完成单组 `last.pt`，必须保持配置和代码哈希不变。

训练包含 step 0 验证；每 340 步固定验证／训练子集监测，保存逐数据集 loss 和准确率。每组最后对选定模型跑完整验证集，完成后自动绘图；本轮不自动打开测试集。

## ⚠️ 解释边界

这是有监督多任务可行性试验，不是跨数据集 zero-shot；SEED 为被试内。五组能比较对齐来源、读出及 InfoNCE/Q-Former 的冻结策略，但尚无等总预算、无文本对齐的匹配新控制。HMC 作者具体随机名单、TUAB/TUEV 跨库身份、基模预训练重叠风险仍未解决。旧分数不得直接当作新协议的匹配对照。

用户已有 ECG、专利、架构草图、图件、编辑器设置和所有 ignored 数据／权重／日志未作为本轮清理对象。旧 LEAF 架构草图不代表当前实现。
