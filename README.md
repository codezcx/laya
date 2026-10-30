# EEG → BSR-VideoMAE → 单入口文本条件 JEV

当前方法是四组匹配的 VideoMAE 读出试验：mean pooling、learnable CLS、32-query cross-attention、32-query Q-Former 风格读出。文本只送入 JEV；BSR 编码器和读出不接收 task description、question 或标签。旧 LEAF 实现归档于 Git `741fe34`，不复用旧下游权重。

## 📋 实验入口

- [当前协议与四组设计](docs/PROTOCOL.md)：同一 17 库划分、2040 步、单 seed、四卡各一组；2b 不训练。
- [当前状态与必要历史对照](docs/RESULTS.md)：保留 LEAF 完整训练／测试成绩和论文比较的局限，不冒充新结果。
- [配置](configs/benchmark_train.json)、[任务及选项](configs/benchmark_tasks.json)、[Git 归档索引](docs/ARCHIVE.md)。
- 输出根目录：`artifacts/benchmark/bsr_jev_readouts_20261030/`；`pipeline.json` 为控制器状态，`train/<arm>/progress.json` 为各组实时状态。

## 🔧 启动与验证

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

不要重复启动已有输出目录。控制器先检查四张 GPU 空闲、全部计划训练输入，再进行四组各 17 步真实 smoke（覆盖所有数据集、跨过 JEV 解冻点、检查磁盘重载），通过后从发布权重重新初始化并启动正式训练。控制器与子进程在终端退出后继续运行；节点掉线仍需人工确认后恢复。`--resume` 仅用于 `eeg_align.train_joint` 的未完成单组 `last.pt`，必须保持配置和代码哈希不变。

训练包含 step 0 验证；每 340 步固定验证／训练子集监测，保存逐数据集 loss 和准确率。每组最后对选定模型跑完整验证集，不在这轮自动读取测试信号或测试标签做前向。四组都完成后再按验证结果选方案。

## ⚠️ 解释边界

这是有监督多任务、跨被试的可行性试验，不是跨数据集 zero-shot。HMC 仍使用冻结的 109 份记录子集；TUAB/TUEV 的跨库身份重合风险未消除；BSR 预训练包含 PhysioMI 等数据，精确被试重叠未完成审计。完整输入窗口、固定划分和原有 QC 均不改写。

用户已有 ECG、专利、架构草图、图件、编辑器设置和所有 ignored 数据／权重／日志未作为本轮清理对象。旧 LEAF 架构草图不代表当前实现。
