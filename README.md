# EEG → JEV：固定学习率的 VideoMAE LoRA-r64 对照

当前只新增 `infonce_qformer_lora64`：在 InfoNCE＋Q-Former→JEV 全参组（完整验证 BAcc 44.15%）的基础上，将 VideoMAE 全参更新替换为 rank=64 LoRA。encoder 学习率保持 `1e-6`；renderer／SharedScale 仍按原阶段训练，Q-Former、连接器与 JEV 原有 LoRA／head 的训练设置不变。不是从 44.15% 成品 checkpoint 续训。

使用同一个 InfoNCE step 1700 父 encoder，Q-Former／连接器／原生 JEV 初值与全参组逐项核对。LoRA 的 rank=alpha=64、dropout=0，覆盖 12 层的 Q/K/V、注意力输出和两层 FFN，共 72 个线性层；原 VideoMAE 参数全部冻结。第二阶段仍只使用分类交叉熵。[迁移范围与对照设计](docs/PROTOCOL.md)。

2026-11-01：SEED 已改为论文的被试内 trial 顺序 9/3/3，处理窗口 22,455/7,875/7,560；三份均含全部 15 位被试，trial 不重叠。其余 16 库及完整 HMC151 划分不变。新九库完整验证为 61,327 窗。[新划分](artifacts/benchmark/multitask_splits_seedpaper_20261101/summary.json)、[协议](docs/PROTOCOL.md)。旧四读出结果保留为历史对照，不覆盖数据、检查点或分母。

## 📋 实验入口

- [当前协议与 LoRA 对照](docs/PROTOCOL.md)：17 库、2040 步、单 seed；SEED 被试内，另八个 benchmark 跨被试，2b 不训练。
- [当前状态与必要历史对照](docs/RESULTS.md)：保留 LEAF 完整训练／测试成绩和论文比较的局限，不冒充新结果。
- [表征分析与可视化](docs/REPRESENTATION_ANALYSIS.md)：四组相同验证窗口，检查任务内类别混叠、文本距离、被试效应和 query 冗余；不访问测试集。
- [配置](configs/benchmark_train.json)、[任务及选项](configs/benchmark_tasks.json)、[Git 归档索引](docs/ARCHIVE.md)。
- 新实验输出根目录：`artifacts/benchmark/aligned_videomae_jev_lora64_20261102/`；`pipeline.json` 为控制器状态，`train/infonce_qformer_lora64/progress.json` 为实时状态，`matched_control_audit.json` 记录与全参组的匹配检查。
- 上游对比学习配置：[contrastive_train.json](configs/contrastive_train.json)；已完成输出 `artifacts/benchmark/contrastive_videomae_20261101/`。原始产物保留，不覆盖或重新训练父实验。

## 🔧 启动与验证

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

不要重复新建已有输出目录。控制器先检查完整配置、显存余量及全部计划训练输入，再进行 17 步真实 smoke（覆盖所有数据集、冻结／解冻、磁盘重载）。通过后重新加载同一对齐 best 权重，正式训练不继承 smoke 更新。当前只用 GPU 0；启动需至少 72,000 MiB 空闲，不停止其他进程。控制器脱离终端运行。

节点回收后，可保持训练配置与源码哈希不变，仅覆盖新节点的调度卡号：

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --resume --gpus 0
```

该入口仅恢复当前配置的未完成 checkpoint，要求源码与配置哈希一致；不要同时启动多个恢复控制器。上一轮五组已全部完成，其配置及恢复入口保存在 Git `828a17d`，产物保留在 `aligned_videomae_jev_20261102`。历史重跑使用隔离 worktree，不覆盖当前源码。

训练包含 step 0 验证；每 340 步固定验证／训练子集监测，保存逐数据集 loss 和准确率。每组最后对选定模型跑完整验证集，完成后自动绘图；本轮不自动打开测试集。

## ⚠️ 解释边界

这是固定学习率、固定预算下的适配方式对照，不是 LoRA 的最优超参数搜索，也不是跨数据集 zero-shot。SEED 为被试内。HMC 作者具体随机名单、TUAB/TUEV 跨库身份、基模预训练重叠风险仍未解决。未同步修改学习率、训练预算、其他模块冻结范围或评估协议。

用户已有 ECG、专利、架构草图、图件、编辑器设置和所有 ignored 数据／权重／日志未作为本轮清理对象。旧 LEAF 架构草图不代表当前实现。
