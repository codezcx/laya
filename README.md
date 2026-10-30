# EEG → 文本条件化 LEAF → JEV

当前实验：9 个 benchmark + 8 个 train-only 数据集联合微调一个 LEAF→JEV 模型，使用冻结的跨被试划分，2b 排除。用户已同意保留 HMC 本地子集及 TUAB/TUEV 候选身份重合，先做验证性实验；不将本轮成绩宣称为严格无泄漏 benchmark 或跨数据集 zero-shot。

## 📋 当前状态

- 当前已启动 step 2040 → 4080 的探索性续训，GPU 2、3，恢复模型、Adam 状态和双卡 RNG；原数据划分不变。低峰值学习率回升后衰减，每 340 步记录固定训练／验证子集指标，不自动重测测试集。见[续训协议](docs/PROTOCOL.md#续训阶段20414080)、[续训配置](configs/benchmark_continue.json)和[当前运行状态](artifacts/benchmark/leaf_jev_multitask_continue_20261030/pipeline.json)。
- 新训练入口、数据转换和文本接口已实现；真实 train-only 前向／反向检查覆盖 9 个 benchmark，LEAF 梯度有效，单卡峰值约 53 GiB。
- 首轮训练在第 42 步因 CHB_MIT 原缓存全零窗口退出，并非训练完成。全量 QC 已完成：排除 1,144 个无效训练窗口后，使用 953,425 条训练、49,069 条验证、49,431 条测试；原始缓存、被试划分及旧日志保留。
- 修复版已完成 2040/2040 步训练及九库完整验证／测试；选中 step 2040，49,431 条测试窗口的九库等权平均 Accuracy 为 51.71%、Balanced accuracy 为 50.59%。各库分类数和成绩见 [完整测试汇总](docs/RESULTS.md#完整测试结果分类数与准确率)。
- 同一 step 2040 checkpoint 的九库完整训练集评估已完成（467,518 个窗口）：等权平均 Accuracy 56.52%、Balanced accuracy 56.74%，未更新权重、未重复测试；[完整结果](artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/complete.json)已填入[对照表倒数第二列](docs/RESULTS.md#论文-baseline分类数与-balanced-accuracy)。
- [LaBraM／CBraMod／REVE 对照与被试划分](docs/RESULTS.md#论文-baseline分类数与-balanced-accuracy)和[划分／窗口核查](docs/RESULTS.md#与论文的划分及窗口一致性核查)已补录：本轮不是论文协议的严格复现；SEED 模式、HMC 子集、部分窗口数量和取整边界存在差异，未改写原成绩。
- 首阶段固定预算 2040 次更新，各数据集均衡轮转；按固定验证子集选模，其 macro balanced accuracy 从 30.29% 提升至 49.11%，与完整测试结果分开记录。续训追加 2040 步，原成绩和图件仍对应首阶段。
- 只初始化发布 LEAF 和原生 JEV adapter，不复用见过本轮测试被试的旧 2a 微调权重。
- VideoMAE 对照尚未启动；旧 BSR 代码已从工作树移除，Git `14c60cf` 可恢复，数据、权重和日志保留。

## 📚 入口和记录

- [当前协议、划分与风险](docs/PROTOCOL.md)
- [当前结果与历史对照](docs/RESULTS.md)
- [训练 loss 与验证曲线](artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_validation_curves.png) · [各数据集验证曲线](artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation_by_dataset.png)
- [17 个数据集各自的训练 loss 曲线](artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_loss_by_dataset.png) · [曲线口径和原始 CSV](docs/RESULTS.md#训练与固定验证子集曲线)
- [训练配置](configs/benchmark_train.json) · [17 个任务及选项](configs/benchmark_tasks.json)
- [固定划分审计](artifacts/benchmark/multitask_splits_20261029_final/summary.json)
- [真实模型组件检查](artifacts/benchmark/leaf_jev_multitask_preflight_20261029/complete.json)
- [全量信号 QC](artifacts/benchmark/multitask_signal_quality_20261029/summary.json)
- [续训运行状态](artifacts/benchmark/leaf_jev_multitask_continue_20261030/pipeline.json) · [首阶段完成记录](artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/pipeline.json)
- [Git 和产物归档](docs/ARCHIVE.md)

## 🔧 启动

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --continuation-config configs/benchmark_continue.json
```

这是当前续训入口，不要重复启动同一目录。控制器核验父 checkpoint 和冻结缓存，检查新增步骤的真实输入，再恢复模型／优化器／RNG、复现父模型验证结果、训练至 4080；续训禁止测试集前向。仅当旧进程退出、已有本阶段 `train/last.pt` 且未完成时，才可加 `--resume`。首阶段从发布模型初始化的命令和原风险说明见[协议](docs/PROTOCOL.md)。不修改或重建本次正在使用的缓存。

## 📦 其他用户工作

ECG-R1 独立调试见[使用说明](docs/ECG_R1_DEBUG.md)。[既有 LEAF 架构草图](docs/MODEL_ARCHITECTURE.md)、[方法草稿](docs/EEGJEV_PATENT.md)以及用户已有 ECG、专利、图件、编辑器设置未作为本轮实验代码修改或提交。
