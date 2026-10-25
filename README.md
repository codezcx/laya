# Task-conditioned EEG → LEAF → JEV

当前保留一套 EEG-FM-Benchmark 联合训练代码：11 个分类数据集，文本指定任务与选项，被试隔离的第 0 折。

当前实验比较 2a 的原输入管线与 LEAF 原始预处理。两组各用一张 H200，从同一个冻结 BN 的 step 600 checkpoint 恢复模型、优化器和随机状态，续训 200 步至 step 800。两组 BN 都冻结，LEAF 参数和 JEV LoRA 继续训练；只有一组替换 2a 输入，其余十个任务保持一致。有效 batch 88、microbatch 22，沿用原 3000 步学习率曲线。step 600、700、800 使用相同验证集。

- [当前训练协议](docs/PROTOCOL.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

当前输出：`artifacts/benchmark/cross_subject_fold0_leaf_preprocessing/`。每组的 `metrics.jsonl`、`LIVE.md` 和 `monitor.json` 记录过程；TensorBoard 自动保存到该目录的 `tensorboard/`，包含总体/各数据集的准确率和 loss，BA 单独记录。启动器拒绝覆盖已有运行目录。

原信号和划分在 `datasets/benchmark/signals/`，2a 的 LEAF 处理缓存位于 `datasets/benchmark/leaf_2a_preprocessing/`。已完成的 BN 对照、线性探针和历史 TensorBoard 保留在 `artifacts/benchmark/cross_subject_fold0_bn_control/`。更早的代码通过 Git 恢复，旧权重和日志在仓库外归档。
