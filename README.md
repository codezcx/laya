# Task-conditioned EEG → LEAF → JEV

当前保留一套 EEG-FM-Benchmark 训练代码：文本指定任务与选项，被试隔离的第 0 折。完整任务目录保留 11 个数据集，本轮只训练 2a／2a＋SEED 的匹配对照。

当前四组：`single_frozen`、`joint_frozen`、`single_finetuned`、`joint_finetuned`，分别使用 GPU 0–3 的 H100 80GB。全部从相同的发布版 LEAF／原生 JEV 初始化，固定 BN 和 LEAF dropout；微调组从第 101 步更新 LEAF。每组 600 步，每步相同的 8 条 2a 样本，联合组额外加 8 条 SEED，损失为各任务平均 CE 之和。先比较固定第 600 步的 2a 验证 BA，再决定后续修复；不自动启动 PCGrad 或 11 数据集训练。

- [当前训练协议](docs/PROTOCOL.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

当前输出：`artifacts/benchmark/cross_subject_fold0_task_mixture/`。每组的 `metrics.jsonl`、根目录的 `LIVE.md` 和 `monitor.json` 记录过程；TensorBoard 自动保存到该目录的 `tensorboard/`。四组真实模型 smoke 通过、初始状态和全部 2a 抽样序列一致后才开始正式训练。启动器拒绝覆盖已有运行目录。

原信号和划分在 `datasets/benchmark/signals/`，四组共用的 2a 缓存位于 `datasets/benchmark/leaf_2a_preprocessing/`（含被试级无标签归一化，非严格单试次 zero-shot）。本轮不使用测试集。已完成的 BN 对照和输入管线对照分别保留在 `artifacts/benchmark/cross_subject_fold0_bn_control/`、`artifacts/benchmark/cross_subject_fold0_leaf_preprocessing/`；此前协议在 Git `5200248`，旧数据和权重不删除。
