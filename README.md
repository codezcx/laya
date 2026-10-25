# Task-conditioned EEG → LEAF → JEV

当前实验：EEG-FM-Benchmark 的11个分类数据集混合训练，文本指定任务和选项。先跑跨被试联合第0折：普通数据集留一位测试被试、一位验证被试；TUAB/SleepEDF按被试分10组，分别留测试组和验证组。不是已完成的全折 Benchmark 复现。

训练实时输入 EEG：前100步冻结 LEAF，随后解冻，与 JEV LoRA、投影器、选择头联合训练。当前做学习率对照：从相同初始权重重训，加入90步学习率预热和余弦衰减，最低为峰值的2%；LEAF在解冻后单独预热90步。两张H200，每卡microbatch22、累积2次，有效batch仍为88。旧固定学习率实验停在日志step1087，权重和日志保留。SEED-VIG、ThingsEEG2仍待数据和输出协议。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

入口复用原始数组、检查被试隔离，完成双卡冒烟测试、联合训练、最佳 checkpoint 完整评估。状态：`artifacts/benchmark/cross_subject_fold0_lr_cosine/pipeline.json`；指标（含各参数组实际学习率）：`artifacts/benchmark/cross_subject_fold0_lr_cosine/jev/metrics.jsonl`。原始信号缓存与划分在忽略目录 `datasets/benchmark/signals/`。

