# Task-conditioned EEG → LEAF → JEV

当前保留一套 EEG-FM-Benchmark 联合训练代码，11 个分类数据集，文本指定任务与选项，采用被试隔离的第 0 折。当前实验比较 BatchNorm 正常更新与固定统计量：两组各用一张 H200，从同一个 step 100 checkpoint 开始，其余配置一致。

LEAF 解冻训练、JEV LoRA 始终开启；AdamW 加预热和余弦学习率衰减。有效 batch 88、microbatch 22。先比较到 step 1200，保持原 3000 步学习率曲线。完整保存参数、BN buffers、优化器、scheduler 和随机状态，并在保存后检查预测可复现。

- [当前训练协议](docs/PROTOCOL.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

当前输出：`artifacts/benchmark/cross_subject_fold0_bn_control/`。`LIVE.md` 和 `monitor.json` 汇总两组变化；每组的 `metrics.jsonl` 保存 loss、准确率、学习率和 BN 状态。原始信号缓存与划分在 `datasets/benchmark/signals/`。旧代码通过 Git 恢复，旧实验权重和日志已移到仓库外归档。

