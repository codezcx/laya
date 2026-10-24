# Task-conditioned EEG → LEAF → JEV

当前实验：将 EEG-FM-Benchmark 的11个分类数据集接入同一个冻结 LEAF＋JEV，以文本指定任务、随机排列选项进行联合 SFT。SEED-VIG 回归和 ThingsEEG2 匹配任务等待输出协议与数据，尚未加入。新实验使用固定被试划分，不是完整 LOSO 复现。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [任务、问题与选项](configs/benchmark_tasks.json)
- [MI 数据审计](docs/MI_DATASETS.md)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

入口完成四卡特征准备、冒烟测试、联合训练、最佳 checkpoint 的完整验证与测试评估；输出已存在时拒绝覆盖。状态：`artifacts/benchmark/pipeline.json`；训练指标：`artifacts/benchmark/jev/metrics.jsonl`。数据清单和特征在忽略目录 `datasets/benchmark/`。历史 checkpoint、日志和缓存保留，旧协议通过 Git 查询。

