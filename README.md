# EEG → 任务＋类别文本对齐 → JEV

当前实验：先评估原生 LEAF 在2a四分类、SEED三分类的文本匹配表现，再冻结LEAF的8个256维query，联合训练JEV连接模块、LoRA和决策头。随机打乱选项，按被试独立验证。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [MI 数据审计](docs/MI_DATASETS.md)

```sh
CUDA_VISIBLE_DEVICES=2 laya/.venv-eeg/bin/python -m eeg_align.leaf_data
# Inspect artifacts/leaf_2a_seed/baseline.json before launching:
laya/.venv-eeg/bin/python scripts/run_joint_training.py
```

状态：`artifacts/leaf_2a_seed/pipeline.json`。当前日志在 `artifacts/leaf_2a_seed/jev/metrics.jsonl`。入口先检查双卡训练与验证，再运行正式训练；输出目录已存在时拒绝覆盖。

数据缓存和被试划分保存在忽略目录 `datasets/leaf_2a_seed/`；历史 checkpoint、日志不删除，旧实验实现由 Git 保存。
