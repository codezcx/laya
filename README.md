# EEG → 任务＋类别文本对齐 → JEV

当前实验：共享 REVE＋MI/SEED 独立对比投影头，仅监督九个运动想象数据集和 SEED 三分类；SEED-VII 只保留零损失诊断。每轮完整遍历49,193条 MI trial，不放回、不丢弃，共1,538步；情绪数据继续混合。使用 GPU2、3，不自动接续 JEV。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [MI 数据审计](docs/MI_DATASETS.md)
- [SEED 通道坐标](docs/SEED_COORDINATES.md)

```sh
laya/.venv-eeg/bin/python scripts/run_joint_training.py --contrast-only --mi-seed-loss-only --split-contrast-heads --contrast-output contrast_mi_seed_split --contrast-smoke-output contrast_mi_seed_split_smoke
```

状态：`artifacts/mi_emotion/pipeline.json`。当前日志在 `artifacts/mi_emotion/contrast_mi_seed_split/metrics.jsonl`。入口先检查双卡训练与验证，再运行正式训练；输出目录已存在时拒绝覆盖。

数据缓存和被试划分保存在忽略目录 `datasets/mi_emotion/`；历史 checkpoint、日志不删除，旧实验实现由 Git 保存。
