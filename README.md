# EEG → 任务＋类别文本对齐 → JEV

当前实验：共享 REVE＋MI/MAT 独立对比投影头，对齐左右手运动想象和静息／心算文本。完整保留49,193条 MI trial；MAT 按被试划分。使用 GPU2、3，不自动接续 JEV。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [MI 数据审计](docs/MI_DATASETS.md)

```sh
laya/.venv-eeg/bin/python scripts/run_joint_training.py --contrast-only --split-contrast-heads --contrast-output contrast --contrast-smoke-output contrast_smoke
```

状态：`artifacts/mi_mat/pipeline.json`。当前日志在 `artifacts/mi_mat/contrast/metrics.jsonl`。入口先检查双卡训练与验证，再运行正式训练；输出目录已存在时拒绝覆盖。

数据缓存和被试划分保存在忽略目录 `datasets/mi_mat/`；历史 checkpoint、日志不删除，旧实验实现由 Git 保存。
