# EEG → 任务＋类别文本对齐 → JEV

当前方案联合训练九个运动想象数据集、SEED 三分类和 SEED-VII 七分类。先训练 REVE 与任务＋类别文本对齐，再冻结 REVE，训练 JEV 的投影器、LoRA 和选择题输出头；2/3/7 个选项随机换序。训练仅使用 GPU2、3。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [MI 数据审计](docs/MI_DATASETS.md)
- [SEED 通道坐标](docs/SEED_COORDINATES.md)

```sh
laya/.venv-eeg/bin/python scripts/run_joint_training.py
```

状态：`artifacts/mi_emotion/pipeline.json`。两阶段日志分别在 `artifacts/mi_emotion/contrast/metrics.jsonl` 和 `artifacts/mi_emotion/jev_restart/metrics.jsonl`。入口先检查双卡训练与验证，再运行正式训练；输出目录已存在时拒绝覆盖。

数据缓存和被试划分保存在忽略目录 `datasets/mi_emotion/`；历史 checkpoint、日志不删除，旧实验实现由 Git 保存。
