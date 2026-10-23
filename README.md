# EEG → 任务＋类别文本对齐 → JEV

当前实验：用 MI＋MAT 对比学习最终 REVE 接入 JEV-27B-VL。冻结 REVE，训练连接模块、原生 JEV LoRA 和决策头；任务对应问题，随机交换 A/B 选项。使用 GPU2、3 联合训练，按被试独立验证。

- [训练协议和数据划分](docs/PROTOCOL.md)
- [结果与日志位置](docs/RESULTS.md)
- [MI 数据审计](docs/MI_DATASETS.md)

```sh
laya/.venv-eeg/bin/python scripts/run_joint_training.py --jev-only --encoder-checkpoint artifacts/mi_mat/contrast/last.pt
```

状态：`artifacts/mi_mat/pipeline.json`。当前日志在 `artifacts/mi_mat/jev/metrics.jsonl`。入口先检查双卡训练与验证，再运行正式训练；输出目录已存在时拒绝覆盖。

数据缓存和被试划分保存在忽略目录 `datasets/mi_mat/`；历史 checkpoint、日志不删除，旧实验实现由 Git 保存。
