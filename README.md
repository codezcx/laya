# EEG 与任务＋类别文本对齐

当前方案将冻结 REVE 的 EEG tokens，经一个可训练聚合投影器，映射到冻结 Qwen3-Embedding-0.6B 的文本空间。对齐完整的任务＋类别描述，五类联合检索；不设独立任务头，不生成答案文本。必要对照使用相同 EEG 投影结构和分类交叉熵。

- [实验协议](docs/PROTOCOL.md)：数据划分、文本、损失、训练及选模。
- [结果记录](docs/RESULTS.md)：本轮状态及此前完成的分类接口结果。
- `eeg_align/`：缓存、模型、训练、评估；`eeg_tasks/`：必要预处理。
- `configs/text_descriptions.json`：五类完整文本；其余配置保存数据和编码器路径。

从仓库根目录运行：

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -v
laya/.venv-eeg/bin/python scripts/download_text_encoder.py
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/prepare_text_bank.py
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_alignment.py
```

已有文本向量和实验目录时，准备/训练入口会拒绝覆盖。下载脚本可校验已由其他节点下载的权重。正式启动前检查文本表示诊断。

运行产物位于 `artifacts/eeg_text_alignment/`。`runs/status.json` 记录阶段，每组的 `metrics.jsonl` 记录 loss 和分开的 MI、情绪验证指标。四卡分别训练对比对齐/CE 对照各两个 seed。

仓库只保留当前方案及必要对照，历史代码通过 Git 恢复。前一方案完整提交为 `471cd15` 的父版本及其历史；准备阶段快照为 `471cd15`。`artifacts/`、`laya/`、`third_party/` 中的数据、权重、环境和日志不纳入 Git，也不随代码清理删除。
