# EEG → JEV-27B-VL：基线配对的信号属性问答

当前训练使用 OpenBMI 同一trial的提示前基线和运动想象片段：6道功率变化辅助题＋1道左右手分类题；情绪六类属性问答保持不变。冻结REVE与语言基座，更新投影器与JEV已有LoRA，**LoRA始终开启**。两段独立编码，通过明确的baseline/task文字边界送入语言decoder。当前分类题固定A=左手、B=右手，属性题仍随机选项；这是针对随机选项训练固定字母塌缩的单变量对照。

- [训练协议与指标](docs/PROTOCOL.md)
- [数据格式和问题清单](docs/ATTRIBUTE_DATASET.md)
- [当前结果与数据审计](docs/RESULTS.md)
- [表示探针与采样改进依据](docs/PAIRED_DIAGNOSTICS.md)
- [FBCSP／REVE特征探测](docs/FEATURE_PROBES.md)

数据在 `datasets/eeg_attribute_qa/` 和 `datasets/eeg_attribute_qa_validation/`，均Git ignore，每题独立JSONL、共享信号数组。只将input字段输入模型，不能加入evidence/provenance/target。训练208,462题、验证57,100题，按被试隔离。

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/plot_jev_training.py
```

启动器先核对数据/缓存/初始化，再做真实四卡两步smoke，成功后运行600步。已有run目录时拒绝覆盖。当前产物集中在 `artifacts/jev_paired_qa/`：`status.json`、`run/metrics.jsonl`、`run.log`、`run/best.pt`和`run/last.pt`。初始化独立保存在`initialization.pt`，旧实验日志/checkpoint按用户授权清理，清单见`cleanup.json`；数据缓存与预训练模型保留。

核心实现：`eeg_align/paired_cache.py` 分段REVE缓存；`eeg_align/attribute_data.py` 混合选项、采样与指标；`eeg_align/jev.py` 分段文本与信号组装；`eeg_align/train_jev.py` 四卡训练与断点恢复。旧方案只通过Git历史保留代码。
