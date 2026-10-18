# EEG → JEV-27B-VL：信号属性问答

当前训练阶段让 JEV 从 REVE tokens 回答频谱、空间和时间属性问题。使用真实 JEV-27B-VL 语言 decoder 与已有 LoRA；**LoRA 始终启用**。冻结 REVE 和语言基座，四卡共同更新投影器与 LoRA，以三选一交叉熵训练。

从上一轮源验证选出的 `artifacts/jev_eeg_coverage/runs/lora_seed42/best.pt`（4380步）继续，保留已训练投影器与 LoRA。本阶段只学习属性问答，不把属性成绩称为最终 MI/情绪分类成绩。

- [训练协议](docs/PROTOCOL.md)
- [当前结果与匹配基线](docs/RESULTS.md)
- [数据格式、问题清单](docs/ATTRIBUTE_DATASET.md)
- [FBCSP／REVE 特征探测](docs/FEATURE_PROBES.md)

训练问答位于 `datasets/eeg_attribute_qa/`；独立源验证问答位于 `datasets/eeg_attribute_qa_validation/`，均已 Git ignore。每题独立 JSONL，共享信号文件。只将问题、选项与 EEG tokens 输入模型，计算依据和原始类别不进入 prompt。

主要代码：`eeg_align/attribute_data.py` 加载和置换选项；`eeg_align/jev.py` 组装可变问题文本并限制到 A/B/C；`eeg_align/train_jev.py` 四卡训练、验证、断点恢复；`scripts/build_attribute_dataset.py` 与 `verify_attribute_dataset.py` 生成和校验数据。

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/plot_jev_training.py
```

启动器先校验输入并进行真实四卡两步 smoke，再开始1200步正式训练。产物 `artifacts/jev_attribute_qa/`：`status.json` 记录阶段，`run/metrics.jsonl` 记录loss及属性指标，`run.log`为运行日志。目录已存在会拒绝覆盖。

上一轮完整分类训练已结束，其结果只作为当前阶段的初始化及后续对照。此前实现归档于 Git `6b42e3d`，旧数据、权重、日志和环境不删除。五类分类评估及接口诊断保留为匹配下游诊断，不自动启动旧训练或目标集评估。
