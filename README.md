# EEG → JEV-27B-VL：基线配对的信号属性问答

当前阶段重建 MI 数据：同一 trial 的提示前 fixation 与提示后运动想象片段配对，生成 6 道功率变化辅助题和 1 道左右手分类题。情绪的六类属性问答保留。旧四卡训练已按用户要求停止，检查点与日志保留；**本次不自动重启训练**。

- [数据格式、问题与构建方法](docs/ATTRIBUTE_DATASET.md)
- [当前协议与模型接入待办](docs/PROTOCOL.md)
- [数据审计与匹配基线](docs/RESULTS.md)
- [FBCSP／REVE 特征探测](docs/FEATURE_PROBES.md)

活动数据位于 `datasets/eeg_attribute_qa/` 与 `datasets/eeg_attribute_qa_validation/`，均已 Git ignore。每题独立 JSONL，共享信号数组。当前 MI 仅使用有本地连续原始记录的 OpenBMI，沿用源被试划分。模型输入只允许 `input` 字段，不能包含 evidence / provenance / target。

主要代码：`eeg_tasks/attribute_questions.py` 定义问题及答案；`scripts/build_attribute_dataset.py` 从原始 MAT 构建配对并安全替换旧 MI；`scripts/verify_attribute_dataset.py` 校验哈希并独立重算全部 MI 答案。

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest tests.test_attribute_questions -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/verify_attribute_dataset.py datasets/eeg_attribute_qa
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/verify_attribute_dataset.py datasets/eeg_attribute_qa_validation
```

旧单窗 REVE cache 没有基线，不兼容新 schema。`eeg_align/attribute_data.py` 明确拒绝新 schema 进入旧训练接口。`scripts/run_jev.py` 同样在启动前拒绝；训练适配完成前不要启动。已有 JEV/REVE 组件保留供本方案接入，LoRA 仍须始终启用。

旧方案归档于 Git `7ab8f21`；原始信号、检查点、日志、第三方环境不参与清理。仅经用户授权删除活动目录内已替换的旧 MI 问答和共享 MI 数组。
