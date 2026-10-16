# EEG → Qwen

工作区只保留当前方案：冻结 REVE 编码原始 EEG，通过可训练 connector 接入冻结 Qwen，输出情绪三类和运动想象两类。参数量匹配的信号分类器是同一方案的必要对照。

- `eeg_align/`：当前 token 缓存、模型、训练和外部评估。
- `eeg_tasks/`：当前方案所需的预处理、通道映射和标签读取辅助函数；旧模型与训练流程已删除。
- `scripts/`：当前训练启动、审计和小样本验证。
- `configs/`：当前使用的数据与模型路径。
- `tests/`：当前方案的接口与指标检查。
- [当前实验协议](docs/MULTITASK_BRIDGE_V31_PROTOCOL.md)：固定 3,000 步的训练预算检查。
- [最近完成的结果](docs/MULTITASK_BRIDGE_V30_RESULTS.md)：500 步对照及外部结果。

从仓库根运行，Python 为 `laya/.venv-eeg/bin/python`。当前已有训练进程，不要重复启动；状态在 `artifacts/multitask_bridge_v31/status.json`。仅查看复现审计可运行：

```sh
laya/.venv-eeg/bin/python scripts/audit_bridge_budget_prefix_v31.py
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests
```

历史方案通过 Git 查看或恢复，不在主工作区复制多套代码。清理前完整快照是 `6642113`，例如 `git show 6642113:docs/EXPERIMENT_INDEX.md`。需要重跑历史实验时使用独立 worktree，避免覆盖当前训练代码。

当前仍有少量历史版本号文件名作为正在运行实验的兼容入口；其中仅保留当前方案用到的函数，没有旧方案训练实现。运行中的 v31 会校验训练器哈希，因此完成前不改其路径、实现或配置。

`artifacts/`、`laya/`、`third_party/` 不纳入 Git。原始数据、checkpoint、日志和缓存不能通过 Git 恢复，本次代码清理保留它们。后续成果以当前协议和结果记录维护，旧代码只留在提交历史。
