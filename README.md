# EEGJEV research workspace

当前工作分支：`experiment/multitask-text-prototypes`。

- `eeg_align/`：历史 REVE→Qwen decoder / MI 实验。
- `eeg_tasks/`：新增多任务 EEG→文本原型分类；与历史模型独立。
- `configs/`：显式数据/模型路径、任务与超参数。
- `docs/`：版本化实验协议与结果摘要。
- `tests/`：数据划分、损失语义与输入处理检查。
- `artifacts/`：日志、缓存、权重、预测；不进Git。
- `third_party/`、`laya/`：外部仓库/模型；不纳入本仓库Git。

基线提交 `658a0b5` 保存本次多任务修改前的已有代码。保留原 `.JEV-27B-VL.git.backup-*`，不覆盖旧历史；未建立远程或推送。

协议见 [docs/MULTITASK_PROTOCOL_V1.md](docs/MULTITASK_PROTOCOL_V1.md)。所有命令从仓库根运行，Python用 `laya/.venv-eeg/bin/python`。先运行 `python -m eeg_tasks.data`，随后 `python scripts/run_multitask_prototypes.py`。每次重跑需新的输出目录，代码拒绝覆盖已经完成的训练。模型选择只看源验证，目标评估需要先保存冻结checkpoint哈希。

`eeg_tasks.predict`支持原始 `(3,T)` C3/Cz/C4 numpy信号，返回情绪和MI的条件分类分数；不把未校准的跨任务分数伪装成任务置信度。

持续目标：先建立MI/情绪跨数据集证据，再扩展心算、压力。任务已知分类、任务识别、未见任务泛化分别评估，不混用zero-shot概念。
