# EEG → JEV-27B-VL

当前方案使用真正的本地 JEV-27B-VL 语言 decoder、原始 JEV LoRA 与决策输出。REVE 编码 EEG 后，每个 512 维 token 经可训练投影器变为 5120 维，插入 JEV 的问题文本，单次 forward 输出 A–E 五类 logits。

**所有组始终启用 JEV LoRA。冻结仅表示不更新参数，绝不表示关闭 adapter。**

当前改为完整覆盖训练：两个匹配组各用两卡，比较只更新投影器与更新投影器＋原始JEV LoRA。REVE均冻结，固定两轮覆盖、每组4380步，全局batch48；用完整源验证集选模。真实模型的梯度同步和精确断点恢复已验证，两组正式训练已启动。总体跨数据集目标尚未达成。

- [实验协议](docs/PROTOCOL.md)
- [结果和状态](docs/RESULTS.md)
- [源训练集属性问答](docs/ATTRIBUTE_DATASET.md)：频谱、空间、时间比较，每种问题独立 JSONL；数据在 Git 忽略的 `datasets/eeg_attribute_qa/`，本次仅构造数据，尚未接入训练。
- [FBCSP／REVE 特征探测](docs/FEATURE_PROBES.md)：独立 CPU 诊断任务与被试信息，不改动 JEV 训练。
- `eeg_align/jev.py`：严格加载语言权重、已有 LoRA、决策槽位，拼接 EEG/text embeddings。
- `eeg_align/signals.py`：原始信号缓存、可训练 REVE；`cache.py`：冻结 REVE tokens。
- `eeg_align/train_jev.py`、`evaluate_jev.py`：训练和源选模后的评估。

从仓库根目录运行（准备产物和实验目录已存在时会拒绝覆盖）：

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/audit_jev.py
PYTHONPATH=. laya/.venv-eeg/bin/python -m eeg_align.signals
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
```

实际 checkpoint：`laya/JEV-27B-VL/`，adapter：其 `adapter_vllm/`。当前训练产物：`artifacts/jev_eeg_coverage/`；`runs/status.json` 记录阶段，各组 `metrics.jsonl` 记录 loss、MI 和情绪验证指标。

只保留当前实现和必要对照。此前“任务＋类别文本相似度”方案（非 JEV 推理）完整版本在 Git `0e6c506`；原始对比方案代码不再留在工作区。忽略目录中的数据、权重、日志和环境保留，不通过 Git 管理或随代码清理删除。

接口诊断入口：`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_interface_diagnostic.py`；状态与日志在 `artifacts/jev_interface_diagnostic/`。训练小集准确率只能证明接口拟合能力，不能作为zero-shot指标。
