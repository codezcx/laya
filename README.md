# EEG → JEV-27B：配对信号与冻结联合读出

当前实现使用冻结 REVE 编码提示前参考段和运动想象段，经 MLP 接入 JEV-27B 的语言主干；已有 JEV LoRA 始终启用。固定决策头训练出现单类输出，已停止。改进方案冻结第100步模型，将最后决策位置的5120维表示与3072维REVE池化特征拼接，拟合二分类Ridge读出。

已完成端到端源验证和外部2b评估：联合读出58.70% / 63.84%，同checkpoint原头50.20% / 50.60%；REVE单独读出58.50% / 64.96%。联合方案恢复信号相关分类，但未证明语言分支优于REVE。当前无活跃训练或评估，不建议继续原600步配方。

- [最终结果与限制](docs/RESULTS.md)
- [协议、标签预算与复现](docs/PROTOCOL.md)
- [数据格式和问题清单](docs/ATTRIBUTE_DATASET.md)
- [失败分析与匹配对照](docs/PAIRED_DIAGNOSTICS.md)
- [早期task-only特征诊断](docs/FEATURE_PROBES.md)

当前结论仅适用于固定问题、固定A=左手/B=右手的二分类MI。没有证明情绪分类改善、任意问题理解或自动识别任务。REVE预训练清单包含2b，不能称为全链路未见目标数据的零样本结果。

## 复现冻结联合读出

从仓库根目录运行，需本机已保存的模型和忽略的实验产物。输出目录必须不存在；以下是复现入口，不会自动启动训练。

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. laya/.venv-eeg/bin/python scripts/evaluate_paired_classification.py \
  --checkpoint artifacts/jev_paired_qa/diagnostics/fixed_step100.pt \
  --readout-head artifacts/jev_paired_qa/diagnostics/fixed_step100_readout/combined/combined_all_training_labels_head.json \
  --external-input artifacts/jev_paired_qa/external_2b \
  --cases matched shuffled_within_subject \
  --output artifacts/jev_paired_qa/reproduce_2b
```

重新审计已有结果不占GPU：

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/report_readout_evaluation.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 laya/.venv-eeg/bin/python -m pytest tests -q
```

逐trial预测、逐被试成绩、固定评估清单及哈希集中在 `artifacts/jev_paired_qa/readout_evaluation/`。数据位于 `datasets/eeg_attribute_qa/` 和 `datasets/eeg_attribute_qa_validation/`，均被Git忽略；每题独立JSONL，共享信号数组。只能向模型提供input字段，不能加入evidence/provenance/target。

代码只保留当前方法及必要匹配对照。`scripts/run_jev.py` 是已失败训练对照的复现入口，不是当前推荐训练方案；日志和checkpoint保留在 `control_sparse_labels/`、`control_random_options/`、`control_fixed_mapping/`。Git保存代码历史，忽略的数据和权重不能从Git恢复，不作为代码清理对象。
