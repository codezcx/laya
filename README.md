# EEG → JEV-27B：配对信号与冻结联合读出

当前训练是 **MI左右手分类 + SEED/FACED负性/中性/正性分类共同监督**，六道MI属性题和六道情绪属性题作为辅助目标。冻结REVE和Qwen基座，更新投影器、JEV语言LoRA及共享决策行；LoRA始终启用。每个全局batch包含32条MI分类、48条情绪分类、54条属性题。loss分别占40%、40%、10%、10%。

当前300步四卡试验，每50步分别评估MI、SEED、FACED分类；选模只用两类最终分类的平衡准确率各50%，属性分数不能替代分类表现。数据和运行细节见协议。联合训练结果见实际验证文件；先前63.84%是仅MI冻结联合读出控制，不是本次联合训练结果。

当前另运行**只关闭属性辅助损失**的匹配对照，MI与情绪分类均保留。分类权重仍各0.4，全部batch与计算顺序保留；用同一训练入口的 `--attribute-loss-weight 0` 开关控制，详见协议末节。对照产物位于 `artifacts/jev_no_attribute_loss/`。

- [最终结果与限制](docs/RESULTS.md)
- [协议、标签预算与复现](docs/PROTOCOL.md)
- [数据格式和问题清单](docs/ATTRIBUTE_DATASET.md)
- [失败分析与匹配对照](docs/PAIRED_DIAGNOSTICS.md)
- [当前情绪属性排查](docs/EMOTION_DIAGNOSTICS.md)
- [早期task-only特征诊断](docs/FEATURE_PROBES.md)

先前读出控制仅适用于固定问题的二分类MI。新训练显式提供任务类型，尚未证明情绪分类改善、任意问题理解或自动识别任务。REVE预训练清单包含2b，不能称为全链路未见目标数据的零样本结果。

## 启动当前联合训练

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
```

入口先验证数据和标签补充文件，完成四卡两步smoke后启动300步训练；已有run/smoke时拒绝覆盖。当前日志为 `artifacts/jev_paired_qa/run/metrics.jsonl`，stdout为 `run.log`，运行状态为 `status.json`。曲线入口 `scripts/plot_jev_training.py`。首次构建分类补充文件用 `scripts/build_emotion_classification.py`，目录已有文件则拒绝覆盖。

## 复现先前冻结联合读出控制

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

代码只保留当前方法及必要匹配对照。之前三个训练对照的日志和checkpoint保留在 `control_sparse_labels/`、`control_random_options/`、`control_fixed_mapping/`。Git保存代码历史，忽略的数据和权重不能从Git恢复，不作为代码清理对象。
