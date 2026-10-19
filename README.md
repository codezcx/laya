# 运动想象 → REVE → JEV：左右手分类

当前使用9个源运动想象数据集的全部EEG通道和完整导出trial时长。冻结REVE与Qwen基座，训练新投影器、原有JEV LoRA和A/B决策头，每个trial计算一次左右手分类交叉熵。不同通道数与时长形成可变长度token，以mask和按长度分组处理。

训练49,193条；按被试隔离的验证12,508条；2b的6,520条仅作跨数据集最终测试。未知电极位置不猜测、不静默删除EEG通道；本轮暂缓Brandl2020和GuttmannFlury2025_MI。

- [当前协议](docs/PROTOCOL.md)
- [当前结果与运行位置](docs/RESULTS.md)
- [数据审计与排除原因](docs/MI_DATASETS.md)

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
```

入口依次完成四卡冻结REVE缓存、数据哈希核验、四卡两步检查、3个epoch训练及验证集最佳checkpoint的2b测试。拒绝覆盖已有输出。数据缓存:`datasets/mi_tokens/`；输出:`artifacts/mi_jev/`；阶段:`status.json`；逐步loss和固定验证指标:`run/metrics.jsonl`。每200步及每轮末验证，按各源数据集验证BA的平均值选模型，同时报告准确率。

TUAB已在第265步按用户要求停止，代码与停止状态归档于Git`7335cfd`，日志/权重保留在`artifacts/tuab_jev/`。不加载其训练权重。所有旧数据、日志、checkpoint保留，不作为代码清理对象。

验证代码:`PYTHONPATH=. laya/.venv-eeg/bin/python -m pytest tests -q`。
