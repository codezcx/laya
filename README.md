# LEAF Tower → JEV：同种子个人适配曲线

_2026-11-25；固定上一轮的四名用户和全部超参数，只修正随机初始化种子并补齐等步数评估。_

## 📋 当前实验

共享 LEAF Tower、原连接器、JEV27B 全部冻结。只更新连接器最后一层上的个人 LoRA：rank4、40,960参数，学习率1e-4，100条支持样本、100步、batch16/micro4。三组为普通个人 LoRA、source ERM 初始化、FOMAML 初始化；后两组直接读取上一轮已完成的权重，不重跑离线训练。

| GPU | 目标用户 | 本卡依次运行 |
| --- | --- | --- |
| 0 | 2a A08 | 普通 LoRA、source ERM、FOMAML |
| 1 | 2a A09 | 普通 LoRA、source ERM、FOMAML |
| 2 | HMC s2 | 普通 LoRA、source ERM、FOMAML |
| 3 | HMC s6 | 普通 LoRA、source ERM、FOMAML |

普通 LoRA 的随机 A 统一用 seed42，零 B；它必须与两种源训练的最初参数指纹一致。源训练已经学到的 A/B 不重置。每名用户的支持集、完整query、训练批次和选项排列保留历史设置。

第0、5、10、20、50、100步保存完整support/query指标、预测和个人参数。固定第100步结束，不按query选最好checkpoint。两任务各2人，只有相同步数下所有配对用户和方法齐全后才计算任务均值、两任务等权均值。

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.run --start
```

- [固定协议](docs/PROTOCOL.md)
- [进度与曲线位置](docs/RESULTS.md)
- [方法选择依据](docs/PERSONALIZATION_METHOD_REVIEW.md)
- [当前配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf_personal_matched_curves_20261125/`。四卡每卡一名用户，完成即退，保留6小时安全上限；不是要求跑满6小时。上一轮完整结果及离线实现已归档Git `4a664b1`，其初始化、所有ignored数据/权重/日志均保留。

## ⚠️ 结果边界

原test个人query已被历史观察，本轮是开发性配对分析，不是独立测试。原valid不读取信号、不参与适配；个人梯度仅来自本人100条support。新普通LoRA换成共同seed42，不能把上一轮普通LoRA分数当本轮结果。每任务仅2人，不作总体显著性结论。
