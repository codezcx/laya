# LEAF Tower → JEV：个人表征适配与支持集稳定性

_2026-11-25；检验个人更新是否受限于连接器，以及支持样本能否提供可重复的个体信息。_

---

## 🎯 当前实验

同一个已训练的 InfoNCE＋LEAF Tower→JEV parent，同一批四名用户，每人100条支持样本。比较连接器 LoRA 与同一连接器 LoRA＋Tower 全参更新；JEV和原连接器始终冻结。Tower 全参为19,557,632参数，个人LoRA为40,960参数。不是重训共享模型或实施FedP²EFT。

| GPU | 用户 | 运行顺序 |
| --- | --- | --- |
| 0 | 2a A08 | Tower全参＋连接器LoRA → 连接器LoRA |
| 1 | 2a A09 | Tower全参＋连接器LoRA → 连接器LoRA |
| 2 | HMC s2 | Tower全参＋连接器LoRA → 连接器LoRA |
| 3 | HMC s6 | Tower全参＋连接器LoRA → 连接器LoRA |

两组保持seed42、lr1e-4、100步、batch16/micro4、支持/query成员、文本和选项排列。Tower保持eval模式，BN运行统计与Dropout策略不变；这不妨碍计算权重梯度。完整support/query在0/5/10/20/50/100步记录，固定100步报告，不按query选点。

另用冻结特征检查类别均衡的支持集前后半、随机半集及源被试跨记录匹配。源数据只用原train；目标用户只使用support，不索引目标query特征设计统计量。

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.diagnostics
laya/.venv-eeg/bin/python -m personal.run --start
```

## 🔗 协议与产物

- [固定协议](docs/PROTOCOL.md)
- [结果与进度](docs/RESULTS.md)
- [当前假设及解释边界](docs/PERSONALIZATION_METHOD_REVIEW.md)
- [配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf_personal_tower_diagnostics_20261125/`。四卡每卡一人，完成即退，6小时安全上限。此前同种子ERM/FOMAML曲线已完成，归档于Git `1d78125`；所有ignored数据、checkpoint、日志与第三方环境保留。

## ⚠️ 结果边界

原test个人query是已观察的开发性评估，不是独立盲测；原valid本轮不使用。每任务仅两名目标用户，不推广为总体结论。个体可识别不等于特征有益于任务，HMC同夜稳定不等于跨夜稳定。
