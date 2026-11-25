# LEAF Tower → JEV：个人 LoRA 位置与样本量

_2026-11-25；同一历史parent、21名用户、固定个人query，比较更新位置和支持样本数量。_

---

## 🎯 当前实验

个人训练只更新LoRA，原LEAF Tower、原连接器、JEV权重与缓冲全部冻结。每名用户、每个方法、每种样本量均从同一parent独立重置，不混合不同用户训练。

| 方法 | LoRA位置 | 个人参数 | 支持样本数 |
| --- | --- | ---: | --- |
| last_only | 连接器第二层5120→5120 | 40,960 | 100 |
| first_only | 连接器第一层256→5120 | 21,504 | 100 |
| both | 两层连接器 | 62,464 | 24、48、100 |
| tower_both | Tower前向线性权重＋两层连接器 | 462,848 | 24、48、100 |

2a全部9人、HMC4人、SEED4人、Nakanishi2015四人，共168次个人适配。统一rank4、scale1、lr1e-4、100步、batch16/micro4；JEV CE监督。24⊂48⊂100，query不随样本量变化。固定step100为主结果；0/5/10/20/50/100完整记录，不按query选点。

## ⚙️ 运行与记录

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.run --start
```

四卡每卡加载一份JEV，共42次适配/卡，12小时安全上限。先完成100条的位置对照，再完成48、24条的样本量对照。部分A05条件分到相同型号的其他GPU以消除长尾，不改变初始化或样本。

- [固定协议](docs/PROTOCOL.md)
- [进度与结果入口](docs/RESULTS.md)
- [假设与结论边界](docs/PERSONALIZATION_METHOD_REVIEW.md)
- [当前配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

输出 `artifacts/benchmark/leaf_personal_lora_locations_20261125/`；其中 `report.md`、`summary.json`、CSV与曲线自动更新。此前Tower全参pilot及统计诊断已归档Git `31a36d9`，不保留另一套执行框架；所有旧数据、权重、缓存、日志与环境保留。

## ⚠️ 解释范围

这些query属于已观察的探索性个人评估，不是独立盲测。2a A01–A05与Nakanishi query在历史共享训练池内；SEED用户见过，但query原始trial未用于共享训练。共享训练未见的2a/HMC用户另行汇总。SEED/HMC的样本数指窗口/epoch，不是独立trial/夜。
