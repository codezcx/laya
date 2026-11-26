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

2026-11-25追加同一方法的rank对照：两层连接器rank16（Tower冻结），以及Tower rank16＋两层连接器rank4。各21人、100条support，新增42次；复用当前rank4条件作对照，不重复训练。这是固定训练配方下的rank比较，不是宣称各rank已充分调优。

## ⚙️ 运行与记录

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.run --start
```

四卡每卡加载一份JEV，共42次适配/卡，12小时安全上限。先完成100条的位置对照，再完成48、24条的样本量对照。部分A05条件分到相同型号的其他GPU以消除长尾，不改变初始化或样本。

追加队列使用同一`personal.worker`，不修改活跃代码或原配置：

```bash
laya/.venv-eeg/bin/python -m scripts.queue_personal_rank --start
```

先等待当前四卡队列退出、100条的配对对照全部齐全且GPU释放，再依次运行两个rank条件；预计新增3–4小时，新增训练总上限4小时，原12小时预算不变。若前序失败或用户停止，不自动继续。追加输出为`artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/`，含独立计划、状态与配对报告；所有数据/优化/冻结检查复用当前实现。

2026-11-26迁移到8卡节点：原rank4的168条件全部完成，不重跑。迁移时rank16已完成8/42，剩余34条件重新分配到8卡，其中4个从80/90步断点恢复。仅改变物理GPU分配；原计划、样本、初始化、优化器和训练配方不变。迁移前首次执行以下命令，之后仍用`--start`续跑：

```bash
laya/.venv-eeg/bin/python -m scripts.queue_personal_rank --resume-gpus 0 1 2 3 4 5 6 7
laya/.venv-eeg/bin/python -m scripts.queue_personal_rank --start
```

`execution.json`保存独立的执行分配及来源审计；不改写旧checkpoint。继续沿用原追加队列09:56 UTC截止时间，不重新计时。实际启动状态见[进度记录](docs/RESULTS.md)。

[结果文档](docs/RESULTS.md)已更新为完整结果：rank4的168条件、rank16的42条件均完成，旧rank队列已结束。上面的迁移命令和时间是历史执行记录，不需重新运行。

2026-11-26新增MAML实验放在隔离worktree，不修改本工作树训练代码：前四卡为`personal-fomaml100`的60轮末层LoRA实验；后四卡为`personal-maml-order`的5轮两层linear机制矩阵，比较全参/LoRA、一阶/二阶、inner5/20/100。后者lr1e-4、有效batch16、100support、rank4、个人100步不变，仅microbatch改为1以运行二阶，全部新组相同。当前队列及检查见[结果文档末节](docs/RESULTS.md#maml追加实验入口)。不要用这两个不同范围/预算的运行直接作同条件比较。

- [固定协议](docs/PROTOCOL.md)
- [进度与结果入口](docs/RESULTS.md)
- [假设与结论边界](docs/PERSONALIZATION_METHOD_REVIEW.md)
- [当前配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

输出 `artifacts/benchmark/leaf_personal_lora_locations_20261125/`；其中 `report.md`、`summary.json`、CSV与曲线自动更新。此前Tower全参pilot及统计诊断已归档Git `31a36d9`，不保留另一套执行框架；所有旧数据、权重、缓存、日志与环境保留。

## ⚠️ 解释范围

这些query属于已观察的探索性个人评估，不是独立盲测。2a A01–A05与Nakanishi query在历史共享训练池内；SEED用户见过，但query原始trial未用于共享训练。共享训练未见的2a/HMC用户另行汇总。SEED/HMC的样本数指窗口/epoch，不是独立trial/夜。
