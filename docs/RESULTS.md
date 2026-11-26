# 个人 LoRA 位置、样本量与rank：实验结果记录

_结果快照：2026-11-26；rank4全部168/168、追加rank16全部42/42完成。_

---

## 📋 本轮范围

21名用户：2a9、HMC4、SEED4、Nakanishi2015四人。100条support时比较first-only、last-only、both、Tower-LoRA＋both；后两组再比较48/24条。每条件独立初始化100步，精确成员见[协议](PROTOCOL.md)。

rank4已完成168/168次适配及1,008个评估点；rank16两组各21/21完成，共42份结果和252个评估点。旧rank队列已结束。前四张卡另在隔离worktree运行末层LoRA的FOMAML100/ERM对照，不与本次两层linear新实验混淆。

当前观察：连接器rank4→16的四任务平均47.74%→48.14%（+0.40个百分点）；Tower rank4→16、连接器固定rank4，48.12%→48.62%（+0.51个百分点）。均为同一21人的配对结果；任务/个体收益不一致，不能据此宣布最优rank。

### 统一比较口径

- 结构：已训练的LEAF Tower → 两层连接器（256→5120→5120）→ JEV-27B。所有原参数和buffer冻结，只更新指定位置的个人LoRA；Tower LoRA不是Tower全参微调。
- 每个用户、方法、支持量从同一parent独立重置，不把21人混在一起做个人微调。parent为`leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360；完整SHA见[协议](PROTOCOL.md)。
- 损失为JEV最终选项logits的交叉熵。学习率1e-4、AdamW、batch16/micro4、100个更新步、seed42、LoRA缩放alpha/r=1均固定。个人标签仅用于support梯度，query不参与梯度。
- 主结果取固定step100，不挑最高query点。step0/5/10/20/50/100均保存完整support和query评估；本文百分数为BAcc，明确标记Acc的列除外。变化单位为百分点。
- 同一用户不同方法的支持/查询ID、每步训练样本和选项安排保持一致；24⊂48⊂100，query不随支持量改变。24/48/100条对应约66.67/33.33/16个等效epoch，并非等epoch比较。
- 按用户平均，再按任务等权平均；不是按窗口数加权。所有query均为历史已观察的探索性个人评估，不是独立盲测。

## 📊 已完成rank4：位置与支持量

数据来自[rank4完整报告](../artifacts/benchmark/leaf_personal_lora_locations_20261125/report.md)和[汇总JSON](../artifacts/benchmark/leaf_personal_lora_locations_20261125/summary.json)。下表每行均覆盖同一21名用户、四个任务；不是单个最优被试。

| 更新位置　　　　　　　　　 | 每人支持样本 | 个人可训练参数 | 四任务等权query BAcc（%） | 四任务等权support Acc（%） | 相对未适配变化（百分点） |
| ----------------------------| -------------:| ---------------:| --------------------------:| ---------------------------:| -------------------------:|
| 第二层连接器LoRA　　　　　 | 100　　　　　| 40,960　　　　| 47.91　　　　　　　　　　 | 56.22　　　　　　　　　　　| +4.16　　　　　　　　　　|
| 第一层连接器LoRA　　　　　 | 100　　　　　| 21,504　　　　| 45.79　　　　　　　　　　 | 49.83　　　　　　　　　　　| +2.04　　　　　　　　　　|
| 两层连接器LoRA　　　　　　 | 100　　　　　| 62,464　　　　| 47.74　　　　　　　　　　 | 56.37　　　　　　　　　　　| +3.99　　　　　　　　　　|
| Tower LoRA＋两层连接器LoRA | 100　　　　　| 462,848　　　 | 48.12　　　　　　　　　　 | 58.72　　　　　　　　　　　| +4.37　　　　　　　　　　|
| 两层连接器LoRA　　　　　　 | 48　　　　　 | 62,464　　　　| 47.33　　　　　　　　　　 | 59.17　　　　　　　　　　　| +3.58　　　　　　　　　　|
| Tower LoRA＋两层连接器LoRA | 48　　　　　 | 462,848　　　 | 47.86　　　　　　　　　　 | 62.30　　　　　　　　　　　| +4.11　　　　　　　　　　|
| 两层连接器LoRA　　　　　　 | 24　　　　　 | 62,464　　　　| 44.82　　　　　　　　　　 | 65.77　　　　　　　　　　　| +1.07　　　　　　　　　　|
| Tower LoRA＋两层连接器LoRA | 24　　　　　 | 462,848　　　 | 44.83　　　　　　　　　　 | 70.40　　　　　　　　　　　| +1.08　　　　　　　　　　|

未适配四任务等权BAcc约43.75%。该平均包含共享训练已见用户和未见用户，不能整体称为新用户泛化。

### 每个数据集的100条结果

前五列的窗口数、成员和共享划分来自本轮parent所用历史缓存，而非此前20261029九库表或原生LEAF复现。评分只覆盖“个人评估人数”列指定的用户，不是整个库的全部测试窗口。SEED/HMC计数分别是处理后窗口/睡眠epoch，不是独立原始trial/夜。

| 数据集 | 历史缓存窗口总数（训／验／测） | 任务种类（分类数） | 共享划分／本轮个人划分 | 历史缓存被试总数（训／验／测） | 个人评估人数 | 未适配BAcc（%） | 第二层LoRA（%） | 第一层LoRA（%） | 两层LoRA（%） | Tower＋两层LoRA（%） |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCIC-IV-2a | 5,184（2,880／1,152／1,152） | 运动想象（4类） | 共享固定跨被试；个人T→E场次 | 9（5／2／2） | 9 | 47.18 | 51.54 | 49.65 | 51.74 | 52.12 |
| SEED | 37,890（22,455／7,875／7,560） | 情绪识别（3类） | 共享被试内原始trial隔离；个人原train→原test | 15（15／15／15） | 4 | 56.60 | 58.83 | 57.31 | 58.69 | 57.45 |
| HMC | 136,925（92,454／22,421／22,050） | 睡眠分期（5类） | 共享跨被试；个人同夜前60%支持池→隔开2个epoch后的后段 | 151（103／24／24） | 4 | 60.20 | 69.66 | 66.08 | 69.51 | 71.23 |
| Nakanishi2015 | 1,800（1,800／0／0） | SSVEP（12类） | 共享全库train；个人分层100条support→剩余80条query | 10（10／0／0） | 4 | 11.01 | 11.61 | 10.12 | 11.01 | 11.66 |

SEED的15名用户在三个split均出现，因此15（15／15／15）不是45名独立用户。完整原始成员和窗口ID见[本轮计划](../artifacts/benchmark/leaf_personal_lora_locations_20261125/plan.json)。

个人评估成员与query数量：2a A01–A09各288条E场次trial；SEED s1–s4各504个窗口；HMC s7/s9/s2/s6分别411/358/340/344个epoch；Nakanishi2015 s1–s4各80条trial。

### 共享训练未见用户单独统计

仅包含2a A06–A09和HMC s7/s9/s2/s6，共8名用户、两个任务，先各自按用户平均再按任务等权；排除2a A01–A05、SEED及Nakanishi2015。

| 更新位置 | 每人支持样本 | 两任务等权query BAcc（%） | 相对未适配变化（百分点） |
| --- | ---: | ---: | ---: |
| 第二层连接器LoRA | 100 | 60.09 | +8.46 |
| 第一层连接器LoRA | 100 | 56.78 | +5.15 |
| 两层连接器LoRA | 100 | 60.01 | +8.38 |
| Tower LoRA＋两层连接器LoRA | 100 | 61.26 | +9.64 |
| 两层连接器LoRA | 48 | 60.71 | +9.08 |
| Tower LoRA＋两层连接器LoRA | 48 | 60.89 | +9.27 |
| 两层连接器LoRA | 24 | 58.05 | +6.42 |
| Tower LoRA＋两层连接器LoRA | 24 | 57.39 | +5.76 |

共享未见用户不等于从未用于开发的盲测：其中A06/A07和HMC s7/s9属于历史原valid，且这些个人query已经被多轮观察。

## 🔍 rank4→rank16：完整配对结果

两组均完成21/21用户、100条support、固定100步；不挑query最好点。每个数据集的完整窗口数、分类数及共享/个人划分见上表，下面将数据集放在列中，数值均为query BAcc（%）。

| 更新条件 | 2a（9人） | SEED（4人） | HMC（4人） | Nakanishi2015（4人） | 四任务等权 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两层连接器：连接器 rank4 | 51.74 | 58.69 | 69.51 | 11.01 | 47.74 |
| 两层连接器：连接器 rank16 | 52.39 | 58.08 | 71.92 | 10.17 | 48.14 |
| 两层连接器升秩变化（百分点） | +0.66 | -0.61 | +2.41 | -0.84 | +0.40 |
| Tower＋两层连接器：Tower rank4，连接器rank4 | 52.12 | 57.45 | 71.23 | 11.66 | 48.12 |
| Tower＋两层连接器：Tower rank16，连接器rank4 | 53.67 | 58.26 | 72.69 | 9.87 | 48.62 |
| Tower＋两层连接器升秩变化（百分点） | +1.54 | +0.81 | +1.46 | -1.79 | +0.51 |

### 拟合与负迁移

| 条件 | 参数量 | 2a support Acc：r4→r16 | SEED support Acc：r4→r16 | HMC support Acc：r4→r16 | SSVEP support Acc：r4→r16 | query下降人数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 两层连接器升秩 | 249,856 | 58.22→65.67 | 81.75→86.75 | 75.00→83.25 | 10.50→10.00 | 8/21 |
| Tower升秩，连接器r4 | 1,664,000 | 61.89→67.44 | 84.50→86.50 | 78.25→83.75 | 10.25→10.25 | 8/21 |

连接器升秩：2a +0.66、SEED −0.61、HMC +2.41、SSVEP −0.84个百分点。Tower升秩：2a +1.54、SEED +0.81、HMC +1.46、SSVEP −1.79个百分点。support拟合增强并不保证query改善；SSVEP仍接近12分类随机水平（8.33%），此处不能声称已找到瓶颈原因。

精确未舍入均值和每个评估点见[rank16配对报告](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/report.md)、[汇总JSON](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/summary.json)。原始逐用户文件全部保留；此前12人阶段性快照在Git历史中可追溯。

## ⚠️ 当前结论与未完成问题

1. 位置：100条时，rank4联合Tower组四任务平均48.12%，两层连接器47.74%，第二层单独47.91%；增加位置的平均收益有限，不能说参数更多必然更好。联合组在HMC更好，但在SEED低于连接器组。
2. 支持量：本次固定100步下，24条的总体表现低于48/100条；48与100相近且随任务变化。样本量和等效epoch同时变化，未比较超过100条，不能宣布100是全局最优。
3. 升秩：目前连接器rank16更擅长拟合support，但query收益不一致。支持拟合增强而query变差，是过拟合或负迁移的线索，不是已证明的唯一原因。尚不能断言rank4容量不足是主要瓶颈。
4. Tower升秩：完整平均+0.51个百分点，21人中8人下降；Tower LoRA不等于全参Tower。
5. 暴露与可靠性：2a A01–A05和Nakanishi query处于历史共享train池；SEED用户已见但query原始trial未进入共享train；HMC是同夜时间段迁移而非跨夜。单seed结果仅支持探索性判断，没有显著性或稳定性保证。

rank对照已完成，不再追加重复训练。下一轮用户指定只更新两层linear，比较全参与LoRA、一阶与二阶MAML、5/20/100步内循环；Tower/JEV冻结。新实验使用独立worktree，见文末入口，不把末层LoRA旧MAML结果当作两层连接器的新对照。

## ▶️ 已启动与检查

2026-11-25 19:27 UTC启动四卡队列，代码提交`3833925`。控制器PID为2467824；四个worker PID为2468775–2468778。12小时上限至2026-11-26 07:27 UTC，所有条件提前完成则直接退出；按当前速度估计约10–12小时，非完成承诺。

截至19:30 UTC，四个worker均已完成至少5步真实梯度更新，正在逐次保存完整query评估。首批为A06/A07/A09的Tower-LoRA＋两层连接器，以及A08的第一层连接器对照；后续按固定计划覆盖每人全部条件。当前每步约7–8秒，显存约50GiB/卡。此时的单用户早期点不能作为方法均值。

36项单元测试通过。四卡的真实模型检查均通过：零增量恒等、有效更新、原权重及buffer不变、精确保存重载；Tower联合组确实更新Tower LoRA，第一层连接器组未更新Tower。历史baseline的逐样本预测亦通过一致性检查。完整检查记录在各`jobs/worker*/smoke_*.json`。

## 🔄 追加rank16对照

同日用户批准新增42次个人适配：21人分别运行两层连接器rank16，以及Tower rank16＋连接器rank4；均为同一100条support、100步。当时原rank4训练继续，不修改活跃配置。追加条件安排在原队列结束并释放GPU后自动接续，原估计另需3–4小时，新增训练上限4小时。

19:42 UTC排队时新增9项测试通过：21人及所有样本/批次配对、仅指定rank改变、配置在worker导入前绑定、完整配对后才计算均值、失败前序不启动，以及真实LEAF Tower的98个rank16目标共1,601,536参数均有有限梯度，原参数没有梯度。当时尚未执行完整JEV GPU检查；后续connector16实际反向检查已通过，见下面八卡记录。Tower16整条训练链尚待该阶段启动，不把CPU检查当成完整GPU训练结果。

全套45项测试已通过。追加代码提交`6a6ee06`；19:42 UTC排队控制器PID2870072已启动，状态`waiting_for_rank4`，未占用GPU。两个追加计划共42条件已落盘，追加与原队列的源hash检查均通过。原队列首批已有条件完成100步、进入下一条件；已完成2a条件的纯训练约12分钟，加评估后约14分钟。当前估计原队列还需约10–12小时，追加两组约3–4小时；不是完成时间保证。

- [追加队列状态](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/queue.json)
- [rank4/rank16配对报告](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/report.md)
- [追加实验配置](../configs/personal_rank_followup.json)

### 2026-11-26迁移到八卡

新节点8×H100 80GB，旧进程退出、共享锁释放。rank4已完成168/168；rank16的connector16完成8/21，tower16尚未开始。剩余34条件只重新分配物理GPU，4个旧断点为A05 step90、SEED s1 step90、s2 step80、s3 step90，恢复原权重与优化器，不从零重跑。

49项测试通过。迁移清单、旧队列快照及逐条件执行来源单独记录，旧计划和checkpoint不改写；已完成文件保存摘要用于结束核验。原训练参数和09:56 UTC截止时间不变。

06:53 UTC新控制器PID304820启动，worker PID为305598、305602、305604、305609、305611、305615、305617、305619，物理GPU0–7各一个。调度代码提交`81a8539`。迁移前8个已完成条件的168个文件及4个中断checkpoint摘要全部核对一致。06:57 UTC八卡真实JEV反向检查全部通过：新增249,856参数有真实更新、原参数/buffer冻结、零增量恒等及保存重载一致；各worker已进入恢复训练或step0完整评估。

两层连接器rank16剩余13个条件，随后自动执行Tower rank16＋连接器rank4的21个条件。按配对rank4实测时间分配后的关键路径合计约75分钟，尚需模型加载、实际rank16速度及调度开销；启动时保守预计1.5–2小时，不是完成承诺。每卡一次独立训练一个用户，不将八卡合成更大的batch。

06:58 UTC已确认八卡均发生实际训练更新：A05及SEED s1/s3由90推进到100步，SEED s2由80推进到90步；其他四卡新条件已完成至少5步。A05最终评估和保存完成，GPU3已进入下一个HMC用户。当前每步约8–9秒，控制器与来源校验正常。

- [八卡分配及迁移审计](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/execution.json)
- [八卡真实更新检查](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/executions/eight_gpu/connector16/)

## 📊 自动结果入口

- [自动报告](../artifacts/benchmark/leaf_personal_lora_locations_20261125/report.md)
- [配对与暴露分层汇总](../artifacts/benchmark/leaf_personal_lora_locations_20261125/summary.json)
- [队列状态](../artifacts/benchmark/leaf_personal_lora_locations_20261125/queue.json)
- [精确实验计划](../artifacts/benchmark/leaf_personal_lora_locations_20261125/plan.json)
- [位置对照曲线](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/location_curves.png)
- [支持准确率和support/query loss](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/support_query_curves.png)
- [样本量曲线](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/support_size_curves.png)

自动报告仅在同一步、同样本量、全部配对方法和用户齐全时填完整任务均值。本文rank16均值现已使用完整21人；历史时间线中的中途状态只用于追溯，不代表当前未完成。

## 📚 已完成参考，不是新结果

前一轮四人固定100步：只调第二层连接器LoRA的两任务等权BAcc65.31%，加Tower全参为66.36%。2a60.42%→63.37%，HMC70.21%→69.36%，四人两升两降；全参support Acc99.75%，query loss后期升高。这仅提供“限制Tower更新是否更稳”的实验动机，不能预测本轮LoRA必然更好。

前一轮完整记录及统计诊断保存在Git `31a36d9` 与 `leaf_personal_tower_diagnostics_20261125/`，未删除。

## 🔄 MAML追加实验入口

- 前四卡：末层LoRA，FOMAML inner100／同计算量ERM，2a A08/A09和HMC s2/s6；[当前队列](../artifacts/benchmark/leaf_personal_fomaml100_20261126/queue.json)。尚未完成，不能填最终BAcc。
- 后四卡新请求：只更新两层linear，全参与LoRA；一阶／二阶MAML，inner5/20/100；目标个人适配保存0/5/10/20/50/100完整曲线，匹配终点另列。实现及可行性检查在隔离worktree `../eegjev-worktrees/personal-maml-order/`；启动状态待真实检查通过后更新。
