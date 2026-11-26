# 个人 LoRA 位置与样本量：进度与结果

_2026-11-25；本轮168个个人条件单独记录，不拿旧全参结果冒充LoRA结果。_

---

## 📋 本轮范围

21名用户：2a9、HMC4、SEED4、Nakanishi2015四人。100条support时比较first-only、last-only、both、Tower-LoRA＋both；后两组再比较48/24条。每条件独立初始化100步，精确成员见[协议](PROTOCOL.md)。

截至2026-11-26迁移前，原rank4的168条件已全部完成；rank16完成8/42，剩余34条件待续跑。完整结果见下方自动报告；早期启动记录保留时间标记，不代表当前仍在等待。独立Tower检查已确认98个LoRA目标、400,384新增参数均有有限梯度，原参数无梯度；零增量推理与原Tower一致。连接器参数数目21,504／40,960／62,464；联合Tower组462,848。

## ▶️ 已启动与检查

2026-11-25 19:27 UTC启动四卡队列，代码提交`3833925`。控制器PID为2467824；四个worker PID为2468775–2468778。12小时上限至2026-11-26 07:27 UTC，所有条件提前完成则直接退出；按当前速度估计约10–12小时，非完成承诺。

截至19:30 UTC，四个worker均已完成至少5步真实梯度更新，正在逐次保存完整query评估。首批为A06/A07/A09的Tower-LoRA＋两层连接器，以及A08的第一层连接器对照；后续按固定计划覆盖每人全部条件。当前每步约7–8秒，显存约50GiB/卡。此时的单用户早期点不能作为方法均值。

36项单元测试通过。四卡的真实模型检查均通过：零增量恒等、有效更新、原权重及buffer不变、精确保存重载；Tower联合组确实更新Tower LoRA，第一层连接器组未更新Tower。历史baseline的逐样本预测亦通过一致性检查。完整检查记录在各`jobs/worker*/smoke_*.json`。

## 🔄 追加rank16对照

同日用户批准新增42次个人适配：21人分别运行两层连接器rank16，以及Tower rank16＋连接器rank4；均为同一100条support、100步。当时原rank4训练继续，不修改活跃配置。追加条件安排在原队列结束并释放GPU后自动接续，原估计另需3–4小时，新增训练上限4小时。

新增9项测试通过：21人及所有样本/批次配对、仅指定rank改变、配置在worker导入前绑定、完整配对后才计算均值、失败前序不启动，以及真实LEAF Tower的98个rank16目标共1,601,536参数均有有限梯度，原参数没有梯度。实际27B JEV的rank16反向检查在每卡启动训练时由现有worker自动执行，尚未执行，不把CPU检查当成完整GPU训练结果。

全套45项测试已通过。追加代码提交`6a6ee06`；19:42 UTC排队控制器PID2870072已启动，状态`waiting_for_rank4`，未占用GPU。两个追加计划共42条件已落盘，追加与原队列的源hash检查均通过。原队列首批已有条件完成100步、进入下一条件；已完成2a条件的纯训练约12分钟，加评估后约14分钟。当前估计原队列还需约10–12小时，追加两组约3–4小时；不是完成时间保证。

- [追加队列状态](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/queue.json)
- [rank4/rank16配对报告](../artifacts/benchmark/leaf_personal_lora_rank_followup_20261125/report.md)
- [追加实验配置](../configs/personal_rank_followup.json)

### 2026-11-26迁移到八卡

新节点8×H100 80GB，旧进程退出、共享锁释放。rank4已完成168/168；rank16的connector16完成8/21，tower16尚未开始。剩余34条件只重新分配物理GPU，4个旧断点为A05 step90、SEED s1 step90、s2 step80、s3 step90，恢复原权重与优化器，不从零重跑。

49项测试通过。迁移清单、旧队列快照及逐条件执行来源单独记录，旧计划和checkpoint不改写；已完成文件保存摘要用于结束核验。原训练参数和09:56 UTC截止时间不变。此处先记录迁移准备，实际8卡启动及首批更新在确认后补记。

## 📊 自动结果入口

- [自动报告](../artifacts/benchmark/leaf_personal_lora_locations_20261125/report.md)
- [配对与暴露分层汇总](../artifacts/benchmark/leaf_personal_lora_locations_20261125/summary.json)
- [队列状态](../artifacts/benchmark/leaf_personal_lora_locations_20261125/queue.json)
- [精确实验计划](../artifacts/benchmark/leaf_personal_lora_locations_20261125/plan.json)
- [位置对照曲线](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/location_curves.png)
- [支持准确率和support/query loss](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/support_query_curves.png)
- [样本量曲线](../artifacts/benchmark/leaf_personal_lora_locations_20261125/figures/support_size_curves.png)

仅同一步、同样本量、全部配对方法和用户齐全才填均值。原始逐用户数据随结果产生立即保存；不将未完成均值或单个用户临时分数作最终结论。

## 📚 已完成参考，不是新结果

前一轮四人固定100步：只调第二层连接器LoRA的两任务等权BAcc65.31%，加Tower全参为66.36%。2a60.42%→63.37%，HMC70.21%→69.36%，四人两升两降；全参support Acc99.75%，query loss后期升高。这仅提供“限制Tower更新是否更稳”的实验动机，不能预测本轮LoRA必然更好。

前一轮完整记录及统计诊断保存在Git `31a36d9` 与 `leaf_personal_tower_diagnostics_20261125/`，未删除。
