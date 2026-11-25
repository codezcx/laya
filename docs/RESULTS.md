# 个人 LoRA 位置与样本量：进度与结果

_2026-11-25；本轮168个个人条件单独记录，不拿旧全参结果冒充LoRA结果。_

---

## 📋 本轮范围

21名用户：2a9、HMC4、SEED4、Nakanishi2015四人。100条support时比较first-only、last-only、both、Tower-LoRA＋both；后两组再比较48/24条。每条件独立初始化100步，精确成员见[协议](PROTOCOL.md)。

当前尚无完整新结果。独立Tower检查已确认98个LoRA目标、400,384新增参数均有有限梯度，原参数无梯度；零增量推理与原Tower一致。连接器参数数目21,504／40,960／62,464；联合Tower组462,848。

## ▶️ 已启动与检查

2026-11-25 19:27 UTC启动四卡队列，代码提交`3833925`。控制器PID为2467824；四个worker PID为2468775–2468778。12小时上限至2026-11-26 07:27 UTC，所有条件提前完成则直接退出；按当前速度估计约10–12小时，非完成承诺。

截至19:30 UTC，四个worker均已完成至少5步真实梯度更新，正在逐次保存完整query评估。首批为A06/A07/A09的Tower-LoRA＋两层连接器，以及A08的第一层连接器对照；后续按固定计划覆盖每人全部条件。当前每步约7–8秒，显存约50GiB/卡。此时的单用户早期点不能作为方法均值。

36项单元测试通过。四卡的真实模型检查均通过：零增量恒等、有效更新、原权重及buffer不变、精确保存重载；Tower联合组确实更新Tower LoRA，第一层连接器组未更新Tower。历史baseline的逐样本预测亦通过一致性检查。完整检查记录在各`jobs/worker*/smoke_*.json`。

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
