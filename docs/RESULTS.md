# 100条个人适配：运行状态与结果

_2026-11-25；复用旧InfoNCE＋LEAF Tower→JEV step1360，原13库parent已停用。_

## 📋 当前运行

输出：`artifacts/benchmark/leaf_infonce_personal_2a100_20261125/`。控制器PID3630498已重启，4名worker为3630508–3630511；26项单元测试通过。A08抽查预处理信号与历史缓存逐值一致；恢复旧评估batch=8后，A08/A09全部576条历史query的JEV logits逐值复现。正式GPU更新和完整结果以实时文件为准，目前尚未报告本轮适配分数。

- [队列](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/queue.json)
- [每人support/query清单](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/plan.json)
- [控制器日志](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/controller.log)
- [实验协议](PROTOCOL.md)

四个用户A06/A07（原valid）和A08/A09（原test），每人T场次100条适配、E场次完整288条评估；四卡并行，每人独立个人模块。保持100步、batch16/micro4和lr1e-4。尚无新100条结果，不能写成已经提升。

重启后的[真实模型检查](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/smoke.json)已通过：原共享forward逐值一致、所有个人训练模块均有非零有限梯度、参数确实更新、个人权重重载逐值一致、共享模型权重保持冻结。ICL两种前向也通过有限值检查。

正式从信号重新编码后的A08/A09基线检查同样通过，576条logits最大差均为0，见各自`baseline_audit.json`。07:39:49 UTC已确认A06残差adapter进入正式step10/100，CE=2.3343、梯度范数0.8527，队列无错误；其余进度以`progress_gpu*.json`为准。

首次启动的保护检查发现，沿用近期框架的推理batch=4会因bf16批形状带来细微logit差异，A08有3/288条、A09有2/288条预测与历史不同，因此自动停止。只恢复历史推理batch=8后，两人logits最大差均为0，训练batch/学习率/步数均未改。失败启动的日志、基线和任何部分状态保留在`leaf_infonce_personal_2a100_20261125_startup_audit/`，本轮重新初始化、不继承其个人状态。详见[诊断记录](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125_startup_audit/diagnosis.json)。

## 📊 已有参照与比较限制

旧A08 residual＋冻结JEV：20条支持，完整E场次288条query，不适配BAcc53.82%，适配后64.93%，支持训练Acc70.00%。来源：[旧结果](../artifacts/benchmark/leaf_personal_ce_alltest_20261123/episodes/bcic_2a_s8_residual_frozen_k5_seed42/complete.json)。

本轮使用相同共享parent与旧输入；当前9方法训练采样/选项策略与那次20条CE有差异，因此旧64.93%是历史参照，不把变化全部归因于100条。正式主要比较本轮同人的不适配基线与适配结果，并分别报告原valid/test组。

## 📦 停止与保留

原58条队列`leaf13_personal_2a58_20261125`已按用户要求停止，四份residual完整结果保留：同人均值不适配54.96%、20条52.23%、58条54.07%；仅A01/A02/A03/A07原valid子集共230条query，与当前人群/parent不同，不能直接比较。其他模块部分checkpoint保留，不自动续跑。源码/配置见Git `9955da9`。

更早`leaf13_personal_20261124`队列同样保持停止，245份结果与26个完整用户×数据集方法组保留。未删除数据、权重、环境、日志或缓存。

13库共享JEV完成分数57.14%、原生LEAF49.59%保留各自产物和Git记录；它们是不同模型/初始化/协议下的13库均值，不作为本轮个人适配提升的分母。
