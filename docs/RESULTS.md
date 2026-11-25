# 同种子个人适配曲线：进度与结果

_2026-11-25；已启动四卡正式运行，最新进度以queue及各GPU进度文件为准。_

运行源码Git `6612d99`，控制器PID1658237；GPU0–3的worker PID为1658339、1658340、1658341、1658342。四卡真实27B检查全部通过，历史未适配预测已逐样本复现；四名用户第0步完整评估已落盘，2a两人已完成第5步评估。不是仅提交了排队命令。

## 📋 本轮固定范围

四名用户：2a A08/A09，HMC s2/s6。每人三组：普通LoRA、source ERM初始化、FOMAML初始化；每组100条support、100步。统一最初seed42，学习率1e-4、rank4、冻结encoder及全部个人采样规则不变。源初始化复用上一轮权重，不再训练。完整细节见[协议](PROTOCOL.md)。

- [运行状态](../artifacts/benchmark/leaf_personal_matched_curves_20261125/queue.json)
- [固定样本、批次及权重来源](../artifacts/benchmark/leaf_personal_matched_curves_20261125/plan.json)
- [同人配对汇总](../artifacts/benchmark/leaf_personal_matched_curves_20261125/summary.json)
- [逐用户曲线数据](../artifacts/benchmark/leaf_personal_matched_curves_20261125/curve_points.csv)

## 📊 曲线

运行中在0、5、10、20、50、100步完整评估。下面的图在有新曲线点时自动更新；只有同一步三组和两人齐全才填任务均值，否则留空，不是准确率为0。

- [完整query BAcc与相对普通LoRA的增益](../artifacts/benchmark/leaf_personal_matched_curves_20261125/figures/adaptation_curves.png)
- [完整support准确率、support loss、query loss](../artifacts/benchmark/leaf_personal_matched_curves_20261125/figures/support_query_curves.png)

30项测试通过：历史预处理逐值一致、所有参数约束、精确用户ID与batch、共同seed42参数指纹、源权重完整性、配对汇总不混用户及图件生成。正式GPU检查另见各用户目录的smoke.json。

## ⚠️ 历史结果不是本轮结果

上一轮普通LoRA／FOMAML的两任务等权BAcc为64.54%／64.67%，完整结果与实现归档Git `4a664b1`。上一轮普通LoRA按用户改变随机A；本轮统一seed，因此需等本轮重新计算后比较，不能拿旧普通LoRA分数冒充新对照。

原test个人query用于开发性诊断，不是独立测试；不据曲线挑最好step，固定100步结束。原valid本轮不用。报告各任务两人均值及两任务等权均值，不只报告某个用户的提升。
