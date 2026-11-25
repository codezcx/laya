# 个人 LoRA 元初始化：完整结果

_2026-11-25；四卡任务已结束，12份唯一个人结果齐全，重复普通LoRA对照一致。_

## 📊 同人完整 query 的平均 BAcc

| 方法 | 2a A08/A09（%） | HMC s2/s6（%） | 两任务等权（%） |
| --- | ---: | ---: | ---: |
| 不适配 | 50.87 | 57.80 | 54.34 |
| 普通个人 LoRA | 59.72 | 69.36 | 64.54 |
| source ERM 初始化＋个人训练 | 59.38 | 68.07 | 63.72 |
| FOMAML 初始化＋个人训练 | 60.07 | 69.27 | 64.67 |

每人100条支持样本、100步个人训练。2a每人完整query288条，HMC为340/344条。完整数据分配及算法见[协议](PROTOCOL.md)。来源：[完整汇总](../artifacts/benchmark/leaf_personal_meta_init_20261125/summary.json)、[队列完成记录](../artifacts/benchmark/leaf_personal_meta_init_20261125/queue.json)、各 jobs 下 result.json。

## 🔍 解释与下一步

FOMAML相对普通个人LoRA的两任务等权改善仅0.13个百分点，相对source ERM改善0.95个百分点；4名用户、1个种子，不能据此宣布稳定收益。原test个人query是开发探索集合，非独立盲测。

排查确认权重更新与断点恢复有效、共享模型保持冻结，没有证据表明元梯度失效。尚有两个需要控制的问题：离线只优化5步内适配、最终比较100步，可能掩盖早期差异；普通个人LoRA的A用42＋用户order×101初始化，而两组源训练初始化都从seed42起步。下一轮应统一起始seed，并在相同步数上比较同人完整query曲线，不同时改变学习率、rank或encoder。

原始权重、每步训练loss、最终预测、样本清单与运行日志全部保留：
`artifacts/benchmark/leaf_personal_meta_init_20261125/`。本轮未保存中途个人query曲线，不能从最后一次结果倒推出早期表现。
