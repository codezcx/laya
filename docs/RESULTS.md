# 个人 LoRA 元初始化：进度与结果

_2026-11-25；四卡机制pilot，尚无正式新分数。_

## 📋 固定运行范围

2a：源A01–A05，目标A08/A09；HMC：8名源用户，目标s2/s6。每个任务两组离线方法，各60episode；之后100条个人support适配100step。每组同时重新运行普通个人LoRA基线。完整协议见[PROTOCOL](PROTOCOL.md)。

- [固定样本及batch清单](../artifacts/benchmark/leaf_personal_meta_init_20261125/plan.json)
- [队列与PID](../artifacts/benchmark/leaf_personal_meta_init_20261125/queue.json)
- [实时按任务汇总](../artifacts/benchmark/leaf_personal_meta_init_20261125/summary.json)
- [GPU0：2a元学习日志](../artifacts/benchmark/leaf_personal_meta_init_20261125/gpu0.log)
- [GPU1：2a普通源训练日志](../artifacts/benchmark/leaf_personal_meta_init_20261125/gpu1.log)
- [GPU2：HMC元学习日志](../artifacts/benchmark/leaf_personal_meta_init_20261125/gpu2.log)
- [GPU3：HMC普通源训练日志](../artifacts/benchmark/leaf_personal_meta_init_20261125/gpu3.log)

28项单元测试通过，包括一阶meta梯度数值、原train隔离、同人配对、60×6批次预算、历史预处理逐值一致。真实模型检查与正式进度另以 `preflight/`、`jobs/*/smoke.json`、`progress_gpu*.json` 为准，不把smoke loss当正式成绩。

2a与HMC的真实27B预检查均通过：零LoRA恒等、临时更新隔离、source-query梯度更新共同初始化、共享权重不变、精确重载。四卡运行会各自再次做检查，然后开始正式离线训练。

## 📊 同人历史参照

| 目标人群 | 支持条数/人 | 完整query条数 | 未适配BAcc | 旧普通个人LoRA BAcc |
| --- | ---: | ---: | ---: | ---: |
| 2a A08/A09 | 100 | 288/288 | 50.87 | 59.72 |
| HMC s2/s6 | 100 | 340/344 | 57.80 | 69.36 |

来源：[前轮42份完整结果](../artifacts/benchmark/leaf_personal_multitask_20261125/summary.json)。完整前轮实现及报告在Git `9a96ae0`；本轮不复制旧分数冒充新计算。

## ⚠️ 结果解释

按相同用户比较普通个人LoRA、source ERM初始化、FOMAML初始化后的完整query BAcc。共享模型全部冻结，仅40,960个人参数更新；原test是开发探索集合，非独立盲测。元学习是否有效，要看是否优于等数据和计算量的ERM，不只看是否优于未适配。
