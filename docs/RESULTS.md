# 跨任务个人适配：运行状态与固定参照

_2026-11-25；连接器LoRA与正确标签ICL；当前扩展实验已启动，尚无完整新结果。_

## 📋 当前运行

控制器PID2548033，四个worker为2548151–2548154。输出`artifacts/benchmark/leaf_personal_multitask_20261125/`。固定2a9人、SEED4人、HMC4人、SSVEP4人，共21个用户×任务、42份结果；每人100条支持样本。

- [固定样本清单及暴露标记](../artifacts/benchmark/leaf_personal_multitask_20261125/plan.json)
- [队列](../artifacts/benchmark/leaf_personal_multitask_20261125/queue.json)
- [实时配对结果](../artifacts/benchmark/leaf_personal_multitask_20261125/summary.json)
- [四任务真实模型检查](../artifacts/benchmark/leaf_personal_multitask_20261125/smoke.json)
- [协议、数据规模与划分](PROTOCOL.md)

28项单元测试通过；四库各2条输入与parent历史预处理逐值一致。真实27B检查四任务全部通过：旧共享forward逐值对照、个人LoRA非零有限梯度、实际更新、精确重载、共享参数冻结，以及100示例ICL完整前向。当前首批2a用户进入正式基线/适配流程，进度以`progress_gpu*.json`为准，不把smoke当作准确率结果。

| 任务 | 每个EEG的token数 | 100示例上下文token数 | ICL峰值分配显存GiB |
| --- | ---: | ---: | ---: |
| 2a | 16 | 3,216 | 50.89 |
| SEED | 16 | 3,016 | 50.77 |
| HMC | 120 | 13,476 | 57.42 |
| SSVEP | 18 | 3,699 | 51.26 |

上述为支持样本上的2条smoke前向显存，并非完整query耗时或最终分数。HMC完整30秒、所有100个示例均保留；未按显存缩减任务输入。

## 📊 上一轮固定参照

旧四人(A06–A09)每人100支持、288query，9方法×4人共36份结果于09:21 UTC完成，用时约104分钟。当前只保留用户选中的两方法参照；完整九方法结果和源码归档Git `8bc8e94`及原产物。

| 方法 | 原valid A06/A07 BAcc | 原test A08/A09 BAcc | 四人平均BAcc |
| --- | ---: | ---: | ---: |
| 不适配 | 35.24 | 50.87 | 43.06 |
| 连接器LoRA | 40.97 | 59.72 | 50.35 |
| 正确标签ICL | 38.02 | 62.33 | 50.17 |

来源：[旧完整汇总](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/summary.json)。两方法差0.17个百分点，不代表显著或稳定胜出。扩展实验对A06–A09保持原support/query、个人seed、更新预算；不复制旧结果冒充新完成。

## ⚠️ 解释与保留

本轮会分别报告未见被试、已见被试但query不在共享train池、query也在原train池三类；不输出混合的跨数据集“泛化平均”。2a A01–A05与SSVEP query在原train池，属于已见数据探索；SEED用户已见，query为原test独立试次；HMC和2a A06–A09未进本次共享train池，但有历史评估/发布权重暴露风险。不能称新的盲测。

旧九方法summary中soft_prompt_kd A06的−5.55e−15个百分点是舍入误差，解释为持平；当前汇总负迁移人数采用1e−9容差，未改写历史产物。

旧58条13库队列保持停止；所有原数据、权重、缓存、环境与日志保留，未删除。当前移除的是已归档的其他个人方法源码/专属测试，不影响原结果。
