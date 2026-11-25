# 100条个人适配：四人九方法最终结果

_2026-11-25；旧InfoNCE＋LEAF Tower→JEV step1360；本轮已完成，作为扩展实验的固定参照。_

## 📋 完成状态

`artifacts/benchmark/leaf_infonce_personal_2a100_20261125/`于09:21 UTC正常完成，4人×9方法共36份结果，四卡已释放，总用时约104分钟。所有结果的parent、代码、support/query清单已核对。每人100条T场次support、完整288条E场次query；固定100步，不使用query梯度或选点。

## 📊 最终BAcc

| 方法 | 原valid A06/A07 | 原test A08/A09 | 四人平均 |
| --- | ---: | ---: | ---: |
| 不适配 | 35.24 | 50.87 | 43.06 |
| 连接器LoRA | 40.97 | 59.72 | 50.35 |
| 正确标签ICL | 38.02 | 62.33 | 50.17 |
| 残差adapter＋蒸馏 | 40.97 | 57.12 | 49.05 |
| Soft prompt＋残差adapter | 41.15 | 56.94 | 49.05 |
| 残差adapter | 40.80 | 56.94 | 48.87 |
| Soft prompt | 37.50 | 55.56 | 46.53 |
| Soft prompt＋蒸馏 | 36.81 | 55.73 | 46.27 |
| KV prefix | 35.24 | 50.87 | 43.06 |
| 打乱标签ICL | 31.08 | 26.74 | 28.91 |

连接器LoRA与ICL均值仅相差0.17个百分点，不能据4人单seed推断稳定优势。原valid/test均有历史暴露，报告开发性结果。summary中soft_prompt_kd的A06差值为浮点舍入误差−5.55e−15个百分点，应解释为持平，不是真实负迁移。

- [完整汇总](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/summary.json)
- [固定清单](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/plan.json)
- [真实模型检查](../artifacts/benchmark/leaf_infonce_personal_2a100_20261125/smoke.json)
- [当前协议](PROTOCOL.md)

## ⚠️ 历史边界

旧A08的20条residual实验为53.82→64.93%，但使用类别均衡batch和固定选项；此次100条使用随机抽样及选项置换，不是纯样本数消融。之前13库parent的57.14%也不是本轮个人适配基线。

初次启动因推理batch=4而未复现历史bf16 logits，保护检查停止；恢复历史eval_batch=8后A08/A09全部576条logits逐值一致。失败启动及诊断保留于`leaf_infonce_personal_2a100_20261125_startup_audit/`，无状态继承。

13库58条队列已停止并归档Git `9955da9`；更早245份13库个人结果及全部数据、checkpoint、环境、日志均保留。用户新要求只扩展连接器LoRA与ICL到2a全体及其他任务，旧九方法源码以Git归档。
