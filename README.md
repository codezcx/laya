# LEAF Tower → 用户条件化 adapter → 冻结 Qwen-JEV

当前方法是 CNAPS-inspired amortized feature adaptation：少量带标签EEG支持样本经DeepSets上下文提取器G与条件映射H，产生用户特定FiLM调制。离线训练共享参数；新用户只计算上下文，不训练个人模型。最终分类来自JEV，不是类别文本相似度。

## 当前实验

两组配对机制验证，当前处于启动检查阶段；真实阶段以运行目录的 `pipeline.json` 为准。

| GPU | 组别 | 离线训练 | 新用户适配 | JEV |
| --- | --- | --- | --- | --- |
| 0、1 | tower_frozen | G/H＋条件化残差adapter | 只做前向 | 全冻结 |
| 2、3 | tower_full | 同上＋LEAF Tower全参 | 只做前向 | 全冻结 |

- 相同父模型InfoNCE→Tower→JEV best step1360、seed42、340次更新和episode流。
- 每步来自一名原训练被试：support每类5条，query全局16条；query CE经冻结JEV反传。每组两卡数据并行，各8条、microbatch4。
- 原训练的2a五名＋PhysioMI六十九名被试用于共享训练；固定验证为2a被试6、PhysioMI被试70。
- 最后仅2a被试8探索性评估：T session 20条support，E session全部288条query；不做全测试被试/多shot/多seed扫参。
- G/H/adapter LR1e-4，Tower全参组LR1e-6；JEV基础权重、已有语言LoRA、connector/head始终冻结。两组Tower BN/dropout保持eval。
- 新增模块为共享参数；用户缓存128维上下文，不维护个人优化器。旧个人CE实验与取消标记不改动。

## 启动与查看

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_personalization.py --start
# 仅用于未取消的意外中断，不能重复启动活跃任务：
laya/.venv-eeg/bin/python scripts/run_personalization.py --start --resume
```

输出 `artifacts/benchmark/leaf_amortized_film_20261123/`。流程为输入检查与准备 → 两组各4步真实GPU短测 → 配对/冻结/重载审计 → 两组正式并行。短测权重不进入正式实验。`train/{tower_frozen,tower_full}/progress.json` 显示进度，`history.json` 记录loss，`valid_step*.json` 记录验证，完成后写入 `summary.json`。

## 模型来源与边界

LEAF纯信号Tower有19,557,632个前向参数，四秒EEG产生16×256 tokens。没有原LEAF文本encoder/Q-Former；新增FiLM属于本轮用户条件化模块，不是重新加载原LEAF FiLM。固定MPNet类别向量只编码support标签，不做相似度分类，不给query输入真值。

父模型在17库训练、九库验证，完整验证BAcc47.83%、测试前向0。旧个人CE pilot中2a被试8的冻结LM＋个人输出adapter为64.93%，含100步个人更新，与本轮前向适配不等计算预算。该用户结果已经被观察，本轮不能称盲测或严格zero-shot。

- [当前结构、数据与实验协议](docs/PROTOCOL.md)
- [当前状态与历史参考结果](docs/RESULTS.md)
- [历史实现和产物归档](docs/ARCHIVE.md)

实验设计技能用于配对与泄漏检查，Markdown技能用于保持协议与实现一致。旧方案归档到Git `05403f7`；数据、权重、日志、环境和第三方目录均保留。用户已有ECG、专利、图件、编辑器改动不在本次提交内。
