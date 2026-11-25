# LEAF Tower → JEV：个人 LoRA 元初始化

_2026-11-25；当前只有 first-order MAML 初始化方案及必要的普通训练、普通个人 LoRA 对照。_

## 📋 当前实验

共享 LEAF Tower、原连接器、JEV27B及其原有LoRA/决策头全部冻结。只训练连接器最后一层上的个人LoRA：rank4、40,960参数。先从源用户学习易于适配的初始化，再给目标用户用自己的100条支持样本训练100步。

| GPU | 离线方法 | 任务 | 目标用户 |
| --- | --- | --- | --- |
| 0 | FOMAML | 2a | A08、A09 |
| 1 | 等计算量 source ERM | 2a | A08、A09 |
| 2 | FOMAML | HMC | s2、s6 |
| 3 | 等计算量 source ERM | HMC | s2、s6 |

每个worker另重跑普通个人LoRA基线；重复基线应一致。每个数据集分别学习初始化，不是把两个任务混成一次元训练。60个源用户episode，5个support梯度批次＋1个query梯度批次，共360批次；源query也是原train，不是目标测试集。个人参数lr1e-4、AdamW、100步、batch16/micro4保持不变。

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.plan
laya/.venv-eeg/bin/python -m personal.run --start
```

- [协议与隔离规则](docs/PROTOCOL.md)
- [进度与结果](docs/RESULTS.md)
- [方法选择依据](docs/PERSONALIZATION_METHOD_REVIEW.md)
- [当前配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf_personal_meta_init_20261125/`。四卡各常驻一个冻结服务器模型；完成即退，6小时上限，checkpoint支持同代码同预算恢复。不自动扩大用户数或扫seed。

## ⚠️ 解释边界

这是两任务、每任务两名用户的机制pilot。目标用户都是原test，被历史观察过；原valid本轮不用。只用原train学习公共初始化；目标用户的100条support用于其个人梯度，独立query仅在适配前后完整评估，不选点。不能称盲测、zero-shot或已证明普遍有效。

前一轮42份LoRA/ICL结果及研究评审已归档Git `9a96ae0`；当前移除ICL执行分支和无关任务专属配置，所有原数据、权重、环境、日志及用户未提交文件保留。共享 `jev_train/` 与 `leaf_repro/` 仅保留依赖和可复现参照。

实验设计技能用于同人配对与计算预算，资源技能用于四卡实测，模型/文档技能用于冻结梯度检查及协议记录；技能引用见协议。
