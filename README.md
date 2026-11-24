# LEAF Tower → JEV-27B：13 库联合训练

_当前方案：2026-11-24；使用已完成原生 LEAF 复现的同一份13库预处理数据和样本划分。_

只用 LEAF 信号 Tower，接两层 MLP 和原始 JEV-27B 的语言主干、发布版 LoRA 与 choice 决策行。训练任务文本与候选选项进入 JEV；不使用 LEAF 的 MPNet、FiLM 或 Q-Former。

## 📋 运行

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m jev_train.run --start
```

四卡同步数据并行；先做只访问train的13步短测，正式阶段重新初始化，再训练2040步。梯度阶段、完整checkpoint恢复、初始化一致性、源码/数据哈希、四卡评估分片均有检查。

- [当前协议与13库计数](docs/PROTOCOL.md)
- [运行状态与原生LEAF参照结果](docs/RESULTS.md)
- [训练配置](configs/jev13_train.json)、[任务选项](configs/jev13_tasks.json)、[评估角色](configs/jev13_evaluation.json)
- [Git归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf13_jev27b_20261124/`。直接读取 `datasets/leaf13/manifest.json` 指向的HDF5，不重复滤波、归一化、插值或重分配样本。

## ⚠️ 比较边界

保留历史JEV的2040步、全局batch16、学习率和分阶段更新规则。Tower从历史发布版instruction checkpoint初始化，不从本次LEAF第50轮或第100轮续接；没有额外对比学习阶段。只更新Tower、MLP、JEV已有LoRA和choice权重/偏置，不全参更新27B主干。

依照当前仓库JEV约定，原test用于开发评估及选点，原valid暂缓；报告开发分数，不称独立测试。原生LEAF参照运行使用valid选点、test末尾评估，且训练预算与初始化不同，不能把两者分数差归因于单一结构。

原生LEAF复现已完整归档在Git `73e26f9`；`leaf_repro/`保留为本轮同数据参照与共享数据准备工具，不再自动运行。旧的独立Qwen/对比/个人适配实验仍只在历史worktree及Git中，不在当前树恢复。
