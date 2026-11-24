# LEAF 原生多任务指令训练复现

_当前方案：2026-11-24，13 个现有数据集的部分复现。_

使用官方预训练 Tower、FiLM、8-query/4-layer Q-Former、冻结 MPNet 文本原型及原生交叉熵；不接入 JEV/Qwen，不沿用历史对齐后 Tower。训练集混合训练一个模型，验证与测试分开。

## 📋 运行

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m leaf_repro.run --start
```

控制器有单实例锁，按顺序完成数据准备、四卡短测、100-epoch 联合训练、完整测试。意外中断后重跑入口会跳过已完成被试，并从 Lightning 的完整优化器检查点恢复。原始数据、历史权重与日志均保留。

- [协议与数据边界](docs/PROTOCOL.md)
- [当前状态](docs/RESULTS.md)
- [运行配置](configs/leaf_reproduction.json)
- [Git 归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf_native13_20261124/`；数据与逐样本划分：`datasets/leaf13/`。当前运行的 13 库名单不会因后台新下载完成而中途改变。

## ⚙️ 与论文复现的边界

复用作者发布的无监督预训练 Tower，本轮不重跑六库预训练。原始指令数据组合为 16 库，本轮只使用已找到的 13 库；其中 Cho2017、HighGamma、OpenBMI-SSVEP 使用现有 MOABB 处理版，SSVEP 只有公开 train blocks。不能称作完整论文复现。

按用户本轮“使用他们的训练范式”的要求，恢复 LEAF 的 train/valid/test 角色；不沿用历史 Qwen 原 test 开发选点的约定。历史划分文件不变，也不把历史用过的数据称作全新盲测。

旧 Qwen/JEV 与个人适配的 58 个专属代码、配置和测试已从当前树移除，完整保存在 Git `5611b49`，需要重跑时使用隔离 worktree；没有删除 ignored 数据、模型、日志、环境或第三方安装。
