# LEAF 信号 Tower → MLP → Qwen-JEV

当前方法：仅取 LEAF 的信号编码器，保留两个 Transformer 分支的全部 token，经 MLP 输入 Qwen-JEV。任务说明、问题与候选选项只交给 JEV；不用 MPNet、FiLM 或 LEAF Q-Former。

## 📋 当前实验

- 一个共享模型，GPU 0–3 四卡同步数据并行；全局 batch 16，每卡 4。
- 沿用 17 库训练、九库验证与固定划分，2b 不训练；SEED 被试内 trial 9/3/3，其他八库跨被试。
- 2040 步；前 340 步仅连接器，之后 LEAF Tower 全参＋JEV 已有 LoRA＋choice head；JEV base 冻结。
- encoder LR 1e-6 不变，其他学习率和训练预算不因四卡放大。
- 本轮准备/启动状态以 `artifacts/benchmark/leaf_tower_jev_20261123/pipeline.json` 为准；不把 smoke 当正式成绩。
- 结束评估最佳 checkpoint 的完整验证集；测试集不自动运行。

## 🔧 使用

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

CPU 输入检查和真实四卡 smoke 通过后自动进入正式训练；从发布 LEAF/JEV 重新初始化，不继承 smoke 或旧下游参数。已有运行不重复启动，中断恢复见协议。

## 📚 文档与边界

- [模型结构、四卡语义、数据与训练协议](docs/PROTOCOL.md)
- [当前状态与必要历史结果](docs/RESULTS.md)
- [历史实现及产物归档](docs/ARCHIVE.md)

源码与配置运行期有哈希保护。旧 VideoMAE/对比学习方法通过 Git `6247e46` 的隔离 worktree 复现，数据、权重、日志、环境和第三方目录均保留。

这仍是有监督适配试验，不是严格跨数据集 zero-shot；TUAB/TUEV 候选身份冲突、预训练重叠等风险尚未消除。用户已有 ECG、专利、图件、编辑器文件未作为本轮清理对象。
