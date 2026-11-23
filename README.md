# LEAF Tower 文本对齐 → MLP → Qwen-JEV

当前方法：先用多正例 InfoNCE 或平衡 Sigmoid 将纯 LEAF Tower 与固定类别文本向量对齐，再只迁移 Tower，保留全部 token 经 MLP 输入 Qwen-JEV。EEG 路径不用 FiLM 或 LEAF Q-Former；对比阶段的 MPNet 只提供冻结文本向量，不接收 EEG，也不进入下游 JEV。

## 📋 当前实验

- 对比阶段两组分别用 GPU 0、1 并行；随后两组下游依次用 GPU 0–3 四卡同步训练。每个下游全局 batch 16，每卡 4。
- 沿用 17 库训练、九库验证与固定划分，2b 不训练；SEED 被试内 trial 9/3/3，其他八库跨被试。
- 2040 步；前 340 步仅连接器，之后 LEAF Tower 全参＋JEV 已有 LoRA＋choice head；JEV base 冻结。
- encoder LR 1e-6 不变，其他学习率和训练预算不因四卡放大。
- 无新增对比学习基线已完成，完整验证 BAcc 43.41%。新运行见 `artifacts/benchmark/leaf_tower_contrastive_20261123/pipeline.json`，不把 smoke 当正式成绩。
- Tower 实测 19,557,632 参数（19.56M），不含连接器或 JEV；两阶段配置分别记录，新增预训练带来额外计算预算。
- 结束评估最佳 checkpoint 的完整验证集；测试集不自动运行。

## 🔧 使用

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_contrastive.py --start
```

CPU 输入检查和真实短测通过后自动进入正式训练。对比阶段从发布 Tower 初始化；下游仅加载已完成对比训练的 best Tower，MLP/JEV 重新初始化。不会继承 smoke、旧 SFT 或对比投影头。中断恢复使用同一命令加 `--resume`，不自动覆盖产物。

## 📚 文档与边界

- [模型结构、四卡语义、数据与训练协议](docs/PROTOCOL.md)
- [当前状态与必要历史结果](docs/RESULTS.md)
- [历史实现及产物归档](docs/ARCHIVE.md)

源码与配置运行期有哈希保护。旧 VideoMAE 方法通过 Git `6247e46` 的隔离 worktree 复现，纯 Tower 无对比基线见 `ef07d2c`。当前仅维护 Tower 文本对齐及其必要无对比控制，数据、权重、日志、环境和第三方目录均保留。

这仍是有监督适配试验，不是严格跨数据集 zero-shot；TUAB/TUEV 候选身份冲突、预训练重叠等风险尚未消除。用户已有 ECG、专利、图件、编辑器文件未作为本轮清理对象。
