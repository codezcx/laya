# LEAF Tower 文本对齐 → MLP → Qwen-JEV

当前保留的方法：先用多正例 InfoNCE 将纯 LEAF Tower 与固定类别文本向量对齐，再只迁移 Tower，保留全部 token 经 MLP 输入 Qwen-JEV。EEG 路径不用 FiLM 或 LEAF Q-Former；对比阶段的 MPNet 只提供冻结文本向量，不接收 EEG，也不进入下游 JEV。Sigmoid 是已完成的对比预训练参考，其后续 JEV 实验已按用户要求取消。

## 📋 当前实验：冻结服务器的个人少样本校准

在已完成的 InfoNCE→LEAF Tower→MLP/JEV 上，新增配对 B/C 控制：B 仅更新输出端残差 adapter（33,088 参数），C 仅更新 Tower 最后两层双分支 Q/V LoRA-r8（32,768 参数）。JEV、连接器、固定文本向量及本地文本投影头全部冻结。只用个人 support 标签做多正例 InfoNCE，不经 JEV 反传。

- 首轮只测运动想象：2a 每类 5/10/20 trial、T session 校准→E session 评估；PhysioMI 每类 5/10、前四个想象 run 校准→后两个评估。
- 每个配置 3 个校准采样种子，保留 0-shot；23 个留出被试、288 次适配。GPU 0/1 跑 B，GPU 2/3 跑 C，各 72 次，独立进程而非 DDP。
- B/C 均从最终 JEV 阶段 best step 1360 的同一 Tower 出发；只用原训练被试重拟合公共文本投影头，再冻结。不是把第一阶段 51.39% 的 encoder 和另一阶段 JEV 混接。
- 固定 100 次更新、batch 16、LR 1e-4；query 不参与微调、停止或超参数选择。逐 trial 保存预测，分别报告直接文本与冻结 JEV 的 Acc/BAcc、负迁移和耗时。
- 新运行目录 `artifacts/benchmark/leaf_personalization_20261123/`，先 prepare→四卡验证被试短测→四卡正式 episode→配对审计→`summary.json`/`summary.md`。以 `pipeline.json` 为准；未完成不填成绩。

## 📋 已完成的共享模型

- InfoNCE 下游四卡训练和完整验证已完成，best step 1360，九库等权 BAcc 47.83%、Acc 50.01%。全局 batch 16，每卡 4。
- 2026-11-23 用户选择保留 InfoNCE：Sigmoid→JEV 短测在 step 7 保存并停止，正式训练未启动，队列已关闭。所有权重/日志保留，四张卡已释放。
- 沿用 17 库训练、九库验证与固定划分，2b 不训练；SEED 被试内 trial 9/3/3，其他八库跨被试。
- 2040 步；前 340 步仅连接器，之后 LEAF Tower 全参＋JEV 已有 LoRA＋choice head；JEV base 冻结。
- encoder LR 1e-6 不变，其他学习率和训练预算不因四卡放大。
- 无新增对比学习基线已完成，完整验证 BAcc 43.41%。新运行见 `artifacts/benchmark/leaf_tower_contrastive_20261123/pipeline.json`，不把 smoke 当正式成绩。
- Tower 实测 19,557,632 参数（19.56M），不含连接器或 JEV；两阶段配置分别记录，新增预训练带来额外计算预算。
- 结束评估最佳 checkpoint 的完整验证集；测试集不自动运行。

## 🔧 使用

当前个人校准入口（只启动一次；意外中断可显式加 `--resume`，已完成或取消会拒绝重启）：

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_personalization.py --start
```

个人校准先检查原划分与独立 trial 身份，source-only 拟合投影头，验证被试上做 B/C 三步短测，核对冻结、梯度、保存重载及配对批次后才正式执行。每个 episode 从同一父 checkpoint 重置；恢复跳过已完成 episode，未完成的重跑，不累积其他用户的更新。旧 `run_contrastive.py` 队列仍被 `cancelled.json` 关闭，不重启 Sigmoid。

## 📚 文档与边界

- [模型结构、四卡语义、数据与训练协议](docs/PROTOCOL.md)
- [当前状态与必要历史结果](docs/RESULTS.md)
- [历史实现及产物归档](docs/ARCHIVE.md)

源码与配置运行期有哈希保护。旧 VideoMAE 方法通过 Git `6247e46` 的隔离 worktree 复现，纯 Tower 无对比基线见 `ef07d2c`。当前仅维护 Tower 文本对齐及其必要无对比控制，数据、权重、日志、环境和第三方目录均保留。

这仍是有监督适配试验，不是严格跨数据集 zero-shot；TUAB/TUEV 候选身份冲突、预训练重叠等风险尚未消除。用户已有 ECG、专利、图件、编辑器文件未作为本轮清理对象。
