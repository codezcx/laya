# LEAF Tower 文本对齐 → MLP → Qwen-JEV

当前保留的方法：先用多正例 InfoNCE 将纯 LEAF Tower 与固定类别文本向量对齐，再只迁移 Tower，保留全部 token 经 MLP 输入 Qwen-JEV。EEG 路径不用 FiLM 或 LEAF Q-Former；对比阶段的 MPNet 只提供冻结文本向量，不接收 EEG，也不进入下游 JEV。Sigmoid 是已完成的对比预训练参考，其后续 JEV 实验已按用户要求取消。

## 📋 当前实验：JEV 分类损失与语言 LoRA 对照

上一轮本地 InfoNCE B/C 已完成，代码归档在 Git `e71c334`，结果保留于 `artifacts/benchmark/leaf_personalization_20261123/`。当前在同一父 checkpoint、同一 support/query 上检验语言侧适配是否有益。

| GPU | 信号侧 | LM 更新 | 损失 |
| --- | --- | --- | --- |
| 0 | B：输出残差 adapter，33,088 参数 | 全冻结 | JEV CE |
| 1 | 同 B | 已有语言 LoRA-r32，217,579,520 参数 | JEV CE |
| 2 | C：Tower Q/V LoRA-r8，32,768 参数 | 全冻结 | JEV CE |
| 3 | 同 C | 已有语言 LoRA-r32，217,579,520 参数 | JEV CE |

- 四组均冻结原 Tower、JEV 基础权重、connector、choice head、文本读出；只切换语言 LoRA 是否更新，不是27B全参微调。
- 沿用2a每类5/10/20 trial、T→E；PhysioMI每类5/10、前四→后两想象run，23名目标被试。
- 第一轮seed42，每组48次、总计192次；与旧InfoNCE比较须筛同seed42，不能混用三seed均值。
- 信号LR1e-4、100步、有效batch16不变；microbatch4累积4次。新增LM LR5e-6，各参数组分别clip1、AdamW weight decay0.01、恒定LR。
- loss仅为JEV CE，直接文本仅作诊断；记录support/query指标、support CE、冻结/更新与重载审计。每次从父权重重置，不累积其他用户的更新。
- 输出 `artifacts/benchmark/leaf_personalization_jev_ce_20261123/`；只读复用旧输入→四卡验证被试短测→正式四组→审计和汇总。
- 更新LM需要服务器参与反传，不是纯边缘端方案。本轮没缩小backbone，不能证明“27B太大”。

## 📋 已完成的共享模型

- InfoNCE 下游四卡训练和完整验证已完成，best step 1360，九库等权 BAcc 47.83%、Acc 50.01%。全局 batch 16，每卡 4。
- 2026-11-23 用户选择保留 InfoNCE：Sigmoid→JEV 短测在 step 7 保存并停止，正式训练未启动，队列已关闭。所有权重/日志保留；释放的四张卡现用于上方个人校准对照。
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

个人校准先检查原划分与独立 trial 身份，只读复用上一轮输入和文本头，不重新拟合；验证被试上做四组三步短测，核对冻结/更新、梯度、重载及配对批次后才正式执行。每个 episode 从父权重重置；恢复跳过已完成 episode，未完成的重跑。旧 `run_contrastive.py` 队列仍被关闭，不重启 Sigmoid。

## 📚 文档与边界

- [模型结构、四卡语义、数据与训练协议](docs/PROTOCOL.md)
- [当前状态与必要历史结果](docs/RESULTS.md)
- [历史实现及产物归档](docs/ARCHIVE.md)

源码与配置运行期有哈希保护。旧 VideoMAE 方法通过 Git `6247e46` 的隔离 worktree 复现，纯 Tower 无对比基线见 `ef07d2c`。当前仅维护 Tower 文本对齐及其必要无对比控制，数据、权重、日志、环境和第三方目录均保留。

这仍是有监督适配试验，不是严格跨数据集 zero-shot；TUAB/TUEV 候选身份冲突、预训练重叠等风险尚未消除。用户已有 ECG、专利、图件、编辑器文件未作为本轮清理对象。
