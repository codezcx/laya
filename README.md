# LEAF Tower → 个人残差 adapter → 冻结 Qwen-JEV

当前扩展已验证的个人监督适配：每位用户独立从同一共享checkpoint开始，只更新个人输出残差adapter，最终预测由JEV输出。不是用户条件化共享训练，也不以文本相似度分类。

## 📋 当前实验

夜间另已排队三个同方案方法对照：AdaBN＋个人CE、Tower内部SSF、个人CE＋MEMO式测试时适配；不执行shot/学习率扫描。代码在隔离worktree `../eegjev-worktrees/personal-method-controls`（分支 `experiment/personal-method-controls`，实现 `460d31e`），不修改本目录运行中的代码/config。119项自动测试通过；待下述23人基线完成、GPU释放后，逐方法先GPU短测，再四卡跑同一23人。截止北京时间2026-11-24 06:50:52（UTC前日22:50:52）。[夜间队列状态](artifacts/benchmark/leaf_personal_methods_20261123/pipeline.json)；结果将在同目录 `summary.md/json` 自动汇总。MEMO/AdaBN是TTA，不称原始联合自监督TTT复现。

原测试被试共23人：BCIC-IV-2a被试8、9；PhysioMI被试89–109。每类5条（每人20条）支持样本，seed42，100步；本轮不做多shot、多seed或TTT。每位用户的adapter、优化器均重置，用户之间不累积更新。

2026-11-23 16:41:33 UTC四卡正式任务已启动；105项自动测试和两名验证用户GPU短测通过。当前状态与进度见[结果记录](docs/RESULTS.md)及运行目录，不将未完成用户记为结果。

| 模块 | 参数状态 | 作用 |
| --- | --- | --- |
| LEAF Tower，19,557,632参数 | 冻结 | 产生16×256 EEG tokens |
| 个人残差adapter，33,088参数 | 更新 | 256→64→256，GELU，残差零初始化 |
| 原MLP connector，27,535,872参数 | 冻结 | 映射到JEV输入空间 |
| JEV，包括已有语言LoRA和choice head | 全冻结 | 给出分类结果；支持集CE梯度穿过它 |

LR1e-4、AdamW、weight decay0.01、梯度裁剪1、有效batch16=microbatch4×4，全部沿用旧个人CE pilot。父模型仍为InfoNCE→Tower→JEV best step1360，不从后来G/H＋FiLM实验迁移权重。

## ⚙️ 启动与进度

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_personalization.py --start
# 意外中断后，仅重启未完成用户；完整用户跳过，不恢复用户中途优化器：
laya/.venv-eeg/bin/python scripts/run_personalization.py --start --resume
```

输出目录：`artifacts/benchmark/leaf_personal_ce_alltest_20261123/`。
流程：输入/父权重校验 → 两名原验证被试各3步GPU短测 → 冻结/重载/批次一致性审计 → 四卡独立并行23名目标用户（6/6/6/5人）→ 完整配对汇总。短测权重不进入正式实验。

`pipeline.json`显示阶段；`worker_0.json`至`worker_3.json`显示进度；
`episodes/*/support_history.json`记录支持集loss；
每名用户完成后保存`adapter.pt`、`predictions.pt`、`complete.json`；
全部完成才生成`summary.json`和`summary.md`。

## 🔍 解释边界

2a采用T session支持、E session完整测试；PhysioMI采用R04/R06/R08/R10支持、R12/R14完整测试。原训练/验证用户不混入主结果，未选中的支持池样本不训练、不加入测试。报告每用户适配前后BAcc、每库被试等权均值、两库等权均值及负迁移人数。

旧2a被试8的64.93%只是一名用户；此次重跑该人并扩展其余22人，不预设均值会提升。原共享模型见过的用户不能称新用户；LEAF上游预训练身份重叠尚未完全排除。本轮是带标签个人适配，不是zero-shot或原始完整benchmark测试。

- [当前协议](docs/PROTOCOL.md)
- [结果与历史对照](docs/RESULTS.md)
- [Git归档](docs/ARCHIVE.md)

上一轮G/H＋FiLM实现归档到Git `36b33d0`；本轮从隔离worktree检查旧个人CE实现 `05403f7`，只恢复所需残差分支。忽略的数据、权重、日志、环境、第三方安装和原取消标记均保留。用户现有ECG、专利、图件等改动不在提交范围。
