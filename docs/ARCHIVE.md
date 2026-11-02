# Git 与实验产物归档

当前维护文本对齐 VideoMAE→JEV 的 LoRA-r64 固定学习率控制，见 [协议](PROTOCOL.md)。上游 InfoNCE／Sigmoid 对齐组件保留；五组迁移、旧 LEAF 适配及发布 BSR 直接接 JEV 的历史配置由 Git 归档恢复。

| 提交 | 可恢复的历史状态 |
| --- | --- |
| `828a17d` | 五组对齐 encoder→JEV 迁移、四卡节点恢复入口；LoRA-r64 修改前代码与配置 |
| `704ad26` | 本轮替换前快照：LEAF 冻结／BSR 解冻适配、两组对比学习、历史结果表、直到 step 3656 的训练曲线工具 |
| `741fe34` | 完整 LEAF 17 库训练、续训、完整训练集评估、结果表及研究历程；本轮替换前归档 |
| `14c60cf` | 旧 BSR 2a 对照与用户取消的 2b 队列；此次仅在隔离 worktree 检查其 loader／resampler，不复用下游权重 |
| `3033ac9` | 历史 official Cross LEAF→JEV 和独立 query 模块 |
| `999380b` | 旧多任务组合与自定义 fold-0 评估 |
| `5200248`、`521f865` | 历史输入预处理及 BN 控制 |
| `cad9178` | 更早完整实验总账 |

历史实现、配置和独立方案文档通过 Git 恢复；只在隔离 worktree 查看或复跑，不能覆盖当前 live job 的源码。旧 `EXPERIMENT_HISTORY.md` 五问前言和研究历程完整保存在 `741fe34`，当前方法的五问定位已重写进 PROTOCOL。

此前移除了已在 Git 归档的独立 LEAF 训练入口、instruction 缓存生成器及旧配置。2026-11-01 曾引入最小冻结 LEAF 适配器；2026-11-02 替换为对齐 VideoMAE 五组后移除此旧适配器及非当前训练分支，可由 `704ad26` 恢复。共享数据切窗、冻结划分、QC、JEV、检查点继续使用。所有 ignored 数据、模型、日志、环境、第三方目录均未删除。

主要历史产物仍在原位置：

- `artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/`：首阶段完整训练／验证／测试、训练集评估、图件。
- `artifacts/benchmark/leaf_jev_multitask_continue_20261030/`：未完成的 4080 步续训日志及检查点；切换前日志最后为 3446。
- `artifacts/benchmark/multitask_splits_20261029_final/`、`datasets/multitask_20261029/`、`artifacts/benchmark/multitask_signal_quality_20261029/`：原 HMC109 数据版；仅作为历史来源，当前未变信号可只读复用。
- `artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/`：完整 HMC151、SEED 跨被试的四读出结果与表征分析。
- `artifacts/benchmark/encoder_adaptation_seedpaper_20261101/`：已完成的 LEAF 冻结／发布 BSR 解冻 JEV 适配，两组完整验证。
- `artifacts/benchmark/aligned_videomae_jev_20261102/`：五组已完成训练和完整验证；其中 InfoNCE＋Q-Former 全参组是本轮 LoRA-r64 的固定控制。
- `artifacts/benchmark/contrastive_videomae_20261101/`：本轮五组的父权重来源；InfoNCE／Sigmoid 的 best 均为 step 1700，第二阶段校验相同划分后仅迁移 encoder。
- `artifacts/benchmark/multitask_splits_seedpaper_20261101/`、`datasets/multitask_seedpaper_20261101/`：当前 SEED 论文 trial 协议，其他 16 库沿用 HMC151 版，不用旧下游权重 warm start。
- `artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028/`、`official_cross_leaf_jev_20261027*/`、`full_eval_fold0_20261027_checked/` 等：更早实验，不应混作当前结果。
- 2026-10-25 归档目录位于 `/inspire/hdd/global_user/zhangchenxi-253108310322/eegjev_archive/20261025_pre_bn_control/`，`index.json` 记录原路径映射；副本为 `artifacts/archive_index.json`。

用户未提交的 AGENTS、gitignore、ECG、专利、图件、编辑器设置及其他工作不包含在本次实验提交内。此次实验提交采用用户批准的命令级 `Codex <codex@local>` 身份，不更改全局 Git 配置，不自动 push。
