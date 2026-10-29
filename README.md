# EEG → 文本条件化 LEAF → JEV

当前实验：9 个 benchmark + 8 个 train-only 数据集联合微调一个 LEAF→JEV 模型，使用冻结的跨被试划分，2b 排除。用户已同意保留 HMC 本地子集及 TUAB/TUEV 候选身份重合，先做验证性实验；不将本轮成绩宣称为严格无泄漏 benchmark 或跨数据集 zero-shot。

## 📋 当前状态

- 新训练入口、数据转换和文本接口已实现；真实 train-only 前向／反向检查覆盖 9 个 benchmark，LEAF 梯度有效，单卡峰值约 53 GiB。
- 全量信号转换进行中。正式更新数、评估结果以运行目录为准，不把预处理或组件检查写成已完成训练。
- 固定预算 2040 次更新，各数据集均衡轮转；按固定验证子集选模，结束后跑 9 库完整验证／测试。
- 只初始化发布 LEAF 和原生 JEV adapter，不复用见过本轮测试被试的旧 2a 微调权重。
- VideoMAE 对照尚未启动；旧 BSR 代码已从工作树移除，Git `14c60cf` 可恢复，数据、权重和日志保留。

## 📚 入口和记录

- [当前协议、划分与风险](docs/PROTOCOL.md)
- [当前结果与历史对照](docs/RESULTS.md)
- [训练配置](configs/benchmark_train.json) · [17 个任务及选项](configs/benchmark_tasks.json)
- [固定划分审计](artifacts/benchmark/multitask_splits_20261029_final/summary.json)
- [真实模型组件检查](artifacts/benchmark/leaf_jev_multitask_preflight_20261029/complete.json)
- [运行状态](artifacts/benchmark/leaf_jev_multitask_pilot_20261029/pipeline.json)
- [Git 和产物归档](docs/ARCHIVE.md)

## 🔧 启动

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --prepared
```

该命令只在全部数据缓存已完成且没有同名运行时使用；删去 `--prepared` 会先准备数据。控制器依次做缓存校验、文本检查、双卡 smoke、正式训练和最终评估，失败则停止并写入日志。不要重复启动同一目录。

## 📦 其他用户工作

ECG-R1 独立调试见[使用说明](docs/ECG_R1_DEBUG.md)。[既有 LEAF 架构草图](docs/MODEL_ARCHITECTURE.md)、[方法草稿](docs/EEGJEV_PATENT.md)以及用户已有 ECG、专利、图件、编辑器设置未作为本轮实验代码修改或提交。
