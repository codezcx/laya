# Task-conditioned EEG → LEAF → JEV

当前保留一套 LEAF→JEV SFT 训练代码：按照 EEG-FM-Benchmark 原仓库的 Cross masks，每个数据集独立训练并在留出被试上测试。每折重新加载发布权重；测试集只在固定最后一轮评估，不参与早停、调参或 checkpoint 选择。

新配方固定 20 epochs、batch 16，训练 LEAF 有效路径、连接层、JEV LoRA 和 choice head，冻结 Qwen 基座及 LEAF BN。仅使用 GPU 2、3。10 个标准 PKL 数据集共准备了 426 个 fold×seed；首批先运行 2a/2b 的 seed 0、fold 0。划分准备完毕不等于全部训练已经排队或完成。这是跨被试 SFT，不是跨数据集 zero-shot。

- [当前训练协议](docs/PROTOCOL.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_leaf_jev_20261027 --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0
```

当前输出：`artifacts/benchmark/official_cross_leaf_jev_20261027/`。根目录 `pipeline.json` 记录实际执行范围和状态，每折保存 `metrics.jsonl`、`progress.json`、`tensorboard/`、checkpoint 和完整测试预测。启动器拒绝覆盖已有目录；两步 smoke 只用训练样本。

官方划分锁定于 `artifacts/benchmark/official_splits_20261027/`，直接读取原始 benchmark PKL，不复用旧被试合并、排除坏样本或被试级归一化的 manifests。SEED-VIG、ThingsEEG2、Nakanishi 的标准源数据尚未就绪，CHB-MIT 坏样本问题另列处理。具体限制见训练协议。旧实验已归档至 Git `999380b`；旧数据、权重和日志均保留。

历史 custom fold-0 全量评估已完成，产物保留在 `artifacts/benchmark/full_eval_fold0_20261027_checked/`，协议/脚本可从 Git `999380b` 恢复；其结果不能混入本轮官方 Cross 的新结果。
