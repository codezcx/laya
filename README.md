# EEG → BSR-VideoMAE → 32-token resampler → JEV

当前保留一套 BSR→JEV SFT 训练代码，比较单层 cross-attention 与四层 Q-Former-style，两者均压缩为 32 个 token。按照 EEG-FM-Benchmark 原仓库的 Cross masks，每个数据集独立训练并在留出被试上测试。每折重新加载发布权重；测试集只在固定最后一轮评估，不参与早停、调参或 checkpoint 选择。

固定 20 epochs、batch 16，训练 BSR 有效路径、压缩器、连接层、JEV LoRA 和 choice head，冻结 Qwen 基座。仅使用空闲的 GPU 2、3，不抢占已有任务。首批 2a/2b × 两种压缩器，seed 0、fold 0，共 4 组；队列先完成 4 组两步 smoke，再从发布权重开始正式训练。这是跨被试 SFT；BSR 预训练包含 2b，不能宣称无重叠 zero-shot。

- [当前训练协议](docs/PROTOCOL.md)
- [既有 LEAF 架构草图（非本轮 BSR 实现）](docs/MODEL_ARCHITECTURE.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- EEGJEV 专利方法草稿：[Word 排版版](docs/EEGJEV_PATENT.docx) · [Markdown 源稿](docs/EEGJEV_PATENT.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028 --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0
```

当前输出：`artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028/`。根目录 `pipeline.json` 记录实际执行范围和状态，每组保存 `metrics.jsonl`、`progress.json`、`tensorboard/`、checkpoint 和完整测试预测。启动器拒绝覆盖已有目录；两步 smoke 只用训练样本，不向正式训练传递权重。

官方划分锁定于 `artifacts/benchmark/official_splits_20261027/`，直接读取原始 benchmark PKL。当前只启用已核验的 2a/2b 信号适配器，保留原始通道与完整 epoch，不做 LEAF 的 65 通道插值或额外稳健归一化。LEAF 训练实现及协议已归档至 Git `3033ac9`，更早实验见 `999380b`；旧数据、权重、环境和日志均保留。
ECG-R1 的独立 SFT 调试环境及 VS Code 使用方法见 [ECG-R1 调试说明](docs/ECG_R1_DEBUG.md)。该入口使用 `.venv-ecg-r1-py311`，不替代上述主训练流程。

历史 custom fold-0 全量评估已完成，产物保留在 `artifacts/benchmark/full_eval_fold0_20261027_checked/`，协议/脚本可从 Git `999380b` 恢复；其结果不能混入本轮官方 Cross 的新结果。

## 🔧 32-token 压缩组件

两种可训练压缩器：[单层 cross-attention / 四层 Q-Former-style](eeg_align/token_resampler.py)，共用[配置](configs/token_resamplers.json)。将 BSR-VideoMAE 的池化前特征 `[B,1568,768]` 转成 `[B,32,768]`，再交给输入宽度为 768 的 JEV connector。压缩器是新初始化参数，已纳入训练器的独立优化器组。

```sh
laya/.venv-eeg/bin/python scripts/check_token_resamplers.py --out artifacts/token_resamplers/my_check
```

该命令只是随机特征组件检查，不启动训练。正式队列使用上方 `run_benchmark.py`。也可用 `--tokens /path/to/features.npz` 检查已提取的 `tokens` 和可选布尔 `valid`；输出目录必须不存在。训练接入和限制见[当前协议](docs/PROTOCOL.md)，运行状态见[结果文档](docs/RESULTS.md)。
