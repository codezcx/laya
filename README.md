# Task-conditioned EEG → LEAF → JEV

当前保留一套 LEAF→JEV SFT 训练代码：按照 EEG-FM-Benchmark 原仓库的 Cross masks，每个数据集独立训练并在留出被试上测试。每折重新加载发布权重；测试集只在固定最后一轮评估，不参与早停、调参或 checkpoint 选择。

新配方固定 20 epochs、batch 16，训练 LEAF 有效路径、连接层、JEV LoRA 和 choice head，冻结 Qwen 基座及 LEAF BN。仅使用 GPU 2、3。10 个标准 PKL 数据集共准备了 426 个 fold×seed；首批先运行 2a/2b 的 seed 0、fold 0。划分准备完毕不等于全部训练已经排队或完成。这是跨被试 SFT，不是跨数据集 zero-shot。

- [当前训练协议](docs/PROTOCOL.md)
- [模型架构图与组件清单](docs/MODEL_ARCHITECTURE.md)
- [结果与指标说明](docs/RESULTS.md)
- [历史记录归档位置](docs/ARCHIVE.md)
- EEGJEV 专利方法草稿：[Word 排版版](docs/EEGJEV_PATENT.docx) · [Markdown 源稿](docs/EEGJEV_PATENT.md)
- [任务、问题与选项](configs/benchmark_tasks.json)

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_leaf_jev_20261027 --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0
```

当前输出：`artifacts/benchmark/official_cross_leaf_jev_20261027/`。根目录 `pipeline.json` 记录实际执行范围和状态，每折保存 `metrics.jsonl`、`progress.json`、`tensorboard/`、checkpoint 和完整测试预测。启动器拒绝覆盖已有目录；两步 smoke 只用训练样本。

官方划分锁定于 `artifacts/benchmark/official_splits_20261027/`，直接读取原始 benchmark PKL，不复用旧被试合并、排除坏样本或被试级归一化的 manifests。SEED-VIG、ThingsEEG2、Nakanishi 的标准源数据尚未就绪，CHB-MIT 坏样本问题另列处理。具体限制见训练协议。旧实验已归档至 Git `999380b`；旧数据、权重和日志均保留。
ECG-R1 的独立 SFT 调试环境及 VS Code 使用方法见 [ECG-R1 调试说明](docs/ECG_R1_DEBUG.md)。该入口使用 `.venv-ecg-r1-py311`，不替代上述主训练流程。

历史 custom fold-0 全量评估已完成，产物保留在 `artifacts/benchmark/full_eval_fold0_20261027_checked/`，协议/脚本可从 Git `999380b` 恢复；其结果不能混入本轮官方 Cross 的新结果。

## 🔧 32-token 压缩组件

新增两种可切换的可训练压缩器：[单层 cross-attention / 四层 Q-Former-style](eeg_align/token_resampler.py)，共用[配置](configs/token_resamplers.json)。默认将 BSR-VideoMAE 的池化前特征 `[B,1568,768]` 转成 `[B,32,768]`，再交给输入宽度为 768 的 JEV connector。两种模块均为随机初始化，不是已经训练好的对齐器，也没有修改上面的 LEAF 训练入口。

```sh
laya/.venv-eeg/bin/python scripts/check_token_resamplers.py --out artifacts/token_resamplers/my_check
```

该命令默认使用随机特征，只检查前向/反向并保存两个输出 NPZ；也可用 `--tokens /path/to/features.npz` 输入已提取的 `tokens` 和可选布尔 `valid`。输出目录必须不存在。接口、训练接入和限制见[当前协议](docs/PROTOCOL.md#-32-token-压缩组件对照)，验证记录见[结果文档](docs/RESULTS.md#-32-token-组件验证2026-10-28)。
