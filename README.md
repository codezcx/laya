# EEG → 文本条件化 encoder → JEV

当前目标：采用 `xw1216/EEG-FM-Bench` 的数据定义与被试划分，先训练一个 LEAF→JEV 多数据集联合模型，再做 BSR-VideoMAE 单全局 token 对照。用任务描述、问题和候选标签控制任务，不为每个数据集分别训练一套模型。

**联合训练尚未启动。** 原始数据尚未齐全，部分预处理和标签需核验；缺失路径或首轮子集范围待用户确认。完整清点及启动门槛见[当前协议](docs/PROTOCOL.md)。

## 📋 当前状态

- 旧 2a 两组已完成：cross-attention 25.69%，Q-Former-style 25.00%，详见[结果记录](docs/RESULTS.md)。
- 旧 2b 两组已按用户要求停止，控制器状态为 `stopped`，无 active / queued 作业。没有删除任何数据、checkpoint、环境或日志。
- 新 benchmark 已克隆并固定在 `third_party/EEG-FM-Bench-xw1216`，提交 `325398d7d057ecc1216fb3510d70c16eb60337cc`；尚未运行该仓库的数据处理。
- 新 2a 验证/测试被试在旧训练集中出现过，因此新实验不继承旧 2a 下游权重。
- VideoMAE 没有原生 CLS；第二阶段拟使用 mean pooling 生成一个全局 token，不把第一个 patch 当 CLS。这只缩短 JEV 输入，不减少 VideoMAE 内部计算。

## 📚 协议和代码迁移

- [当前协议、数据路径与缺口](docs/PROTOCOL.md)
- [已完成结果与停止记录](docs/RESULTS.md)
- [Git 和产物归档索引](docs/ARCHIVE.md)
- [既有 LEAF 架构草图](docs/MODEL_ARCHITECTURE.md)：历史草图，不是新训练已完成的证据
- EEGJEV 方法草稿：[Word](docs/EEGJEV_PATENT.docx) · [Markdown](docs/EEGJEV_PATENT.md)

旧 BSR→32-token→JEV 实现和完整协议已保存于 Git `14c60cf`。工作树中的 `scripts/run_benchmark.py`、`configs/benchmark_train.json` 仍是旧接口，不能用于新实验；不要重启已取消的 2b。缺失数据/范围确认后再整体替换旧实现，不积累多套独立实验代码。

历史运行产物保留于 `artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028/`，其 `pipeline.json` 是停止状态的证据。更早 LEAF 实现保存于 Git `3033ac9`，历史检查使用隔离 worktree；不直接复用旧数据划分或缓存。

## 🔧 其他用户工作

ECG-R1 的独立 SFT 调试环境与 VS Code 使用方式见 [ECG-R1 调试说明](docs/ECG_R1_DEBUG.md)。该入口使用 `.venv-ecg-r1-py311`，不属于本轮 EEG 联合训练。未将用户已有的 ECG、专利、图件、编辑器设置或其他未提交改动纳入本次提交。
