# BSR → 32-token resampler → JEV：官方 Cross SFT

_2026-10-28；当前只保留此训练入口及两种压缩器对照。旧 LEAF 协议在 Git `3033ac9`。_

---

## 🎯 目标与划分

比较 cross-attention 与 Q-Former-style 压缩器能否在相同下游训练预算下支持 EEG 分类。固定 BSR 发布权重、原生 JEV adapter、数据、样本顺序、选项排列策略、训练步数与学习率；不匹配两种压缩器的参数量或 FLOPs，不把差异单独归因于 self-attention。

使用 `artifacts/benchmark/official_splits_20261027/` 已锁定的官方 Cross 被试划分，原仓库提交 `b36719588abec452e4c2b564f25ede1c0b1c03bc`。[^1] 首批仅 seed 0 / fold 0，每个数据集独立训练：

| 数据集 | 类别数 | 训练 trial / 被试 | 测试 trial / 被试 | 每组正式 updates |
| --- | ---: | --- | --- | ---: |
| BNCI2014001 / 2a | 4 | 2,304 / 2–9 | 288 / 1 | 2,880 |
| BNCI2014004 / 2b | 2 | 1,240 / 2–9 | 160 / 1 | 1,560 |

两种压缩器各跑两数据集，共 4 组、8,880 次正式 updates。原始 PKL、split masks、被试及标签计数均按哈希核验。保持官方预处理所选 session，不用 T/E 划分替换 Cross。其他数据集的 masks 仍保留，但当前 BSR 适配器不自动启用它们。

## ⚙️ 模型与输入

BSR checkpoint：`eegckpt/BSR-VideoMAE-AS-Final-v3/ckpts/B1_regen_21e31e56baf2.pt`，SHA256 `21e31e56baf23c0d5446e750ac4eb54a5032d4fe34d2845788e6d22c373bba36`。复用该包的 R1 renderer 与 SharedScale adapter，严格加载 encoder/renderer/adapter 权重；不使用其分类头。

- 保留完整 benchmark epoch：200 Hz 时 2a 为 `[B,22,800]`，2b 为 `[B,3,901]`；不截短 2b，不补额外基线。
- 保留原始通道与 PKL 信号尺度，不做 65 通道插值、被试级标准化或新的 per-trial 稳健归一化。
- R1 使用幅度 STFT，`n_fft=200`，按完整 epoch 长度选择 hop，使输出恰为 16 帧；未测通道对应网格填零。
- SharedScale 对每个 clip 的 RGB、时空整体求均值/标准差，应用发布的 gain/bias、sigmoid、224×224 resize，再做 ImageNet normalization。它不汇总其他 trial 或测试被试统计。
- VideoMAE 读取 16 帧，保留池化前 `[B,1568,768]`；随后压缩为 `[B,32,768]`，connector 将宽度映射至 5120。
- JEV 使用问题、排列后的类别选项与最后的 `[decision]:`；真实标签只用于 CE target，不进入文本。

本地实现：[OnlineBSR](../eeg_align/online_bsr.py)、[压缩器](../eeg_align/token_resampler.py)、[训练器](../eeg_align/train_joint.py)。外部源文件哈希随运行记录并由运行时 guard 校验，外部源码/权重不修改。旧 Transformers 的 `q_bias/v_bias` 无损转换为新 Linear bias；新显式 key bias 固定为 0，保持原参数化。

## 🔧 两种压缩器

| 配置 | 每层操作 | 层数 | 输出 | 参数量 |
| --- | --- | ---: | --- | ---: |
| `cross_attention` | cross-attention + FFN | 1 | `[B,32,768]` | 7,115,520 |
| `qformer` | self-attention + cross-attention + FFN | 4 | `[B,32,768]` | 37,834,752 |

两组都使用新初始化的 32 个可学习 query、12 个头、pre-LN、残差、4 倍 FFN、输出 LayerNorm、dropout=0。Q-Former-style 不等于完整原版 BLIP Q-Former 或预训练权重。queries 不由文本或真实类别标签产生。输入 mask 中 True 表示有效 token，全空样本报错。

## 📋 固定训练配方

| 设置 | 值 |
| --- | --- |
| epochs / effective batch / microbatch | 20 / 16 / 4 |
| LR：压缩器 / connector | 1e-4 / 1e-4 |
| LR：JEV LoRA / choice / BSR | 5e-6 / 1e-5 / 1e-6 |
| 优化器 / weight decay / clipping | AdamW / 0.01 / norm 1 |
| LR schedule | 3% warmup + cosine，最低倍率 0.02 |
| BSR 预热冻结 | 前 min(100, 总步数÷10) 步 |
| 正式可训练 | BSR 有效路径、压缩器、connector、原生 JEV LoRA、choice |
| 始终冻结 | Qwen 基座、BSR 逆 renderer、VideoMAE key bias |
| 精度 | BSR/STFT/adapter 参数 fp32，VideoMAE CUDA autocast bf16；JEV 原有混合精度 |

每 epoch 遍历全部 train 行一次，保留尾批；不复制 smoke 权重。初始化时记录完整可训练参数和 encoder 指纹，相同 seed/variant 跨数据集必须相同；两种 variant 的共享参数指纹也必须相同。构建模型后重置 RNG，避免参数量不同改变训练随机流。

每步检查 resampler 梯度有限且非零；BSR 解冻前不得产生梯度，解冻后必须产生梯度。checkpoint 包含压缩器、encoder 可训练参数、LoRA、connector、head、buffers 和 RNG；训练中包含 optimizer/scheduler。最终在训练样本上扰动全部可训练张量，重载真实磁盘 checkpoint，逐元素复现 logits 后才允许测试。

## 🔐 测试隔离与解释边界

官方 Cross 没有额外验证集；原 trainer 的逐 epoch test-as-validation 不用于本轮选择。[^2] 固定最后一轮后才访问完整外层 test，不早停、不搜索，不根据反转选项表现挑结果。将来调参需另设外层训练被试内部的验证集。

BSR Stage-B provenance 包含 `bnci2014_004`；与本轮 2b 测试被试/trial 的重叠尚未排除。2a 不在该 Stage-B 列表中也不能证明全部预训练未见它。因此这是**下游 Cross SFT 对照，不是已证实无预训练泄漏的 zero-shot**。2b 三个双极通道放到 C3/Cz/C4 网格也不会变成高密度单极 EEG。幅度 STFT 丢失相位，压缩器可能进一步损失任务信息，必须用最终任务指标检验。

报告完整 test 预测、accuracy、BA、kappa、F1，二分类加 AUROC/AUPRC；反转选项作为诊断，不替换主指标。step 0 记录固定训练小批量诊断，逐步 batch accuracy 不是验证集准确率。

## 🚀 启动与状态

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_bsr_jev_32tokens_20261028 --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0
```

启动器默认仅上述首批范围；每组先跑两步真实 train-only smoke，第 2 步解冻 BSR。全部 smoke 通过后，自动开始 4 组正式训练；`--smoke` 则只跑检查。仅使用 GPU 2、3，并在每次启动前要求该卡显存占用 ≤1024 MiB、利用率 ≤5%，否则等待，不终止外部任务。任何 worker/哈希/初始化/重载检查失败，队列失败并停止，不继续排训练。

`pipeline.json` 记录 active / queued / completed；每组有 log、`progress.json`、`metrics.jsonl`、TensorBoard、`last.pt`。测试完成才生成正式 `complete.json` 和 `test_predictions.npz`。目录拒绝覆盖。环境新增 `opencv-python==4.11.0.86` 是外部包导入依赖，本轮不调用 CLAHE；本地 wheel 与 PyPI 公布 SHA256 `6b02611523803495003bd87362db3e1d2a0454a6a63025dc6658a9830570aa0d` 一致。

资源、Hugging-science、Transformers 技能用于资源/预处理/权重审查，Markdown 技能用于同步协议；检查通过不等于准确率提高。旧 LEAF 实现、结果文档均在 Git `3033ac9`；其日志、数据、环境和 checkpoint 未删除。

[^1]: Official Cross splitter. https://github.com/Dingkun0817/EEG-FM-Benchmark/blob/b36719588abec452e4c2b564f25ede1c0b1c03bc/utils/dataset_split.py
[^2]: Official trainer. https://github.com/Dingkun0817/EEG-FM-Benchmark/blob/b36719588abec452e4c2b564f25ede1c0b1c03bc/utils/trainer.py
