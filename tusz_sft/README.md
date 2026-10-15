# TUSZ → EEG RL 8B → LoRA 决策分类

当前训练入口为 `python -m tusz_sft.train`（实现：`train_decision.py`），默认加载本地 **Qwen3-VL-8B EEG RL checkpoint**：

```
/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss/DataEngine/output/eeg_rl_hf_8b_v3render/step_60
```

`step_60` 是固定起点，不代表最优 RL step。底座只读，不使用原 JEV-27B 权重或适配器。历史实现保留在 `train_legacy.py` 和 [LEGACY_PIPELINE.md](LEGACY_PIPELINE.md)。

## 模型

图像与固定问题经过 Qwen3-VL 多模态主干，取最后一个有效 prompt token 的 hidden state，送入 **Linear(4096, 2)**，输出 `P(bckg)` 和 `P(seiz)`。

- 从原 `lm_head` 的 A/B token 行初始化决策头，然后移除完整词表输出层。推理只需一次 forward，不生成文本。
- 保留 EEG RL 后的主干权重；冻结视觉塔、视觉投影、语言原权重。在语言 decoder 的 q/k/v/o、gate/up/down projection 上训练新 LoRA，默认 rank 16、alpha 32、dropout 0。
- 决策头单独训练；采用交叉熵，类别权重由训练计数决定并显式乘逐样本 loss，避免 microbatch=1 时权重被均值归一化抵消。
- 主干 BF16、LoRA/决策头 FP32，SDPA，非重入 gradient checkpointing。每张 GPU 保存一份主干，DDP 同步可训练参数。
- A=`bckg`，B=`seiz`。非发作不等于正常 EEG，也不直接监督诊断解释。

## 数据和输入

`render_v3.py` 对齐本地 AAAI 的 `DataEngine/tusz_adapter.py` 和 `render_images_v3.py`：19 通道，200 Hz，每张 10 秒；整段记录做 MNE FFT 重采样，转 µV 和 float32 后做共同平均参考。不额外滤波、不逐窗缩放；12×8 英寸、110 DPI PNG、负电位向上、通道间距相当于 200 µV。真实 EDF 的数组与原适配器逐元素一致；绘图通过 PNG 字节一致性检查。

本流程严格要求完整 19 通道，原适配器允许缺失通道补零，因此这里只覆盖一个更严格的记录子集。标签来自 `.csv_bi`：任何正发作重叠即为发作；默认训练排除部分重叠边界窗，丢弃不足 10 秒的尾段。图片和 prompt 不包含标签、受试者 ID、路径或发作标注。

当前目录 `artifacts/eeg_decision_experiments/` 使用固定种子抽取 4096 个官方 train 来源窗口，再按受试者划分为 **3366 张训练 / 730 张内部验证**。这不是完整 TUSZ 基准。内部验证未参加当前 LoRA 梯度更新，但初始 EEG RL 模型可能见过这些患者/窗口，分数不能声称是从未见过患者的外部泛化。官方 dev/eval 未用于本轮调参。已核对的旧语料清单覆盖了 dev 的 51/53 名合格患者，eval 的 42/43 名；内部验证 730 窗中有 274 窗出现在可用旧 SFT 清单。详细审计见实验目录的 `prior_corpus_overlap.json` 和 `official_split_prior_overlap.json`。

频谱与波形保持相同窗口、标签、顺序和划分。`prepare_spectrum.py` 画 19 通道 STFT 功率谱：Hann 2 秒窗、0.25 秒步长、0.5–70 Hz，共享 −30～40 dB 色标（相对于 1 µV²/Hz），不逐图归一化。STFT 中心覆盖 1～9 秒，边缘保留空白。双图模式同时输入同一窗口的波形和频谱。

## 实验安排

`run_suite.py` 管理两组各四卡的任务，前一阶段全部完成后才开始下一阶段：

| 阶段 | GPU 0–3 | GPU 4–7 |
|---|---|---|
| 1 | 波形，LoRA LR 1e-5 | 波形，LoRA LR 5e-5 |
| 2 | 频谱，LoRA LR 1e-5 | 频谱，LoRA LR 5e-5 |
| 3 | 波形+频谱双图 LoRA | 先只训练决策头，完成后接原生 8B 波形对照 |

每组 3 epochs，microbatch 1、梯度累积 8、有效 batch 32，头学习率 1e-4，固定种子。双图的 LoRA 学习率根据波形内部验证选择；双图使用更多视觉 token，计算量并不与单图相同。

先评估原始 RL 表示配 A/B 决策头的效果，step 0 也可以成为最佳 checkpoint。每 100 步和每轮末验证，以内部验证 AUROC 选择 checkpoint；分类阈值固定为 0.5。这里按类别抽样，不能据此推断真实部署时的概率校准或阳性预测值。

- 调度状态：`artifacts/eeg_decision_experiments/suite_state.json`。
- 实际日志：`artifacts/eeg_decision_experiments/logs/`。
- 曲线、指标和权重：`artifacts/eeg_decision_experiments/runs/<实验名>/`。

遇到失败，调度器停止推进；不自动覆盖或重启失败任务。

## 命令

在工作区根目录执行，环境为 `laya/.venv-eeg/bin/python`。

```bash
# 四卡单组：使用新输出路径。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 scripts/train_eeg_decision.sh \
  --config configs/waveform_lr1e5.json --out artifacts/my_waveform_run

# 八卡单组：累积改成 4，保持有效 batch=32。
NPROC_PER_NODE=8 scripts/train_eeg_decision.sh \
  --config configs/waveform_lr1e5.json --grad-accum 4 --out artifacts/my_8gpu_run

# 恢复尚未完成的原运行：只接受 latest.json 指向的 checkpoint；保持全部设置一致。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 scripts/train_eeg_decision.sh \
  --config configs/waveform_lr1e5.json --out /absolute/run \
  --resume /absolute/run/checkpoints/step-000100

# 单图推理；双图 checkpoint 则依次传 waveform.png spectrogram.png。
laya/.venv-eeg/bin/python -m tusz_sft.predict_decision \
  --checkpoint /absolute/run/checkpoints/step-000100 --image /absolute/image.png

# 只检查数据、processor 和图像 token，不加载模型权重。
laya/.venv-eeg/bin/python -m tusz_sft.train \
  --config configs/waveform_lr1e5.json --out artifacts/my_input_check --check-input-only
```

`--base` 可换成其他本地 Qwen3-VL checkpoint，头维度由实际权重推导。目前实测为 8B；32B 可配置但未实测资源消耗。此加载器不支持 Qwen3.5/JEV-27B 架构。

## 保存与验证

每个 checkpoint 包含 `adapter/`、`decision_head.safetensors`、`processor/`、`decision_config.json` 和 `training_state.pt`（优化器、scheduler、每 rank RNG、训练位置）。只训练头的对照没有 adapter。记录底座路径、配置/索引哈希和分片大小/修改时间，加载时核对；临时目录保存完整后再改名。训练 batch 丢弃不足一整个分布式 microbatch 的尾部，每轮数量记录在 preflight 中；梯度累积的最后不足组按实际组大小归一化。

已完成八卡两步训练、断点恢复到第三步；重新加载 step 2 后，22 个样本的概率与保存前逐项相同。单元测试覆盖 padding、A/B 决策头等价性和梯度、v3 绘图、STFT 频率与功率单位、标签和通道处理。这些是工程正确性证据，不是泛化结果。

关键实现：`decision_model.py`、`decision_data.py`、`train_decision.py`、`predict_decision.py`、`prepare.py`、`prepare_spectrum.py`、`run_suite.py`。

验证环境：Python 3.10，PyTorch 2.11.0+cu128，Transformers 5.13.1，PEFT 0.19.1，MNE 1.12.1，NumPy 2.2.6，SciPy 1.15.3，Matplotlib 3.10.9。CerebraGloss 公开代码许可保留在 `third_party/CerebraGloss/LICENSE`；本地原实现保留不动。

## 锁定模型后的额外评估

`prepare_holdout.py` 从官方 dev 中剔除可用旧 SFT/RL 清单的相同 10 秒窗口，再固定抽取 512 发作 + 512 非发作窗（47 名患者，124 个边界窗），生成匹配的波形/频谱/双图输入。其中 45 名患者出现在旧语料清单；这是窗口排除后的测试，不是严格患者独立测试。剩余真正未出现在这些旧清单中的 dev 患者只有 2 名、283 个非发作窗，无法单独评价二分类 AUROC。

为提供没有本地 EEG SFT/RL 历史的对照，阶段 3 的右侧队列从本地原生 `CerebraGloss/DataEngine/models/Qwen3-VL-8B-Instruct` 训练同样的波形决策模型。它在只训练头的任务结束后使用同组四卡和梯度累积 8，有效 batch 为 32；与其他单图 LoRA 组使用相同批次设置，避免另一组双图训练期间出现长时间空闲。

`finalize_experiments.py` 等全部七组训练正常完成后，自动对内部验证选出的 checkpoint 和初始 EEG RL 决策模型做固定 dev 评估。评估两组各四卡并行，不回写训练选择。状态在 `evaluation_state.json`，结果在 `evaluations/`；最终生成 `RESULTS.md`、`comparison.csv/json` 和 `comparison.png`。AUROC 区间按患者 bootstrap 1000 次计算，不包含训练随机种子变异。

最终验收执行 `python -m tusz_sft.audit_experiments`：核对全部训练步数、实际配置、源权重签名、适配器范围与有限值，独立复算所选 checkpoint 的验证指标及八组 dev 指标，并比较 CSV/JSON 报告；只有全部任务完成且无训练/评估进程残留时才写出 `completion_audit.json`。最终图表也会标明旧 EEG RL 语料的患者重叠限制。

实验已经完成：七组训练各 3 epochs / 318 optimizer steps，八组固定 dev 评估已完成并独立复算。最终指标见 [RESULTS.md](../artifacts/eeg_decision_experiments/RESULTS.md)，核验记录见 [completion_audit.json](../artifacts/eeg_decision_experiments/completion_audit.json)。收尾增加了恢复训练目录、latest 指针、训练身份与完成状态校验；无效恢复请求在写入实验元数据前失败。
