# REVE → Qwen 原始 EEG 分类

本模块独立于 `tusz_sft` 的波形图片实验。没有实例化图像 encoder、视觉 merger 或完整 lm_head。
任务为左右手运动想象，输出顺序固定为 `left_hand / right_hand`，对应 A/B。

## 模型与路径

所有本地路径在 `configs/reve_qwen_mi.json`：

- REVE：`/inspire/hdd/global_user/zhangchenxi-253108310322/BSR-VideoMAE/reve-base`
- 电极位置：同级 `reve-positions`
- Qwen：`/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss/DataEngine/output/eeg_rl_hf_8b_v3render/step_60`
- 数据：`/inspire/dataset/moabb-processed/v1/EEG-processed/BNCI2014_004`

加载器只读取 Qwen safetensors 中 `model.language_model.*` 的张量，并逐项检查键名与形状。
原 lm_head 只读 A/B 两行来初始化可训练分类头。没有使用 JEV-27B 权重或旧波形图 LoRA。

```
EEG (B,3,T), 200 Hz
  → frozen REVE-base (B,3,P,512)
  → flatten channel/time (B,K,512)
  → LayerNorm → Linear(512,4096) → GELU → Linear(4096,4096)
  → [chat prefix embeddings, EEG embeddings, question/assistant prefix embeddings]
  → Qwen3VLTextModel, causal attention, sequential position IDs
  → last valid input hidden state → Linear(4096,2), no bias
  → CrossEntropyLoss(logits, label)
```

文本使用原 checkpoint 的 chat template；没有新增词表项。EEG 不经过图像 processor 或图像网格位置编码。
REVE 已包含电极/时间位置特征，语言侧给拼接序列分配普通顺序位置。

## 两阶段

- `warmup`：冻结 REVE、Qwen；训练 connector 和分类头。Qwen 的 forward 保留计算图。
- `lora`：从 warmup checkpoint 加载连接层/分类头；新增 Qwen attention 和 MLP 的 LoRA。
  REVE 仍冻结。默认 r=16、alpha=32，LoRA lr=1e-5，连接层/分类头 lr=1e-4。
- loss 都是二分类交叉熵，不加对比学习或 CoT loss。
- 默认 batch=4/GPU，梯度累积4，epochs=3。这些是初始超参数，不是已验证的最优设置。
- `--init-checkpoint` 是权重 warm start，重新建立 optimizer；**不是中断点精确恢复**。
- 保存 best/last，按 validation accuracy 选择 best；test 不参与训练或 checkpoint 选择。

## 数据处理与已知限制

数据入口读取可信本地 pickle，核对通道、200 Hz、imagery 标签和有限值；每个 trial 每通道
独立 z-score 后 clip 到 ±15，不使用跨 trial 或测试集统计。这是明确记录的首版预处理选择，
不是对上游预训练全流程逐行等价的保证；会移除绝对幅值及部分通道间幅值差异。
保留完整 trial，尾部用末值补齐到 REVE 的 200-sample patch、180-sample stride，不静默截断。
不同真实长度分别经过 REVE，再在语言输入末尾 padding，避免 encoder 看见批量补零。

2b 的 C3/Cz/C4 是双极导联，位置银行使用同名位置近似。来源字段有已知 `inferred_file_by_run`
问题，正式实验前需核对原始文件对应关系；不能宣称数据来源已完全验证。prepare 生成
`audit.json` 并将 `provenance_verified` 标记为 false。探索性训练需要显式传
`--allow-unverified-provenance`，该选择保存到 run.json。

当前划分是**同被试跨会话**：session 0/1train 用于训练，2train 用于验证，3/4test 用于测试。
不是跨被试泛化。原测试集已有历史实验结果，不视为全新未见测试集。
仅用二分类监督不代表获得丰富的 EEG–文本语义理解，下一步结果应与冻结 REVE 的简单分类器对照。

## 使用

在仓库根目录执行；现有 `laya/.venv-eeg/bin/python` 环境已具备依赖。

```bash
# 读取检查全部数据，生成固定划分（不训练）
laya/.venv-eeg/bin/python -m eeg_align.data --config configs/reve_qwen_mi.json

# 接口自检：两条合成 EEG，一次 backward/update，不是正式训练
bash scripts/train_reve_qwen.sh --smoke --output artifacts/reve_mi/my_smoke
laya/.venv-eeg/bin/python -m eeg_align.verify artifacts/reve_mi/my_smoke

# A 阶段：连接模块预热（以下是探索性数据配置）
bash scripts/train_reve_qwen.sh --stage warmup --epochs 3 \
  --allow-unverified-provenance --output artifacts/reve_mi/warmup

# B 阶段：加载 A 的 best，再训练新的语言侧 LoRA
bash scripts/train_reve_qwen.sh --stage lora --epochs 3 \
  --init-checkpoint artifacts/reve_mi/warmup/best \
  --allow-unverified-provenance --output artifacts/reve_mi/lora

# 单独评估，输出 accuracy / balanced accuracy / confusion matrix / loss
bash scripts/train_reve_qwen.sh --stage lora --evaluate test \
  --init-checkpoint artifacts/reve_mi/lora/best \
  --allow-unverified-provenance --output artifacts/reve_mi/test_lora
```

多卡入口：在命令前设 `NPROC_PER_NODE=4`（以实际可见卡数为准）。每卡复制完整模型，
这是 DDP，不是模型分片；有效 batch 为每卡 batch × 累积次数 × 卡数。
评估按 rank 分片但不重复补样本，最终汇总混淆矩阵。
每个 output 必须是新的运行目录，防止覆盖已有结果。

checkpoint 只保存 `bridge.safetensors`、配置及 B 阶段的 `adapter/`；冻结基座仍依赖上述本地路径。
推理调用 `load_model(..., checkpoint=..., stage="lora", trainable=False)` 后以相同预处理输入
`eeg` 和 `lengths`，对返回的 `logits` 做 softmax/argmax 即可。不进行自回归生成。
