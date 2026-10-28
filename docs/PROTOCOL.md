# LEAF → JEV：官方 Cross 划分上的 SFT

_2026-10-27 固定的新协议，替代旧 fold-0 四组诊断。SFT 是训练方式，跨被试是测试方式。_

## 🎯 目标与划分

每个数据集独立微调，在该折完全留出的被试上测试。此次不是多数据集联合训练，也不是整个目标数据集未见的跨数据集 zero-shot。

使用官方仓库提交 `b36719588abec452e4c2b564f25ede1c0b1c03bc` 的原始 Cross splitter：LOSO；TUAB/SleepEDF 为被试级 10-fold。保留原 `usage`（尤其 SEED、CHB-MIT）、subject ID 和全部 source row indices，不额外减少一个外层验证被试。[^1]

划分锁定于 `artifacts/benchmark/official_splits_20261027/`，每折含行号、被试、数量、SHA256。训练前重新核验 PKL 和 mask。10-fold 在 splitter 调用前固定 RNG，为所有模型提供共同 masks；不宣称与论文未公开的逐行 fold manifests 完全相同。

| seed 0 / fold 0 | SFT 被试 | SFT trial | 测试被试 | 测试 trial |
| --- | --- | ---: | --- | ---: |
| 2a | 2–9 | 2,304 | 1 | 288 |
| 2b | 2–9 | 1,240 | 1 | 160 |

## 🔐 测试隔离规则

| 环节 | 使用数据 | 禁止事项 |
| --- | --- | --- |
| 梯度、训练诊断、smoke | 本折 train mask | 输入测试被试 trial |
| 归一化 | 当前单个 trial | 汇总测试被试/测试集统计 |
| checkpoint | 固定最后一轮 | 按测试分数选 best |
| 最终测试 | 完整 test mask | 截断、挑样本、反馈调参 |
| 下一折 | 重新加载发布权重 | 沿用上一折 SFT 权重 |

官方 Cross 没有独立验证集；原 trainer 每轮评估 `test_loader` 并把它写成 validation，最终返回最后一轮模型。本轮沿用 masks，但不沿用逐轮看测试集的行为：固定 20 epochs 后才做最终测试，不早停、不搜索。若以后调参，需在外层训练被试内部做内层验证，选定后重新训练全部外层 train，不能使用外层 test。[^2]

## ⚙️ 模型与固定配方

原 benchmark PKL 完整 epoch → 单 trial 稳健归一化/200 Hz/65 通道映射 → 任务描述条件化 LEAF → 256 维 query tokens → 5,120 维连接层 → JEV 文本塔及 choice head。语言输入为 question、随机排列选项和末尾 decision 位置；真实标签只用于 CE target，不进入文本。没有图像编码器。

| 设置 | 值 |
| --- | --- |
| 可训练 | LEAF 有效路径、连接层、JEV LoRA、choice 权重/偏置 |
| 冻结 | Qwen 原始大矩阵，LEAF BN 与 dropout |
| epochs / batch / microbatch | 20 / 16 / 8 |
| 采样 | 每 epoch 打乱全部 train 行，每行一次，保留尾批 |
| 优化器 / weight decay / clipping | AdamW / 0.01 / norm 1 |
| LR：连接层 / LoRA / choice / LEAF | 1e-4 / 5e-6 / 1e-5 / 1e-6 |
| 冻结 LEAF 预热 | 前 min(100, 总步数÷10) 步 |
| LR schedule | 3% warmup + cosine，最低倍率 0.02 |
| seeds / GPU | 0、1、2 / 仅 2、3 |

2a 第 0 折 2,880 updates；2b 第 0 折 1,560 updates。20 epochs/batch 16 取原入口默认值；其余沿用现有模型经验，预先固定。官方没有 LEAF/JEV 适配器或超参数，因此是官方划分上的新模型实验，不是精确复现论文模型配方，也不是全量 27B SFT。[^3]

每折新进程加载发布 LEAF、原生 JEV adapter、seed 对应的随机连接层；同 seed 的初始参数/buffers 指纹必须一致。严禁恢复历史 EEG 多任务 checkpoint。只允许同 dataset/seed/fold、代码、数据、配方的断点续训。

## 📊 输入与输出完整性

直接读取原 PKL，不复用旧缓存的删行、CHB 被试合并或被试级归一化。长 epoch 按有序 ≤10 秒片段处理，保留全部时间。坏信号触发失败，不静默排除。

逐步保存训练 loss、在线 batch accuracy、梯度、学习率与 TensorBoard；在线 batch accuracy 不是完整训练集最终准确率。测试只使用最后一轮，主指标固定为正常选项顺序；反转顺序只作诊断，不能二选一挑高分。

保存全部测试行号、标签、概率、反转概率、混淆矩阵、accuracy、BA、kappa、weighted F1；二分类额外 AUROC/AUPRC。使用官方 metric 函数；其单类别 AUROC/AUPRC 回退为 0.5，应结合类别支持数解释，不能当实测排序能力。

`last.pt` 保存可训练参数、完整 buffers、RNG；训练中另含 optimizer/scheduler，最终去掉优化器以控制存储。冻结基座从本地原生权重恢复。真实磁盘 checkpoint 在训练样本上重载并逐元素复现 logits 后才能输出测试结果。中断/失败不标记完成。

## ⚠️ 范围与限制

- 10 个标准 PKL 数据集共 426 个 fold×seed 仅代表划分就绪。首批只启动 2a/2b seed 0、fold 0；实际状态见 pipeline。
- SEED-VIG、ThingsEEG2 缺标准数据/相应接口；Nakanishi 替代缓存 cohort 与论文不同，未冒充标准 PKL；CHB-MIT 坏样本尚待处理。
- 2a/2b 保留官方预处理所选 session，不用 T/E 会话划分替代 Cross。
- CHB 原 ID 01/21 的同一生物学被试问题未消除；保留原 ID 只保证 ID 隔离。
- 本地 CHB/TUAB 等 cohort 数量与论文有差异，不能标为论文精确数据复现。
- 过去已观察部分被试结果；本轮保证当前 SFT fold 的训练/选择隔离，不宣称研究过程中从未接触这些被试。
- 发布 LEAF 的预训练数据重叠尚未排除；下游隔离不等于 foundation pretraining 全流程无泄漏。
- 2b 三通道插值为 65 通道不增加新的实测空间信息。现有适配器的这一限制仍然存在。

## 🔧 执行与归档

先用训练行运行两步真实模型 smoke；通过后从发布权重开始正式训练：

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_leaf_jev_20261027_smoke --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0 --smoke
laya/.venv-eeg/bin/python scripts/run_benchmark.py --out artifacts/benchmark/official_cross_leaf_jev_20261027 --datasets BNCI2014001 BNCI2014004 --seeds 0 --folds 0
```

不加筛选会排入全部 426 个 folds，计算代价极大；完成吞吐和数据检查前不自动这样启动。每折有 `metrics.jsonl`、`progress.json`、`tensorboard/`、`last.pt`、`test_predictions.npz`、`complete.json`。worker 支持 `--resume` 在原目录续训，拒绝跨折加载。

旧实现/ledger/历史评估协议在 Git `999380b`，用隔离 worktree 重跑；没有删除旧数据、权重、日志或环境。实验设计、资源检查、Transformers 技能辅助隔离检查与运行准备，Markdown 技能用于同步文档；这些程序不代表实验结论已有效。[^4]

## 🔧 32-token 压缩组件对照

2026-10-28 新增的组件验证，不改变上面的 LEAF 官方 Cross 训练配方。目标是把 BSR-VideoMAE **池化前**的 `[B,1568,768]` 压缩为 `[B,32,768]`；不能先均值池化成 `[B,768]`，再把它当作 1568 个 token。输入应保留 encoder 的位置编码信息。这里不另外引入文本或标签作为 query。

| 配置名 | 每层操作 | 层数 | 输出 | 参数量 |
| --- | --- | ---: | --- | ---: |
| `cross_attention` | cross-attention + FFN | 1 | `[B,32,768]` | 7,115,520 |
| `qformer` | self-attention + cross-attention + FFN | 4 | `[B,32,768]` | 37,834,752 |

两种实现均使用 32 个可学习 query、12 个注意力头、pre-LN、残差、4 倍 FFN 和输出 LayerNorm，dropout 为 0。Q-Former-style 只实现 query resampler，不包含 BLIP 的文本分支、预训练目标或发布权重；两组都是新初始化、需要训练。两组匹配输出长度及宽度，但**不匹配参数量和计算预算**，因此未来准确率差异不能单独归因于 self-attention。

实现为 [`TokenResampler`](../eeg_align/token_resampler.py)，配置为 [`token_resamplers.json`](../configs/token_resamplers.json)。`valid[B,N]` 中 True 表示有效输入；padding 在归一化前清零且不参与 attention。全空样本报错；返回的 `valid32[B,32]` 全 True。支持可变 N，N 小于 32 时属于重采样/扩展而非压缩。不要直接对 LEAF 原有的 8 个 query 宣称“压缩到 32”。

### 接入 JEV

```python
from eeg_align.token_resampler import ResamplerConfig, TokenResampler, ResampledDecisionModel
from eeg_align.jev import JEVBridge

config = ResamplerConfig.from_json("configs/token_resamplers.json", "qformer")
resampler = TokenResampler(config).to(weight.device)
# text、weight、bias、tokenizer 由现有 load_jev 加载；不复用 256 维 LEAF connector。
bridge = JEVBridge(text, weight, bias, tokenizer, input_dim=config.hidden_dim)
model = ResampledDecisionModel(resampler, bridge)
# bsr_tokens 是 encoder 的池化前特征；valid 是可选的输入 token mask。
result = model(bsr_tokens, valid, labels=labels)
```

`ResampledDecisionModel` 同时注册压缩器和 JEV bridge，使 `model.parameters()` / `state_dict()` 包含压缩器。正式训练仍须把 `model.resampler.parameters()` 纳入优化器，并随 checkpoint 保存 `config.to_dict()`。如果要联合微调 encoder，还须注册 encoder 并保留特征提取的计算图。现有 `train_joint.py` 使用 LEAF 专用参数组，**不会自动启用这两个组件**；本次没有修改该入口，也没有运行新的 SFT。

压缩后还需要 connector 做 `768 → 5120` 的特征维度映射；32 是 token 数，不是 embedding 维度。1568 → 32 减少了 49 倍的信号 token 数，不代表端到端提速 49 倍；encoder 和文本提示的计算仍然存在。

### 功能验证与限制

```sh
laya/.venv-eeg/bin/python -m pytest tests/test_token_resampler.py -q
laya/.venv-eeg/bin/python scripts/check_token_resamplers.py --out artifacts/token_resamplers/my_check
```

检查脚本默认在 CPU 上对同一份随机 `[2,1568,768]` 输入执行两个模块，保存 `cross_attention.npz`、`qformer.npz` 及带配置/代码哈希的 `summary.json`。`--tokens` 可读取已提取特征的 NPZ，键为 `tokens[B,N,D]` 和可选布尔 `valid[B,N]`；输出目录拒绝覆盖。仅做前向、反向，无优化器更新，不加载 BSR encoder 或 27B JEV。单次耗时包含冷启动，不用作两个架构的速度排名。

本轮只验证形状、mask、梯度、checkpoint 往返和下游接口；不能据此宣称保留了任务信息、实现文本对齐或改善 EEG 准确率。Hugging-science 技能用于区分预训练权重与新模块的验证范围，Markdown 技能用于同步说明。[^4]

[^1]: Official Cross splitter. https://github.com/Dingkun0817/EEG-FM-Benchmark/blob/b36719588abec452e4c2b564f25ede1c0b1c03bc/utils/dataset_split.py
[^2]: Official trainer. https://github.com/Dingkun0817/EEG-FM-Benchmark/blob/b36719588abec452e4c2b564f25ede1c0b1c03bc/utils/trainer.py
[^3]: Official training entry. https://github.com/Dingkun0817/EEG-FM-Benchmark/blob/b36719588abec452e4c2b564f25ede1c0b1c03bc/run_finetuning.py
[^4]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
