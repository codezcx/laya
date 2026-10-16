# 当前协议：真实 JEV-27B-VL 的 EEG 接入与微调

## 模型与始终启用的 JEV LoRA

本地 `laya/JEV-27B-VL`，配置为 Qwen3.5、语言隐藏维度 5120。只实例化语言 decoder，按 checkpoint index 严格检查并加载全部语言参数；不加载视觉 encoder、完整 lm_head 或 MTP。随后加载 `adapter_vllm/adapter_model.safetensors` 中全部语言 LoRA A/B，使用官方 rank32、alpha64，原有 adapter 权重作为初始化，**所有实验/训练/推理路径始终启用**。

从原始 lm_head 提取 A–E 五行，加上对应的已训练 lm_head LoRA 增量，恢复 JEV 有效决策权重，并加 decision_head.json 中对应 bias；该有效头固定。只计算五类输出，无自回归生成，也不启动 System2。官方通用校准温度不用于此 EEG 训练，输出准确率不受共同温度影响，不宣称概率已校准。

冻结完整语言基座。LoRA 可训练组更新现有语言 LoRA A/B，不是另加一个随机 adapter，也不关闭原 adapter。载入时核对所有 adapter 键均被语言层或决策头消费。

## 信号接入

REVE-base 输出每个 512 维 token。投影器为 LayerNorm(512)→Linear(512,5120)→GELU→Linear(5120,5120)。插入形式：prefix embeddings、有效 EEG embeddings、suffix embeddings；仅在整个序列尾部 padding，重新计算 attention mask、position IDs 和最后决策位置。

prefix/suffix 使用官方 bare-v1 的 `[kind] choice / [state] / [question] / [options] / [decision]:` 格式，具体内容在 `eeg_align/jev.py`。五个候选包含任务＋类别：情绪负性/中性/正性、左手/右手运动想象。不输入正确任务提示或标签。训练用五类交叉熵，推理用 JEV A–E logits argmax。

此前对比学习的投影器输出图级 1024 维文本检索向量，与此处逐 token 的 5120 维接口不同，因此本轮初始化新的投影器，没有声称继承那一投影器的训练成果。文本 embedding 模型不参与本轮。

## 四卡训练设置

| GPU | 设置 | seed | JEV LoRA 是否生效 | 更新参数 |
|---|---|---:|---|---|
| 0 | projector | 42 | 始终生效 | 投影器 |
| 1 | lora | 42 | 始终生效 | 投影器＋现有语言 LoRA |
| 2 | lora | 43 | 始终生效 | 投影器＋现有语言 LoRA |
| 3 | lora_encoder | 42 | 始终生效 | 投影器＋现有语言 LoRA＋REVE |

每组 500 optimizer steps，前 50 步只更新新投影器，预热期间 adapter 仍参与 forward。预热统一复用冻结 REVE fp16 token 缓存，之后解冻组改为从原始 EEG 在线编码。其他组继续使用缓存。REVE 即使更新权重也使用 eval 模式，关闭随机训练行为。

batch=12，12 个数据集×类别分组各采样一个，任务各占一半，microbatch=6、累积两次。AdamW，weight decay0.01，梯度裁剪1；投影器 lr1e-4、LoRA lr1e-5、REVE lr1e-6。语言权重 bf16、LoRA/投影器/REVE fp32。语言层启用梯度检查点；保持梯度能穿过冻结 decoder 到投影器。

只训练投影器的对照保留 JEV 能力，不能称为“关闭 JEV”。主要比较 lora seed42 与 lora_encoder seed42；第二个 lora seed 用于观察随机性，但不足以支撑普遍显著性结论。

## 数据、选模与目标集

源数据复用此前按被试隔离的 57,749 训练、12,510 验证窗：Cho2017、Dreyer2023、Lee2019_MI、SEED、FACED。MI C3/Cz/C4，情绪共同30通道；200Hz、1–30Hz、3秒。MI 共同幅度缩放，情绪逐通道标准化。冻结 encoder 用12/120个实际 REVE tokens；原始信号缓存由相同预处理生成，并检查覆盖和哈希。解冻 encoder 输入实测通道数量选择电极位置，不接收类别标签。

考虑 27B 全验证成本，每100步使用一个固定源验证 panel：每个数据集×被试×类别最多2窗，共244窗，固定seed20261016；索引保存到配置。每个被试计算平衡准确率，数据集内平均被试，再任务内平均数据集。分开报告 MI 和情绪，两任务均值仅选模。此 panel 分数不得与此前全验证分数直接混称。

所有组源选模完成后，写入 `source_freeze.json` 锁定 checkpoint。再对每组最佳模型计算完整源验证集与外部2b、SEED-VII-HNS，不根据这些结果重新选模。REVE 解冻组在外部集从原始信号重编码，其余组复用同序 token 缓存。外部报告普通准确率和被试平均平衡准确率，五类联合判断跨任务错误照计。

2b/SEED-VII 此前已反复查看，且 REVE 预训练包含2b；这是探索性跨数据集实验，不是全链路零接触盲测。不同通道数/token数可能暴露任务身份，不把任务路由正确当作语义理解。

## 检查与版本

实机检查真实 JEV 前向、反向、冻结基座无梯度、LoRA和投影器有梯度；检查原始信号 REVE 特征与旧缓存的一致性及 encoder 反向。单元检查涵盖冻结 adapter 后输出不变（证明仍启用）、LoRA 与合并矩阵等价、EEG拼接/决策位置和指标。

checkpoint 文件哈希记录 `artifacts/jev_eeg/checkpoint_audit.json`。训练配置保存 checkpoint audit、数据 manifest 和代码哈希及 Git commit；只保存可训练参数和解冻 encoder。源码在启动前提交。旧方案在 Git `0e6c506`，不在工作区堆积旧实现。
