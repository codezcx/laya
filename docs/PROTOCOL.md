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

## 当前下一步：源训练集上的接口可学习性诊断

上述500步正式训练和外部评估已完成，结果未达目标。特征探针显示REVE在源留出被试上可解码左右手，下一步先检查语言接口是否能利用这些特征，不重复扩大失败配方的预算。

在五个源数据集的12个数据集×类别组中各固定抽取5个训练窗口，共60个；seed20261017，记录具体索引和manifest哈希。四卡采用相同初始化seed42、相同样本、相同轮换顺序，每步各组取一个窗口（batch12），200步；只训练投影器，JEV原始LoRA始终启用但冻结。REVE冻结，学习率1e-4、AdamW、weight decay0.01、clip1。每25步评估全部60条**训练样本**。

四个条件形成2×2诊断：

| 条件 | 文本/EEG顺序 | 投影输出 |
|---|---|---|
| original | 原模板，EEG后接完整题目及选项 | 原MLP |
| near | 题目及选项移到EEG前，EEG后只保留结束边界和decision | 原MLP |
| scaled | 原模板 | 每个token方向保留，RMS匹配该模板冻结文本embedding的参考RMS |
| near_scaled | 题目选项在前、EEG靠近decision | 同上RMS匹配 |

移动位置同时改变了提示内容的顺序，所以不能单凭这个对照将收益严格归因于距离或线性attention记忆。所有条件保留同样问题、五个任务＋类别选项、原始JEV有效决策头；不更换类别语义，不提供正确任务提示。

记录初始化投影RMS、参考文本RMS、batch CE、梯度范数、完整训练小集的MI/情绪平衡准确率。最后追加两个干预：同任务内置换窗口与标签的对应关系；用每个任务的平均tokens替换各窗口。二者保留任务及token数、去除或破坏具体trial信息。检查预测是否依赖个体窗口，而不只依赖任务或固定答案偏好。

两任务各自达到95%训练小集准确率只是接口拟合门槛，**不是zero-shot成功标准，也不是泛化证据**。任何改进须随后在完整源训练、源留出及外部测试中验证；不从这60条的成绩直接推断性能。当前诊断不读取2b/SEED-VII，不使用验证/目标集选条件。

入口 `scripts/run_interface_diagnostic.py`，实现 `eeg_align/interface_diagnostic.py`；产物 `artifacts/jev_interface_diagnostic/`。这是当前JEV方案的必要故障诊断，不另保留历史训练实现。

只读核算命令：`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/report_interface_diagnostic.py`。核验源索引仅来自训练分区、分组计数、代码与manifest哈希，并在每组完成后从保存logits独立重算CE、MI/情绪指标和预测分布；未完成组不排名或提前选条件。

## 后续正式训练的数据覆盖准备（尚未启动）

`eeg_align/sampling.py` 提供按分组覆盖轮定义预算的采样器，以修复旧500步配方实际仅覆盖约9.56%训练窗口的问题。每组内部按随机排列取样、耗尽才重新排列；小组会重复采样以维持数据集/类别权重。一个覆盖轮结束前所有训练窗口至少出现一次，不能把“有放回抽样次数/数据集大小”误称实际完整epoch。

该采样器已通过单元检查和全部源训练索引干跑；尚未改动或接入正在运行的JEV接口诊断。正式训练的接口和预算须在200步诊断及输入干预完成后确定；完整源验证应与244窗训练诊断面板区分，外部集仍只在源选模冻结后评估。

覆盖采样器现支持保存/恢复轮次、组内排列、游标和随机数状态，并校验训练索引签名与batch设置。6项测试包含跨覆盖轮恢复与未中断运行逐批完全一致、拒绝分区变化和损坏状态；这只是后续正式训练的采样恢复支持，尚不表示旧500步训练器已支持完整optimizer恢复。
