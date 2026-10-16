# 当前协议：真实 JEV-27B-VL 的 EEG 接入与微调

## 模型与始终启用的 JEV LoRA

本地 `laya/JEV-27B-VL`，配置为 Qwen3.5、语言隐藏维度 5120。只实例化语言 decoder，按 checkpoint index 严格检查并加载全部语言参数；不加载视觉 encoder、完整 lm_head 或 MTP。随后加载 `adapter_vllm/adapter_model.safetensors` 中全部语言 LoRA A/B，使用官方 rank32、alpha64，原有 adapter 权重作为初始化，**所有实验/训练/推理路径始终启用**。

从原始 lm_head 提取 A–E 五行，加上对应的已训练 lm_head LoRA 增量，恢复 JEV 有效决策权重，并加 decision_head.json 中对应 bias；该有效头固定。只计算五类输出，无自回归生成，也不启动 System2。官方通用校准温度不用于此 EEG 训练，输出准确率不受共同温度影响，不宣称概率已校准。

冻结完整语言基座。LoRA 可训练组更新现有语言 LoRA A/B，不是另加一个随机 adapter，也不关闭原 adapter。载入时核对所有 adapter 键均被语言层或决策头消费。

## 信号接入

REVE-base 输出每个 512 维 token。投影器为 LayerNorm(512)→Linear(512,5120)→GELU→Linear(5120,5120)。插入形式：prefix embeddings、有效 EEG embeddings、suffix embeddings；仅在整个序列尾部 padding，重新计算 attention mask、position IDs 和最后决策位置。

prefix/suffix 使用官方 bare-v1 的 `[kind] choice / [state] / [question] / [options] / [decision]:` 格式，具体内容在 `eeg_align/jev.py`。五个候选包含任务＋类别：情绪负性/中性/正性、左手/右手运动想象。不输入正确任务提示或标签。训练用五类交叉熵，推理用 JEV A–E logits argmax。

此前对比学习的投影器输出图级 1024 维文本检索向量，与此处逐 token 的 5120 维接口不同，因此本轮初始化新的投影器，没有声称继承那一投影器的训练成果。文本 embedding 模型不参与本轮。

## 四卡覆盖训练设置

两组均 seed42：GPU0/1只更新投影器，GPU2/3更新投影器和已有JEV语言LoRA。REVE均冻结，复用缓存。原始LoRA始终生效，原始文本模板和投影器结构不变；两组从相同随机投影器初始化，不载入小集诊断权重。

每组两卡同步梯度，全局batch48：12个数据集×类别分组各4条；每卡24条，microbatch6。组内无放回遍历、耗尽再打乱；小组重复采样以保持分组权重。每覆盖轮2190步、105120次抽样，覆盖全部57749条源训练窗口；固定两轮4380步。前50步只更新投影器，随后LoRA组更新已有adapter。

AdamW，weight decay0.01，clip1，投影器lr1e-4、LoRA lr1e-5，恒定学习率。基础语言权重bf16，投影器和LoRA fp32，启用梯度检查点。每100步及轮末原子保存可训练参数、optimizer、采样排列/游标、每rank RNG，恢复时核对配置和实现哈希。使用torchrun，每rank持有完整模型并对可训练梯度求均值。

## 数据、选模与目标集

源数据复用此前按被试隔离的 57,749 训练、12,510 验证窗：Cho2017、Dreyer2023、Lee2019_MI、SEED、FACED。MI C3/Cz/C4，情绪共同30通道；200Hz、1–30Hz、3秒。MI 共同幅度缩放，情绪逐通道标准化。冻结 encoder 用12/120个实际 REVE tokens；原始信号缓存由相同预处理生成，并检查覆盖和哈希。解冻 encoder 输入实测通道数量选择电极位置，不接收类别标签。

每覆盖轮末在完整12510条源验证窗上评估。每个被试计算平衡准确率，数据集内平均被试，再任务内平均数据集。分开报告MI和情绪；两任务均值仅用于在两个轮末checkpoint之间选模，不再使用244窗panel选模。

两组完成后统一冻结最佳checkpoint哈希，再评估外部2b和SEED-VII-HNS。外部报告普通准确率及被试平均平衡准确率；五类联合判断，跨任务错误照计。外部数据不参与本轮梯度更新或checkpoint选择。

2b/SEED-VII 此前已反复查看，且 REVE 预训练包含2b；这是探索性跨数据集实验，不是全链路零接触盲测。不同通道数/token数可能暴露任务身份，不把任务路由正确当作语义理解。

## 检查与版本

实机检查真实 JEV 前向、反向、冻结基座无梯度、LoRA和投影器有梯度；检查原始信号 REVE 特征与旧缓存的一致性及 encoder 反向。单元检查涵盖冻结 adapter 后输出不变（证明仍启用）、LoRA 与合并矩阵等价、EEG拼接/决策位置和指标。

checkpoint 文件哈希记录 `artifacts/jev_eeg/checkpoint_audit.json`。训练配置保存 checkpoint audit、数据 manifest 和代码哈希及 Git commit；保存可训练参数及完整优化器/采样/RNG恢复状态。源码在启动前提交。旧方案在 Git `0e6c506`，不在工作区堆积旧实现。

## 已完成：源训练集上的接口可学习性诊断

上一轮500步正式训练和外部评估已完成，结果未达目标。特征探针显示REVE在源留出被试上可解码左右手，下一步先检查语言接口是否能利用这些特征，不重复扩大失败配方的预算。

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

## 启动和恢复

`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py` 启动两组，并在两组完成后冻结及评估。产物在 `artifacts/jev_eeg_coverage/runs/`。目录已存在时拒绝覆盖。

单组恢复须先确认旧进程终止，使用相同GPU数及参数，通过torchrun运行 `eeg_align.train_jev --resume`。不因观察超时自动重启。恢复受模型、数据manifest、代码哈希和采样配置检查约束。

正式训练前进行真实两卡LoRA连续/中断恢复对照；smoke只有两步和12条源验证样本，禁止用于外部评估。测试记录在 `artifacts/jev_eeg_coverage/smoke/`。历史500步实现归档于Git `b5a7fd4`；旧数据、权重和日志保留。
