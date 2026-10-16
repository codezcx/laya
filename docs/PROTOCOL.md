# 当前协议：单一 EEG 表示对齐任务＋类别文本

## 表示和监督

冻结 REVE-base；复用已经校验的最终层 token 缓存，每 token 512 维。MI 为 12 tokens（C3/Cz/C4），情绪为 120 tokens（30 通道）。有效 token 经过 LayerNorm，再按固定通道/时间顺序补零到 120 tokens；展平后由 Linear(61440,307)、GELU、Dropout(0.1)、Linear(307,1024) 得到 L2 归一化 EEG 表示。此方案依赖固定 montage，不主张任意通道泛化。

文本由冻结 Qwen3-Embedding-0.6B 编码，固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，使用最后一个有效 token 的输出并归一化。类别描述作为 document，不额外加 query instruction。权重下载后校验官方 LFS SHA256，产物保存本地文件哈希。

文本只有任务＋类别，不含数据集名或编造的波形解释。五类为情绪负性/中性/正性，以及左手/右手运动想象，每类 4 条训练措辞、2 条保留措辞；完整内容见 `configs/text_descriptions.json`。保留措辞只做文本检索检查，不参与 EEG 训练或选模。

使用平衡 sigmoid 配对对比损失：同类别的所有文本均为正样本，其余为负样本。分别按每个 EEG anchor、每个文本 anchor 对正负项各占一半求均值，再平均两个方向。温度倒数和偏置可训练。不是一对一 CLIP，也不是旧分类 CE 换名。

推理对五类各自的四条训练文本计算平均余弦相似度，直接取最高类别；任务由该类别推导，不增加任务 head/loss。输入只有 EEG，不提供真实任务提示。当前实验没有 EEG tokens 注入语言 decoder、CoT 或语言生成。

## 必要对照与固定预算

四张 GPU 独立运行：contrastive seeds 42/43、CE seeds 42/43。CE 对照共享完全相同的 EEG 投影结构、初始化方式和采样序列，用可训练归一化五类线性分类器替代文本匹配，优化交叉熵。这个比较检验整套文本对比方案是否优于直接分类；不能单独分离文本语义和损失形式的贡献。

每组固定 2,000 步，AdamW，lr=1e-4、weight_decay=0.01、梯度裁剪 1。每个 batch 从 12 个数据集×类别分组各采样 10 条，共 120 条，MI/情绪各 60 条，包含所有五类及同任务负类。仅训练 LayerNorm、聚合投影器与相似度标定参数（CE 另有分类器），不更新 REVE 或文本编码器。

## 数据与评估

源数据：MI 为 Cho2017、Dreyer2023、Lee2019_MI；情绪为 SEED、FACED。共有 70,259 窗口，57,749 训练、12,510 验证。训练/验证被试隔离，训练入口再检查。200Hz、1–30Hz、3 秒窗；MI 使用共同幅度缩放，情绪逐通道标准化。沿用缓存 `artifacts/multitask_bridge_v30/source_cache`，训练前校验所有 shard 哈希；不因旧目录名复制数据。

每 100 步验证。分别报告 MI、情绪：每个被试计算平衡准确率，数据集内平均被试，再任务内平均数据集。两任务平均仅用于选 checkpoint。主结果为五类联合预测，跨任务误判计错；oracle 已知任务指标仅诊断。

所有模型源验证选模结束后，记录 checkpoint 哈希的 `source_freeze.json`，再评估外部 2b 和 SEED-VII-HNS。外部缓存继承旧提取 provenance，但模型选模证据使用本轮 freeze，不能混用。外部普通准确率、被试平均平衡准确率分别保留。

2b/SEED-VII 已在此前实验中反复查看，且 REVE 预训练包含 2b，因此这是探索性跨数据集实验，不是完全无接触盲测。通道数/token 数可能暴露任务身份，任务路由准确不证明语义理解；已知标签跨数据集泛化也不等于未见任务/类别 zero-shot。

## 产物与版本

文本诊断：`artifacts/eeg_text_alignment/embedding_bank/complete.json`。
训练和自动外部评估：`artifacts/eeg_text_alignment/runs/`。
当前源码在启动前 commit；训练配置记录 Git commit、代码哈希、源 manifest 和文本 bank 哈希。旧方案源码、配置只留 Git 历史，忽略目录内数据和实验产物保留。
