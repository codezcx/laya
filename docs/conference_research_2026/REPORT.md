# EEG 可信性选题：现实证据、问题定义与新颖性审查

更新：2026-10-25。状态：**研究继续，尚未通过立项的新颖性门槛。** 不训练基模、不做 VLM，不先改实验代码。旧“边界不确定性”为优先方向的推荐已撤回，相关草案与数学例子存于 Git 提交 fae0972，不再作为当前研究路线。

## 1. 先纠正任务来源

用户对当前运动想象任务的判断成立：程序按实验事件标记裁窗，标签来自实验要求，不需要逐个估计人工事件边界。

- 本地 2a 构建器调用 MOABB MotorImagery，读取事件类别并截取 1000 个采样点；当前 raw 2a 控制按 MAT 中的事件索引截取 1500 点，再按固定偏移保留数据。代码证据见 [local_code_evidence.json](sources/eeg_trust/local_code_evidence.json)。这些代码没有估计“不确定边界”。
- BCI IV-2a 原始协议提供 trial start、各类别 cue onset 事件码，明确记录实验时间线。[原始说明](https://www.bbci.de/competition/iv/desc_2a.pdf)
- 必须区分“何时展示指令”“实际神经反应何时出现”“程序从哪里裁窗”。前两者不相等，不代表当前监督目标就是神经反应起始时间。
- 不应进一步概括为所有 EEG 标签都由计算机自动生成。本地 CHB-MIT 构建器先读取发作时间，再对发作段与附近非发作段分别裁窗；自动裁窗仍依赖上游事件标注。但这不构成把标注不确定性移植到 MI 任务的理由。

撤回旧方向的理由是**缺少与目标任务对应的观测机制和现实效应证据**，不是已经证明任何医学标注都没有分歧。

## 2. “换数据集优势就变了”有直接证据，但需要准确解释

MOABB 官方公布的左右手运动想象结果，在相同 within-session 五折评估类型下有以下均值变化：

| 数据集 | ShallowConvNet | CSP+LDA | 前者减后者 |
|---|---:|---:|---:|
| BNCI2014_001 | 86.17 ± 13.74 | 82.34 ± 17.26 | +3.83 |
| BNCI2014_004 | 72.36 ± 18.05 | 80.10 ± 14.93 | −7.74 |

指标为 **ROC-AUC × 100，不是 accuracy**；± 是跨被试/会话的标准差，不是置信区间。这是算法管线在各数据集内训练后的均值排序反转，**不是同一个冻结 checkpoint 从一个数据集直接迁移到另一个**。这些汇总值足以确认描述性排名反转，不能独立证明差异显著或所有 EEG 模型不可信。[官方结果](https://moabb.neurotechx.com/docs/paper_results.html)；[提取与核算](sources/eeg_trust/ranking_evidence.json)。

要讨论可信性，必须先指定承诺：

| 承诺 | 应检验的对象 | 不能替代它的证据 |
|---|---|---|
| 新被试仍有用 | 新被试风险及差异分布 | 训练集准确率、随机窗口划分 |
| 到新中心仍有优势 | 同一任务、固定迁移规则下的跨中心比较 | 每个中心重新调参后的排行榜 |
| 概率可靠 | 声明部署分布上的校准/覆盖/选择风险 | softmax 很高、总体准确率高 |
| 真在利用目标神经活动 | 与任务机制对应的混杂控制/干预 | 跨被试准确率、单张注意力图 |
| 可在给定时间预测 | 截至决策时刻实际可用的信息 | 使用整段目标记录统计量的离线分数 |

端到端也不是“不可信”的充分条件；CSP、LDA、Riemannian 等经典管线同样可能失效，且它们仍是应包含的有竞争力对照。

## 3. 已经核实的机制

**预处理影响分数，且高分可能来自非目标成分。** 2025 年 How EEG preprocessing shapes decoding performance 系统改变滤波、参考、基线与伪迹处理，比较 EEGNet 与时点逻辑回归。原文讨论伪迹可提升分类分数、却削弱神经解释。因此“预处理改变准确率”或“更高分未必更有生理意义”已有直接先行工作。[原始论文](https://doi.org/10.1038/s42003-025-08464-3)

本地既有记录显示冻结编码器的 2a source-CV-selected probe 验证准确率由 29.51% 变为 36.11%，但干预同时改变 CAR、滤波、滤波上下文、归一化池等；验证被试已反复查看。这是**需要拆解的混合干预**，不能归因于一个步骤，也不是跨数据集泛化证据。见 [RESULTS](../RESULTS.md)。

**程序准确裁窗，仍可能保留提示相关的捷径。** Latent Alignment 对 PhysioNet 运动任务分析了 cue 后早期片段及眼动相关信号。2026 年 Visual cues in MI-BCI induce biases in EEG classification models with deep learning but not with standard machine learning 直接研究 cue 片段对解码与特征的影响。因此自动裁窗不会自动排除混杂，但该现象已经被研究。[Latent Alignment](https://doi.org/10.1088/1741-2552/adb336)、[2026 原始论文](https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1804143/full)

视觉诱发活动本身也是脑活动。问题是它是否对应声称要解码的运动意图，以及部署时提示是否仍存在。去掉早期片段同时改变时长与神经信息，不能当作完美的去混杂干预。

**某些预测器依赖目标试次集合。** 本地 LEAF shared.pipeline 对形状为 (B,C,T) 的数组计算不指定 axis 的百分位数和 median/IQR；统计量由整批试次共同决定。当前 matched control 也明确声明按被试整批无标签 T-session 处理。这是已声明的 transductive 约定，不应指控为隐藏标签泄漏。它的预测形式应写成 F(x; C)：当前试次 x 加统计对齐上下文 C。

Latent Alignment 已分析上下文类别比例敏感性；2026 OSPDIM 进一步研究类别不平衡下的在线对齐。因此“预测依赖同批其他样本”“不平衡破坏对齐”“改成在线处理”均不能单独充当新问题。[Latent Alignment](https://doi.org/10.1088/1741-2552/adb336)、[OSPDIM 预印本](https://arxiv.org/abs/2608.05315)

## 4. 最近工作对新颖性的约束

| 想法 | 直接相关工作 | 不能重复宣称的贡献 |
|---|---|---|
| 比较跨数据集表现 | MOABB；ICLR 2026 Are EEG Foundation Models Worth It? | 排名有差异；基模未必胜过小模型 |
| 标准化任务定义 | NeurIPS 2026 EEG Benchmarking Needs a Task Specification Layer | 首次提出 EEG task specification 或可执行规则层 |
| 诊断迁移为什么不好 | ICML 2026 EEG-FM-Bench | 单靠统一评估、linear probing 与梯度分析构成新框架 |
| 模型编码、使用什么 | NeurIPS 2026 What do EEG Foundation Models Capture from Human Brain Signals? | 只用特征 probing 和 erasure 就首次揭示 EEG 表征 |
| EEG 内部概念解释 | NeurIPS 2026 Mechanistic Interpretability … via Sparse Autoencoders | 加 SAE、steering、年龄/病理纠缠审计即新问题 |
| 选择新用户模型 | ThinkNet 2026 作者版本 | 首次发现源验证排序不能可靠选出新被试最佳模型 |
| 对齐上下文与类别比例 | Latent Alignment 2025；OSPDIM 2026 | 无标签 context 有代价；类比例改变导致对齐失效 |
| EEG 加 conformal | 2018 seizure CP；2026 Making Conformal Predictors Robust… | 首次研究 EEG coverage 或个体化校准 |
| 适应模型的置信度 | CUI 2025；ECP/EACP 2025；2026 TTA coverage 修复预印本 | 泛泛提出 TTA 破坏校准，再做重校准 |
| 连续发作检测 | SzCORE 2024/2025 | 首次要求事件指标、完整背景时间、跨中心验证 |

官方会议来源：[ICLR 比较](https://iclr.cc/virtual/2026/poster/10011450)、[NeuroDoc](https://neurips.cc/virtual/2026/poster/139164)、[ICML benchmark](https://icml.cc/virtual/2026/poster/60932)、[特征审计](https://neurips.cc/virtual/2026/poster/150770)、[SAE](https://neurips.cc/virtual/2026/poster/150909)。目前主要按官方摘要核验；不能把摘要的 representation-causal 扩张为真实神经机制的因果证明，也不能据此指控作者做了这种扩张。

其他来源：[ThinkNet](https://arxiv.org/abs/2609.33967)、[EEG conformal](https://arxiv.org/abs/2602.19483)、[CUI](https://arxiv.org/abs/2502.02998)、[ECP/EACP](https://proceedings.mlr.press/v286/kasa25a.html)、[TTA coverage 预印本](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7602585)、[SzCORE](https://arxiv.org/abs/2402.13005)。arXiv 中的接收说明仅为作者报告，未独立核验 BHI/AIME 会议列表。

## 5. 后续查重改变了立项判断

先前“有限上下文下的新用户受损率认证”普通版本现已排除为新贡献。以用户评价包为独立样本，以受损指示量为损失，预算/管线为选择参数，它可直接落入 Learn then Test。无标签目标前缀与用于认证的独立有标签用户是两类资源，不能混淆。完整归约及条件见 [PROBLEM_DEFINITIONS.md](PROBLEM_DEFINITIONS.md)。[LTT 原文](https://arxiv.org/html/2110.01052)

另一条看似自然的路线——输入管线影响 EEG 基模比较——也存在直接先行工作。2026 年 Matched-Input Estimates Differ in Sign Across Architectures 已做多架构匹配输入分析，并限制其统计结论。只复现这种比较不足以立项。[原文](https://arxiv.org/html/2609.23924)

EEG-Arena 的原始摘要报告覆盖 30 个 EEG 基模、25 个监督基线、57 个任务，并考察预训练和通道配置。因此，泛泛扩大排行榜也不是清楚的新颖性缺口。这里仅核对摘要，未完成其 91 页全文审阅；HTML 请求因长度限制失败，不把失败当全文阅读。[原始摘要](https://arxiv.org/abs/2609.32743)

## 6. 当前问题定义与剩余工作

当前可确认的研究对象是：**固定任务与可用信息条件下，源数据上选出的 EEG 管线对未见用户/中心的相对收益，以及使该收益消失或反转的可观测机制。** 这是已有问题的精确定义，不声称问题本身首创。

现实性证据与若干新颖性排除已取得；有足够新意的机制/方法尚未确定。下一项有价值的证据是从本地已观测的 bundled preprocessing 差异中识别具体机制，随后与直接先行工作比较。仍未运行“上下文预算导致排名反转”的本地受控实验，不把它写成已证实。

旧边界草案存于 fae0972，已排除的受损率候选草案存于 a3b0280；当前不保留多套实验实现。未修改训练代码。原始 54 篇会议条目仍作为背景索引保留于 [PAPERS.md](PAPERS.md)。

流程使用 scientific-critical-thinking 技能，流程参考已记入：[Kassis et al., Scientific Agent Skills (2026)](https://doi.org/10.48550/arXiv.2609.00065)。工具流程不是科研结论证据。
