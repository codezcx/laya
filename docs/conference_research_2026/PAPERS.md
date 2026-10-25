# 定向论文表

本表保留为首轮会议文献背景索引；旧边界课题推荐已撤回。当前 EEG 可信性核查、直接先行工作与候选状态以 [REPORT.md](REPORT.md) 和 [PROBLEM_DEFINITIONS.md](PROBLEM_DEFINITIONS.md) 为准。

截至 2026-10-25。下列 54 篇均核对到官方会议 poster 页面或正式论文集；未逐项核验主会/数据等细分 track。摘要反映作者主张，不能代替复现。D 表示定向原文检查，并不表示证明逐行验证；A 表示摘要。代码链接仅核验页面是否列出，未安装或运行。

## NEURIPS 2026（18 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [ECG Dataset with Multi-Expert Annotations and Delineations](https://neurips.cc/virtual/2026/poster/138906) | 多专家 ECG 分类与波形分界数据 | 新数据入口；核验双专家是否保留独立边界，不能把核验等同重复标注 | A |
| [Benchmarking Sensor-Fault Robustness in Forecasting](https://neurips.cc/virtual/2026/poster/138912) | 以统一传感故障机制评估预测模型 | 干净误差与故障鲁棒性不同；单做 dropout stress test 已不足 | A |
| [How Complete Should a Reference Be? A Benchmark Audit for Fluorescence Spot Detection](https://neurips.cc/virtual/2026/poster/139218) | 荧光点检测中参考完整度与定义的审计 | 不同 reference 可改变 top-1；A 必须超越少数 reference 的敏感性分析 | A |
| [Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty](https://neurips.cc/virtual/2026/poster/139282) | 目标检测参考漏标与不确定性审计 | 摘要称分数敏感但总体排名大致稳定；反驳“标注变化必然翻排名” | A |
| [Automated Hypothesis Discovery for Characterizing Annotation Disagreement](https://neurips.cc/virtual/2026/poster/139723) | 生成并在独立数据检验标注分歧解释 | 分歧不应一律当噪声；该方法使用自动假设生成，非本次主路线 | A |
| [Decide Before You Record: Heterogeneity-Guided Pre-Acquisition Stimulus Selection for Cross-Day Neuroprostheses](https://neurips.cc/virtual/2026/poster/149170) | 在当天神经记录前选择校准刺激 | 采集前决策已有明确课题；数据访问门槛较高，不当低成本默认 | A |
| [Counterfactual Online Conformal Prediction Under Adaptive Logging](https://neurips.cc/virtual/2026/poster/150109) | 预测影响动作和标签可见性时的在线校准 | 反事实覆盖、IPW/DR 与 positivity 已被直接研究 | A |
| [Announced Breaks Separate Conformal Reliability from Frequency Calibration](https://neurips.cc/virtual/2026/poster/151358) | 已知制度切换时刻下的两种覆盖目标 | 概率与实现频率不可混同；摘要包含真实数据无收益结果 | A |
| [ObsConDA: Observability-Constrained Data Assimilation with Control-Space Inference](https://neurips.cc/virtual/2026/poster/152077) | 随稀疏观测几何变化限制数据同化修正空间 | 可支持方向本身是已有方法对象；C 不能仅重复该原则 | A |
| [Structural Blindness in Latent Data Assimilation: Representation Geometry Misleads Sensor Design](https://neurips.cc/virtual/2026/poster/152860) | 潜空间几何诊断对真实传感器效用的盲点 | 传感器选择不能只看 latent 指标；完整假设因全文受限未审计 | A |
| [No Free Alignment: Observability-Aware Alignment for Multimodal Heterogeneous Learning](https://neurips.cc/virtual/2026/poster/153751) | 跨模态对齐的关系级可观测支持 | “不要对齐不受支持的关系”已有明确框架，非新口号 | A |
| [When a Window Is Not an Action: Selective Phase-Script Deliberation for Sliding-Window Human Activity Recognition](https://neurips.cc/virtual/2026/poster/154058) | 活动语义与局部滑窗证据粒度不匹配 | 提出 phase-script 网络；我们不能重复宣称首次发现窗口粒度问题 | A |
| [Hidden Tails: Certifying Tail-Risk Claims under Selective Labels](https://neurips.cc/virtual/2026/poster/154165) | 选择性标签下 CVaR 非识别、sharp 界与认证成本 | 直接排除通用“隐藏标签尾部风险认证”课题 | A |
| [Anytime-Valid Conformal Risk Control](https://neurips.cc/virtual/2026/poster/154243) | 增长校准集上的 anytime 高概率风险控制 | 不同于固定样本平均保证；仅加入 anytime 名称不构成新颖点 | A |
| [Posterior Alternative Calibration in Ambiguous Inverse Problems](https://neurips.cc/virtual/2026/poster/154341) | 模糊逆问题的候选模式概率质量分配 | 重建一致不代表 alternatives 概率正确；本次不做生成器训练 | A |
| [Beyond Imputation: Mask-Adaptive Conformal Prediction via Tree Embeddings on General Missing Data Mechanisms](https://neurips.cc/virtual/2026/poster/154427) | 无插补的缺失模式自适应 conformal | 缺失掩码条件保证已有直接工作；不同有效性级别需要分清 | A |
| [Budgeted Multi-Source Counterfactual Annotation for Off-Policy Evaluation](https://neurips.cc/virtual/2026/poster/155643) | 有成本和偏差的多来源反事实标注分配 | 预算优化已涉及来源误差；不能假定所有额外标签都是真值 | A |
| [Distribution Shift in Missing Data Imputation: A Risk-Based Perspective and Importance-Weighted Correction under MAR](https://neurips.cc/virtual/2026/poster/155709) | MAR 下插补训练分布与全数据风险不匹配 | 需说明哪一个风险；权重修正已是直接基线 | A |

## ECCV 2026（11 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [Hybrid Event–Frame Sensors: Modeling, Calibration, and Simulation](https://eccv.ecva.net/virtual/2026/poster/3755) | 混合事件/帧传感器的统计噪声建模与仿真 | 物理测量机制可成为贡献；需要真实硬件校准，低算力不等于低成本 | A |
| [Rethinking Detection Calibration: A Coordinate Perspective](https://eccv.ecva.net/virtual/2026/poster/4330) | 检测置信度从 box 层面到坐标方向层面 | 重新定义校准对象的实例；不能只把 box 分解当新问题 | A |
| [Unsupervised Source-Free Ranking of Biomedical Segmentation Models Under Distribution Shift](https://eccv.ecva.net/virtual/2026/poster/4464) | 仅用扰动一致性做无标签无源分割模型排序 | 原文有任务不匹配限制；经验相关不等于普适认证 | D |
| [Why Can Accurate Models Be Learned from Inaccurate Annotations?](https://eccv.ecva.net/virtual/2026/poster/4660) | 不准确标签下权重主子空间稳定性 | 有理论+轻量方法，但本次仅核验摘要，未审核定理假设 | A |
| [FoundDP: Revisiting Weak Disparity Observability in Dual-Pixel Depth Estimation](https://eccv.ecva.net/virtual/2026/poster/4903) | 弱视差可观测性下的双像素深度 | 依赖深度基模先验，不作为符合约束的主候选 | A |
| [Rolling Shutter Camera Self-Calibration](https://eccv.ecva.net/virtual/2026/poster/4929) | 从图像序列估计滚动快门内参与读出时间 | 测量时间/几何假设可变成任务；工程验证门槛较高 | A |
| [Mode-Conditioned Residual Calibration for Multi-Object Tracking](https://eccv.ecva.net/virtual/2026/poster/5156) | 跟踪中的模式依赖关联误差校准 | 轻量修正可有效，但偏方法增量，不优先于问题定义 | A |
| [Robustness Meets Uncertainty: Evidential Adversarial Training for Robust Selective Classification](https://eccv.ecva.net/virtual/2026/poster/5235) | 对抗准确率与选择性预测可靠性的联合评测 | 已有 robustness–uncertainty 课题；新的拒判任务要避开简单重合 | A |
| [Coordinate Singularities Break Conformal Coverage for Gaze and Head Pose](https://eccv.ecva.net/virtual/2026/poster/5387) | 坐标奇异性造成 slice 覆盖畸变 | 原文局部高斯近似、exchangeability 和异方差限制；不是任意条件覆盖定理 | D |
| [Multi-Hypothesis Test-Time Adaptation to Mitigate Underspecification](https://eccv.ecva.net/virtual/2026/poster/5455) | 无标签熵最小化的多个欠确定适应轨迹 | 多假设 TTA 已有框架；PDF 提取但本次未定向审核其全文理论 | A |
| [Causal Yet Future-Aware: Dual-Path Temporal Modeling for Online Action Segmentation](https://eccv.ecva.net/virtual/2026/poster/5567) | 在线动作分段通过蒸馏预测未来依赖 | 在线推断不可偷看未来；主贡献为模型学习，供 A 的任务基线参考 | A |

## ICML 2026（9 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [Conditional Coverage Diagnostics for Conformal Prediction](https://icml.cc/virtual/2026/poster/60890) | 把条件覆盖诊断转为分类 excess risk | 仅画分组覆盖图已弱；ERT 应作为校准研究的已有诊断 | A |
| [Conformal Calibration Transfer](https://icml.cc/virtual/2026/poster/61667) | 有标签源空间到无标签配对目标空间的校准迁移 | 不能把换传感器且无目标标签当成未定义的问题 | A |
| [Conformal Prediction for Early Stopping in Mixed Integer Optimization](https://icml.cc/virtual/2026/poster/61715) | 在混合整数优化中用学习与校准决定提前停机 | 决策成本明确的例子；完整 stopping/coverage 假设未核验 | A |
| [LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](https://icml.cc/virtual/2026/poster/63016) | 用线性期望约束控制选择后风险 | 主要动机含基模；作为 selection-conditioned 风险的邻接工作 | A |
| [Learning-To-Measure: In-Context Active Feature Acquisition](https://icml.cc/virtual/2026/poster/63119) | 在历史缺失和少标签条件下主动特征采集 | 涉及预训练，本次不做；作为“测什么”的直接问题参照 | A |
| [Design-Based Anytime-Valid Inference for Randomized Experiments with Delayed Outcomes and Staggered Entry](https://icml.cc/virtual/2026/poster/65233) | 延迟结果和错峰进入实验的 design-based anytime 推断 | filtration 是关键；不能只套标准 martingale 宣称延迟保证 | A |
| [Coverage ≠ Exposure: Auditable Control of Same-Support Tail Failures under Multimodal Missingness](https://icml.cc/virtual/2026/poster/65812) | 相同支持和平均表现下的尾部参数曝光不均 | missingness 的平均性能不能代表尾部；已有训练控制框架 | A |
| [Learning Treatment Allocations with Risk Control Under Partial Identifiability](https://icml.cc/virtual/2026/poster/65829) | 部分识别下学习并认证治疗分配风险 | 说明部分识别可以连接实际决策；不在本次提出临床部署 | A |
| [A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data Analysis](https://icml.cc/virtual/2026/poster/65937) | 区间删失下严格 proper score 与校准 | 机制假设重要；A 应区分 reference 集合与删失分布拟合 | A |

## ICLR 2026（6 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [Robust Decision-Making with Partially Calibrated Forecasters](https://iclr.cc/virtual/2026/poster/10007595) | 只有部分校准保证时的 minimax 决策 | 校准好不一定对所有效用有用；决策相关保证已有理论 | A |
| [ICYM$^2$I: The illusion of multimodal informativeness under missingness](https://iclr.cc/virtual/2026/poster/10007902) | missingness shift 造成模态信息增益估计偏差 | E 的直接先行工作；不能仅比较完整与缺失集平均准确率 | A |
| [Multi-Condition Conformal Selection](https://iclr.cc/virtual/2026/poster/10008124) | 多条件筛选的 conformal FDR 控制 | 条件组合并非简单套一个阈值；常规扩展已有工作 | A |
| [Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding](https://iclr.cc/virtual/2026/poster/10009162) | EEG 运动想象的无梯度 Bayesian TTA | 有现成小模型路线；作为 EEG 方法基线，不是问题定义候选 | A |
| [Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](https://iclr.cc/virtual/2026/poster/10009505) | 标签只在预测集合包含真值时显露的在线覆盖 | 部分反馈已被建模成 adversarial semi-bandit；避免重复 | A |
| [Online Selective Conformal Inference: Errors and Solutions](https://iclr.cc/virtual/2026/poster/10014047) | 选择校准样本的交换性错误及修正 | 必须审查 selection；不能用固定 CP 定理套自适应筛选 | A |

## AISTATS 2026（3 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [Learnability with Partial Labels and Adaptive Nearest Neighbors](https://proceedings.mlr.press/v300/errandonea26a.html) | 部分标签生成机制下的可学习性与自适应近邻 | 全文区分知道生成机制的可重构性与实际未知机制下学习 | D |
| [Mask-Conditional Conformal Prediction: Valid Uncertainty For All Missing Data Mechanisms](https://proceedings.mlr.press/v300/fan26a.html) | 一般缺失机制下 mask-conditional coverage | 全文定理用似然比和绝对连续性；估计权重不自动保持精确有效性 | D |
| [Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity](https://proceedings.mlr.press/v300/mortier26a.html) | 层次预测集合的表示复杂度约束 | “更粗预测”已有问题定义；仅引入树状标签不足 | A |

## COLT 2026（3 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [A Complexity Measure for Active Learning in Multi-group Mean Estimation](https://proceedings.mlr.press/v336/aznag26a.html) | 分组 max-risk 主动均值估计的局部复杂度 | 是更严肃的预算理论基线；记录页数是扩展摘要，不夸大精读 | A |
| [Linear Regression under Missing or Corrupted Coordinates](https://proceedings.mlr.press/v336/diakonikolas26b.html) | Gaussian 设计下逐坐标对抗缺失/污染的回归界 | 原文参数化误差下界非零；不应承诺任意缺失可无偏恢复 | D |
| [Distribution-Free Sequential Prediction with Abstentions](https://proceedings.mlr.press/v336/yu26a.html) | 未知干净分布、对抗注入下的拒答学习 | 上下界和不同 adversary 都需处理，不宜当容易的纯理论题 | A |

## UAI 2026（4 篇）

| 论文 | 已定义的问题 | 对我们选题的约束/启示 | 深度 |
|---|---|---|---|
| [Testing Partially-Identifiable Causal Queries Using Ternary Tests](https://proceedings.mlr.press/v337/bhadane26a.html) | 部分识别因果问题的三元检验与拓扑条件 | 全文说明“不可决定”可以是正式输出；不能把它说成新发明 | D |
| [Overcoming Dependent Censoring in the Evaluation of Survival Models](https://proceedings.mlr.press/v337/lillelund26a.html) | 相关删失下生存模型评价 | IPCW 有机制要求；采用 copula 假设，并非无假设恢复 | A |
| [Coarsening Bias from Variable Discretization in Causal Functionals](https://proceedings.mlr.press/v337/ou26a.html) | 离散化改变因果目标泛函的偏差 | 预处理不仅改变方差；有一阶到二阶修正思路 | A |
| [Explore, Refine, then Commit: Nearly Optimal Multi-Group Mean Estimation with Active Learning](https://proceedings.mlr.press/v337/pandey26a.html) | 固定预算下多组均值的统一置信精度 | 主动复核要与正规 allocation 文献相比，不只比 random | A |

## CVPR 2026 补充（官方索引证据，未计入 54 篇）

| 论文 | 状态与判断 |
|---|---|
| [Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Multimodal_Learning_on_Low-Quality_Data_with_Conformal_Predictive_Self-Calibration_CVPR_2026_paper.html) | CVF 官方搜索索引给出主会 bibliographic record 与摘要；提出表示/梯度自校准，不支持把“低质量多模态 + CP”作为新问题。实时打开 403，未读全文。 |
| [Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](https://openaccess.thecvf.com/content/CVPR2026W/SAUAFG/html/Mitsuoka_Combining_Boundary_Supervision_and_Segment-Level_Regularization_for_Fine-Grained_Action_Segmentation_CVPRW_2026_paper.html) | 明确是 CVPR **Workshop**，不算主会录用；轻量边界监督用于训练，作为 A 的近邻而非相同问题。 |

## 必须对照的非本年度录用证据

| 作品 | 状态 | 为什么关键 |
|---|---|---|
| [How Many Labels Does Model Choice Need? Certificates and Budgets for Selective Prediction](https://arxiv.org/abs/2609.18622) | 2026-09 预印本；D | 已有固定预测/置信排序、未读标签补全、精确停止和证书预算；本次 A 不能重复这些贡献。 |
| [GRAFT-EEG: Gauge-Aware Physical Field Transport for Cross-Dataset Motor-Imagery EEG](https://arxiv.org/abs/2610.23620) | 2026-10 预印本；D | 已有参考 gauge、限制可辨识性与算子扰动界，强烈压缩 C 的新颖空间。 |
| [Diagnosing Temporal Misalignment in Multichannel Time-Series Classification with Minimum Description Length](https://arxiv.org/abs/2609.14595) | 2026-09 预印本；A | 无标签无需重训的时间错位诊断已存在；B 必须处理任务时延与硬件偏移的区别。 |
| [Fortifying Time Series: DTW-Certified Robust Anomaly Detection](https://arxiv.org/abs/2605.07690) | 2026-05 预印本；A | 不能宣称首次提供时序扭曲的认证鲁棒性。 |
| [Handling Annotation Uncertainty in Human Activity Recognition](https://doi.org/10.1145/3341163.3347744) | ISWC 2019；官方节目确认、论文检索文本 | temporal label jitter 与训练时处理早已存在。 |
| [Symmetry-Robust 3D Orientation Estimation](https://openreview.net/pdf?id=rcDYnkG1F8) | 早期先行工作；搜索定位，venue 未独立核验 | quotient orientation 与 conformal 结合不能默认是新题；本次未据此确认某会录用。 |

## 怎样使用这张表

建议先读 A 的四个对照：NeurIPS reference audit、NeurIPS PSDNet、ICML interval-censoring score、Kuboyama 预印本。再看 AISTATS partial-label learnability 以及 UAI ternary tests，校准“可识别”和“可学习”的不同含义。若坚持 EEG 跨蒙太奇，则先读 GRAFT-EEG 的 Proposition 1–2 和实验信息访问限制，再决定是否有不同问题可做。

可机器筛选版本：[CSV](papers.csv)、[JSON](papers.json)。各条目的 source_snapshot 指向本地证据；下载状态见 sources/fulltext_fetch.json。
