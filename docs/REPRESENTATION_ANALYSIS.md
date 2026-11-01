# 文本条件下的 EEG 表征是否过于拥挤？

_2026-11-01 · 当前完整 HMC 版 BSR→JEV 四读出试验 · 验证集只读诊断_

## 🎯 结论先行

你的猜想得到部分支持：2a、PhysioMI 的最终决策表征在同一任务内很集中，不同类别又高度混叠，且多个 checkpoint 严重偏向单一类别。但目前不能把原因归结为“文本条件／类别文本太接近”。原因包括：运动想象在进入 JEV 之前的 BSR 均值表征上已缺乏明显类别分离；TUAB 的类别文本更近但预测更好；TUEV 虽有明显的几何分离，最终分类仍有大量错误。

更准确的诊断是：不同任务的类别可分性、被试差异、读出冗余和最终决策利用能力共同存在问题，不能用一个距离解释全部现象。本次没有训练、修改权重、重新选 checkpoint 或访问测试信号。

## 🔬 分析对象和方法

- 运行目录：`artifacts/benchmark/bsr_jev_readouts_hmc151_20261031`。四组分别使用既定 best：mean_pool step 2040；CLS、cross_attention、qformer step 1700。不是为得到漂亮图另选 checkpoint。
- 九库各固定 256 个验证窗口，合计 2,304 个不同窗口 ID，四组完全相同。重复提取四次不等于 9,216 个独立统计样本。采样使用原 `validation_rows(cap=256)`：保留旧随机 128 窗，再按类别／被试队列补足，不是严格均匀类别样本。
- 实际覆盖验证被试数依次为：2a 2、PhysioMI 19、SEED 1、SEED-VII 2、TUAB 100、TUEV 20、HMC 24、Workload 5、ADFTD 11。不同库的 ID 不跨库合并；原始完整划分见[当前结果](RESULTS.md)。逐类数量和窗口 ID 见[样本清单](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/mean_pool/manifest.json)。
- 沿原验证路径提取五处：BSR token 均值、读出 token 均值、全部有序读出 token 展开、连接器 token 均值、JEV 末尾 `[decision]:` 的 5120 维状态。文本只进入 JEV，不进入 BSR 或读出。
- 真实 EEG 前向只含原任务说明、问题和全部候选项，候选顺序与原验证一致；真值只用于采样、统计和图中着色。所有距离、silhouette、rank 均在原维度计算，不能用二维散点分离程度代替。

关键指标：设 `a` 为同类别窗口间的平均 cosine distance，`b` 为不同类别窗口间的平均距离。`a` 先在每类内平均再对类别等权，`b` 对不同类别对等权；排除自身配对。

```text
cosine distance = 1 - cosine similarity
class gap = (b - a) / max(a, b)
```

gap 接近 0 表示在这个平均距离指标下类内、类间没有明显差别；正值表示异类更远，但不保证每一类都能分开。另保存类别等权 silhouette，考察每个窗口与最近竞争类别的关系。距离全部为描述性指标，没有给出独立样本检验或显著性结论。

## 📊 观察到什么

### 1. 运动想象确实表现为“任务内紧密、类别混叠”

在 mean_pool 中，2a 的平均任务内 cosine distance 为 `7.86e-5`，PhysioMI 为 `1.64e-4`；SEED 为 `1.58e-3`，HMC 为 `1.53e-2`。四组都呈现类似的任务差异，不是只在某一个 checkpoint 出现。

同时，2a 的四组 class gap 为 `-0.004/-0.002/-0.003/-0.005`，PhysioMI 为 `-0.013/-0.011/-0.011/-0.015`，均没有正向的平均类间分离。换成未归一化 Euclidean 距离，运动想象也没有明显正分离，而不是只由 cosine 的归一化产生这个结论。

在本次固定 256 窗子集上，CLS 和 Q-Former 对 2a 均把 256 窗全部判为左手；cross-attention 对 PhysioMI 同样全部判为左手。这是观测到的预测偏置，不是从图中猜测。四组 MI 子集 BAcc 为约 25–26.5%；不要与[完整验证指标](RESULTS.md)混用。

![四组九库的原维度任务内距离、任务内外距离比、类别分离与子集准确率](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/geometry_overview.png)

_图 1：第一列直接比较不同任务条件内部的分散程度；第二列是同任务样本距离与跨任务样本距离之比。第二列同时混入数据集、通道、窗口时长等差异，不能解释成纯文本的因果作用。最后一列是 256 窗诊断子集 BAcc。_

### 2. 全局分簇主要反映任务差异，不等于任务内分类成功

![四组共享 PCA 坐标下的九任务决策表征](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/decision_pca_tasks.png)

_图 2：四组共用一个 PCA 坐标系和轴范围，输入先逐向量 L2 归一化，前两主成分解释 45.3% 的总体方差。颜色与形状表示数据集／任务，而不是同任务中的类别。2a 和 PhysioMI 聚在附近，并不说明它们内部的左／右手等类别已分开。_

进一步看[mean_pool 的任务内类别图](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/within_task_classes_mean_pool.png)：MI 的多类高度重叠；TUAB 的两类沿主要变化方向出现区分；SEED 仍有较多类别重叠，不能因为平均 gap 为正就称三类都清晰分开。其余组：[CLS](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/within_task_classes_cls.png)、[cross-attention](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/within_task_classes_cross_attention.png)、[Q-Former](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/within_task_classes_qformer.png)。

这些任务内图只为显示去掉了各组各任务均值，再为同一任务跨四组拟合共同 PCA；同任务跨组轴范围相同，不同任务则是不同 PCA，不能直接比较图上面积。

### 3. 不能只责怪 JEV 文本条件

![从 BSR 到读出和 JEV 的原维度类别分离](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/separation_by_stage.png)

_图 3：原维度类别分离在各提取位置的变化。All readout 保留全部有序 token，避免均值池化掩盖 slot 间差异。_

冻结 BSR 的均值表征在 2a、PhysioMI 上的 gap 已分别为 `-0.0009/-0.0099`。这一观察说明“问题只发生在文本条件化之后”不符合当前证据。但不能据此断言完整 BSR patch 序列没有运动想象信息：均值仅是一个摘要，仍需正确的冻结特征探针验证。

对纯文本也做了两个不同检查：

- [完整任务条件相似度](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/text_condition_similarity.png)：移除 EEG slots，保留实际完整提示，在末尾取上下文状态。2a 与 PhysioMI 的相似度约 `0.9996–0.9997`；两者说明和问题完全相同，候选项也大量共享，这本身不异常。
- [候选文本距离](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/candidate_text_distance.png)：单独用“任务＋问题＋一个候选类别”的诊断文本提取状态。mean_pool 中，TUAB 类别距离约 `0.005`，比 2a 的 `0.022` 更小；但 TUAB 子集 BAcc 为 `72.6%`，2a 仅 `26.0%`。因此不存在当前证据支持的“类别文本越近，准确率必然越低”。

JEV 的上述状态不是经过 CLIP 式训练的句子 embedding。共同提示模板、末尾标记及语言模型特征的共同方向都可能影响距离。本次只能报告其几何形态，不能把此探针当作文本语义质量的标准评测。

### 4. TUEV 是重要反例：分得开不代表选得对

四组 TUEV 的平均 class gap 为 `0.626–0.679`，类别等权 silhouette 也为正，但子集 BAcc 只有 `22.4–45.1%`。其中 CLS 为 `45.1%`，cross-attention 为 `22.4%`，不能通过二者相近的平均分离指标解释这个差距。

这提示要进一步区分类别局部混叠、最终 decision head 的方向／偏置、候选项语义绑定等因素；本次没有证明具体哪项是原因。参见 [CLS 混淆矩阵](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/confusion_cls.png)与 [cross-attention 混淆矩阵](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/confusion_cross_attention.png)。

### 5. 被试效应和 query 冗余也值得排查

同一被试内比较时，mean_pool 的 Workload gap 从全体混合的 `0.057` 变为被试等权平均的 `0.366`；SEED-VII 从 `0.026` 变为 `0.177`。这与跨被试变化掩盖任务信息的解释相容，但二者比较的配对集合不同，不能当作已证明的因果分解。TUAB、ADFTD 的被试标签固定，没有同被试不同类别对照，明确记为 NA。

Q-Former 的同一窗口内 32 个读出 token，其平均两两 cosine similarity 在九库为 `0.99963–0.99978`；cross-attention 为 `0.98374–0.99380`。因此目前的 32 slots 在方向上高度相似，尤其 Q-Former；不能默认它们已经学到 32 种互补信息。但高相似度仍不能排除小差异携带有效信息，也不等价于矩阵完全相同或精确秩为 1。

参见[被试控制、silhouette 和 slot 相似度图](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures/subject_and_slot_diagnostics.png)。这比单纯增加 query 数量更值得先排查。

## ⚠️ 可以说什么，不能说什么

可以说：在这轮单 seed、小预算 checkpoint 和固定验证子集上，MI 类别混叠、部分任务严重预测偏置、Q-Former slot 方向冗余都有实际证据。

不能说：文本距离导致低准确率；全部 BSR tokens 没有 MI 信息；训练对比学习一定能解决；二维图已经证明模型完全塌缩。SEED 仅有一个验证被试；连续窗口不是独立重复；各任务的数据量、类别数、通道、时长和被试构成都不同。验证集也是原先的选模集，本次不是新的无偏测试证据。

额外的零 slot 控制保留序列长度、位置和任务提示，但在 connector 之前把所有读出向量置零，属于 OOD 干预，不是有效静息 EEG，也不等价于移除 EEG。本轮 MI 的真实输入预测与这个控制的预测大多不同，所以不能把“多数样本预测同一类”直接说成“模型只看文本”。数值保存在 `decisions.json`，未用这个控制挑模型。

接下来较有区分力的实验是冻结特征探针：在既定训练被试上拟合任务分类器，分别读取 BSR 完整 tokens 的合适汇聚、readout 和 JEV 状态，只在既定验证被试上评估。它能帮助区分“信息缺失”和“原决策头未利用信息”。再考虑任务内多正例对齐、跨被试约束或 query 多样性约束，并保留匹配对照；本次没有擅自启动这些训练。

## 🔗 产物与复核

- [全部图件](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figures)：15 组 PNG＋SVG；[图件清单／图注／导出参数](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/figure_manifest.json)。图中数据为实际计算，未用生成式图片替代。
- [原维度统计 CSV](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/geometry.csv)、[预测和诊断 CSV](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/representation_analysis/decisions.csv)，同时保留 JSON。
- 四组子目录保存逐窗 embeddings/probabilities、真实标签、被试、窗口 ID、实际提示和完成清单；PCA 坐标、变换矩阵、均值和解释方差同样保留。
- 提取路径与原 `JEVBridge.forward` 在每库首批 logits 逐元素一致。与历史完整验证保存的同窗口概率相比，最大绝对差为 `0.005913`，9,216 次分类中 argmax 改变为 `0`。概率不是逐 bit 一致；批次组成及数值实现差异不被隐瞒。
- 训练源代码、配置、数据索引、发布权重和 checkpoint 身份均核验；冻结 BSR 哈希未变；没有反向传播或优化步骤。GPU 4、5 的提取进程均正常结束，显存已释放。项目测试 `76 passed`，不是包含第三方依赖的全仓库测试。
- 可复现入口：[extract_representations.py](../scripts/extract_representations.py)、[plot_representations.py](../scripts/plot_representations.py)、[距离诊断测试](../tests/test_representations.py)。提取脚本拒绝覆盖已存在子目录。

只重画当前已保存表征：

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 laya/.venv-eeg/bin/python \
  scripts/plot_representations.py \
  --run artifacts/benchmark/bsr_jev_readouts_hmc151_20261031
```

科学可视化与证据审查技能用于固定样本、保留原维度指标／反例、区分统计观察与因果推断；Matplotlib／scikit-learn 用于真实数值绘图，Markdown 技能用于报告结构。相关工具方法已记入来源，不作为结果有效性的证据。[^skills]

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 ，2026-11-01 核对当前 arXiv 记录。
