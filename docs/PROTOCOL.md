# 全部留出用户的个人残差适配协议

_2026-11-23：扩展成功的个人CE pilot；固定训练预算，仅增加目标用户覆盖。_

---

## 🎯 问题与本轮假设

目标是让已训练的EEG→JEV系统，利用新用户少量标注样本做个人校准，同时保持服务器LM和共享信号模型不变。少量样本可拟合支持集，却不一定提升独立session/run；因此主要观察同一用户在同一query上的适配前后差值，而不是支持集准确率。

旧2a被试8、20条support、100步的B＋冻结LM结果为64.93%（不适配53.82%）。本轮假设是该机制能在更多目标用户上改善表现；不将单用户成绩外推为结论。只扩展该分支，不新增对比损失、TTT、源用户共享训练、LM LoRA或Tower解冻对照。

## ⚙️ 唯一训练路径

个人EEG经过已训练且冻结的LEAF Tower、个人残差adapter、冻结原connector、冻结JEV，得到choice logits。损失为support真实标签对choice logits的交叉熵；梯度经过冻结JEV和connector，只有adapter参数更新。没有文本原型相似度读出。

adapter为 `z + Linear(64,256)(GELU(Linear(256,64)(z)))`，含bias共33,088参数；上投影weight/bias零初始化，初始函数恒等。Tower有19,557,632参数，四秒输入输出16×256 tokens；原connector是LN256→Linear256/5120→GELU→Linear5120/5120，共27,535,872参数。JEV全部冻结，包括基础权重、已有语言LoRA、choice head；仍需云端LM前向和输入梯度计算，不是纯边缘设备本地训练。

父模型固定为 `artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，原验证选中step1360。每个用户重置到同一个adapter初始权重，新建优化器；不继承另一用户的更新。父模型不因个人适配改变。

## 📋 数据与划分

以下是父模型使用的处理后窗口/原被试划分，不是本轮个人support/query数量。当前MI缓存每trial恰好一个4秒窗口。

| 数据集 | 处理后窗口数：总计（父模型原训／验／测） | 任务种类（分类数） | 划分协议 | 被试数：总计（父模型原训／验／测） |
| --- | ---: | --- | --- | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4类） | 原固定跨被试；个人T→E session | 9（5/2/2） |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4类） | 原固定跨被试；个人前四→后两imagery run | 109（69/19/21） |

- 主实验：2a被试8、9；PhysioMI被试89–109，共23人。未将原训练/验证被试算入新用户主结果。
- 2a：每人从T session 288条中按类无放回取5条，共20条支持；E session全部288条query。两人共40条支持、576条query。
- PhysioMI：每人从R04/R06/R08/R10按类无放回取5条，共20条支持；R12/R14全部query。21人共420条支持、632条query。
- 总计460条支持样本选择、1,208条query；支持池剩余样本不训练，也不并入query。
- seed42和每类抽样算法逐条沿用旧InfoNCE个人实验；支持与query ID、记录边界和batch流有哈希核对。
- 短测只用原验证被试2a 6、PhysioMI 70，各3步，取8条query检查工程正确性；不选超参数，不计入主结果。
- 2a六个run的event会重置，使用record/run/event联合键查重；不将同trial多个窗口误当独立样本。
- 预处理和Tower tokens复用已校验的个人输入缓存；不利用query集合统计拟合归一化。问题文本不含query真值。

## 🔒 固定预算与审计

| 设置 | 固定值 |
| --- | --- |
| 每人支持样本 | 5条/类，共20条 |
| 更新预算 | 100步，seed42，仅一个seed |
| 优化器/学习率 | AdamW，1e-4，constant |
| Batch | 16，microbatch4，累积4次 |
| 正则/梯度 | weight decay0.01，clip norm1 |
| 每步抽样 | 类别均衡；在已选20条支持样本内有放回抽取 |
| 选点 | 固定第100步，无query早停/选点 |
| 并行 | 四卡各持同一冻结父模型，独立处理不同用户；不DDP合并用户梯度 |
| 恢复 | 完整用户跳过；未完整用户从父模型重跑，不声称中途精确续训 |

控制器检查四卡各至少60,000MiB空闲、磁盘至少20GiB。启动前单元测试、两名验证被试真实GPU短测全部通过后才进入正式运行。

每次适配核查：初始adapter恒等；共同不适配基线等于旧实验同query；训练batch ID流相同；adapter真实更新；Tower/JEV参数及buffers不变且无参数梯度；保存重载tokens与logits一致；无query选点。源码/config/父权重/缓存哈希改变则停止，不热改训练语义。

旧个人CE运行被用户取消的剩余批次继续保持取消。本轮使用新run根目录，不覆盖或篡改旧manifest、取消标记或结果。旧amortized训练已结束并归档，不保留独立实现分支在当前工作树。

## 📊 输出与解读

根目录 `artifacts/benchmark/leaf_personal_ce_alltest_20261123/`。逐用户记录支持集CE/Acc、query适配前后Acc/BAcc、confusion matrix、耗时、参数数、训练批次与冻结审计、小adapter checkpoint、逐trial预测。

主汇总先在每库按用户等权平均BAcc，再报告两库等权平均；另列23名用户等权平均，避免混淆两种权重。报告改善/下降/持平人数、各用户差值，不只报均值。只有23人全部完成且审计通过才标完整汇总；部分结果不冒充完整结果。

2a只有两名目标用户、一个seed，且被试8已观察过；本轮是探索性扩展，不宣称盲测、显著性或严格zero-shot。上游LEAF预训练身份重叠未完整排除。父模型与EEG-FM-Bench原论文仍有已记录窗口数量差异，详见[结果页](RESULTS.md)。本轮个人query是原test的严格子集，不等于原test整份评测。

## 🔗 实现与过程来源

后续夜间匹配方法对照已在隔离分支 `experiment/personal-method-controls` 实现（`460d31e`）：AdaBN＋CE、内部SSF、CE＋MEMO。原基线所有规则不变；新方法逐一比较同一20条support与同一query，不扫描shot或LR。SSF参数量63,488，MEMO额外使用当前query无标签更新，需分别报告容量与transductive差异。完整方法协议位于 `/inspire/hdd/global_user/zhangchenxi-253108310322/eegjev-worktrees/personal-method-controls/docs/PROTOCOL.md`；主工作树不热改源文件。夜间截止2026-11-23 22:50:52 UTC。

当前入口与全部设置见[配置](../configs/personalization.json)、[个人适配模块](../eeg_align/personalization.py)、[训练入口](../eeg_align/run_personalization.py)、[控制器](../scripts/run_personalization.py)。旧成功pilot见[原结果](../artifacts/benchmark/leaf_personalization_jev_ce_20261123/pilot_summary.json)，历史代码 `05403f7` 在隔离worktree检查复用。

实验设计/资源检查技能用于锁定逐用户配对、泄漏边界和运行容量；Transformers技能用于检查冻结模型仍传递输入梯度；Markdown技能用于同步当前协议。未更换现有模型或依赖。科学资源目录无法访问，不作为模型选择证据；采用现有本地权重。技能过程参考[^skills]，不是本方法的新颖性来源。

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
