# 个人 Tower 适配与支持统计：固定实验协议

_2026-11-25；只改变个人可训练范围，先验证机制，不默认Tower必须冻结。_

---

## 🎯 问题与比较

连接器-only更新可能无法修复编码器中的个人偏差；但开放更多参数也可能拟合噪声。因此直接比较同人、同样本、同预算的连接器LoRA和连接器LoRA＋Tower全参。并独立检查支持集能否稳定描述个人，不把个人识别率当成分类提升。

共享parent为 `leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360，SHA256 `7c58002990c707e8f42c6de411eec61fb2d2b55d21406ab4024975708dba7a6a`。Tower从这个已经联合训练过的parent恢复，不重新使用裸发布权重代替。

| 对照 | 可训练参数 | 保持冻结 |
| --- | --- | --- |
| ordinary_lora | 连接器最后5120→5120层的rank4 LoRA，40,960参数 | Tower、原连接器、JEV |
| tower_full | 同一LoRA＋Tower全参，共19,598,592参数 | 原连接器、JEV |

Tower eval模式维持历史行为：Dropout不启用，BN运行均值/方差不更新；BN可学习仿射参数属于全参梯度。损失均为个人support标签与JEV选项logits的CE，不新增对比损失、蒸馏、正则策略生成器或文本原型分类。

## 📋 数据与隔离

下表为parent历史缓存全量成员，训／验／测顺序，不是本轮全部用于梯度的数据量。

| 数据集 | 处理后试次/窗口总计（训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（训／验／测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,184试次（2,880／1,152／1,152） | 运动想象（4） | 固定跨被试 | 9（5／2／2） |
| HMC | 136,925窗口（92,454／22,421／22,050） | 睡眠分期（5） | 原缓存跨被试 | 151（103／24／24） |

目标为2a A08/A09、HMC s2/s6，均属原test，未进入parent共享train。沿用已完成同种子实验的全部ID、排序、case order与batch计划：

- 2a每人T场次100条support（每类25），完整E场次288条query。
- HMC前60%时段选择100个support epoch（每阶段20）；间隔2个30秒epoch后，后段分别340/344条完整query。
- 个人梯度只用本人support；query不参与梯度、学习率调整、早停或checkpoint选择。
- 原valid不读取信号、不参与本轮；原train只用于冻结统计诊断，不额外训练共享参数。
- 不改变原split标签或成员。原test/valid有历史暴露，本轮结果只作探索性开发结果。

预处理、通道插值、归一化、分段与parent相同；2a为16个256维tokens，HMC完整30秒为120个tokens。全参组每批从预处理信号重新计算Tower，禁止误用冻结token缓存绕过Tower梯度。

## ⚙️ 固定训练设置

| 设置 | 值 |
| --- | --- |
| 初始化 | 共享parent完全相同；LoRA A为seed42，B为零 |
| LoRA rank／scale | 4／1 |
| LoRA学习率／Tower学习率 | 1e-4／1e-4，无scheduler |
| 优化器 | AdamW，weight_decay0.01 |
| 梯度裁剪 | 所有本组可训练参数的联合norm上限1 |
| batch／microbatch／eval batch | 16／4／8 |
| 每人support／训练步数 | 100／100，有放回抽样并核对全部support被使用 |
| 评估点 | 0、5、10、20、50、100，完整support和完整query |
| 主结果 | 固定第100步；不选query最高点 |
| 并行 | GPU0 A08、GPU1 A09、GPU2 HMC s2、GPU3 HMC s6 |
| 每卡顺序 | Tower全参＋LoRA → 普通LoRA，每组重置Tower、LoRA与优化器 |
| 预算 | 8次个人适配，共800训练步，6小时安全上限，完成即退 |

Tower学习率沿用LoRA的数值是为了先做固定设置的机制检查，不声称是全参最优学习率。增加Tower梯度自然会影响联合裁剪的缩放；记录grad norm，不把它误称为完全相同优化轨迹。

共同LoRA初值指纹 `88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542`。每批仍使用 `SeedSequence([42,case_order,step])`；前向随机种子和选项排列沿用历史。正式训练前真实27B smoke检验恒等初始化、有效梯度、冻结JEV/原连接器、Tower是否按组更新、完整个人状态精确重载；smoke后恢复parent，不能把试更新带入训练。

## 📊 支持集稳定性诊断

事先固定三种描述，不按目标query挑选：

| 描述 | 维度 | 计算 |
| --- | ---: | --- |
| tower_scalar | 2 | 所有token/特征合并后的均值、总体标准差 |
| connector_scalar | 4 | 两个linear的输入分别计算整体均值、标准差 |
| tower_featurewise | 512 | 256个特征维各自的均值、标准差 |

使用冻结parent特征，按选中trial/token汇总一阶、二阶矩，再计算总体SD。RMS欧氏距离，无拟合分类器、无目标query标准化。所有匹配两端保持完全相同的类别构成，以排除类别比例差异；类内反应或采集条件仍可能包含在描述里。

1. 源训练被试：2a 5人、HMC 8人。每人100条support与另一记录/时段100条source-query均来自原train。2a为T→E跨场次；HMC为同一夜前后时段，不能称跨夜。
2. 支持集内部：每类按时间排序各取前后半，2a每半48条（每类12），HMC每半50条（每类10）。另作100次随机对半，仅报告重采样敏感性，不能当100名用户或总体置信区间。
3. 目标用户：只从memory-mapped历史特征文件索引100条support；不索引或计算query特征。目标仅各2人，单独报告，不与见过的源用户混成总体识别率。
4. 用另一半/记录的描述匹配最近的个人描述，随机水平为1/人数；距离并列按比例计分。报告同人/异人距离及匹配率。
5. 源记录迁移用整个人profile置换计算精确配对距离p值，六个描述×任务比较做Holm修正；用户是统计单位。识别率不是下游任务BAcc。

`diagnostics/protocol.json`、`results.json`记录选中ID、cache来源、parent与plan指纹、矩阵与解释边界。源数据用来判断是否值得开发条件策略，不把这些诊断自动等同于FedP²EFT证据。

## 💾 记录与解释

输出 `artifacts/benchmark/leaf_personal_tower_diagnostics_20261125/`：

- `plan.json`、`queue.json`：固定成员、batch、源码hash、时间预算与状态。
- `jobs/<用户>/cases/<用户>/<方法>/curves/step*.json/.npz/.pt`：指标、逐样本logits、个人LoRA及全参组Tower权重。
- `last.pt`：优化器、个人全部可训练权重、历史；重启先补完当前评估点。
- `summary.json`、`curve_points.csv`、`figures/`：同一步、所有配对用户/方法齐全才汇总；图不填造缺失点。
- `diagnostics/source_profile_transfer.png/pdf`：源用户跨记录/时段距离矩阵。

主要看同人query的增益及负迁移，support准确率只用于判断拟合。稳定个人统计可能只是montage/记录条件，不自动证明生理特性；全参收益也不自动证明该统计是因果机制。若未提升，先看支持拟合与query曲线，不能同时改lr/rank/encoder后再归因。

旧方法归档Git `1d78125`；不删除任何ignored数据、权重、日志、环境。实验设计/统计技能用于配对和用户级证据，资源技能用于四卡预检，Matplotlib/Markdown技能用于可复核图表与记录；属于工作流辅助，不属于算法创新。[^skills]

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://arxiv.org/abs/2609.00065
