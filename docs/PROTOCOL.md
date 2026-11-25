# 同种子个人适配曲线：固定实验协议

_2026-11-25；只统一初始化种子并补齐等步数曲线，不改变学习率、rank、encoder或个人训练预算。_

## 🎯 目的与比较

上一轮FOMAML相对普通个人LoRA的两任务均值只提高0.13个百分点。先检查它是否在适配早期有收益、但100步时差异消失；不先用新超参数解释或修复结果。此前普通LoRA按用户seed初始化，而源训练统一从seed42开始；这轮消除该差异。

冻结的parent仍为 `leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360，SHA256 `7c58002990c707e8f42c6de411eec61fb2d2b55d21406ab4024975708dba7a6a`。原始结构、损失（支持集CE）和优化器不变。

| 对照 | 个人适配开始时的参数 | 本轮更新 |
| --- | --- | --- |
| 普通 LoRA | seed42随机A、零B | 本人support，100步 |
| source ERM | 上轮60episode完成的初始化 | 本人support，100步 |
| FOMAML | 上轮60episode完成的元初始化 | 本人support，100步 |

所有组最初随机参数的共同指纹为 `88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542`。两种已学习初始化直接复用，不重新置零A/B，不重跑离线阶段。源权重、完成记录、原plan的hash纳入新运行守卫；完整旧实现与结果在Git `4a664b1`。

## 📋 数据与人员

下表是parent历史缓存的完整成员数量，训／验／测顺序；不是本轮全部参与梯度的数据量。

| 数据集 | 处理后试次/窗口总计（训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（训／验／测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,184试次（2,880／1,152／1,152） | 运动想象（4） | 固定跨被试 | 9（5／2／2） |
| HMC | 136,925窗口（92,454／22,421／22,050） | 睡眠分期（5） | 原缓存跨被试 | 151（103／24／24） |

实际仍为2a A08/A09、HMC s2/s6，所有人都是原test被试，未进入parent共享train。严格复用上轮support/query ID、排序、原split标签和case order：

- 2a：每人T场次100条类别平衡support（每类25），完整E场次288条query。
- HMC：每人前60%时间段中取100个support epoch（每阶段20）；间隔2个30秒epoch后，完整后段分别340/344条query。
- 原valid本轮不读取信号、不用于梯度或选点；原train也不重跑梯度。
- 支持与query不重叠原试次/epoch；个人梯度守卫拒绝query。HMC相邻睡眠epoch并非独立用户或场次。

`plan.json` 固化全部成员及100步的batch ID。每名用户原batch RNG为 `SeedSequence([42, case_order, step])`，保留不变；唯一更正是普通LoRA的A不再用42＋case_order×101，而统一42。模型内每个训练batch的随机状态和选项排列也沿用原规则。

## ⚙️ 固定设置

| 设置 | 本轮值 |
| --- | --- |
| 可训练参数 | 连接器最后5120→5120层的LoRA A/B，40,960参数 |
| encoder／原连接器／JEV | 全冻结；梯度经过JEV流向个人LoRA |
| rank／scale | 4／1 |
| learning rate | 1e-4，无scheduler |
| optimizer | AdamW，weight_decay0.01，grad_clip1 |
| batch／microbatch／eval batch | 16／4／8 |
| 每人support／训练步数 | 100／100；有放回抽样，确认100条均被使用 |
| 评估步数 | 0、5、10、20、50、100 |
| 选点 | 固定100步；不早停、不按query取最好点 |
| 并行 | GPU0 A08、GPU1 A09、GPU2 HMC s2、GPU3 HMC s6 |
| 每卡顺序 | 普通LoRA→source ERM→FOMAML，每组重置优化器 |
| 预算 | 12次个人适配，共1200个训练step；完成即退，6小时安全上限 |

预处理保持parent：最小范数双极导联转换、200Hz、逐epoch裁剪及median/IQR归一化、65通道插值；2a保留16个256维token，HMC完整30秒保留120个token。没有更换encoder、特征选择、损失或任务文本。

FOMAML源训练仍是5步inner-loop、60个episode；此轮不解决其与100步个人适配之间的预算差异，只测量差异何时出现。两个任务使用各自的源初始化，不能称为跨任务元学习。

## 📊 记录与汇总

每个评估点保存完整support的Acc/BAcc/CE、完整query的Acc/BAcc/CE、逐样本logits与ID、个人参数指纹及权重。训练loss逐步保存。评估用no_grad并恢复随机状态，不更改优化器、参数或下一批训练样本。

断点在评估前保存；若中断，恢复后先补当前步评估，禁止跳过应有的较早曲线点。所有组初始优化器状态都是空的。真实27B检查验证零LoRA恒等、支持集梯度确实更新、共享参数/缓冲不变及精确重载。

同一步数的所有三组、两名用户齐全后，才输出该任务的配对均值；两任务都齐全后才输出等权平均。主比较为FOMAML与普通LoRA、FOMAML与source ERM的同人差值，不混用不同用户的临时均值。最终仍以固定100步作主结果，前几步曲线作机制诊断，不因观察曲线重新选择训练预算。

输出目录 `artifacts/benchmark/leaf_personal_matched_curves_20261125/`：

- `summary.json`：完整配对的均值、差值、未齐全的组合。
- `curve_points.csv`：逐用户、逐方法、逐步指标。
- `figures/adaptation_curves.png/pdf`：完整query BAcc与相对普通LoRA增益。
- `figures/support_query_curves.png/pdf`：完整support Acc、support CE、query CE。
- `jobs/<用户>/cases/<用户>/<方法>/curves/step*.json/.npz/.pt`：指标、预测及个人权重。
- `last.pt`：可恢复的参数、优化器、历史；`result.json`：固定100步结果。

图使用实际step间距，无平滑、无插值填补缺失结果。尚未齐全的均值位置留空；本轮不从2名用户推断总体显著性。

## ⚠️ 限制与归档

这是单seed、每任务2人的探索性诊断，不是独立盲测。原test和原valid都有历史暴露；本轮不使用valid不意味着它重新成为未观察集。LEAF发布权重预训练暴露未完全审计。新普通LoRA初值已修正，旧59.72%／69.36%不可充当本轮新基线。

实验设计技能用于固定同人配对与变量，资源技能用于确认四卡可用，Matplotlib技能用于实际数值曲线，Markdown技能用于协议记录；这些工程辅助不属于方法创新。方法背景与已核验引用保留在[研究记录](PERSONALIZATION_METHOD_REVIEW.md)。所有ignored数据、checkpoint、日志、缓存和第三方环境保留，旧离线训练代码在Git归档，可用隔离worktree复跑。
