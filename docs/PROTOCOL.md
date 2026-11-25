# 个人 LoRA 位置与支持样本量：固定协议

_2026-11-25；用户授权的21人整夜验证，固定样本和优化设置，先定位再比较样本量。_

---

## 🎯 问题与可归因比较

要回答三个问题：个人修正应该发生在EEG→语言维度映射之前还是之后；再开放Tower内部低秩更新是否有帮助；在固定计算预算下，24/48/100条个人数据的收益如何。

| 方法 | 可训练位置 | 参数量 | 样本量 |
| --- | --- | ---: | --- |
| last_only | 连接器第二层5120→5120的LoRA | 40,960 | 100 |
| first_only | 连接器第一层256→5120的LoRA | 21,504 | 100 |
| both | 两个连接器linear的LoRA | 62,464 | 100、48、24 |
| tower_both | Tower线性权重LoRA＋两个连接器linear LoRA | 462,848 | 100、48、24 |

100条时四组同人比较，额外的48/24条只比较both与tower_both。共21×(4＋2×2)=168次独立个人训练。零更新baseline每人单独计算；其query预测必须与历史21人实验逐样本一致。

这一设计隔离新增位置的作用，但并非相同参数量对照：first_only比last_only更小，tower_both比both更大。不得将收益完全归因于位置而忽略容量。固定rank避免同时扫rank、学习率和位置。

## 📋 固定数据与暴露分层

精确复用 `leaf_personal_multitask_20261125/plan.json` 的21名用户、100条support、完整query及原split标签；新24/48只从这100条中按类别前缀取嵌套子集，不重新抽query。

- 2a：A01–A09。每人T场次100条支持池，完整E场次288条query。A01–A05及其query已在共享train池；A06/A07原valid，A08/A09原test，均未参与共享训练。
- SEED：s1–s4。沿用原train窗口作为支持池、原test完整504窗口query，原始trial分组不重叠。用户均在共享train见过，query trial未进入共享train。
- HMC：s7、s9、s2、s6。前60%时段选择100条，每阶段20；间隔2个epoch后，完整后段query分别411/358/340/344。s7/s9来自历史原valid，s2/s6原test，均是共享训练未见用户；只有同夜时段迁移，不是跨夜。
- Nakanishi2015：s1–s4，12类，每人180条原始trial，历史100支持＋固定80 query。整个数据集属于历史共享train；只能作已见训练池上的机制验证。

本轮明确继续使用用户指定的历史个人被试，包括原valid的A06/A07、HMC s7/s9；不新增valid成员，不改变共享训练评估角色。共享模型不再训练。原始valid/test都存在历史暴露，不能把角色交换解释成新的盲测。

窗口与被试的完整原train/valid/test数量自动写入输出 `report.md`，取自已校验的历史缓存index，不把本轮个人support/query计数冒充基准划分。

## ⚙️ 初始化、优化和梯度范围

parent仍为 `leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360，SHA256 `7c58002990c707e8f42c6de411eec61fb2d2b55d21406ab4024975708dba7a6a`。所有个人条件都从此parent恢复；不从全参pilot或上一个用户的终点继续训练。

| 设置 | 值 |
| --- | --- |
| LoRA rank／scale | 全部4／1，相当于alpha4、alpha/r1 |
| LoRA初始化 | A为确定性seed42，B为0；共享位置在不同组初值完全相同 |
| 个人损失 | JEV最终选项logits的交叉熵 |
| learning rate | 连接器/Tower均1e-4，常数，不调度 |
| optimizer | AdamW，weight_decay0.01，所有可训练参数联合clip norm1 |
| batch／micro／eval batch | 16／4／8 |
| 每人每条件 | 100个optimizer steps，有放回抽样；核对所有支持样本都被使用 |
| 评估 | step0、5、10、20、50、100，完整support与固定完整query |
| 主结果 | 固定100步，不按query选最好点或提前停止 |
| 并行预算 | 4×H100，每卡42条件，12小时上限，完成即退出 |

每个连接器位置分别使用局部CPU seed42初始化，避免增加第一层改变第二层A的随机数。第二层初值指纹仍为 `88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542`。同一用户同样本量的batch ID和选项排列在所有方法间完全相同；同一用户不同方法顺序用固定种子随机化。A05八个条件分摊到四卡，消除21人不能整除四卡的长尾。

Tower LoRA覆盖98个权重：两支输入投影、两支24个Transformer层内的融合QKV、attention输出投影和两层FFN。融合QKV整体用一个rank4增量，不是Q/K/V各自rank4。通过weight parametrization接入，确保`MultiheadAttention`直接读取权重时不会绕过LoRA。

Tower仅新增400,384个LoRA参数可训练。卷积、bias、norm、位置编码、原矩阵、BN运行统计和JEV全部冻结；保持eval-mode/dropout历史策略。只保留前向Tower，不恢复重建头。没有增加原生LEAF文本编码器、Q-Former或新对比损失。

tower_both每批重新从EEG计算tokens；其他组可以缓存冻结tokens。预处理逐值沿用历史：200Hz、逐trial归一化、65通道插值、长片段分段；SEED保留插值后的[-10,10]裁剪。不能因为删除旧任务代码而静默丢掉此步骤。

## 📊 样本量与统计单位

24/48/100为类别尽量均衡的嵌套集。SSVEP的24/48正好每类2/4条，100为每类8或9条；2a分别每类6/12/25条。每个样本量都用相同query，不把未用的support补进query。

固定100步×batch16，等效遍历数分别66.67、33.33、16。这回答的是固定更新预算下支持池大小的效果，不是相同epoch比较；每条样本重复程度不同会影响过拟合。尚未比较超过100条，也未测first/last-only的24/48条，不能宣称全局最优样本量。

主要报告每人适配前后BAcc/Acc/CE、support Acc/CE、同人方法差值、负迁移人数、参数量与时间。先按被试平均再按任务等权平均，不把窗口当独立被试。原训练已见/未见以及query是否处于共享train池分别汇总；方法、样本量和所有配对用户齐全后才输出该组均值。

支持集趋近100%而query loss升高，作为过拟合线索；不据此临时选择最高query点替换step100。单seed结果只用于探索，不能声称总体显著性。

## 💾 检查、记录与归档

每个worker实际验证零增量恒等、有限且有效的梯度、LoRA更新、原参数及buffer未变、精确保存/重载。单元测试覆盖MHA直接读weight、两个连接器位置、样本嵌套、query隔离、精确历史成员、分层汇总与缺失条件不平均。

输出 `artifacts/benchmark/leaf_personal_lora_locations_20261125/`：

- `plan.json`：168条件、精确ID、批次、GPU分配和代码hash。
- `queue.json`、`progress_gpu*.json`、`gpu*.log`：状态和时间。
- `jobs/worker*/cases/<用户>/<方法>_n<数量>/`：last断点、完整个人LoRA、result及每评估点参数/logits/ID。
- `summary.json`、`curve_points.csv`、`report.md`：自动配对汇总、已见/未见分层、完整计数。
- `figures/location_curves.*`、`support_query_curves.*`、`support_size_curves.*`：实际步数和支持量曲线，缺失点留空。

前一轮全参Tower pilot已全部完成并归档Git `31a36d9`。其诊断实现移入Git历史；所有ignored数据、checkpoint、日志、缓存与环境保留。实验设计技能用于配对与重复测量结构，资源技能用于容量预检，Markdown/Matplotlib技能用于可审计报告；Hugging Science目录无EEG匹配，不替换用户已指定的模型。技能归属不作为算法有效性证据。[^skills]

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://arxiv.org/abs/2609.00065
