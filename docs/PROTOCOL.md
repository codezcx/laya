# 旧InfoNCE Tower→JEV：100条个人适配协议

_2026-11-25；用户明确选择旧checkpoint，废止当前13库混合场次个人适配，不重新训练共享模型。_

## 🎯 研究问题与权重

研究同一个目标用户提供100条带标签T场次EEG后，冻结服务器JEV、只更新个人模块，能否改善该用户完整E场次的分类。不是把多个人的support混合训练，也不是每人更新共享LM。

parent：`artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360；SHA256 `7c58002990c707e8f42c6de411eec61fb2d2b55d21406ab4024975708dba7a6a`。它是旧InfoNCE对齐Tower接JEV后训练的共享模型；本轮不重新做对比学习。严格恢复所有可训练权重与buffers，再冻结共享Tower、连接器、JEV已有LoRA和choice head。

## 📊 被试、场次与样本

恢复的旧划分来自[split manifest](../artifacts/benchmark/multitask_splits_seedpaper_20261101/bcic_2a/manifest.json)与[缓存索引](../datasets/multitask_seedpaper_20261101/bcic_2a/index.csv)。

| 数据集 | 旧共享划分窗口：总（训/验/测） | 任务（分类数） | 旧共享划分协议 | 被试：总（训/验/测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4） | A01–A05训练；A06–A07验证；A08–A09测试 | 9（5/2/2） |

当前目标为A06、A07、A08、A09，均未进入该parent的共享训练池；每人完整T/E场次各288条，每类72条。每人从T场次按固定seed42抽每类25条，共100条产生个人梯度；剩余188条T样本本轮不用。全部288条E样本仅评估，不参与梯度、蒸馏教师、早停或选点。4人共400条support、1,152条query。

原valid/test成员不改变；个人support使用是用户新授权的局部适配，不改写共享训练历史。`summary.json`分别汇总`bcic_2a/valid`与`bcic_2a/test`，方法间仅比较相同用户完整组。A06/A07曾参与共享checkpoint选择；A08/A09已有历史评估，故报告开发性个人适配结果，不称从未观察的独立测试。

原始源记录索引、100条ID、288条ID、类别数量、split、输入数组stat和哈希写入`plan.json`。A08/A09的100条支持集包含历史20条，query的ID和顺序保持一致。旧65通道HDF5里A04T被错误跳过96条的问题不影响本轮旧缓存：旧2a缓存18个场次均完整288条，不使用该HDF5。

## ⚙️ 输入与优化

旧缓存为22通道、200Hz、800采样点；恢复原parent对应的逐trial 0.1/99.9百分位裁剪、median/IQR缩放（零IQR用std），然后用原空间矩阵插值到65通道。不重新滤波、不使用跨query统计。A08抽查输入与历史缓存逐值一致；正式A08/A09不适配预测必须复现历史逐样本类别，否则停止。

任务文本及四个选项恢复旧catalog，不使用native13任务描述。Tower输出16×256 tokens，经冻结LayerNorm→Linear(256,5120)→GELU→Linear(5120,5120)进入JEV。共享权重不更新，但梯度可穿过JEV回到个人模块。保留原历史Tower MHA fastpath关闭设置、bf16和梯度检查点。

| 设置 | 值 |
| --- | --- |
| 更新预算 | 每人每个训练方法100步 |
| 有效batch / microbatch | 16 / 4 |
| 评估batch | 8，恢复历史推理形状；ICL保持2 |
| 优化器 | AdamW，lr1e-4，weight_decay0.01 |
| 梯度裁剪 | 1 |
| 随机种子 | 基础42；每用户按case order派生，所有方法配对 |
| 抽样 | 沿用当前方法框架，有放回随机抽16条；固定100步不按epoch停止 |
| 等效遍历量 | 100×16/100＝16遍，不保证每条恰好16次 |
| 训练选项 | 沿用当前方法框架，按样本ID/step/seed置换 |
| 评估 | 未适配、模块step0、固定step100；每10步只存断点和loss |
| 调度 | 4卡各一用户，共36个方法/对照结果；完成即停，上限12小时 |

## 👤 当前方法与必要对照

| 对照 | 更新位置 | 个人参数量 | 目标 |
| --- | --- | ---: | --- |
| 不适配 | 无 | 0 | 同人query基线 |
| residual | Tower后256→64→256残差MLP | 33,088 | CE |
| soft_prompt | EEG后8×5120向量 | 40,960 | CE |
| connector_lora | 第二个连接器linear，rank4增量 | 40,960 | CE |
| kv_prefix | 最后4个full-attention层，每层5个KV槽 | 40,960 | CE |
| soft_prompt_kd | soft prompt | 40,960 | CE＋0.1 KL |
| residual_kd | 残差MLP | 33,088 | CE＋0.1 KL |
| soft_prompt_residual | 两个模块联合 | 74,048 | CE |
| icl | 输入support EEG及真实标签示例，不更新权重 | 0 | 前向 |
| icl_shuffled | 同样示例，置换标签 | 0 | ICL对照 |

KV prefix层51/55/59/63，学习post-RoPE K/V；LM共享参数不变。教师只看support，冻结parent按正/倒选项给出重映射后的均值概率，两个教师都预测对才施加KL；温度1，权重0.1。每个人与每种方法重新初始化，不继承上一人的个人参数。

## ⚠️ 解释边界与运行审计

- 主要指标是同用户同query的适配前后差值、被试等权BAcc、负迁移人数，另存训练support Acc。
- 历史20条的64.93%对应A08的residual＋冻结LM，但旧实现使用类别均衡batch、固定选项顺序和不同个人初始化路径。本轮恢复旧parent和输入，沿用最近9方法的优化框架；不是严格只有样本量改变的复现实验。
- 旧parent不同于57.14%的13库模型，不能跨parent、跨用户直接比较平均值。
- 发布LEAF的预训练/指令暴露未完全审计；“未参与本次共享梯度”不等于从未见过其数据。
- 原58条队列与源码归档Git `9955da9`；其已完成结果、断点和所有数据保持原样，当前不恢复。
- 启动先检查26项单元测试、真实27B梯度/冻结/重载，再进入正式任务；源码、配置、parent、缓存索引及输入stat均有检查。失败停止队列，不静默改超参数。

资源/Transformers技能用于同环境权重与显存验证，实验设计技能用于被试/场次隔离和历史比较边界，Markdown技能用于保留可追溯协议；科学技能引用沿用原记录。[^skills]

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
