# 个人 LoRA 元初始化：固定机制验证协议

_2026-11-25；用户批准实际训练。先验证能否改善同一新用户的100条有标签适配，不增加其他模块。_

## 🎯 问题与假设

普通个人LoRA直接从随机A、零B初始化，优化本人support的CE；它没有显式学习“少量更新后在本人另一批样本上泛化”。这里采用MAML的初始化学习思想[^maml]，只对40,960个个人参数做一阶近似，不做27B模型的二阶反传。假设是学到的初始化可提高个人适配后的query BAcc；不是预设一定优于继续训练。

原已训练模型不从头重训：parent为 `artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360，SHA256 `7c58002990c707e8f42c6de411eec61fb2d2b55d21406ab4024975708dba7a6a`。

## 📋 数据与成员

以下为历史parent缓存的完整分配计数，顺序训／验／测，不是本pilot全部使用的样本。

| 数据集 | 处理后试次/窗口总计（训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（训／验／测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,184试次（2,880／1,152／1,152） | 运动想象（4） | 固定跨被试 | 9（5／2／2） |
| HMC | 136,925窗口（92,454／22,421／22,050） | 睡眠分期（5） | 原缓存跨被试 | 151（103／24／24） |

实际使用：

- 2a源用户A01–A05：每人从T场次取100个类别平衡support，从E场次另取100个平衡source-query。两场次在历史分配中都属于train。
- HMC源用户3、5、10、17、25、28、34、35：按数值ID取最前8名符合条件的train用户。每人前60%时间段取100个support，隔开2个30秒epoch，再从后段取100个source-query。条件是两段均至少20个每类epoch，不按模型分数选人。
- 2a目标用户A08/A09：每人T场次100个support，完整E场次288个query。保留上一轮精确ID、排序、个人seed。
- HMC目标用户s2/s6：每人100个support，原来的完整后段340/344个query，保留2epoch隔离和上一轮精确ID。
- 源用户与目标用户不重合。源support/query均为原train；目标为原test的个人适配协议。原valid本轮不读取信号、不参与梯度或选点。样本成员与原split标签不改。
- HMC100条是睡眠epoch，不是100个独立用户或场次。稀有阶段不足者不进入本pilot源池，存在选择偏差。

所有样本、原split、source-group及完整batch ID序列固定在 `plan.json`。数据/代码hash和array大小、mtime校验；支持源train守卫、个人support守卫两套检查。

## ⚙️ 方法与预算

冻结主干：LEAF Tower → 原连接器 → JEV。个人LoRA加在连接器最后的5120→5120层，`delta(z)=B(Az)`，rank4、scale1、40,960参数。训练和推理都只输入当前EEG＋原问题/选项，不插入有标签示例，不引入类别原型。

| 组别 | 离线阶段 | 新用户阶段 |
| --- | --- | --- |
| 普通个人LoRA | 无 | 按原个人seed随机A、零B；100support训练100步 |
| source ERM | 同一LoRA连续用source-support和source-query做CE | 加载学习后的共同初始化；100support训练100步 |
| FOMAML | 每episode克隆共同初始化；本人support更新临时LoRA，再由本人source-query梯度更新共同初始化 | 加载元初始化；100support训练100步 |

FOMAML逐episode流程：

1. 从源用户循环抽一个人；临时LoRA拷贝共同初始化，inner AdamW状态重置。
2. 本人的support做5次batch16更新。
3. 本人的另一批source-query做1次batch16前向/反向。将临时参数上的query梯度拷给共同初始化，省略inner更新的Jacobian，再做一次outer AdamW更新。
4. 丢弃临时LoRA；下一episode重新从更新后的共同初始化开始。

source ERM按完全相同的用户次序、6个batch的样本、选项置换依次更新同一组参数。两组都是60episode、360个梯度batch、5,760次样本呈现；ERM有360次优化器step，FOMAML有300次inner＋60次outer，这是算法差异，不声称优化器状态轨迹相同。每次batch前固定随机状态，支持断点重现。

| 设置 | 固定值 |
| --- | --- |
| 共用初始LoRA | seed42、随机A/零B，ERM与FOMAML完全相同 |
| 离线预算 | 60episode，每次1源用户、5inner＋1query |
| 离线inner/outer学习率 | 1e-4；outer是新增参数，明确取相同值 |
| 所有优化器 | AdamW，weight_decay0.01，grad_clip1，无scheduler |
| batch / microbatch | 16 / 4 |
| 每人适配 | 100support，100steps；有放回抽样，与旧个人采样ID一致，并确认全部100条被使用 |
| 选点 | 固定离线episode60＋个人step100，禁止query挑最好checkpoint |
| 评估 | 每名目标用户完整query，未适配、初始化后未个人训练、最终适配；另报告support Acc/BAcc |
| 设备 | GPU0/1：2a FOMAML/ERM；GPU2/3：HMC FOMAML/ERM |
| 范围 | 每任务2名目标用户，seed42，不扩展到所有人 |
| 时间 | 6小时上限；完成即退；每episode与每10个人step保存 |

每个worker还独立重跑普通个人LoRA，两卡同用户的普通基线必须一致。基线个人A仍沿用历史用户seed，不假装与共同seed42完全相同；主要因果比较为ERM与FOMAML，二者初值与样本暴露完全一致。历史不适配预测需逐样本复现。

## 🔍 预处理、检查与判据

保持parent预处理：最小范数双极导联转换、200Hz、逐epoch裁剪及median/IQR归一化、65通道插值；2a为16个256维token，HMC保留完整30秒的120个token，不裁短。冻结Tower特征可缓存，样本ID和代码hash不符则拒绝复用。

真实模型检查包括零LoRA恒等、inner更新不改变共同初始化、非零且有限的source-query meta梯度、初始化确实更新、精确重载及共享参数/缓冲未改变。每次梯度不累积到冻结JEV参数，但允许梯度经过JEV流向个人LoRA。

主要结果按任务报告同人BAcc差值、个人support训练表现及负迁移人数。每任务2人只作机制验证，不给小样本显著性结论。必须比较FOMAML相对ERM，而不只比较相对未适配的收益。初始化本身已改变输出，因此同时报告适配前与适配后的分数。

短inner-loop只有5步、部署适配100步，是已知近似限制；本轮固定此预算，不依据目标query追加训练。两个任务分别学习初始化，尚未证明跨任务元迁移。

## ⚠️ 历史暴露与归档

原test已有历史观察，原valid也曾被用过；本轮遵守当前开发角色，不称盲测。原train被试也参与过parent共享训练，LEAF发布权重的预训练暴露未完全审计。个人支持样本来自原test用户是既有、已授权的个人校准流程，不能混称普通无适配测试。

前轮42份结果、ICL代码与协议归档Git `9a96ae0`；当前只保留元初始化及匹配控制。所有ignored产物保留。方法检索依据见[研究记录](PERSONALIZATION_METHOD_REVIEW.md)，后续尝试项不是本轮已执行内容。

实验设计技能用于同人配对、样本隔离、预算匹配；资源及模型技能用于原环境/冻结梯度检查，Markdown技能用于协议记录。Hugging Science目录未找到EEG匹配条目，未因此更换既定模型或下载新权重。技能来源[^skills]不属于模型创新。

[^maml]: Finn, C., Abbeel, P., Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML. https://proceedings.mlr.press/v70/finn17a.html
[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
