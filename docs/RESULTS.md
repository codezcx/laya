# 当前 BSR→JEV 试验与必要历史对照

更新：2026-11-01。当前为 SEED 论文协议下的 LEAF 冻结与 BSR 解冻两组适配试验，[协议](PROTOCOL.md)。此前冻结 BSR 四读出和 LEAF 数字均为历史对照，不是新运行的成绩。

## 📋 当前适配试验

- 运行目录：`artifacts/benchmark/encoder_adaptation_seedpaper_20261101`；本节尚无正式验证结果，以运行中的 `pipeline.json` 和每组 `progress.json` 为准。
- `leaf_frozen`：原生 LEAF 全程冻结；`bsr_unfrozen`：CLS 读出，340 步连接器预热后解冻 BSR 前向路径。GPU 4、5，各 2040 步、global batch 16。
- SEED 改为每个被试／session 的原始 trial 顺序 9/3/3：22,455/7,875/7,560 窗，15/15/15 被试。同一 trial 不跨集合。其他 16 库不变，HMC 仍完整 151 份。
- 新完整验证 61,327 窗。旧 55,978 窗及旧 SEED 跨被试结果不能直接当作本轮冻结效应的对照；尚未完成完整 2×2 适配消融。
- BSR 读出选择依据为旧 HMC151 完整验证：CLS 41.6912% 对 mean_pool 41.6742%，差 0.017 个百分点，不宣称统计显著。

以下各节均描述此前实验；训练／验证／测试身份及原分母保留。

## 📋 当前状态

- 当前实验单元测试：76 项通过（`python -m pytest -q tests`），包含完整 HMC 下载门控、下载进程存活检查、保留其他划分、GPU 排队和表征距离诊断测试。未把第三方仓库独立测试套件当作本项目测试；仓库根目录不限定范围的 pytest 会收集缺少独立依赖的第三方测试，不能据此声称全仓库通过。
- 17 库各一个训练样本已在真实 GPU 上通过发布 BSR 前向、全部四种读出的输出／梯度检查；没有执行验证／测试前向来调试这些输入。
- 2a 等短窗输出 1568×768 tokens，SEED-VII／TUAB 15 秒为 3136×768，HMC／SleepEDF 30 秒为 4704×768，后者保留全部三个片段。
- 2040 步全部计划训练输入（32,640 次样本呈现）检查通过；四组各 17 步、batch=16、microbatch=8 的完整 JEV smoke 全部完成。BSR 权重和 buffers 未变、JEV 冻结／开放阶段梯度契约通过、磁盘重载 logits 精确一致，测试前向次数为 0。
- 四组连接器、原生 JEV LoRA／head、BSR 初始化哈希一致。smoke 峰值显存分别为 mean_pool 53.373 GiB、CLS 54.660 GiB、cross_attention 54.062 GiB、qformer 55.076 GiB。每组共同连接器 30,158,336 参数、JEV LoRA 217,579,520、choice 头 81,936，读出参数见协议。
- 原数据版正式四卡任务于 2026-10-30 21:19:57 UTC 启动，2026-10-31 01:06:14 UTC 完成；各组训练 2040 步，随后加载验证子集选中的权重做完整验证。状态见 [pipeline.json](../artifacts/benchmark/bsr_jev_readouts_20261030/pipeline.json)。每组 `train/<arm>/` 保存 step 0 起的验证、训练监测、loss、checkpoint 和输入审计；测试前向均为 0。
- 完整 HMC 已补齐 151 份记录，官方清单 457 个文件全部通过 SHA-256；[下载校验](../artifacts/downloads/hmc_1_1_completion_20261030/complete.json)。新缓存无跳过事件，全库 QC 通过；有效 train 977,664，valid 55,978，test 56,341。未新增无效窗排除，仍仅排除原 CHB_MIT 1,144 个无效训练窗，验证／测试窗全部保留；[QC 摘要](../artifacts/benchmark/multitask_signal_quality_hmc151_20261031/summary.json)。原数据版结果不改写成完整 HMC 成绩。
- 完整 HMC 版控制器于 2026-10-31 10:27 UTC 启动，18:18:26 UTC 完成，四组返回码均为 0。32,640 次计划训练输入预检、四组各 17 步 smoke、共同初始化、checkpoint 重载和冻结 BSR 检查均通过；正式训练从发布权重重新初始化，各完成 2040 步，smoke 不计入预算。四组随后完成各自 best checkpoint 的完整验证，测试前向仍为 0。[pipeline.json](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/pipeline.json) 和各组 `complete.json` 为完成证据。
- 旧 LEAF 续训日志停在 step 3446；切换前当前节点没有存活的旧训练进程，四张卡空闲。未删除或覆盖旧 checkpoint，不把它当成完成 4080 步。

### 完整 HMC 更新

以下为本轮实际缓存与固定划分，不是作者未公开的随机被试名单。其他 16 库的 manifest 和 records.csv 与原数据版逐字节相同。

| 数据集 | 处理后窗口总数（训／验／测） | 任务种类（分类数） | 划分方式 | 被试总数（训／验／测） |
| --- | ---: | --- | --- | ---: |
| HMC 完整数据 | 136,925（92,454/22,421/22,050） | 睡眠分期（5 类） | pinned 官方函数、seed=42，跨被试 | 151（103/24/24） |

证据：[缓存完成记录](../datasets/multitask_hmc151_20261031/hmc/complete.json)、[新划分摘要](../artifacts/benchmark/multitask_splits_hmc151_20261031/summary.json)。总窗口数与论文一致，但论文三份窗口数为 91,681/22,804/22,440，因此仍不能声称具体划分完全相同。新实验从发布权重重新初始化，不复用旧子集下游 checkpoint；四组在 GPU 2、3 排队，每组 2040 步。

### 完整 HMC 版 BSR 完整验证

九库共 55,978 个验证窗口，以下为九库等权平均，不是测试成绩。四组都是单 seed、小预算 pilot；选中 step 来自既定固定验证子集，不根据本次表征分析重新选模。

| 读出 | 选中 step | 完整验证 Acc（%） | 完整验证 BAcc（%） |
| --- | ---: | ---: | ---: |
| mean_pool | 2040 | 44.96 | 41.67 |
| CLS | 1700 | 39.94 | 41.69 |
| cross_attention | 1700 | 40.99 | 40.25 |
| qformer | 1700 | 40.56 | 40.10 |

证据：[mean_pool](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/mean_pool/valid_step002040_full.json)、[CLS](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/cls/valid_step002040_full.json)、[cross_attention](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/cross_attention/valid_step002040_full.json)、[qformer](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/qformer/valid_step002040_full.json)。MI 仍接近四分类随机水平；不把复杂读出在本轮的小幅分差解释为稳定优劣。

### 原数据版 BSR 完整验证

以下均使用包含 HMC 109 份子集的原划分，完整九库验证共 49,069 窗。指标为九库等权平均，不是完整测试成绩。各组文件名中的 `002040` 是训练终点，实际选中权重以 `selected_step` 为准。

| 读出 | 选中 step | 完整验证 Acc（%） | 完整验证 BAcc（%） |
| --- | ---: | ---: | ---: |
| mean_pool | 2040 | 41.43 | 41.44 |
| CLS | 1020 | 43.34 | 38.98 |
| cross_attention | 1700 | 42.93 | 41.43 |
| qformer | 1700 | 33.58 | 34.72 |

证据：[mean_pool](../artifacts/benchmark/bsr_jev_readouts_20261030/train/mean_pool/valid_step002040_full.json)、[CLS](../artifacts/benchmark/bsr_jev_readouts_20261030/train/cls/valid_step002040_full.json)、[cross_attention](../artifacts/benchmark/bsr_jev_readouts_20261030/train/cross_attention/valid_step002040_full.json)、[qformer](../artifacts/benchmark/bsr_jev_readouts_20261030/train/qformer/valid_step002040_full.json)。

## 🔍 四组表征诊断（2026-11-01）

已完成九库各 256 个固定验证窗口、四组相同样本的只读表征提取与可视化。完整结论、方法和图件见[表征分析](REPRESENTATION_ANALYSIS.md)。MI 任务内类别混叠且部分组预测集中于左手；但 BSR 均值表征上已存在低分离，TUAB／TUEV 又给出反例，不能直接归因于文本距离。Q-Former 的 32 个读出 token 平均两两 cosine similarity 达到约 0.9996–0.9998，另保留被试内类别分离诊断。

此次不改变权重或 best checkpoint，测试前向为 0。新图中的 BAcc 来自诊断子集，不是上面的完整验证分母；与原完整验证保存的同窗口预测相比，9,216 次 argmax 均相同。所有原始表征、样本清单、几何指标和 PNG/SVG 存于当前 run 的 `representation_analysis/`。

## 📊 历史 LEAF 完整训练与测试对照

来源为已完成的 2040 步 LEAF→JEV 共享模型，按固定验证128子集选中 step 2040。完整测试 49,431 窗；完整九库训练评估 467,518 窗。训练评估仅做推理，不更新权重、不重复测试。[原训练完成记录](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/complete.json)、[完整训练评估](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/complete.json)。

表内窗口与被试数描述历史 LEAF 冻结的实际划分，不代表论文 baseline 的具体窗口或被试名单。连续切出的窗口不是独立原始 trial。LaBraM、CBraMod、REVE 和最后一列均为测试 BAcc（%），倒数第二列为我们的完整训练 Accuracy（%）；不同指标不能直接相减。原数据版 BSR 沿用相同划分，但预处理、文本位置和训练阶段不同；完整 HMC 版又改变 HMC 数据及被试归属，均不是对 LEAF 的严格单因素消融。

| 数据集 | Trial／窗口数：总计（训／验／测） | 任务种类（分类数） | 本轮划分方式 | 被试数：总计（训／验／测） | LaBraM | CBraMod | REVE | 历史 LEAF→JEV 训练 Acc | 历史 LEAF→JEV 测试 BAcc |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4 类） | 固定跨被试 | 9（5/2/2） | 34.58 | 35.50 | 41.87 | 45.94 | 46.18 |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4 类） | 固定跨被试 | 109（69/19/21） | 43.19 | 41.80 | 58.82 | 37.13 | 33.73 |
| SEED | 37,890（32,838/2,526/2,526） | 情绪识别（3 类） | 随机跨被试；不同于论文 | 15（13/1/1） | 67.71† | 72.25 | 72.62 | 71.54 | 69.00 |
| SEED-VII | 19,420（15,536/1,942/1,942） | 情绪识别（7 类） | 随机跨被试 | 20（16/2/2） | 26.13 | 26.05 | 27.62 | 41.56 | 36.18 |
| TUAB | 272,320（247,728/12,315/12,277） | EEG 异常检测（2 类） | 原训练集＋评估集按被试分层 | 2,328（2,075/101/152） | 80.47 | 81.55 | 81.18 | 65.76 | 63.68 |
| TUEV | 113,353（87,834/12,473/13,046） | EEG 事件分类（6 类） | 按被试分层 | 370（328/20/22） | 71.53 | 69.41 | 74.27 | 50.36 | 39.55 |
| HMC 本地子集 | 98,867（68,215/15,512/15,140） | 睡眠分期（5 类） | 子集随机跨被试 | 109（75/17/17） | 71.63 | 72.03 | 71.93 | 56.68 | 53.92 |
| Workload | 2,134（1,534/300/300） | 认知负荷识别（2 类） | 随机跨被试 | 36（26/5/5） | 67.77 | 74.12 | 74.67 | 84.68 | 71.78 |
| ADFTD | 7,013（4,743/1,115/1,155） | AD／FTD／正常识别（3 类） | 按被试分层 | 88（64/11/13） | 50.27 | 51.95 | 52.12 | 55.03 | 41.34 |

历史九库等权平均：完整训练 Acc 56.52%、BAcc 56.74%；完整测试 Acc 51.71%、BAcc 50.59%。† 保留论文的预训练重叠标记。baseline 数字来自此前核对的 EEG-FM-Bench 论文 Table 2；当前未重新训练这些 baseline。[论文对应仓库](https://github.com/xw1216/EEG-FM-Bench)。

其他八个库只用于训练，不参加验证／测试。以下是 QC 前缓存数量；CHB_MIT 中 1,144 个全零／无效训练窗口被固定 QC 排除，其他数量不变。

| 数据集 | 缓存窗口总数（训／验／测） | 任务种类（分类数） | 划分方式 | 被试总数（训／验／测） |
| --- | ---: | --- | --- | ---: |
| BNCI2015001 | 2,400（2,400/0/0） | 运动想象（2 类） | 全部 train-only | 12（12/0/0） |
| BNCI2014008 | 33,600（33,600/0/0） | P300（2 类） | 全部 train-only | 8（8/0/0） |
| BNCI2014009 | 5,760（5,760/0/0） | P300（2 类） | 全部 train-only | 10（10/0/0） |
| CHB_MIT | 27,436（27,436/0/0） | 癫痫检测（2 类） | 全部 train-only | 23（23/0/0） |
| SleepEDF | 414,961（414,961/0/0） | 睡眠分期（5 类） | 全部 train-only | 78（78/0/0） |
| AlexMI | 480（480/0/0） | 运动想象／静息（3 类） | 全部 train-only | 8（8/0/0） |
| Zhou2016 | 614（614/0/0） | 运动想象（3 类） | 全部 train-only | 4（4/0/0） |
| Nakanishi2015 | 1,800（1,800/0/0） | SSVEP（12 类） | 全部 train-only | 10（10/0/0） |

## ⚠️ 保留的划分、数据与来源问题

- 2a 被试 1–5/6–7/8–9 与仓库固定名单一致；我们训练窗 2,880，论文 2,784。PhysioMI 被试 1–69/70–88/89–109 与代码一致，但我们的测试窗 1,893，论文 1,803。具体数量差异仍待定位。
- SEED 我们按跨被试 13/1/1，论文是被试内 trial 9/3/3，不能把分数当成相同设置。Workload 的窗口数我们 1,534/300/300，论文 1,537/300/297，也非完全一致。
- SEED-VII、TUAB、TUEV、ADFTD 已对同一本地记录重算官方划分函数：被试归属一致、三份窗口数一致；但作者未公开那次运行的完整名单，因此不能证明与论文逐条一致。
- 四库逐窗核查共 412,106 窗：TUAB 272,320 窗和 ADFTD 7,013 窗边界一致；SEED-VII 7,491/19,420 窗、TUEV 7,781/113,353 窗起点相差 1 个 200 Hz 采样点，源于本地 round 与官方 floor。标签、长度及逐记录窗口数一致。缓存未重建，不声称 5 ms 差异一定无影响。
- 历史 HMC 成绩使用 109/151 份记录，75/17/17。当前完整 HMC 151 份记录已补齐，采用新的 103/24/24 分组；原始测试分母及成绩不改写。当前三份窗口数与论文仍不同，作者具体随机被试名单未确认。
- TUAB/TUEV 19 个候选共享文件名被试标识，5 个存在跨库训练／测试角色冲突，精确身份映射未解决。原用户已同意先做验证性实验，不应宣称严格无泄漏成绩。
- SEED 原 CNT 头部异常已在缓存生成时处理；PhysioMI S104R08 第 12 个零起始事件越界，未生成该窗口，但保留记录及被试。上述旧事实的详细证据保存在 Git `741fe34` 的旧 RESULTS 和原逐记录 JSON。
- BSR 预训练来源与目标库重叠未完整排除；本轮不是跨数据集 zero-shot，也不是 EEG-FM-Bench 论文完全复现。

## 📈 保留的历史曲线与当前日志

历史 LEAF step 0–2040 图件未删除：[loss 和验证总图](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_validation_curves.png)、[各库验证](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation_by_dataset.png)、[各库 loss](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_loss_by_dataset.png)。完整训练集准确率只测过 step 2040，不把训练 batch 曲线称为逐 checkpoint 的完整训练准确率。

当前四组指标在各组 `metrics.jsonl`、`valid_step*_subset.json`、`train_step*_subset.json`，完整验证在 `valid_step002040_full.json` 中通过 `selected_step` 标明实际加载的最佳权重。失败或尚未完成不填写准确率。代码变更、历史研究过程和旧方案恢复方式见 [ARCHIVE.md](ARCHIVE.md)。
