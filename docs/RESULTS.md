# EEG→JEV 实验运行记录

_更新于 2026-10-30；当前为 LEAF 多任务验证实验。包含分类数、完整训练／测试成绩、验证及分数据集 loss 曲线、论文 baseline 和划分／窗口核查；以下旧 BSR 结果单独标注协议。_

---

## 📋 当前 LEAF 多任务实验

固定 9 个 benchmark + 8 个 train-only 数据集，2b 排除，用户已同意保留 HMC 子集和跨 TUAB/TUEV 候选身份重合。17 库全量转换后原有 train 954,569、valid 49,069、test 49,431 个窗口／缓存样本。首轮在第 42 步失败；全量 QC 排除 1,144 个无效训练窗口后，有效 train 为 953,425，验证／测试数量不变。修复轮已完成全部 2040 步及九库完整验证／测试。[协议](PROTOCOL.md)记录精确范围与预算。

### 当前续训：step 2040 → 4080

已按用户要求启动同模型的探索性续训，GPU 2、3，追加 2040 步，原划分不变。父模型、Adam 状态与双卡 RNG 从 step 2040 恢复，采用原峰值 20% 的低峰值学习率回升／衰减；详细设置见[续训协议](PROTOCOL.md#续训阶段20414080)和[当前状态](../artifacts/benchmark/leaf_jev_multitask_continue_20261030/pipeline.json)。每 340 步保存验证及固定训练子集的 Accuracy、BAcc、交叉熵与对应模型，只以验证 BAcc 选模，禁止本阶段测试集推理。下列表格和已有曲线仍对应首阶段，不是续训成绩；新监测只从 step 2040 基线开始记录。

### 完整测试结果：分类数与准确率

模型为一个共享的 LEAF→JEV 多任务 SFT 模型，使用固定跨被试划分。按固定验证子集选择的 checkpoint 为 **step 2040**；完整测试共 **49,431** 个窗口，在 `2026-10-29 15:16:47 UTC`（北京时间 23:16:47）结束，正式测试执行一次。以下是完整测试集成绩，不是每库 128 条的验证子集曲线。

第二列列出本轮窗口总数及训练／验证／测试数量；第四列说明划分方式，第五列单独列出被试总数及训练／验证／测试人数。窗口数量统一按模型输入统计，不将连续记录切出的窗口或重复事件标注误称为独立原始 trial。被试数量按[冻结划分清单](../artifacts/benchmark/multitask_splits_20261029_final/summary.json)中的被试标识统计，各库内部三份划分互斥；跨库身份未去重，因此不汇总唯一被试总数。下表 Accuracy 和 Balanced accuracy 均为测试集指标，补充训练集评估另行记录。

| 数据集 | Trial／窗口数：总计（训／验／测） | 任务种类（分类数） | 本轮划分方式 | 被试数：总计（训／验／测） | Accuracy | Balanced accuracy |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4 类） | 固定跨被试 | 9（5/2/2） | 46.18% | 46.18% |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4 类） | 固定跨被试 | 109（69/19/21） | 33.76% | 33.73% |
| SEED | 37,890（32,838/2,526/2,526） | 情绪识别（3 类） | 随机跨被试；与论文不同 | 15（13/1/1） | 69.24% | 69.00% |
| SEED-VII | 19,420（15,536/1,942/1,942） | 情绪识别（7 类） | 随机跨被试 | 20（16/2/2） | 33.68% | 36.18% |
| TUAB | 272,320（247,728/12,315/12,277） | EEG 异常检测（2 类） | 原训练集＋评估集按被试分层 | 2,328（2,075/101/152） | 64.70% | 63.68% |
| TUEV | 113,353（87,834/12,473/13,046） | EEG 事件分类（6 类） | 按被试分层 | 370（328/20/22） | 45.75% | 39.55% |
| HMC 本地子集 | 98,867（68,215/15,512/15,140） | 睡眠分期（5 类） | 子集随机跨被试 | 109（75/17/17） | 53.76% | 53.92% |
| Workload | 2,134（1,534/300/300） | 认知负荷识别（2 类） | 随机跨被试 | 36（26/5/5） | 77.00% | 71.78% |
| ADFTD | 7,013（4,743/1,115/1,155） | AD／FTD／正常识别（3 类） | 按被试分层 | 88（64/11/13） | 41.30% | 41.34% |
| 九库等权平均 | 566,018（467,518/49,069/49,431） | 多任务 | 按各库划分分别计算 | —（跨库未去重） | 51.71% | 50.59% |

Accuracy 是每库全部测试窗口中预测正确的比例；Balanced accuracy 是该库各类别召回率的平均。九库等权平均先分别计算各库指标，再取算术平均，不按窗口数或被试数加权，也不是将所有数据合并计算的总体准确率。

本轮类别含义如下，与[实际任务配置](../configs/benchmark_tasks.json)一致：

- **BCIC-IV-2a（4 类）**：左手、右手、双脚、舌头运动想象。
- **PhysioMI（4 类）**：左手、右手、双拳、双脚运动想象；本轮不是左右手二分类。
- **SEED（3 类）**：消极、中性、积极情绪。
- **SEED-VII（7 类）**：厌恶、恐惧、悲伤、中性、快乐、愤怒、惊讶。
- **TUAB（2 类）**：正常 EEG、异常 EEG。
- **TUEV（6 类）**：棘波和尖波、广泛性周期性癫痫样放电、周期性偏侧癫痫样放电、眼动、伪迹、背景 EEG。
- **HMC（5 类）**：清醒、REM、N1、N2、N3。
- **Workload（2 类）**：静息、心算。
- **ADFTD（3 类）**：阿尔茨海默病、额颞叶痴呆、认知正常对照。

结果来源：[完整评估 JSON](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/complete.json)、[完成状态](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/pipeline.json)。各库逐样本概率、标签和被试 ID 保存在同目录 `test_<dataset>_predictions.npz`；九个文件的样本数量和预测准确率已核对。

解释边界：这是**数据集内跨被试、多任务监督微调**，不是跨数据集 zero-shot。HMC 仅本地 109/151 份记录；TUAB/TUEV 候选跨库身份重合、基础模型预训练重叠风险仍未解决。该表用于本轮验证性实验，不宣称严格无泄漏或与论文默认结果直接可比。

### 论文 baseline：分类数与 Balanced accuracy

外部结果统一取 EEG-FM-Bench arXiv v3 的 Table 3：全参数多任务微调、平均池化分类头。只保留 LaBraM、CBraMod、REVE 三个 baseline，列出本轮使用的 9 个数据集，不混入 Table 2 的单任务结果、Table 4 的冻结主干结果或 Table 5 的 LoRA 结果。以下成绩单位均为 %；论文 baseline 列为测试 Balanced accuracy，不是普通 Accuracy。论文结果为五次实验的均值；为便于阅读省略原文的 ± 项，不代表没有波动。查阅日期：2026-10-30。[^benchmark]

这些 baseline 是论文报告值，不是我们在本地重跑的结果。我们是 17 库混合、2040 步、单 seed 的验证性实验，参数更新范围也不同；论文使用 14 库、30 epochs。下表仅作数值参照，不据此给出公平排名或优于 baseline 的结论。HMC 的论文列为完整数据结果，我们列为本地子集结果。[^benchmark]

按指定顺序汇总为一张表：数据集 → trial／窗口数量 → 任务种类及分类数 → 划分方式 → 被试数量及划分 → 三个 baseline → 本轮训练 Acc → 本轮测试 BAcc。第二、四、五列描述本轮实际数据，不代表论文 baseline 使用了完全相同的窗口与被试名单；论文与本轮的差异见下一节。倒数第二列为 step 2040 模型在完整训练集上的 Accuracy，最后一列及三个 baseline 列为测试 Balanced accuracy；两种指标不能直接相减作为泛化差距。表格不加粗。

| 数据集　　　 | Trial／窗口数：总计（训／验／测） | 任务种类（分类数）　　　　| 本轮划分方式　　　　　　　 | 被试数：总计（训／验／测） | LaBraM | CBraMod | REVE  | 本轮 LEAF→JEV 训练 Acc | 本轮 LEAF→JEV 测试 BAcc |
| --------------| ----------------------------------:| ---------------------------| ----------------------------| ---------------------------:| -------:| --------:| ------:| -----------------------:| ------------------------:|
| BCIC-IV-2a　 | 5,184（2,880/1,152/1,152）　　　　| 运动想象（4 类）　　　　　| 固定跨被试　　　　　　　　 | 9（5/2/2）　　　　　　　　 | 34.58  | 35.50   | 41.87 | 45.94　　　　　　　　　| 46.18　　　　　　　　　 |
| PhysioMI　　 | 9,837（6,210/1,734/1,893）　　　　| 运动想象（4 类）　　　　　| 固定跨被试　　　　　　　　 | 109（69/19/21）　　　　　　| 43.19  | 41.80   | 58.82 | 37.13　　　　　　　　　| 33.73　　　　　　　　　 |
| SEED　　　　 | 37,890（32,838/2,526/2,526）　　　| 情绪识别（3 类）　　　　　| 随机跨被试；与论文不同　　 | 15（13/1/1）　　　　　　　 | 67.71† | 72.25   | 72.62 | 71.54　　　　　　　　　| 69.00　　　　　　　　　 |
| SEED-VII　　 | 19,420（15,536/1,942/1,942）　　　| 情绪识别（7 类）　　　　　| 随机跨被试　　　　　　　　 | 20（16/2/2）　　　　　　　 | 26.13  | 26.05   | 27.62 | 41.56　　　　　　　　　| 36.18　　　　　　　　　 |
| TUAB　　　　 | 272,320（247,728/12,315/12,277）　| EEG 异常检测（2 类）　　　| 原训练集＋评估集按被试分层 | 2,328（2,075/101/152）　　 | 80.47  | 81.55   | 81.18 | 65.76　　　　　　　　　| 63.68　　　　　　　　　 |
| TUEV　　　　 | 113,353（87,834/12,473/13,046）　 | EEG 事件分类（6 类）　　　| 按被试分层　　　　　　　　 | 370（328/20/22）　　　　　 | 71.53  | 69.41   | 74.27 | 50.36　　　　　　　　　| 39.55　　　　　　　　　 |
| HMC 本地子集 | 98,867（68,215/15,512/15,140）　　| 睡眠分期（5 类）　　　　　| 子集随机跨被试　　　　　　 | 109（75/17/17）　　　　　　| 71.63  | 72.03   | 71.93 | 56.68　　　　　　　　　| 53.92　　　　　　　　　 |
| Workload　　 | 2,134（1,534/300/300）　　　　　　| 认知负荷识别（2 类）　　　| 随机跨被试　　　　　　　　 | 36（26/5/5）　　　　　　　 | 67.77  | 74.12   | 74.67 | 84.68　　　　　　　　　| 71.78　　　　　　　　　 |
| ADFTD　　　　| 7,013（4,743/1,115/1,155）　　　　| AD／FTD／正常识别（3 类） | 按被试分层　　　　　　　　 | 88（64/11/13）　　　　　　 | 50.27  | 51.95   | 52.12 | 55.03　　　　　　　　　| 41.34　　　　　　　　　 |

九库完整训练集评估已于 2026-10-30 17:49:20 UTC（北京时间 10 月 31 日 01:49:20）完成，共 467,518 个窗口。使用与原测试相同的 step 2040 checkpoint、固定选项顺序及 microbatch=4，在 GPU 2、3 只做推理；未重新划分、未更新权重、未重复测试集推理。HMC 仍为本轮冻结的 109 份记录缓存，不接入后续补下载数据。[完整训练集结果](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/complete.json)、[评估身份与窗口数](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/identity.json)及同目录 `train_<dataset>_predictions.npz` 保存了可核验的来源；九库逐窗口 ID、标签、混淆矩阵和准确率已核对，双卡评估前后参数及 buffers 未变。

九库等权平均训练 Accuracy 为 56.52%、Balanced accuracy 为 56.74%；对应原测试指标为 51.71%、50.59%，同口径差距分别为 4.81、6.14 个百分点。各库训练 Balanced accuracy 保存在上述 JSON 中，主表按要求仅在倒数第二列填写训练 Accuracy，不将其与测试 BAcc 直接相减。当前完整训练集只评估了 step 2040，不是不同 step 的训练集准确率曲线。

这里评估的是冻结划分中的全部训练窗口，含有限训练预算下没有实际抽到的窗口，并非只测优化器曾采样的样本。现有[训练日志](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/metrics.jsonl)每库记录了 120 个训练 batch，每个 batch 为 16 次样本呈现，可能重复采样；日志中的 `accuracy` 是训练过程中、参数不断更新时的 batch Accuracy，不能直接充当最终模型的完整训练集准确率。[原完整评估文件](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/complete.json)保留不变，仍仅包含验证集与测试集结果。

† 保留论文标记：对应模型的预训练数据与该 benchmark 数据集存在重叠。论文还报告多分类的加权 F1、Cohen's Kappa，以及二分类的 AUROC、AUC-PR；这里仅汇总各任务共同使用的 Balanced accuracy。[^benchmark]

### 与论文的划分及窗口一致性核查

核查针对已完成的缓存和冻结清单，不修改测试名单、不重新计算模型成绩。必须区分：划分函数一致、具体被试名单一致、窗口数量一致、逐窗口边界一致。前一项不能自动证明后一项。

| 数据集　　 | 被试划分核查　　　　　　　　　　　　　　　　　　 | 样本层面核查　　　　　　　　　　　　　　　　　　　　 |
| ------------| --------------------------------------------------| ------------------------------------------------------|
| BCIC-IV-2a | 与仓库固定名单一致：1–5 / 6–7 / 8–9　　　　　　　| 训练窗口我们 2,880，论文 2,784；验证／测试均为 1,152 |
| PhysioMI　 | 与仓库一致：1–69 / 70–88 / 89–109　　　　　　　　| 测试窗口我们 1,893，论文 1,803；训练／验证数量相同　 |
| SEED　　　 | 不同；我们跨被试 13/1/1，论文被试内 trial 9/3/3　| 不能当作同一协议对照　　　　　　　　　　　　　　　　 |
| SEED-VII　 | 官方函数在本地记录上重算一致；作者实际名单未确认 | 三份窗口数一致；部分边界相差 1 个采样点　　　　　　　|
| TUAB　　　 | 官方函数在本地记录上重算一致；作者实际名单未确认 | 三份窗口数一致；同为 200 Hz 时边界一致　　　　　　　 |
| TUEV　　　 | 官方函数在本地记录上重算一致；作者实际名单未确认 | 三份窗口数一致；部分边界相差 1 个采样点　　　　　　　|
| HMC　　　　| 不同；本地 75/17/17，论文完整数据 103/24/24　　　| 本地仅 109/151 份记录　　　　　　　　　　　　　　　　|
| Workload　 | 使用仓库随机被试函数；论文写分层划分　　　　　　 | 我们 1,534/300/300，论文 1,537/300/297　　　　　　　 |
| ADFTD　　　| 官方函数在本地记录上重算一致；作者实际名单未确认 | 三份窗口数一致；同为 200 Hz 时边界一致　　　　　　　 |

论文数字来自 Table 1 和附录 B.3；PhysioMI 原文测试编号上界写 110，但仓库 `np.arange(89, 110)` 实际取 89–109，本表明确按代码核对。2a、PhysioMI 和 Workload 的数量差异尚未定位，不能擅自归因于过滤规则或论文笔误。[^benchmark]

SEED 的差异来自本地划分适配脚本统一设置 `is_cross_subject=True`，而官方 baseline 配置选择 `finetune_sub_dependent`；这是配置选择偏差，不是论文要求全部数据集跨被试。SEED-VII 则不同：论文总述把它归入被试内，但具体条目写按被试 8:1:1，仓库实现也按被试划分，应保留这一原文不一致的说明。[^benchmark][^benchmark-code]

四库逐窗口核查共覆盖 412,106 个窗口、3,676 条源记录。对同一本地记录执行官方事件解析及划分函数，四库记录的 split 归属均未改变；统一到我们使用的 200 Hz 后，用官方切窗函数重算，并按记录、标签、起点、长度的多重集比对，保留 TUEV 重复标注的计数，而不是仅比较总数。标签、长度、每条记录窗口数均一致，未发现越界窗口。

| 数据集 | 训练／验证／测试窗口数（两者相同） | 官方重算边界核查（200 Hz） |
| --- | --- | --- |
| SEED-VII | 15,536 / 1,942 / 1,942 | 7,491 / 19,420 个窗口起点偏移 1 点 |
| TUAB | 247,728 / 12,315 / 12,277 | 272,320 个窗口全部一致 |
| TUEV | 87,834 / 12,473 / 13,046 | 7,781 / 113,353 个窗口起点偏移 1 点 |
| ADFTD | 4,743 / 1,115 / 1,155 | 7,013 个窗口全部一致 |

原因已定位：本地 [window_positions](../eeg_align/multitask_data.py) 用 `round`，官方 `_milli_sec_to_pts` 用 `math.floor`，造成上述 1 点即 5 ms 的边界偏移。该差异尚未修复或重建缓存，当前成绩仍对应原缓存；不能据此未经实验推断它对准确率的影响。这里确认的是元数据和时间边界，不是声称预处理后的信号数组与作者逐字节一致。[^benchmark-code]

此外，TUAB 论文 Table 1 的 Duration 写 30 s，仓库 `finetune` 配置为 15 s，我们使用后者。公开仓库代码树（194 个文件、提交 `325398d7d057ecc1216fb3510d70c16eb60337cc`）和 Releases 核查中未找到作者实际使用的被试／窗口清单，因此仍不能确认与论文那一次运行逐条相同。SEED-VII 本地每位被试均有 971 个窗口，任取 16/2/2 人都会得到同样三份数量，数量相同本身不能证明名单相同。[^benchmark][^benchmark-code]

本地核查来源：[冻结划分目录](../artifacts/benchmark/multitask_splits_20261029_final/summary.json)，以及 [SEED-VII](../datasets/multitask_20261029/seed_vii/index.csv)、[TUAB](../datasets/multitask_20261029/tuab/index.csv)、[TUEV](../datasets/multitask_20261029/tuev/index.csv)、[ADFTD](../datasets/multitask_20261029/adftd/index.csv) 的逐窗口索引。索引／清单哈希和原始记录大小、修改时间核对通过；本次文档补录未改动这些产物。

### 训练、验证和测试数据规模

| benchmark | train 窗口 | valid 窗口 | test 窗口 |
| --- | ---: | ---: | ---: |
| 2a | 2880 | 1152 | 1152 |
| PhysioMI | 6210 | 1734 | 1893 |
| SEED | 32838 | 2526 | 2526 |
| SEED-VII | 15536 | 1942 | 1942 |
| TUAB | 247728 | 12315 | 12277 |
| TUEV | 87834 | 12473 | 13046 |
| HMC 子集 | 68215 | 15512 | 15140 |
| Workload | 1534 | 300 | 300 |
| ADFTD | 4743 | 1115 | 1155 |

9 库合计训练 467,518 条，8 个额外库原有 487,051 条 train-only 缓存样本，QC 后有效 485,907 条。窗口数含 TUEV 上游保留的重复标注，不等同于独立临床事件数。预处理仅有 PhysioMI 的 `S104R08`（test）第 12 个零起始事件越界未生成窗口，原始记录和被试均保留；其余库未发生事件跳过，详情见逐记录 JSON。不会静默补信号或改测试被试。

已完成 [9 库真实训练样本的组件检查](../artifacts/benchmark/leaf_jev_multitask_preflight_20261029/complete.json)：每库 2 条样本，LEAF→JEV loss 和梯度有限，encoder 梯度非零，峰值显存 52.965 GiB。检查用权重已丢弃，测试访问为 0；这些 loss 不代表任务准确率。

正式配方的 [双卡 smoke](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/smoke/complete.json)完成 2 次更新（CHB_MIT、Nakanishi2015），第二步 encoder 梯度范数 0.11269，峰值显存 54.505 GiB；[磁盘重载检查](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029/smoke/reload_check.json) logits 逐元素完全一致，测试访问 0。正式模型另行初始化，不继承 smoke 权重。当时 35 项单元测试通过。

可训练参数：connector 27,535,872，JEV LoRA 217,579,520，decision head 81,936，LEAF 有效路径 24,177,920；主干仍冻结。训练代码提交 `93ed85d`，文档后续提交不改变运行代码哈希。

SEED 原 CNT 的 6 个正数样本长度字段异常，已通过事件表字节偏移重算，不删除被试。第一次失败缓存保留在 `datasets/multitask_20261029_cnt_header_firstpass/`，当前只使用 `datasets/multitask_20261029/`。

### 第 42 步故障与修复重启

首轮在 `2026-10-29T09:38:37Z` 失败，实际完成 41/2040 次更新。第 42 步的 CHB_MIT batch 包含原始行 11504、11880、11333（被试 12），两 rank 的双极适配报 `All-zero EEG epoch`。只有 step 0 的 `best.pt`，尚未生成可恢复的 `last.pt`；不能声称从第 41 步继续。

[源数据核对记录](../artifacts/benchmark/multitask_signal_quality_20261029/CHB_MIT_source_verification.json)确认源 PKL 的 SHA256 与冻结 manifest 一致；源数据已有 1,144 个全零窗口，与转换后的缓存对应全零行一致。1,130 个来自被试 12，其余分布于 02/03/06/13/15/16/18/20；标签 0/1 各 615/529。上游读入缺失导联时保留初始化零值，是可疑机制，但未追溯到每个原始 EDF 的确切生成原因，不能称为已证明蒙太奇不匹配。

全量 [QC 报告](../artifacts/benchmark/multitask_signal_quality_20261029/summary.json)覆盖 17 库 1,053,069 个窗口，仅发现以上全零问题。CHB_MIT 有效训练数从 27,436 变为 26,292。原始数据和索引未修改，排除项按原始 ID 记录；其他数据集、所有被试划分和 held-out 分母不变。修复后第 42 步重新采样的 16 条全部有效；当前 45 项单元测试通过。

修复轮用同一模型、学习率、采样策略和 2040 步预算从发布权重重启。新增启动前全部计划 batch 的适配检查、每 20 步保存，以及 step 0 和解冻边界保存；异常进度和 rank 故障详情单独落盘。首轮失败产物完整保留。

`2026-10-29T11:00:15Z` [重启核验快照](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/restart_verification.json)：代码 `7a37a39`，已完成 60/2040 步。第 42 步 CHB_MIT 正常完成（loss 0.69834，梯度范数 6.9334）；双卡训练峰值分配显存约 54.20 GiB/卡，无测试评估。已读回确认第 40 步 checkpoint 包含 808 个 optimizer 参数状态、scheduler 和两 rank RNG；快照时第 60 步保存正在进行。这是带时间戳的运行检查，不是最终准确率。

[全部计划输入检查](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/signal_preflight.json)验证 2040 步、32,640 次训练样本呈现，耗时 33.77 s。[新 smoke](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/smoke/complete.json)完成两步，解冻 encoder 梯度范数 0.16390，磁盘重载 logits 完全一致。新旧模型初始化参数、buffer 及 step 0 验证结果均完全相同；step 0 九库验证子集 macro balanced accuracy 为 30.2882%，不是训练后或测试集成绩。

新运行日志：[pipeline.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/pipeline.json)、[train/progress.json](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/progress.json)。文件尚不存在时，不能理解为该阶段已启动。训练完成后以各数据集完整测试输出为准。

### 训练与固定验证子集曲线

2040 个训练 loss 点和 7 次固定子集验证已绘图，并在补齐训练集结果后用同一绘图入口重新导出 PNG/PDF/SVG 和 CSV。仅使用本轮 QC 重启日志，不混入首轮失败日志、smoke 或最终完整验证结果；原始日志只读，来源与输出哈希记录在图件清单中。

![训练 loss 原值和过去 51 步均值总体下降，九库固定子集验证 balanced accuracy 从 30.29% 提升到 49.11%](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_validation_curves.png)

*左：各步交叉熵及过去 51 步平均；右：九库等权平均 accuracy / balanced accuracy。竖虚线为 step 103 解冻 LEAF。验证纵轴缩放至观察范围附近；训练 step 0 没有 loss 记录，不补造。*

![九个 benchmark 的固定验证子集 Accuracy 与 Balanced accuracy，横轴从 step 0 到 2040，各图纵轴独立缩放](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation_by_dataset.png)

*每库始终使用同一组 128 条窗口；评估点为 0、340、680、1020、1360、1700、2040。蓝色圆点为 Balanced accuracy，橙色方点为 Accuracy，不平滑；独立纵轴范围已注明。八个额外 train-only 库没有验证集，不画验证曲线。*

![十七个训练数据集各自的交叉熵 loss 原始 batch 值和过去九次该任务更新的滑动平均，所有数据集均有一百二十次更新](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_loss_by_dataset.png)

*覆盖九个 benchmark 和八个额外 train-only 库。灰色为原始 batch loss，蓝色为该库过去 9 次更新的均值，凑满 9 个点才显示。横轴是全局训练 step，每库 120 个观测点；不是每一步都训练所有数据集。各子图纵轴独立缩放，应比较库内趋势，不能用曲线高度直接判断跨任务难度。*

[训练 CSV](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training.csv)、[逐库验证 CSV](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation.csv)、[宏平均 CSV](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation_macro.csv)、[来源哈希和绘图处理记录](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/figure_manifest.json)；同名 PDF/SVG 也保存在该目录。

| update | 验证 macro balanced accuracy | 验证 macro accuracy |
| --- | ---: | ---: |
| 0 | 30.29% | 37.07% |
| 340 | 38.51% | 43.23% |
| 680 | 44.35% | 47.74% |
| 1020 | 44.21% | 47.83% |
| 1360 | 46.49% | 47.74% |
| 1700 | 48.40% | 50.95% |
| 2040 | 49.11% | 52.26% |

前 51 步和后 51 步的平均训练 loss 为 1.4821 / 0.9774。不同任务的类别数不同，总 loss 的波动不宜直接解释为训练不稳定；应结合分任务图。PhysioMI、TUEV 等验证表现仍有波动，九库宏平均提升不代表每库单调改善。这里只描述单次运行，不提供跨 seed 置信区间。当前没有保存验证 loss，因此不能画验证 loss 曲线；以上均不是最终完整测试成绩。

复现命令（已有图时须明确追加 `--overwrite`；只读日志，不加载模型、不改训练代码）：

```bash
laya/.venv-eeg/bin/python scripts/plot_training_curves.py
```

图表的证据保留和呈现方式遵循[协议中的软件过程说明](PROTOCOL.md#-软件过程说明)，已保留相应软件参考文献。

## 📊 历史 BSR：已完成 2a 和已取消 2b

原定 2a/2b，seed 0、官方 Cross fold 0；每个数据集分别运行 cross-attention 与 Q-Former-style，输出均为 32 tokens，正式训练固定 20 epochs。2a 两组完成；用户要求不再训练 2b 后，向已核验的控制器 PID 1373666 发送 SIGTERM，由控制器终止其两名 2b worker。控制器已记录 `status=stopped`、`active={}`、`queued=[]`，相关进程已退出，未终止其他任务。所有既有数据、日志和 checkpoint 保留。

| 数据集 | 压缩器 | 正式 updates | 测试 trial | 最终 accuracy |
| --- | --- | ---: | ---: | --- |
| 2a | cross-attention | 2,880 | 288 | 25.6944% |
| 2a | Q-Former-style | 2,880 | 288 | 25.0000% |
| 2b | cross-attention | 最后进度 474 / 1,560 | 未正式测试 | 用户取消 |
| 2b | Q-Former-style | 最后进度 424 / 1,560 | 未正式测试 | 用户取消 |

2a cross-attention 的预测计数为 `[283, 0, 5, 0]`，Q-Former-style 为 `[288, 0, 0, 0]`，均接近四分类随机水平。两者反转候选顺序后的准确率都是 25%，语义类别一致率都是 0；这提示明显的输出偏置，但尚不能仅凭这些现象确定其原因。完整结果读取各组 `complete.json` 和 `test_predictions.npz`；不能把选项诊断当作另一次模型选择。

后续按用户要求切换到 xw1216/EEG-FM-Bench 的多数据集联合 SFT，先 LEAF，再 BSR-VideoMAE，当前状态见本文首节。不得加载本轮 2a 下游权重：旧训练被试 2–9 包含新协议验证被试 6、7 和测试被试 8、9。原始基础模型的预训练重叠仍需另行审计。

## 🔍 已完成的输入与组件检查

官方 source / masks 哈希核验通过。仅从训练行检查真实输入：2a `[2,22,800]`、2b `[2,3,901]`，均为 200 Hz；保留完整 epoch 和原始通道，不使用旧 LEAF 插值/归一化缓存。

先前随机特征检查在 `artifacts/token_resamplers/smoke_20261028/`：两组均为 `[2,1568,768] → [2,32,768]`，输出与梯度有限；参数量分别 7,115,520 和 37,834,752。该产物不是训练权重或真实 EEG 准确率。

## ✅ 历史启动核验：08:57 UTC

当前实现提交 `5835493`；全仓库 65 项测试通过。控制器 PID `1373666`，输出目录为上文路径。GPU 3 启动时被外部任务占用，待其空闲后才启动本轮 worker，没有终止或修改其他任务。

4 组真实 train-only smoke 全部完成，第 2 步解冻后的梯度如下；每组磁盘 checkpoint 重载 logits 逐元素一致，测试访问均为 0：

| 数据集 | 压缩器 | BSR 梯度范数 | 压缩器梯度范数 |
| --- | --- | ---: | ---: |
| 2a | cross-attention | 0.739894 | 4.524652 |
| 2a | Q-Former-style | 0.296867 | 2.367886 |
| 2b | cross-attention | 0.057141 | 0.317723 |
| 2b | Q-Former-style | 0.200537 | 0.784290 |

同 seed/variant 初始化一致，两种 variant 的共享参数指纹也一致；共享 SHA256 为 `ffb60049ecac0b8f3b6c183f69b4a2da9c41dc822e8f17cd5aa9353f647d15e8`。

该历史核验时刻，GPU 2 上的 2a cross-attention、GPU 3 上的 2a Q-Former-style 均完成 16/2,880 updates；loss 有限，压缩器梯度非零，测试访问为 0。正式前 100 步处于 BSR 冻结预热，之后自动解冻。这不是当前状态：2a 已结束，2b 已取消，队列不再自动启动任何训练。

这只是带时间戳的启动快照，最新进度应读取 `pipeline.json` 和每组 `progress.json`，不能将早期训练 batch accuracy 当作最终测试结果。

## ⚠️ 解释边界与历史

BSR 的 Stage-B 包含 2b，预训练被试/trial 重叠未排除；下游被试隔离不等于 foundation pretraining 无泄漏。不将本轮称为干净 zero-shot，也不把两组准确率差异只归因于 self-attention（参数量不匹配）。本轮完整旧协议见 Git `14c60cf:docs/PROTOCOL.md`；[当前协议](PROTOCOL.md)记录正在进行的 xw1216 多数据集迁移。

LEAF 当前轮以前的协议和启动记录在 Git `3033ac9`；旧实际产物保留于 `artifacts/benchmark/official_cross_leaf_jev_20261027*/`。更早结果见[归档索引](ARCHIVE.md)。此次没有删除旧权重、数据、缓存或日志。

[^benchmark]: Xiong, W., Li, J., Li, J., Zhu, K., & Jiang, C. (2026). *EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation and Diagnostic Analyses of EEG Foundation Models*. arXiv:2508.17742v3, Table 1–5 and Appendix B.3–B.5. https://arxiv.org/html/2508.17742v3 （本轮查阅：2026-10-30）。

[^benchmark-code]: xw1216. *EEG-FM-Bench*, commit `325398d7d057ecc1216fb3510d70c16eb60337cc`. https://github.com/xw1216/EEG-FM-Bench/tree/325398d7d057ecc1216fb3510d70c16eb60337cc 。关键文件：`data/processor/builder.py`、`data/dataset/seeds/seed.py`、`data/dataset/seeds/seed_vii.py`、`data/dataset/tue/tuab.py`、`data/dataset/workload.py`、`assets/conf/baseline/reve/reve_unified.yaml`。
