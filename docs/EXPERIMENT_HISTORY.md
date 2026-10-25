# EEG → REVE / LEAF → JEV 实验总账

核对日期：2026-10-25。本文汇总已经运行过的实验、各轮 JEV 的训练数据及已经留下结果文件的泛化评估。按实验方案归类；续训、换节点、重复种子和 smoke test 不各自算成新方法。

这是历史索引，不恢复旧实现到当前工作树。当前代码和协议以 [PROTOCOL.md](PROTOCOL.md)、[RESULTS.md](RESULTS.md) 为准；旧实现通过 Git 恢复，产物通过 [ARCHIVE.md](ARCHIVE.md) 定位。

## 1. 阅读口径

- **早期 Qwen-8B ≠ JEV-27B**：最早的 REVE→语言塔使用 CerebraGloss 的 Qwen3-VL-8B，后来才接原生 JEV-27B。
- **对比学习分类 ≠ JEV 分类**：前者根据 EEG 与候选文本的相似度选类别；后者将信号 token 输入语言塔，由决策头回答。
- **跨被试验证 ≠ 跨数据集测试**：训练过某数据集、评估其留出被试，属于前者。整个目标数据集未参加本轮下游训练，才属于后者。
- 下文“未参加训练”指该轮下游监督训练。REVE 预训练包含 2b，LEAF 的预训练暴露也未全部排除，因此不能据此宣称基础模型完全没见过目标数据。
- 选项交换测试检查文本选项与输出位置的对应关系；EEG 错配检查信号依赖。这两者都不是新的跨数据集测试。
- “未确认外测”表示本次核对没有找到可对应的完成结果，不把仅有 heldout 配置或用户曾提出的测试请求算作已完成。
- 表内结果标明选中 checkpoint 的验证分数或具体外测分数；不同数据池、划分、窗口/trial 口径不能混作统一排行榜。

## 2. 数据池定义与名称对应

| 数据池 | 完整数据集列表 | 用途及说明 |
| --- | --- | --- |
| M3：早期三源 MI | Cho2017、Dreyer2023、Lee2019_MI（OpenBMI） | 左右手二分类；早期通常使用 C3/Cz/C4 |
| E2：早期情绪源 | SEED、FACED-NEMAR | 映射为负性／中性／正性情绪；FACED 不是后期 SEED-VII |
| M9：后期九源 MI | BNCI2014_001（2a，仅左右手）、Chang2025、Cho2017、Dreyer2023、Kumar2024、Lee2019_MI（OpenBMI）、PhysionetMI、Weibo2014、Zhou2016 | 全部可用 EEG 通道，实际训练清单共 49,193 trials；验证 12,508 trials |
| E3/7：后期情绪源 | SEED（三分类）、SEED-VII（完整七分类） | 用于 MI＋情绪联合对齐与后续 JEV |
| MAT | EEGMAT / MAT | 静息与心算二分类；不是情绪分类 |
| L2：LEAF 双数据集 | 2a（完整四分类）、SEED（三分类） | 2a 包含左手、右手、双脚、舌；不同于 M9 中的 2a 二分类 |
| B11：LEAF benchmark | BNCI2014001（2a）、BNCI2014004（2b）、BNCI2015001、BNCI2014008、BNCI2014009、CHB-MIT、TUAB、SleepEDF、SEED、EEGMAT、Nakanishi2015 | 11 个数据集联合训练；当前只运行跨被试 fold 0 |

命名统一：`BNCI2014_001` 与 `BNCI2014001` 都是 2a；`BNCI2014_004` 与 `BNCI2014004` 都是 2b；`Lee2019_MI` 就是本项目所称 OpenBMI。BNCI2015001 是右手／双脚二分类，不是左右手。

M9 的九源名称同时由历史 `mi_jev_tuab_lr_continued/config.json` 和现存 `datasets/mi_tokens/train.jsonl` 核对。`datasets/mi_index/train.jsonl` 还列过 Brandl2020、GuttmannFlury2025_MI，但这两者因当时坐标缺失被训练配置排除，**不能计入 M9 已训练数据**。Chang2025 索引条数与最终训练清单也不同，应以实际训练 manifest 为准。

## 3. REVE 直接监督训练及编码器对照

主结构：`EEG → REVE → 连接层 → 语言塔 → 分类交叉熵`。REVE-only 对照不经过语言塔。

| 编号 | 已运行方案 | 下游训练数据 | 验证／泛化测试 | 结果与状态 | 归档目录（相对归档 artifacts） |
| --- | --- | --- | --- | --- | --- |
| R01 | 早期 REVE→Qwen-8B；冻结编码器、训练连接层及 LoRA | M3；开发期也有 Cho＋Dreyer 两源组 | 源域留出被试；Lee 整域开发验证；外部 2a 左右手、2b E-session | 有完整报告；源域可学习，外部迁移较弱。属于 Qwen-8B，不是原生 JEV | `reve_mi_multi`、`mi_zero_shot_v1` |
| R02 | REVE-only：冻结 Linear probe vs 全量微调 | M3，和对应 Qwen-8B 组固定划分一致 | 内部测试；外部 2a 左右手及 2b E-session | 内部 pooled accuracy 63.21% / 67.81%；2b 61.44% / 62.18% | `reve_only_multi/train_20261015_1404` |
| R03 | 同一 Qwen 模型不同阶段重新训练线性探针 | M3 的训练特征 | 相同内部测试、外部 2a/2b | 分别读 REVE、connector、Qwen EEG token、最后读出；2b 可读性在后段下降，不能只归因于最终 Linear | `reve_stage_diagnostic/run_20261015_1444` |
| R04 | 冻结接口小样本拟合、分类与属性问答对照 | Cho2017＋Dreyer2023 | 源验证、零信号、错配、选项交换；该轮不读 2b | tiny 组能拟合，较大源数据组未通过当时门槛 | `mi_interface_v2` |
| R05 | REVE→JEV 单窗属性问答 | M3＋E2 | 五个源数据集的留出被试属性题；错配诊断 | 已运行至有 step 800 验证；旧 MI 前后段题后来被替换 | `jev_attribute_qa/run` |
| R06 | 配对基线 MI 属性问答，后改为 MI＋情绪分类联合监督 | **Lee2019_MI＋SEED＋FACED-NEMAR** | 源验证、错配、选项交换；冻结读出在外部 2b 的评估 | MI 基线/任务配对只用 OpenBMI；稀疏分类标签控制和联合组均有记录 | `jev_paired_qa/control_sparse_labels`、`jev_paired_qa/run`、`jev_paired_qa/readout_evaluation` |
| R07 | 关闭属性辅助损失的对照 | Lee2019_MI＋SEED＋FACED-NEMAR | 相同源验证 | 已启动且保存至 step 100 验证；本次未找到完成标记，不写成完整收敛结果 | `jev_no_attribute_loss/run` |
| R08 | REVE→JEV，仅 TUAB 正常／异常 | TUAB | TUAB 留出验证，记录级汇聚窗口 logits；不是跨数据集 | 最佳验证记录级 accuracy 78.41%，step 150；随后按要求停止 | `tuab_jev/run` |
| R09 | REVE→JEV，固定 A=左手/B=右手 | M9 | 九源留出被试；后续完成外部 2b、Emotiv、Beetl A/B、Shin | 初始组类别塌缩；改用 TUAB 的优化设置后可训练，最佳源宏平均 BA 71.25% | `mi_jev/run`、`mi_jev/resumed`、`mi_jev_tuab_lr`、`mi_jev_tuab_lr_resumed`、`mi_jev_tuab_lr_continued` |
| R10 | REVE-only M9 对照：空间池化、全部 token 均值池化、解冻编码器 | M9 | 九源验证；部分组有 2b 外测；另有留出 2a 的专门对照 | 最佳源宏平均 BA：空间 LP 70.91%、均值 LP 57.91%、全量微调 81.30%；优化预算不同 | `mi_reve_probe`、`mi_reve_probe_mean`、`mi_reve_finetune` |
| R11 | 直接 JEV 随机交换左右手选项 | M9 | 九源正常／反序选项验证 | 用于检查固定位置捷径；随后改为先预训练 REVE 再接 JEV | `mi_jev_option_shuffle` |

R06 的数据已直接核对：训练 MI 8,800 trials，全部来自 Lee2019_MI；情绪 24,477 窗口＝SEED 4,050＋FACED-NEMAR 20,427。Cho/Dreyer 当时的 task-only 缓存没有配对基线，未进入该版 MI 训练。不能因为它们存在于旧 source cache，就说这一轮也训练了它们。

R09 中目录名 `tuab_lr` 指借用 TUAB 的优化设置，**不代表混入 TUAB 数据或使用 TUAB 微调权重**；其训练数据仍是 M9。

## 4. REVE 文本对齐与后续 JEV

### 4.1 早期文本原型与五数据集 JEV

| 编号 | 已运行方案 | 训练数据 | JEV 是否参加训练 | 已做泛化评估／说明 | 归档目录 |
| --- | --- | --- | --- | --- | --- |
| C01 | 文本 CE、sigmoid 对齐、随机原型、可学习分类头 | 起初 Cho＋Dreyer＋SEED，后扩到 M3，另加入 FACED | 否；文本模型只离线编码描述 | Lee 整域验证、2b、SEED-VII 的 Happy/Neutral/Sad 子集；不是 SEED-VII 完整七分类 | `multitask_prototypes_v1`、`multitask_prototypes_v2_*` |
| C02 | 查询／空间池化、REVE 末两层适配、频谱融合、完整源池、分任务读出 | 各组使用 Cho/Dreyer/Lee 的指定组合＋SEED；后续情绪组加入 FACED | 这些表征／读出对照不应直接算成 JEV 成绩 | 各组的源选择规则不同；报告中分别列有 2b、SEED-VII-HNS 外测 | `channel_prototypes_v5`、`spatial_prototypes_v6`、`spatial_repeats_v7`、`reve_tail_v8`、`trial_aggregation_v9`、`spectral_fusion_v10_*`、`full_sources_v12_*`、`task_readouts_v13_*` |
| C03 | “任务＋类别”文本对齐，CE/contrastive 两组、seed 42/43 | M3＋E2 | 否 | 源验证及 2b、SEED-VII-HNS；另有 heldout wording 文本模板 | `eeg_text_alignment/runs` |
| C04 | 五选项 JEV：只训投影器、投影器＋LoRA、再加 REVE 解冻；LoRA seed 42/43 | **M3＋E2**，共五个数据集 | 是；原生 JEV LoRA 始终启用，“只训投影器”仅指不更新 LoRA | 五源留出被试及外部 2b、SEED-VII-HNS；固定 A–E 同时表示情绪三类与 MI 两类 | `jev_eeg/runs/{projector_seed42,lora_seed42,lora_seed43,lora_encoder_seed42}` |
| C05 | 五数据集覆盖采样长训练：投影器 vs 投影器＋LoRA | **M3＋E2** | 是 | 五源完整验证及同一 2b、SEED-VII-HNS；LoRA 组外部 2b accuracy 56.87%、HNS 被试宏平均 BA 37.70% | `jev_eeg_coverage/runs/{projector_seed42,lora_seed42}` |

更早的 `emotion_*`、`faced_*`、`mi_covariance_v23`、`mi_reve_scaling_v24`、`mi_fusion_v25`、`task_subject_disentangle`、`multitask_bridge_v30/v31` 等目录还保留了情绪多源、空间／频谱读出、任务与被试解耦及桥接准备实验。这些属于支撑性表征探索；本文不根据目录名称把它们一律计为 JEV 训练，也不把所有组误记为同一数据划分。具体逐组旧报告由 [ARCHIVE.md](ARCHIVE.md) 索引。

### 4.2 九源 MI 预对齐、情绪混合与 MAT 混合

| 编号 | 已运行方案 | 对齐阶段训练数据 | 后续 JEV 的训练数据 | 验证／泛化测试与结果 | 归档目录 |
| --- | --- | --- | --- | --- | --- |
| C06 | MI 分类 CE 预训练 → 随机选项 JEV | M9 | **M9** | 九源留出被试，正常／反序选项；最佳 checkpoint 正常顺序宏平均 BA 74.84%。配置保留 2b，但本次未确认该组新增 2b 完成外测 | `mi_precondition_ce`、`mi_jev_precondition_ce` |
| C07 | MI 文本对比预训练 → 随机选项 JEV | M9 | **M9** | 对齐阶段宏平均 BA 76.53%；JEV 最佳组正常／反序 77.14% / 77.64%。不能当作独立外测成绩 | `mi_precondition_contrast`、`mi_jev_precondition_contrast` |
| C08 | MI＋SEED＋SEED-VII 联合对齐 → JEV | M9＋SEED＋SEED-VII | **M9＋SEED＋SEED-VII** | 各源留出被试、窗口/trial 指标、选项交换；SEED-VII 已参与训练，因此这轮不是 SEED→SEED-VII 跨数据集测试 | `mi_emotion/contrast`、`mi_emotion/jev`、`mi_emotion/jev_restart` |
| C09 | 保留完整 MI 覆盖重新对齐 | M9＋SEED＋SEED-VII | 未确认使用这版 encoder 重新完成 JEV；上行 JEV 的 config 指向最初 `contrast/best.pt` | 完整源验证；最佳记录 MI BA 54.38%。不能把 `jev_restart` 自动归给此 encoder | `mi_emotion/contrast_full_mi` |
| C10 | 只保留 MI 损失 | 有效监督为 M9；情绪损失权重为零 | 未确认后续 JEV | MI 最佳记录 BA 80.37%，用于定位联合优化影响 | `mi_emotion/contrast_mi_loss_only` |
| C11 | 只混 SEED 三分类；共享头 vs 分任务头 | 有效监督为 M9＋SEED，SEED-VII 权重为零 | 未确认这两版分别接 JEV | 最佳记录 MI BA 53.63% / 53.45%；配置里仍出现 SEED-VII 文本，不代表其有监督梯度 | `mi_emotion/contrast_mi_seed`、`mi_emotion/contrast_mi_seed_split` |
| C12 | 丢弃对比头，冻结 REVE，重新拟合 MI/SEED 线性探针 | M9＋SEED，同一划分 | 否 | 验证 MI BA 61.35%、SEED 57.04%；这是 probe，不是 JEV | `mi_emotion/probe_mi_seed_encoder` |
| C13 | MI＋MAT 分任务头对齐 → JEV | M9＋MAT | **M9＋MAT** | 对齐最佳记录 MI BA 77.05%；JEV 最佳 step 400：MI 79.16%、MAT trial BA 62.50%。MAT 验证仅 8 trials；未确认新增外部数据集测试 | `mi_mat/contrast`、`mi_mat/jev` |

C07 的对齐使用 Qwen3-Embedding-0.6B 编码任务＋类别描述，温度 0.07、双向多正样本 InfoNCE。例如：`Task: motor imagery recognition. Imagined movement: left hand.`。JEV 训练是另一个阶段，不能用对齐阶段的相似度分类分数替代。

C13 的 JEV config 明确加载 `mi_mat/contrast/last.pt`，不是对齐的 `best.pt`。上表分别报告各阶段选中结果，不将其当作同一 checkpoint 的严格前后比较。

## 5. LEAF 路线：每轮 JEV 训练数据

使用发布的 `leaf-v1.0-instruct-mpnet-base.ckpt`，不是从头复现 LEAF 的完整预训练。

| 编号 | 已运行方案 | JEV 下游训练数据 | 验证／泛化测试 | 结果与状态 | 产物目录 |
| --- | --- | --- | --- | --- | --- |
| L01 | 原生 LEAF 分类基线 | 不训练 JEV，也不做本轮 LEAF SFT | L2 的固定验证集 | 2a 四分类 37.15%；SEED 窗口 65.00%、trial 67.41% | 归档 `leaf_2a_seed/baseline.json` |
| L02 | 冻结 LEAF，训练连接层＋JEV LoRA | **2a 四分类＋SEED 三分类**；训练 3,456＋22,734 条 trial/窗口 | L2 留出验证，正常／反序选项；外部 Cho/OpenBMI/PhysionetMI | 完成 600 步；2a 54.34%；SEED 窗口 68.52%、trial 77.04% | 归档 `leaf_2a_seed/jev`、`leaf_2a_seed/external_mi` |
| L03 | 11 数据集联合 SFT，恒定 LR | **B11 全部数据集** | 同 11 数据集留出被试的固定验证子集；不是跨数据集测试 | 记录至 1087 步，最近权重 1000；最高验证宏平均 BA 50.05%（600） | 归档 `benchmark/cross_subject_fold0`；当前保留 `benchmark/reference_constant_lr` |
| L04 | warmup＋cosine LR | **B11 全部数据集** | 同 fold 0 验证 | 节点中断于早期，保留 step 100 共同初始化；不能当完整 LR 消融结果 | 归档 `benchmark/cross_subject_fold0_lr_cosine`；当前 `benchmark/reference_cosine_partial` |
| L05 | 同一初始化的 BN 更新／BN 冻结对照 | **B11 全部数据集** | 同 fold 0 验证；两组到 1200；未做独立 test | step 1200 宏平均 BA 51.85% / 55.39%；冻结组最好 step 600 为 55.59% | 当前 `benchmark/cross_subject_fold0_bn_control` |
| L06 | 相同 step 600 起点，仅替换 2a 为 LEAF 原始预处理 | **B11 全部数据集**；仅 2a 输入改变 | 相同 fold 0 验证，续训至 800；未做独立 test | 已完成。step 800：2a baseline 24.65% / raw 32.99%；整体 BA 53.40% / 54.70% | 当前 `benchmark/cross_subject_fold0_leaf_preprocessing` |

B11 不是 11 个独立模型：训练的是一个共享模型，由任务文本指定分类问题、随机排列候选选项。前 100 步冻结 LEAF，之后训练 LEAF、连接层和 JEV LoRA；“冻结 BN”仅冻结运行统计量，不等于冻结全部 LEAF 参数。

当前 B11 训练池 476,032 个 epochs/windows，完整验证池 58,362，训练中固定监控子集 9,748；test 已划分，但不因此算作已完成测试。仅 fold 0，不是完整 LOSO/10-fold。TUAB 的当前被试划分也不同于其官方 train/eval 划分。

L06 的 raw 预处理包含 CAR、滤波、重采样及按被试汇总的无标签归一化，是组合改变；验证被试统计参与归一化属于 transductive/batch 处理，不能表述为严格单 trial 在线预处理。

## 6. 已有结果文件的泛化性测试清单

### 6.1 跨数据集／外部数据评估

| 训练模型／训练源 | 外部评估数据 | 已确认结果 | 不能混淆的地方 | 证据（归档 artifacts 下） |
| --- | --- | --- | --- | --- |
| 早期 M3 → Qwen-8B；匹配 REVE-only | 2a 左右手，2b E-session | 各组已完成；见原报告 | 2b E-session 2,840 trials；不是后来 6,520 条的整池口径 | `mi_zero_shot_v1/RESULTS.md`、`reve_only_multi/train_20261015_1404/RESULTS.md` |
| M3＋E2 → 五选项 JEV，含覆盖采样组 | 2b E-session；SEED-VII-HNS | 有 `external.json`；例如覆盖采样 LoRA 组 2b 56.87%、HNS 被试 BA 37.70% | HNS 只取 Happy/Neutral/Sad，并映射三种 valence，不是七分类 | `jev_eeg/runs/*/external.json`、`jev_eeg_coverage/runs/*/external.json` |
| OpenBMI＋E2 配对问答模型及新读出 | 2b | 原 JEV 头 50.60%；联合新读出最高列 63.84%；匹配 REVE-only 64.96% | 重新训练读出的提升不能记为原 JEV 输出头的提升；新读出未胜过 REVE-only | `jev_paired_qa/readout_evaluation/REPORT.md` |
| M9 → 固定选项 JEV，step 2307 | 2b 留出数据池 | **6,520 trials，accuracy 64.95%** | 全部 2b 不参与该轮下游训练；与 2,840 E-session 不能直接横比 | `mi_jev_tuab_lr_continued/test_2b.json` |
| 同一 M9 JEV 权重 | 本地 `mne_data/zhangchenfei/` Emotiv 数据 | **880 trials，accuracy 34.09%** | 本地 14 通道外部记录；保留原数据标签映射，不按分数翻标签 | `emotiv_jev_zero_shot/config.json`、`metrics.json` |
| 同一 M9 JEV 权重 | Beetl2021_A、Beetl2021_B、Shin2017A | **51.00%（100）、48.33%（180）、51.26%（1,740）** | 括号为 trial 数；该轮只取左右手标签 | `external_mi_jev_zero_shot/config.json`、`metrics.json` |
| L2 → LEAF＋JEV | Cho2017、OpenBMI、PhysionetMI | **61.35%（2,000）、75.82%（2,200）、69.99%（993）** | 未用这些数据做该轮 SFT；使用现有 MI 验证子集，不是各数据集全量测试集；LEAF 预训练暴露未排除 | `leaf_2a_seed/external_mi/config.json`、`results.json` |

AlexMI 曾被提出用于外测，但本次在对应归档中未找到可核实的独立完成结果，因此不列作已完成。OpenBMI、Zhou2016 对 **M9 模型**属于训练数据集，不能称跨数据集 zero-shot；OpenBMI 对只训练 **L2 的 LEAF＋JEV**才是未参与该轮下游 SFT 的数据集。

SEED-VII 的身份随实验改变：早期 HNS 是外部目标；C08 完整 SEED-VII 已加入训练，这时其留出被试成绩只能叫跨被试验证。

### 6.2 跨被试、通道和语言接口诊断

| 类型 | 已做实验 | 解释范围 |
| --- | --- | --- |
| 跨被试 | M3、M9、MI＋情绪、MI＋MAT、L2，以及 B11 fold 0 的留出被试验证 | 主要用于选择 checkpoint；被反复查看的验证集不是最终独立测试 |
| 留一数据集与通道对照 | REVE-only 不含 2a 的训练；同一批 2a 比较全部 22 通道与 C3/Cz/C4 | 评估通道改变对同一模型的影响；这是 REVE-only 对照，不能记成 JEV 新训练 |
| 选项重排 | MI 的 A/B 交换；MI＋情绪、MI＋MAT、LEAF 多类别选项重排 | 检验类别文本与位置映射，不证明未见语义类别识别 |
| EEG 错配／置零 | 早期 Qwen、JEV 问答、冻结读出等 | 判断是否依赖输入信号，不是独立外测 |
| 阶段读出 | REVE、连接层、Qwen token；丢弃对比头重训 probe；LEAF query probe | 定位线性可读信息，不等同于完整模型性能 |
| 已知答案提示 | 当前 LEAF/JEV 222 个文本绑定案例 | 类别被刻意写在文本中，仅测选项遵循；绝不能报告为 EEG 准确率 |
| 预处理与频率正控制 | LEAF 2a 原始处理、Nakanishi 频率读出、任务指令变体 | 用于诊断表示和输入管线；不是新的未见数据集测试 |

## 7. 结果入口与可复核性

历史产物统一根目录：

```text
/inspire/hdd/global_user/zhangchenxi-253108310322/eegjev_archive/20261025_pre_bn_control/artifacts/
```

当前 benchmark 产物根目录：

```text
/inspire/hdd/global_user/zhangchenxi-253108310322/eegjev/artifacts/benchmark/
```

上文归档目录均相对第一个根目录；当前 benchmark 目录相对当前 `artifacts/`。归档中的旧绝对路径不一定仍可直接打开，应通过 `artifacts/archive_index.json` 映射。原始日志、数据、checkpoint 没有因写本表而删除或修改。

关键证据入口：

- [早期 REVE→Qwen 跨数据集结果](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_zero_shot_v1/RESULTS.md)
- [REVE-only 固定划分对照](../../eegjev_archive/20261025_pre_bn_control/artifacts/reve_only_multi/train_20261015_1404/RESULTS.md)
- [配对问答后的冻结读出外测](../../eegjev_archive/20261025_pre_bn_control/artifacts/jev_paired_qa/readout_evaluation/REPORT.md)
- [M9 JEV 训练配置](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_jev_tuab_lr_continued/config.json)
- [M9 JEV 外部 MI 测试](../../eegjev_archive/20261025_pre_bn_control/artifacts/external_mi_jev_zero_shot/metrics.json)
- [MI＋情绪 JEV 的实际初始化](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_emotion/jev_restart/config.json)
- [MI＋MAT JEV 配置](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_mat/jev/config.json)
- [LEAF 双数据集 JEV 结果](../../eegjev_archive/20261025_pre_bn_control/artifacts/leaf_2a_seed/jev/best.json)
- [LEAF 双数据集模型的外部 MI 测试](../../eegjev_archive/20261025_pre_bn_control/artifacts/leaf_2a_seed/external_mi/results.json)
- [当前 B11 协议](PROTOCOL.md)、[当前结果记录](RESULTS.md)、[历史归档总入口](ARCHIVE.md)

## 8. 尚不能从这些实验得出的结论

1. 尚未证明单一信号模型能自动识别任意未知任务；大多数后期实验由文本明确给出任务。
2. 选项随机排列有效，不等于二分类训练后就能识别任意没见过的第三类信号。
3. M9 源验证改善不保证外部 MI 稳定迁移；Beetl/Shin/Emotiv 的结果保留了这一差距。
4. 当前 B11 尚无独立 test 或完整跨验证成绩；不能用其固定验证子集代替这些结果。
5. 三条路线没有统一数据、划分、训练预算和预训练暴露，不能从历史表直接宣称 REVE、LEAF 或 JEV 的普遍优劣。
