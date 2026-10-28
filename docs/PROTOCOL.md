# xw1216 多数据集、文本条件化 EEG→JEV：迁移准备

_2026-10-28；目标已更新，联合训练尚未实现或启动。缺失数据和训练范围待确认。_

---

## 📋 当前状态

旧 BSR→32-token→JEV 的 2a 两组已完成；2b 两组按用户要求停止。控制器记录 `stopped`，没有 active 或 queued 作业。旧实现及完整配方已保存到 Git `14c60cf`，实际结果见[结果记录](RESULTS.md)。旧入口暂留在工作树，不能用于新协议；待数据范围明确后一次性替换，不并存两套训练实现。没有删除权重、日志、数据、缓存或环境。

新 benchmark 已稀疏克隆到 `third_party/EEG-FM-Bench-xw1216`，检出 `data/`、`common/`、`assets/conf/` 和根目录文件，固定提交 `325398d7d057ecc1216fb3510d70c16eb60337cc`。这与旧 `Dingkun0817/EEG-FM-Benchmark` 是不同仓库。只检查源码，没有执行其预处理或训练入口。[^1]

## 🎯 目标与顺序

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| 旧 2a 对照 | 完成已有训练，不追加 2b | 已结束 |
| 第一阶段 | 一个 LEAF→JEV 模型联合微调全部纳入数据集 | 尚未启动 |
| 第二阶段 | 相同划分、文本和训练预算，改用 BSR-VideoMAE | 尚未启动 |

这轮是多数据集监督微调后、各数据集内部的跨被试测试，不是跨数据集 zero-shot。若之后要声明跨数据集泛化，需另设完全不参与训练和调参的目标数据集。基础模型的预训练/指令调优数据重叠也要单独审计。

两个阶段分别从发布基础权重和原生 JEV adapter 初始化；不继承此次 2a 下游 checkpoint。旧训练包含 2–9 号，而新 2a 协议把 6、7 号作为验证、8、9 号作为测试，因此继续训练会污染新划分。[^2]

## 🔐 划分和数据门槛

采用固定提交的 `finetune` builder；SEED / SEED-V 显式选择跨被试模式，不照搬示例中的 `finetune_sub_dependent`。2a 为 train 1–5 / valid 6–7 / test 8–9；PhysioMI 为 train 1–69 / valid 70–88 / test 89–109。[^2][^3]

其余数据集不能按统一的 trial 随机比例代替原方法：有的按被试抽样，有的按被试的类别分布做平衡。原通用被试抽样依赖 RNG 和输入被试顺序，准备阶段需保存原始记录顺序、seed、实际被试列表、行索引和哈希；生成后冻结，不能每轮重新划分。TUEV 会重新组合原 train/eval，再进行被试级分层划分。[^4]

训练前必须验证：

- 每个数据集的 train / valid / test 被试集合两两不相交；不能仅检查 trial ID。
- 同一原始记录的重叠窗口、重复记录、派生缓存不能跨集合。
- 若不同 TUE 子集存在同一患者或同一记录，需做跨数据集泄漏审计；不得因为各单数据集 split 正确就默认联合训练安全。冲突时报告，不能静默改测试集。
- 在转换到 LEAF 所需通道/采样率之前保存 benchmark 原始 epoch 定义；模型适配步骤单独记录，不把旧 PKL 直接称作新 benchmark 数据。
- 验证集用于选择 checkpoint，测试集只在选择结束后评估；所有纳入数据集逐一报告结果，同时报告等数据集权重的汇总指标。

## 📦 14 个数据集的本地清点

以下只是文件定位，不代表已完成数据完整性、标签、预处理或被试划分验证。搜索覆盖当前 `mne_data`、共享 `/inspire/dataset` 及已知模型数据目录；“未定位”不表示整个服务器一定没有。缩写 `MNE` 指 `/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data`，`NB` 指 `/inspire/dataset/neural-bench/v1`。任务含义按源码核对。[^1]

| 数据集 | builder / 类别数 | 本地候选来源 | 当前缺口 |
| --- | --- | --- | --- |
| BCIC-IV-2a | `bcic_2a` / 4 | `MNE/BCIC-IV-2a` | 原仓库读取 SET；现有 MAT/GDF 转换与事件时刻需核验 |
| PhysioMI | `motor_mv_img` / 4 | `MNE/eegbci/MNE-eegbci-data/files` | 核验完整被试、runs 4/6/8/10/12/14 |
| Mimul-11 | `mimul_11` / 3 | `/inspire/dataset/moabb-processed/v1/EEG-processed/Jeong2020` | 仅定位到其他预处理版；原 BrainVision 文件未定位 |
| SEED | `seed` / 3 | `MNE/SEED/SEED/SEED_EEG` | 原始/预处理目录存在，需核验所需 SET 与事件 |
| SEED-V | `seed_v` / 5 | 未定位 | 需要原始数据路径 |
| SEED-VII | `seed_vii` / 7 | `MNE/SEED-VII/EEG_raw` | CNT 文件存在，标签与记录完整性待核验 |
| TUAB | `tuab` / 2 | `MNE/tuh/tuh_eeg_abnormal`、`NB/Lopez2017Tuab` | 核验版本、记录和来源划分 |
| TUEV | `tuev` / 6 | `NB/Harati2015Tuev/edf` | 核验 REC 标注和跨 TUE 子集重复 |
| Siena | `siena_scalp` / 2 | 未定位 | 需要 EDF、发作标注路径 |
| HMC | `hmc` / 5 | `MNE/HMC/recordings` | EDF 与分期标注存在，完整性待核验 |
| TUSL | `tusl` / 3 | 未定位 | 需要 TUSL 数据；不能用 TUSZ 替代 |
| Things-EEG-2 | `things_eeg_2` / 2 | `NB/Gifford2022Large/download` | 有 EEG ZIP；所需行为标签未定位 |
| Workload | `workload` / 2 | `MNE/EEGMat` | EDF 存在，元数据与预处理待核验 |
| ADFTD | `adftd` / 3 | `NB/Miltiadous2023Dice/download` | SET、参与者元数据和预处理版本待核验 |

两个易混淆的标签定义：Mimul-11 默认 `finetune` 是 reach / grasp / twist 三分类，不是 11 分类；Things-EEG-2 在这里是 5 秒序列的 non-target / target 二分类，并依据行为回答过滤序列，需要 `behaviour/sub-*/ses-*/behavioral_data.npy`，不能从图像类别 ID 臆造该标签。[^5][^6]

必须先确认缺失路径，或取得用户允许后缩小首轮数据范围；不能把可用子集报告成全部 14 个数据集。当前没有发起额外大规模数据下载或解压。

## ⚙️ 文本条件化和联合优化设计

| 输入 / 参数 | 第一阶段约定 |
| --- | --- |
| EEG 输入 | LEAF 信号编码器与任务条件 query；原始通道、窗口转换需记录 |
| LEAF 文本 | 任务描述的固定 MPNet embedding，不含本 trial 标签或被试身份 |
| JEV 文本 | 任务描述、问题、该任务全部候选标签，末尾 `[decision]:` |
| JEV 信号输入 | LEAF query token 经可训练 connector 映射到语言宽度 |
| 优化对象 | LEAF 有效路径、connector、原生 JEV LoRA 和 choice 参数 |
| 冻结部分 | Qwen 基座及不在当前信号路径上的重建模块 |
| 数据混合 | 单一共享模型；拟采用等数据集采样，记录每个数据集实际曝光量 |
| 损失 | 对当前候选位置做 CE；标签仅作为监督 target |
| 候选顺序 | 训练时随机排列并同步重映射 target，评估另报选项顺序一致性 |

不为每个数据集单独训练一套 JEV 或独立最终分类头。文本决定问题和可选语义，不能将 ground truth 放入 prompt。均衡采样和具体步数是尚待实现的训练配方，不是已经运行的配置。

LEAF 使用已下载的 `LEAF/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt`。历史接口检查在隔离 worktree `eegjev-worktrees/leaf-3033ac9-inspect` 完成；未把旧划分、旧训练权重或旧文本缓存无条件复制到新实验。LEAF 最长窗口、长记录分段和有效 token mask 仍需在真实样本 smoke 中验证。

## 🔧 VideoMAE 单 token 方案

本地 `BSR-VideoMAE-AS-Final-v3/code/bsri/model.py` 的 `BSRIEncoder.forward` 使用全部 token 的均值；`Token0BSRIClassifier` 注释也明确 token 0 是 patch，不是 CLS。现有权重没有可直接提取的 CLS token。

第二阶段拟使用 `hidden.mean(dim=1, keepdim=True)`，将 `[B,1568,768]` 池化为 `[B,1,768]` 后输入 JEV connector。该表示称为 mean-pooled global token，不称 CLS，也不增加随机 CLS 后冒称预训练表示。它减少 JEV 序列长度，不减少 VideoMAE 内部 token 数量。此阶段尚未实现或启动；LEAF 完成后再按相同 split 和文本运行，不继承旧 2a 微调权重。

## 📍 下一步和启动条件

1. 用户提供缺失数据路径，或明确允许首轮只使用已核验的子集。
2. 完成各数据集完整性、标签、epoch 和被试划分检查，生成锁定 manifest。
3. 在 Git 已归档旧状态的前提下替换旧训练实现，清理 superseded 配置/测试，保留用户未提交工作与所有忽略数据。
4. 执行小样本多任务前向/反向、梯度和 checkpoint 重载检查；通过后才启动 LEAF 正式联合训练。
5. LEAF 完成后实施同协议的 VideoMAE 单全局 token 对照。

Hugging-science 技能用于资源和协议检查，Markdown 文档技能用于区分完成、计划和阻塞状态；所用技能库的方法引用见文末。[^7] 目录匹配不等于数据已可训练；本文件不构成已启动训练的证据。

[^1]: xw1216/EEG-FM-Bench，固定源码。https://github.com/xw1216/EEG-FM-Bench/tree/325398d7d057ecc1216fb3510d70c16eb60337cc
[^2]: BCIC-IV-2a builder。https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/bcic/bcic_2a.py
[^3]: SEED 与 PhysioMI builder。https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/seeds/seed.py ，https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/motor_mv_img.py
[^4]: 被试级随机及类别平衡划分。https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/processor/builder.py
[^5]: Mimul-11 默认标签。https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/mimul_11.py
[^6]: Things-EEG-2 行为标签与事件定义。https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/things_eeg_2.py
[^7]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065，核验时最新修订为 2026-09-02。https://doi.org/10.48550/arXiv.2609.00065
