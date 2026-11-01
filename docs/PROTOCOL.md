# LEAF 冻结与 BSR 解冻：统一划分的 JEV 适配试验

_更新：2026-11-01。当前运行 `encoder_adaptation_seedpaper_20261101`，配置见 [benchmark_train.json](../configs/benchmark_train.json)。_

---

## 🎯 要回答的问题

1. What is the problem? 现有 EEG→JEV 的任务表现不均衡，需检验冻结的原生 LEAF 能否保持可读信息，以及 BSR 端到端适配是否有帮助。
2. Why is it important? 避免把连接器训练不足、输入适配或冻结策略的问题，误认为 EEG 基模无法支持多任务语义决策。
3. Why is it hard? 被试、任务、通道数和窗口长度不同；大模型只获得有限更新，输入接口与编码器适配存在混杂。
4. Prior limitations? 历史 LEAF 与 BSR 的冻结策略、文本入口和数据版本不同；历史 SEED 使用跨被试划分，而论文采用被试内 trial 划分。
5. Proposed components? 同一数据清单、采样序列、JEV 初始权重、训练预算及选模规则，运行冻结原生 LEAF 与解冻 BSR+CLS 两组适配控制。

本轮是两种适配方案的可行性试验，不是完整“编码器×冻结状态”2×2 单因素实验。SEED 协议同时改变，不能直接用旧 SEED 分数归因冻结效果，也不把现成连接器结构当作 novelty。

## 🔌 两组结构与冻结范围

| 项目 | leaf_frozen | bsr_unfrozen |
| --- | --- | --- |
| EEG 编码器 | 发布 LEAF instruct MPNet checkpoint | 发布 BSR R1＋SharedScale＋VideoMAE |
| 编码器适配 | 全程冻结，包括原生 Q-Former、BN buffers | 1–340 步冻结，341–2040 步解冻前向路径 |
| EEG 读出 | 原生任务条件 Q-Former，每 ≤10 秒 8×256 token；长窗拼接 | 新学习 CLS＋一层 self-attention/FFN，全窗口输出 1×768 |
| 连接器 | LN→Linear(256,5120)→GELU→Linear(5120,5120) | LN→Linear(768,5120)→GELU→Linear(5120,5120) |
| 文本入口 | 任务描述的固定 MPNet embedding 进入 LEAF；完整问题选项进入 JEV | 文本只进入 JEV |
| GPU | 4 | 5 |

```mermaid
flowchart LR
    accTitle: EEG encoder adaptation controls
    accDescr: Two independent models share locked data and JEV initialization. LEAF remains frozen and task conditioned; BSR learns its forward path after bridge warmup.
    eeg_input[EEG 完整窗口] --> leaf_encoder[冻结 LEAF 和原生 Q-Former]
    eeg_input --> bsr_encoder[可解冻 BSR 和 CLS]
    leaf_text[任务描述 embedding] --> leaf_encoder
    leaf_encoder --> leaf_bridge[LEAF 连接器]
    bsr_encoder --> bsr_bridge[BSR 连接器]
    leaf_bridge --> leaf_jev[独立 JEV 实例一]
    bsr_bridge --> bsr_jev[独立 JEV 实例二]
    task_prompt[任务说明和问题选项] --> leaf_jev
    task_prompt --> bsr_jev
```

BSR 解冻包含 `renderer.eeg2pix`、SharedScale `log_gain/bias` 和整个 VideoMAE；不训练未参与前向的 `renderer.pix2eeg` 重建头。VideoMAE 使用 activation checkpointing。不是仅加 LoRA，也不是只训练最后一层。

CLS 按前一轮 HMC151 完整验证九库平均 BAcc 选择：41.6912%，mean_pool 为 41.6742%；仅差 0.017 个百分点，不宣称显著优势。这里只复用读出结构，CLS、连接器重新初始化，不复用旧下游 checkpoint。

LEAF 路径从 Git `741fe34` 的隔离 worktree 核对，仅保留当前控制所需的适配器，共享现有 sampler、trainer、JEV 和检查点工具，不恢复旧独立训练入口。两组 JEV 初始化完全相同，但连接器输入维度不同，不宣称连接器初始参数一致。

## 📥 数据与划分

17 库不变：9 个 benchmark＋8 个 train-only。2b 不训练。额外训练库仍为 BNCI2015001、BNCI2014008、BNCI2014009、CHB_MIT、SleepEDF、AlexMI、Zhou2016、Nakanishi2015。

只改变 SEED：严格调用 pinned 官方 `SeedBuilder._divide_by_uniform_label` 核对每位被试每次 session 的 15 个原始 trial，按顺序 1–9／10–12／13–15 分为训练／验证／测试。不是随机切窗口，允许同一被试出现在三份集合，但禁止同一 trial 跨集合。此规则与论文附录 B.3 一致。[^bench]

下表是本轮实际处理窗口及划分，不是论文作者当次运行名单；窗口不能一概视为独立原始 trial。

| 数据集 | 处理窗口：总计（训／验／测） | 任务种类（分类数） | 划分协议 | 被试：总计（训／验／测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4 类） | 固定跨被试 | 9（5/2/2） |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4 类） | 固定跨被试 | 109（69/19/21） |
| SEED | 37,890（22,455/7,875/7,560） | 情绪识别（3 类） | 被试内，原始 trial 顺序 9/3/3 | 15（15/15/15；重复参与） |
| SEED-VII | 19,420（15,536/1,942/1,942） | 情绪识别（7 类） | 随机跨被试 | 20（16/2/2） |
| TUAB | 272,320（247,728/12,315/12,277） | EEG 异常检测（2 类） | 原训练集＋评估集按被试分层 | 2,328（2,075/101/152） |
| TUEV | 113,353（87,834/12,473/13,046） | EEG 事件分类（6 类） | 按被试分层 | 370（328/20/22） |
| HMC | 136,925（92,454/22,421/22,050） | 睡眠分期（5 类） | 完整记录随机跨被试 | 151（103/24/24） |
| Workload | 2,134（1,534/300/300） | 认知负荷（2 类） | 随机跨被试 | 36（26/5/5） |
| ADFTD | 7,013（4,743/1,115/1,155） | AD／FTD／正常识别（3 类） | 按被试分层 | 88（64/11/13） |

SEED 原始 trial 为 675（405/135/135），来自 45 份 session 记录。处理窗口总数不变。其余 16 库 manifest、records 和信号索引逐文件不变，HMC 已补齐 151 份并维持现有 103/24/24 名单。

- 划分：`artifacts/benchmark/multitask_splits_seedpaper_20261101`；summary SHA256 `9a4d302b25413d5e06f728e6fd9caad7b1b3dbf1590e1e7d03121dd2cc19a222`
- 缓存：`datasets/multitask_seedpaper_20261101`；只重建 SEED 索引，全部信号只读链接原缓存，不重写 EEG
- QC：`artifacts/benchmark/multitask_signal_quality_seedpaper_20261101`；summary SHA256 `45e504a715cc5fdbd42b78a28d0edddc735f139063c985af55cd5f0bc236b98c`
- 新 QC 继承已核验的原信号完整性审计，检查源报告哈希、所有数组大小／mtime、SEED 无无效窗及重新分配后的计数；未声称本次重新扫描全部波形
- 17 库有效训练 967,281 窗，九库验证 61,327 窗，九库测试 61,375 窗；仍只排除 CHB_MIT 已知 1,144 个无效训练窗
- 严格校验非 SEED 被试不重叠，以及 SEED 原始 event ID、trial 完整覆盖、窗口身份唯一与 9/3/3 分配；不能全局关闭泄漏检查

重划分入口：[prepare_seed_trial_split.py](../scripts/prepare_seed_trial_split.py)。旧目录保留；禁止拿旧下游 SFT 权重在新划分上继续训练，因为部分样本角色已改变。

## 🔧 输入适配与共同训练设置

LEAF 保留原来的 65 通道插值、逐窗稳健 IQR 归一化、SEED/SEED-VII 插值后裁剪 ±10、200 Hz 和 ≤10 秒分段／100 点补齐。MPNet embedding 仅含任务描述，不含本 trial 类别；来源为已校验的 `multitask_text_20261029`。

BSR 保留已验证的微伏单位、69 通道网格、发布 200 点矩形窗 STFT 幅值谱、16 帧、SharedScale 和 ImageNet 归一化，不采用 LEAF 的 IQR 或 65 通道插值。≤10 秒输出 1568×768；15 秒、30 秒分别拼接 3136、4704 个 token，再由 CLS 汇总。保留原始样本，不截断尾部；短 P300 窗仅按现有规则右侧反射补齐。双极信号的最小范数表示不是恢复真实单极电压。

共同设置：

- 单 seed=42；2040 步，全局 batch=16，microbatch=8，评估 microbatch=8
- 17 库周期随机轮转，库内类别均衡、被试均匀后抽窗口；逐步 sample IDs 和选项排列相同
- 每库 120 个 batch／1920 次样本呈现，不是完整训练 120 个 epoch
- 1–340 步只更新连接器及可训练读出；341–2040 步开放 JEV 原生 LoRA、choice 头；BSR 同时开放前向编码器，LEAF 仍冻结
- JEV 大模型基础权重始终冻结。连接器／读出 LR 1e-4，JEV LoRA 5e-6，choice 头 1e-5，BSR 编码器 1e-6
- AdamW，weight decay=.01，clip norm=1；每个参数组自开放日起 warmup 62 步，再 cosine 衰减至峰值 2%
- 两组从发布权重重新开始，独立 JEV 实例；不继承 smoke 或旧微调参数
- 实际可训练参数量、梯度范数、样本 ID 哈希、LR、loss、batch Accuracy、显存、耗时写入每组日志
- 测试集不参与训练、选模和本轮自动评估

## 📊 验证、恢复与启动门控

从 step 0 开始记录验证；每 340 步评估固定每库 ≤512 验证窗、≤128 训练监控窗。以九库等权平均验证 BAcc 选 checkpoint，同分保留更早步。最终加载 best checkpoint 跑完整 61,327 窗验证并保存预测。训练 batch 指标不等于完整训练集准确率。

两组共用当前 `eeg_align.train_joint` 和 `scripts/run_benchmark.py`。控制器先检查指定 GPU 空闲、2040 步全部计划输入，再运行每组 17 步完整 JEV smoke，覆盖全部数据集并跨过缩短的冻结阶段。必须验证 LEAF 参数／buffers 不变，BSR 开放阶段存在梯度及实际权重变化，磁盘重载 logits 一致，然后正式重新初始化。任一失败不启动后续阶段；保留另一独立任务，不抢占其他 GPU。

代码／配置有运行期哈希保护；未完成的单组可用 `--resume` 恢复，但必须保持源码、配置、数据及检查点身份不变。不要一边运行一边改动导入模块或配置。控制器脱离终端继续执行；节点掉线后需先核实再恢复。

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

输出 `artifacts/benchmark/encoder_adaptation_seedpaper_20261101/`。控制器状态为 `pipeline.json`，实时进度为 `train/<arm>/progress.json`，逐任务曲线在两组完成后自动生成。已有输出目录不可重复启动。

启动阶段曾用 microbatch=2 检查两组，均在第 9 步主动安全停止，峰值显存分别约 52.5/54.3 GiB；未进入正式训练。为利用显存余量改为 microbatch=8，并从发布权重重新执行全部预检和 smoke。早期记录完整保留于 `encoder_adaptation_seedpaper_20261101_micro2_smoke`，不当作完成的 17 步 smoke，也不拼入正式训练曲线。

## ⚠️ 解释边界与历史对照

这是有监督多任务试验，不是跨数据集 zero-shot。SEED 为被试内，其余八个 benchmark 为跨被试，不能把九库一概称作跨被试。

两组在原生预处理、文本入口、token 数和连接器宽度上仍不同；本轮缺少同新划分的 LEAF 解冻和 BSR 冻结两个格子，因此不能单独量化冻结的因果效应。后续若需要严格归因，补齐相同协议的 2×2 控制，而不是将旧分数当作匹配基线。

TUAB/TUEV 的 19 个候选重合 ID（5 个跨库角色冲突）、基模预训练被试重叠、额外缓存来源风险仍未解决。HMC 151 份完整性已解决，但论文作者当次随机被试名单未取得，处理窗口分母不全相同，不声称完全复现论文。

旧冻结 BSR 四读出结果及 LEAF 成绩只保留为必要历史对照，见 [RESULTS](RESULTS.md)。旧表征图见 [REPRESENTATION_ANALYSIS](REPRESENTATION_ANALYSIS.md)，仅描述旧划分／旧 checkpoint，不是新实验结果。旧实现的隔离复现规则见 [ARCHIVE](ARCHIVE.md)。

## 📚 来源与过程说明

本轮使用实验设计、资源检测、hugging-science 和 Markdown 文档技能检查混杂、启动门控与审计记录；不作为模型有效性证据。相关方法工具说明见 Kassis 等的 Scientific Agent Skills。[^skills]

[^bench]: Xiong et al. EEG-FM-Bench，附录 B.3。https://arxiv.org/html/2508.17742v3#A2.SS3 。实现固定于 xw1216/EEG-FM-Bench commit `325398d7d057ecc1216fb3510d70c16eb60337cc`：https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/seeds/seed.py 。

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 。
