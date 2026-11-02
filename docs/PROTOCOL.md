# EEG→JEV：固定学习率的 VideoMAE LoRA-r64 对照

_更新：2026-11-02。当前运行 `aligned_videomae_jev_lora64_20261102`，配置见 [benchmark_train.json](../configs/benchmark_train.json)。_

---

## 🎯 要回答的问题

1. What is the problem? InfoNCE＋Q-Former→JEV 全参组完整验证 BAcc 为 44.15%，需要检验限制 VideoMAE 的可更新参数是否有帮助。
2. Why is it important? 在相同训练预算和学习率下区分参数更新方式的影响，避免同时调参导致无法归因。
3. Why is it hard? LoRA 与全参的有效权重更新并不相同，固定学习率只能回答本设定下的差异，不能证明某种方法的最优性能。
4. Prior limitations? 父 InfoNCE 的 47.49% 没有 JEV；不同协议的 LEAF 或旧 VideoMAE 数字不是本轮匹配控制。
5. Proposed components? 复用同一 InfoNCE encoder、新初始化且种子匹配的 Q-Former／连接器／原生 JEV，只将 VideoMAE 更新替换为 rank=64 LoRA。

第二阶段仅使用任务分类交叉熵，不继续叠加 InfoNCE 损失。名字中的 InfoNCE 表示第一阶段权重来源。仍是有监督多任务试验，不是 zero-shot，不把通用连接器或 LoRA 本身当作 novelty。

## 🔌 LoRA 结构与匹配控制

| 配置项 | 全参控制 | 本轮 LoRA-r64 |
| --- | --- | --- |
| 实验标识 | infonce_qformer_unfrozen | infonce_qformer_lora64 |
| VideoMAE | 全参更新 | base 冻结，仅训练 LoRA A/B |
| renderer／SharedScale | 预热后训练 | 相同，不额外冻结 |
| encoder 组学习率 | 1e-6 | 1e-6 |
| Q-Former／连接器 | 新初始化，参与训练 | 相同初值与训练方式 |
| JEV | 原生 LoRA＋choice head 按阶段训练 | 相同 |
| 训练预算 | 2040 步 | 2040 步 |
| 完整验证 BAcc | 44.15% | 待完成 |

本轮只新增一个实验，使用 GPU 0，启动前要求至少 72,000 MiB 空闲。LoRA 的 rank=64、alpha=64，缩放系数 alpha/r=1，dropout=0；A 为 Kaiming-uniform，B 为零，初始增量为零。12 层每层注入 query/key/value、attention.output.dense、intermediate.dense、output.dense，共 72 个线性层；bias、LayerNorm 和 patch embedding 等原 VideoMAE 参数全部冻结，不合并低秩增量到 base。LoRA 共 10,616,832 个参数；另保留原 renderer／SharedScale 的 26,862 个可训练参数。

控制为 `aligned_videomae_jev_20261102/train/infonce_qformer_unfrozen`，不是从其选中的下游 checkpoint 继续训练。配置中的 `matched_control` 固定控制组配置／初值／完成记录 SHA-256；只允许运行标识、GPU 调度和 VideoMAE LoRA 设置差异。训练数据、全部学习率、优化器配置、预算、评估规则等任一其他变化均拒绝启动。逐步训练样本 ID 哈希也与原组核对。

权重来自 `artifacts/benchmark/contrastive_videomae_20261101/train/{infonce,sigmoid}/best.pt`，两组均按第一阶段固定验证子集选中 step 1700，不重新挑选更有利于第二阶段的父 checkpoint：

| 来源 | best.pt SHA-256 | 完整验证 BAcc（第一阶段，无 JEV） |
| --- | --- | ---: |
| InfoNCE | `74b6d0b2c6996aea75de9b95c824ce519c1141900778ffb69d40fb117a97f41b` | 47.49% |
| Sigmoid | `eb3664908e664a10dc59aeca7595d062225f8fe91d2135a03fe4437f18395387` | 46.55% |

证据：[InfoNCE 完成记录](../artifacts/benchmark/contrastive_videomae_20261101/train/infonce/complete.json)、[Sigmoid 完成记录](../artifacts/benchmark/contrastive_videomae_20261101/train/sigmoid/complete.json)。

仅迁移训练后的 `renderer.eeg2pix`、SharedScale、VideoMAE 及全部 encoder buffers，包括未进入普通 state_dict 的 STFT／电极索引 buffers。重建专用 `pix2eeg` 保持原发布权重，不参与前向和训练。源 checkpoint 的 CLS、投影层、scale/bias、文本向量、优化器与 RNG 都不迁移。

本轮只使用 InfoNCE 父权重。Q-Former、连接器和原生 JEV 从与全参组相同种子重新初始化，逐项核对参数哈希，不迁移父 CLS／投影或全参组下游权重。上表两个父成绩保留为来源参考，不能当作新模型 step 0 应达到的成绩。

CLS 是新学习的全局 query＋一层 self-attention/FFN，不是 VideoMAE 的第一个 patch，也不是逐帧 CLS。Q-Former 为 4 层 query self-attention＋cross-attention＋FFN，32 个 learnable queries、768 维、12 heads；不是预训练 BLIP-2 权重。连接器均为 LN→Linear(768,5120)→GELU→Linear(5120,5120)。任务说明、问题和候选选项只进入 JEV，真实类别只用于监督。

## 📥 数据与划分

17 库不变：9 个 benchmark＋8 个 train-only。2b 不训练。额外训练库仍为 BNCI2015001、BNCI2014008、BNCI2014009、CHB_MIT、SleepEDF、AlexMI、Zhou2016、Nakanishi2015。

沿用上一阶段的 SEED 划分：严格调用 pinned 官方 `SeedBuilder._divide_by_uniform_label` 核对每位被试每次 session 的 15 个原始 trial，按顺序 1–9／10–12／13–15 分为训练／验证／测试。不是随机切窗口，允许同一被试出现在三份集合，但禁止同一 trial 跨集合。此规则与论文附录 B.3 一致。[^bench]

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

重划分入口：[prepare_seed_trial_split.py](../scripts/prepare_seed_trial_split.py)。旧目录保留；只允许迁移与当前划分、QC、缓存索引完全一致的对齐权重；不能使用旧 HMC109／SEED 跨被试下游权重。

## 🔧 共同训练设置

- 保留第一阶段输入适配：微伏、69 通道网格、200 Hz、200 点 STFT 幅值谱、16 帧、SharedScale、ImageNet 归一化；长窗分段后全部拼接，不裁尾部
- 单 seed=42；2040 步，global batch=16，microbatch=8，eval microbatch=8
- 17 库周期随机轮转，库内类别均衡／被试均匀；每库 120 个 batch、1920 次样本呈现，不是 120 个完整 epoch
- 1–340 步仅训练新读出和连接器；341–2040 步开放 JEV 原生 LoRA、choice head、VideoMAE LoRA、renderer 和 SharedScale
- VideoMAE 基础参数全程冻结；JEV 基础权重始终冻结；重建专用 pix2eeg 始终冻结
- AdamW：读出／连接器 LR 1e-4，JEV LoRA 5e-6，choice head 1e-5，encoder 组 1e-6；weight decay .01，clip norm 1
- 每个参数组自开放日起 warmup 62 步，随后 cosine 衰减至峰值 2%；沿用现有日程，不为某组单独调参
- 正式训练重新加载相同父 best 权重，不继承 smoke 更新；不从 44.15% 下游 checkpoint 续训
- 保存每步每库 loss、batch Acc、LR、梯度、样本哈希与显存，供后续训练曲线和表征分析
- 本轮只新增一个 2040 步实验；与全参控制使用相同父对齐权重，不另外增加对齐训练

## 📊 验证与启动门控

从 step 0 记录固定每库 ≤512 验证窗和 ≤128 训练监控窗；每 340 步重复。按九库等权验证 BAcc 选择 best，同分保留较早 step。结束后仅对 best 跑完整 61,327 窗验证，保存逐窗预测，自动绘制 loss 与验证曲线。测试集保持关闭，训练子集监控不能称为完整训练集准确率。

启动顺序：全部计划训练输入 CPU 检查；17 步真实 GPU smoke；核对源权重、匹配控制初值、逐步样本、VideoMAE base 不变、LoRA／renderer／SharedScale 梯度及更新、磁盘重载；全部通过后正式训练。只读加载需通过源 checkpoint 文件哈希、选中步数、未访问测试集、相同 split/QC/cache/类别语义检查，并逐项核对所有 encoder 前向参数与 buffers。

`transfer_audit.json` 记录父 checkpoint、encoder 哈希和迁移范围；`initial_state.json` 记录新读出、连接器、JEV 初值。源码／配置在运行期有 hash guard；源 checkpoint 文件大小与时间戳也受监控。已有任务运行期间不得改其导入模块，恢复只用相同身份的未完成 `last.pt`。

节点更换仅通过启动器 `--start --resume --gpus 0 1 2 3` 覆盖调度槽位，不改已保存配置中的 GPU 列表或训练身份。跳过已完成组，保留中断的审计和预测快照，并使用原训练器恢复；若已到 2040 步，则不新增优化器更新，只重新执行完整验证。完整验证不跨中断拼接部分统计，测试集仍不访问。

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

输出 `artifacts/benchmark/aligned_videomae_jev_lora64_20261102/`。控制器状态 `pipeline.json`，分组进度 `train/infonce_qformer_lora64/progress.json`。同目录不可重复启动；控制器脱离终端运行。失败时保留全部产物，不杀其他用户任务。

## ⚠️ 解释边界与历史复现

旧 LEAF 适配入口与配置已在 Git `704ad26` 归档，五组迁移及四卡恢复入口在 `828a17d`。当前工作树维护同一方法的 LoRA 适配控制及其共享组件。父对齐代码仍作为第一阶段保留，历史运行必须从对应提交的隔离 worktree 复现，不能覆盖当前源码。

SEED 被试内，其余八个 benchmark 跨被试；TUAB/TUEV 19 个候选共享 ID、其中 5 个跨库角色冲突仍未解决，基模预训练重叠及额外缓存来源风险也未排除。HMC 151 份已补齐，但作者随机名单未确认。不声称完全复现 EEG-FM-Bench 或严格无泄漏。

第一阶段已用于选择两组编码器，第二阶段再用同一验证集选择 JEV checkpoint，因此这些是探索性验证成绩；最终测试不能用于继续选模。当前还没有“相同总预算、无对齐初始化”的匹配新控制，不能仅凭与旧结果的差值证明语言对齐的因果收益。历史结果见 [RESULTS](RESULTS.md)，恢复方式见 [ARCHIVE](ARCHIVE.md)。

## 📚 来源

[^bench]: Xiong et al. EEG-FM-Bench，附录 B.3。https://arxiv.org/html/2508.17742v3#A2.SS3 。实现固定于 xw1216/EEG-FM-Bench commit `325398d7d057ecc1216fb3510d70c16eb60337cc`：https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/seeds/seed.py 。
