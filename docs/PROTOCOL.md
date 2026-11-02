# EEG→JEV：文本对齐编码器的五组迁移试验

_更新：2026-11-02。当前运行 `aligned_videomae_jev_20261102`，配置见 [benchmark_train.json](../configs/benchmark_train.json)。_

---

## 🎯 要回答的问题

1. What is the problem? 直接解冻发布 VideoMAE 接入 JEV 的一轮验证 BAcc 仅 31.41%，需要检验先做 EEG–文本对齐能否改善 JEV 的信号利用。
2. Why is it important? 区分 EEG 表征、token 读出和 JEV 适配问题，避免把一轮失败推广为信号编码器不可用。
3. Why is it hard? 固定类别文本的信息量有限，MPNet 对齐空间不等于 JEV 隐空间，多任务、小样本、不同通道及长度仍带来困难。
4. Prior limitations? 对比学习 47.49%／46.55% 没有加载 JEV；旧冻结 BSR 组的 SEED 协议又不同，不能直接作为匹配对照。
5. Proposed components? 迁移两组对齐 best encoder，做“对齐损失来源×读出结构”四组冻结控制，再加 InfoNCE＋Q-Former 解冻控制。

本轮五组都接 JEV，第二阶段仅使用任务分类交叉熵，不继续叠加 InfoNCE／Sigmoid 损失。名字中的 InfoNCE／Sigmoid 表示第一阶段权重来源。仍是有监督多任务试验，不是 zero-shot，也不把通用连接器结构本身当作 novelty。

## 🔌 五组结构与初始化

| 实验标识 | 对齐权重来源 | 新读出 | EEG 编码器策略 | GPU |
| --- | --- | --- | --- | ---: |
| infonce_cls_frozen | InfoNCE best，step 1700 | CLS，1 token | 全程冻结 | 0 |
| infonce_qformer_frozen | InfoNCE best，step 1700 | Q-Former，32 tokens | 全程冻结 | 1 |
| sigmoid_cls_frozen | Sigmoid best，step 1700 | CLS，1 token | 全程冻结 | 4 |
| sigmoid_qformer_frozen | Sigmoid best，step 1700 | Q-Former，32 tokens | 全程冻结 | 5 |
| infonce_qformer_unfrozen | InfoNCE best，step 1700 | Q-Former，32 tokens | 1–340 步冻结，341–2040 步解冻 | 6 |

用户已允许与现有测评共卡；本次选用当前余量更大的五张卡，不停止、迁移或重启已有测评进程。每次启动子进程前检查至少 72,000 MiB 空闲；实际峰值以五组真实 smoke 为准。

权重来自 `artifacts/benchmark/contrastive_videomae_20261101/train/{infonce,sigmoid}/best.pt`，两组均按第一阶段固定验证子集选中 step 1700，不重新挑选更有利于第二阶段的父 checkpoint：

| 来源 | best.pt SHA-256 | 完整验证 BAcc（第一阶段，无 JEV） |
| --- | --- | ---: |
| InfoNCE | `74b6d0b2c6996aea75de9b95c824ce519c1141900778ffb69d40fb117a97f41b` | 47.49% |
| Sigmoid | `eb3664908e664a10dc59aeca7595d062225f8fe91d2135a03fe4437f18395387` | 46.55% |

证据：[InfoNCE 完成记录](../artifacts/benchmark/contrastive_videomae_20261101/train/infonce/complete.json)、[Sigmoid 完成记录](../artifacts/benchmark/contrastive_videomae_20261101/train/sigmoid/complete.json)。

仅迁移训练后的 `renderer.eeg2pix`、SharedScale、VideoMAE 及全部 encoder buffers，包括未进入普通 state_dict 的 STFT／电极索引 buffers。重建专用 `pix2eeg` 保持原发布权重，不参与前向和训练。源 checkpoint 的 CLS、投影层、scale/bias、文本向量、优化器与 RNG 都不迁移。

CLS 和 Q-Former 都从固定种子重新初始化，避免“已训练 CLS 对随机 Q-Former”的初始化混杂。同结构读出跨来源初值一致，三组 Q-Former 初值一致；五组连接器与原生 JEV LoRA／head 初值一致。因此 47.49%／46.55% 不能当作新模型 step 0 应达到的成绩。

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
- 1–340 步仅训练新读出和连接器；341–2040 步开放 JEV 原生 LoRA 与 choice head，第五组同时开放 BSR 前向路径
- 编码器解冻是完整前向参数更新，不是 encoder LoRA；JEV 基础权重始终冻结
- AdamW：读出／连接器 LR 1e-4，JEV LoRA 5e-6，choice head 1e-5，第五组 encoder 1e-6；weight decay .01，clip norm 1
- 每个参数组自开放日起 warmup 62 步，随后 cosine 衰减至峰值 2%；沿用现有日程，不为某组单独调参
- 正式训练重新加载相同父 best 权重，不继承 smoke 更新；第五组与 InfoNCE＋Q-Former 冻结组仅 encoder 适配策略不同
- 保存每步每库 loss、batch Acc、LR、梯度、样本哈希与显存，供后续训练曲线和表征分析
- 本轮新增五组各 2040 步；父对齐训练是额外预算，不与无对齐旧组宣称等总预算

## 📊 验证与启动门控

从 step 0 记录固定每库 ≤512 验证窗和 ≤128 训练监控窗；每 340 步重复。按九库等权验证 BAcc 选择 best，同分保留较早 step。结束后仅对 best 跑完整 61,327 窗验证，保存逐窗预测，自动绘制 loss 与验证曲线。测试集保持关闭，训练子集监控不能称为完整训练集准确率。

启动顺序：全部计划训练输入 CPU 检查；五组各 17 步真实 GPU smoke；核对源权重、相同初始化、逐步样本、冻结／解冻梯度及权重变化、磁盘重载；全部通过后并行正式训练。只读加载需通过源 checkpoint 文件哈希、选中步数、未访问测试集、相同 split/QC/cache/类别语义检查，并逐项核对所有 encoder 前向参数与 buffers。

`transfer_audit.json` 记录父 checkpoint、encoder 哈希和迁移范围；`initial_state.json` 记录新读出、连接器、JEV 初值。源码／配置在运行期有 hash guard；源 checkpoint 文件大小与时间戳也受监控。已有任务运行期间不得改其导入模块，恢复只用相同身份的未完成 `last.pt`。

节点更换仅通过启动器 `--start --resume --gpus 0 1 2 3` 覆盖调度槽位，不改已保存配置中的 GPU 列表或训练身份。跳过已完成组，保留中断的审计和预测快照，并使用原训练器恢复；若已到 2040 步，则不新增优化器更新，只重新执行完整验证。完整验证不跨中断拼接部分统计，测试集仍不访问。

```bash
laya/.venv-eeg/bin/python -m pytest -q tests
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start
```

输出 `artifacts/benchmark/aligned_videomae_jev_20261102/`。控制器状态 `pipeline.json`，分组进度 `train/<arm>/progress.json`。同目录不可重复启动；控制器脱离终端运行。失败时保留独立在跑的组及全部产物，不杀其他用户任务。

## ⚠️ 解释边界与历史复现

旧 LEAF 适配入口与配置已在 Git `704ad26` 归档，当前工作树只维护这五组迁移及其上游对齐组件。父对齐实验代码仍作为本方法第一阶段保留，历史运行必须从对应提交的隔离 worktree 复现，不能覆盖当前源码。

SEED 被试内，其余八个 benchmark 跨被试；TUAB/TUEV 19 个候选共享 ID、其中 5 个跨库角色冲突仍未解决，基模预训练重叠及额外缓存来源风险也未排除。HMC 151 份已补齐，但作者随机名单未确认。不声称完全复现 EEG-FM-Bench 或严格无泄漏。

第一阶段已用于选择两组编码器，第二阶段再用同一验证集选择 JEV checkpoint，因此这些是探索性验证成绩；最终测试不能用于继续选模。当前还没有“相同总预算、无对齐初始化”的匹配新控制，不能仅凭与旧结果的差值证明语言对齐的因果收益。历史结果见 [RESULTS](RESULTS.md)，恢复方式见 [ARCHIVE](ARCHIVE.md)。

## 📚 来源

[^bench]: Xiong et al. EEG-FM-Bench，附录 B.3。https://arxiv.org/html/2508.17742v3#A2.SS3 。实现固定于 xw1216/EEG-FM-Bench commit `325398d7d057ecc1216fb3510d70c16eb60337cc`：https://github.com/xw1216/EEG-FM-Bench/blob/325398d7d057ecc1216fb3510d70c16eb60337cc/data/dataset/seeds/seed.py 。
