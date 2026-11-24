# LEAF Tower → JEV-27B：13 库验证实验协议

_2026-11-24；当前实现为jev_train/。数据来自已完成原生LEAF复现；历史JEV训练语义来自5611b49。_

## 🎯 当前实验

用户确认使用原始JEV-27B，在刚完成的13库组合上训练我们的决策模型。它是LEAF信号Tower→LayerNorm→Linear(256,5120)→GELU→Linear(5120,5120)→JEV语言主干→[decision]位置choice logits；无图像输入、无LEAF文本编码器/FiLM/Q-Former，也无新对比训练阶段。

Tower沿用历史JEV实验的发布版instruction初始化：`leaf-v1.0-instruct-mpnet-base.ckpt`，SHA256 `f9ca9cabbd734d2691fa6748adf3e78056f895d4031f7edc481e7d7b6822f3fa`。只加载tower.*并移除不参与forward的重建头。不使用本轮LEAF第50/100轮或历史EEG→JEV下游权重。发布权重的下游暴露范围未独立审计，不声称干净zero-shot。

JEV使用本地`laya/JEV-27B-VL`原始语言权重和adapter_vllm；载入现有rank32/alpha64 LoRA，choice A–P对应的16行决策权重融合发布版lm_head LoRA。大语言主干的原始参数冻结；连接器、Tower、现有LoRA和16行决策权重/偏置按阶段更新。

## ⚙️ 保持历史训练参数

| 项目 | 当前设置 |
| --- | --- |
| 训练预算 | 2040步；四卡全局batch16、每卡microbatch4，seed42 |
| 采样 | 每13步覆盖13库；库内按类别及被试均衡随机取样，复用历史算法 |
| 第1–340步 | 仅更新新连接器；Tower/LoRA/decision head不更新 |
| 第341–2040步 | 连接器＋Tower全参＋JEV已有LoRA＋choice head联合更新 |
| 峰值LR | connector 1e-4；Tower 1e-6；JEV LoRA 5e-6；choice head 1e-5 |
| 优化器 | AdamW，weight_decay0.01；62步warmup，cosine最低比例0.02；裁剪梯度范数1 |
| Tower模式 | 保持历史eval/dropout/BatchNorm策略；eval模式不等于参数冻结 |
| 损失 | 当前样本有效A–P选项的交叉熵，不使用相似度分类损失 |
| 文本 | 任务、问题、候选选项；训练选项顺序按样本ID/step/seed置换，标签相应重映射；不输入答案或被试ID |
| 输入 | 已归一化65通道、200Hz；保留原生4秒或10秒窗口，不再次预处理 |
| EEG tokens | 两分支各每0.5秒一个token：4秒16个、10秒40个，均为256维 |
| 评估并行 | 四卡按样本索引分片，汇总后核对无重复/遗漏，再计算每库Acc/BAcc |
| 评估频率 | step0、每340步与末步；每库固定最多512个开发样本；训练监控每库128条 |
| 选点 | 原test固定子集的13库等权BAcc最高；相同时保留更早step |
| 完整评估 | 选中checkpoint访问原test全部53,363窗口；逐样本概率与每被试指标保存 |
| 短测 | 13步覆盖全部任务，只访问train；与正式模型重新初始化哈希一致，短测权重不继承 |
| 断点 | 每20步和阶段边界保存优化器/RNG/模型参数与buffers；后续用--resume恢复 |

本轮是沿用旧JEV预算的验证实验，2040×16共32,640次训练样本抽取（可以重复），不是完整遍历144,553条训练窗口100次。原生LEAF采用自然样本采样、100epochs、global1024，其初始化/采样/更新量均不同，不能把差值解释为只改变语言主干的因果效果。

## 📋 评估角色与数据契约

沿用用户对JEV后续实验的现行约定：只使用原train产生梯度，原test用于开发曲线/选点，原valid暂缓，不改变manifest内任何成员和split名称。原valid与原test已经在原生LEAF实验被观察过；暂缓不构成全新盲测。所有选点后的原test分数标记development、independent_test_estimate=false。

manifest SHA256固定为`60647f3ae8a5fb9cd286a6958cc758d4a394a6c2f748ec8c9379e908400c9f2a`；启动审计窗口数、标签范围、划分间样本不重叠和训练信号有限值，记录源文件stat与每样本ID。signal()逐次检查源shard未改变。类别顺序逐库与固定的LEAF tasks.yaml核对，尤其OpenBMI/SHU为Right/Left，SEED各库不能互套旧标签编号。

HighGamma的文本说明为实际运动；PhysioNet沿用原处理器的二分类编号，但选项如实写T1/T2 motor event，不把双拳/双脚及执行运动包装成纯左右手想象。

## 📊 数据与划分

以下是本轮实际名单与全量预处理后的计数，来自 `datasets/leaf13/manifest.json`（2026-11-24）。数量不从历史 EEG-FM-Bench 表搬用。被试总数为库内唯一人数；train 与 valid 的被试集合重叠，因此三个被试数不能相加求总数。[^data]

| 数据集 | 本轮处理后窗口：总（训/验/测） | 任务（分类数） | 本轮划分协议 | 被试：唯一总数（训/验/测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 5,088（3,148/788/1,152） | 运动想象（4） | A01–A07 中每第5样本验证；A08–A09测试 | 9（7/7/2） |
| OpenBMI-MI | 21,600（13,440/3,360/4,800） | 运动想象（2） | 前42人训验池，后12人测试 | 54（42/42/12） |
| SHU-MI | 11,988（7,712/1,929/2,347） | 运动想象（2） | 前20人训验池，后5人测试 | 25（20/20/5） |
| HighGamma | 6,742（3,761/941/2,040） | 左/右运动解码（2） | 前10人训验池，后4人测试；去掉feet | 14（10/10/4） |
| Cho2017 | 9,880（6,464/1,616/1,800） | 运动想象（2） | 排除32/46/49，剩余前40/后9人 | 49（40/40/9） |
| Shin2017A | 1,680（1,056/264/360） | 运动想象（2） | 排除S05，剩余前22/后6人 | 28（22/22/6） |
| PhysioNet | 19,571（11,447/2,862/5,262） | T1/T2运动执行与想象（2） | 前80人训验池，后29人测试；R03–R14 | 109（80/80/29） |
| FACED | 10,332（6,720/1,680/1,932） | 情绪（9） | 000–099训验池，100–122测试 | 123（100/100/23） |
| SEED | 38,070（22,545/7,905/7,620） | 情绪（3） | 每人trial 1–9/10–12/13–15 | 15（15/15/15） |
| SEED-IV | 37,575（26,025/5,340/6,210） | 情绪（4） | 每人每session trial 1–16/17–20/21–24 | 15（15/15/15） |
| SEED-VII | 69,742（34,000/18,542/17,200） | 情绪（7） | 每人每session trial 1–10/11–15/16–20 | 20（20/20/20） |
| OpenBMI-SSVEP | 10,800（6,720/1,680/2,400） | SSVEP（4） | 前42人训验池，后12人测试；仅公开train blocks | 54（42/42/12） |
| Workload | 2,134（1,515/379/240） | 负荷（2） | 前32人训验池，后4人测试 | 36（32/32/4） |

所有“训验池”都先按官方被试/样本顺序拼接，再取从index0开始每5个样本中的一个为valid，其余train；不是每名被试重新从0计数。

## ⚠️ 已知差异与限制

- 缺少 SEED-V、BCIC-Upperlimb、BCIC-Speech；下载结果不会中途加入已启动的13库训练。
- Cho2017/HighGamma/SSVEP 的 MOABB 缓存已经有200Hz、0.1–75Hz与60Hz notch处理，再执行LEAF的0.3–40Hz/尺度归一化；不能恢复滤掉的信息，不能宣称与作者原NPZ逐样本相同。逐trial源路径、run、标签另存到shard旁的sources.json。
- SSVEP现有两session公开train共10,800条，未包含公开test blocks。与后续个人/跨被试划分的“test”不是同一个概念。
- PhysioNet本地缺少的执行运动runs从PhysioNet公开源补齐，再调用官方R03–R14处理器。官方代码将T1/T2统一记为Left/Right，但其中也包含双拳/双脚及实际运动；不应称为纯左右手想象。
- FACED按代码的100/23被试划分，不按论文表格可能存在的边界笔误。SEED采用当前官方修正后的Negative/Neutral/Positive顺序；SHU严格采用当前tasks.yaml的Right/Left映射，后续对原始标注语义仍应单独核验。
- 复用官方按被试或按被试/划分估计的robust scaling，包含同一评估组内未标注信号统计；不是单trial在线归一化。保留原始short-window滤波与大量插值通道的警告，不擅自改变方法。
- 13库的留出测试仍来自指令训练涉及的同名数据集，不是13个全新数据集的zero-shot；跨数据集zero-shot应额外用未参与指令训练的Dreyer2023、Weibo2014等独立数据评估。[^heldout]
- 官方主结果还包含每库单独的Tower＋MLP监督微调；本轮未运行该阶段，不能把direct-inference结果与其最佳监督微调结果等同。


## 🔧 工程与历史参照

原生LEAF13库复现已完成，完整源码与协议归档Git `73e26f9`。原生LEAF使用无监督预训练Tower、MPNet＋IQF，完整测试Task+Targets BAcc为第50轮49.59%、第100轮48.85%；其valid/test角色与本轮JEV不同。原生复现实现保留为同数据参照与公共准备工具，不与JEV同时重复训练。

当前JEV训练器、权重加载、分阶段LR、选项重排、梯度同步与小checkpoint语义继承Git `5611b49`，仅恢复当前需要的模块，未恢复旧独立实验。17项单元测试和13步四卡真实短测均已通过；控制器随后自动进入正式阶段。

资源检查/Transformers技能用于本地权重与四卡容量核查，Markdown技能用于记录可复核的训练参数和数据边界；科学模型目录未检索到适用EEG条目，未引入其他模型或服务。[^skills]

[^native]: LEAF固定官方代码。https://github.com/Jiang-Muyun/LEAF/tree/08ee6577ca4e9517536e73ba3053e676c028a379
[^data]: LEAF官方数据处理与划分。https://leaf-bci.github.io/docs/datasets/instruction-tuning-datasets.html
[^heldout]: LEAF官方独立数据集评测。https://leaf-bci.github.io/docs/datasets/held-out.html
[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
