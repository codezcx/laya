# LEAF Tower 语言决策训练与个人适配协议

_2026-11-24：原生8B共享SFT，原test作为开发集；后文保留已完成的个人联合适配协议。_

## 📋 后续实验：原 test 用作开发集

2026-11-24用户明确决定：希望在探索阶段观察原test上随训练步数变化的表现，并据此选步数、checkpoint和方法；原valid待方案确定后再使用。接受此用途安排，不交换原始划分名单、被试或缓存标签。

| 原始文件划分 | 后续用途 | 是否用于梯度更新 |
| --- | --- | --- |
| `train` | 训练和训练指标监测 | 是 |
| `test` | 开发过程评估、选步数/checkpoint、比较方法 | 否 |
| `valid` | 用户确定方案后再评估 | 否 |

原test承担的是模型选择中的开发/验证角色，其选中成绩标记为“原test开发集BAcc/Acc”，不作为未参与调参的独立benchmark测试结果。反复据留出集表现挑选设置也会引入选择偏差，不能因未反传就宣称其评价独立[^evaluation_roles]。本项目原valid已经在历史及当前实验中多次评估、选点，后续即使暂不使用，也不能将其描述为从未接触过的盲测；如需这样的最终评估，须另外预留未用于开发的数据并排查被试/预训练重叠。

实施状态：4B已完成，旧实现归档`ad4c054`后统一更新共享训练器、控制器及绘图。新策略见[配置](../configs/evaluation_policy.json)和[运行门禁](../eeg_align/evaluation_protocol.py)：test只能评估，valid禁止前向，train独占梯度。评估ledger记录完成的每库评估次数及全局样本数（包含重复评估），随checkpoint保存/恢复，不再固定写test次数为0。旧4B结果不重标。

抽样沿用原512算法（固定随机128起点，再按类别/被试队列扩充），但底层只请求原test；不改原始行标签。`test_selection_ids.json`保存固定ID，后续选点前逐条核对；完整预测保存`test_*_predictions.npz`。`test_step*_subset/full.json`显式记录split、development用途和评估范围；完整原test结果同样不是独立测试。原始数据配置不变，保证InfoNCE父权重及缓存身份核验；新run的effective config明确覆盖旧selection/test_policy声明。

新运行须记录`selection_split=test`、原划分ID及样本指纹、评估频率和全量/子集范围；方案未确定前不自动执行valid最终评价。该用途调整不授权改变学习率、batch、训练预算、缓存或被试划分，也不自动将个人适配query纳入support。实验设计与文档技能用于区分文件标签和统计用途、保留暴露历史，过程引用沿用文末[^skills]。

[^evaluation_roles]: scikit-learn developers. Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html

## 📋 当前控制：原生 Qwen3-VL-8B 指令微调

本轮“用我们的JEV微调”指沿用现有EEG指令决策训练流程，不是用27B JEV生成答案或蒸馏。模型为官方Qwen3-VL-8B-Instruct[^qwen8b]，仅加载内置`Qwen3VLTextModel`；不运行视觉encoder、ECG代码或远程Python。复制在`third_party/Qwen3-VL-8B-Instruct`，官方revision及校验值固定在[语言主干配置](../configs/language_backbones.json)。启动器检查四份权重完整SHA256和配置/tokenizer/index的Git blob哈希；源目录不修改。这不是此前测速的ECG-R1-8B派生权重。

共用[训练实现](../eeg_align/train_joint.py)、[抽样/预算配置](../configs/benchmark_train.json)和[任务文本](../configs/benchmark_tasks.json)。旧JEV加载器和前向保持哈希不变；小模型只替换主干加载及连接器宽度，不新建另一套训练器。

| 项目 | 本轮设置 |
| --- | --- |
| EEG初始化 | 已完成多正例InfoNCE的best step1700 Tower；不是JEV SFT step1360 |
| 语言初始化 | 官方Qwen3-VL-8B-Instruct；无JEV/ECG-R1 LoRA |
| 连接器 | 新LayerNorm(256)→Linear(256,4096)→GELU→Linear(4096,4096) |
| 语言更新 | 新LoRA r32、alpha64、dropout0；Q/K/V/O及MLP gate/up/down；基础权重冻结 |
| 决策更新 | 原生A–P词表行初始化16行读出，bias初始化0；允许训练，不是随机分类头 |
| 损失和输出 | 随机排列候选选项的choice交叉熵；单次forward，不逐字生成，不新增对比/蒸馏损失 |
| 数据 | 原17库训练，9库原test开发评估；缓存、QC、subject/session划分及task文本不变 |
| 预算 | 2040步，seed42，global batch16，micro4，四卡同步；前340步仅连接器 |
| 学习率 | connector 1e-4、语言LoRA 5e-6、head 1e-5、Tower 1e-6；原warmup62/cosine/wd0.01 |
| 开发评估 | 从step0起每340步，原test固定每库512上限；九库等权BAcc选点，同分取更早步数 |
| 最终报告 | 选中checkpoint的完整原test开发分数；不执行原valid前向 |

8B权重有独立`lm_head.weight`，读出从该矩阵抽取A–P行；微调读出是独立FP32参数，不更新词嵌入或其余词表行。语言LoRA为标准A随机、B全零，初始不改变原生主干。Tower继续使用原eval/冻结buffers策略。17步短测逐个覆盖全部训练库，只评估train，检查实际梯度、四rank一致性和保存重载；正式阶段重新初始化，不继承短测更新。每20步保存续训状态。

四卡实际加载语言基础参数7,568,405,504；connector 17,834,496、语言LoRA 87,293,952、choice head 65,552、Tower 19,557,632，联合可训练124,751,632。审计见本轮`initial_state.json`。4B历史语言基础参数4,022,468,096、联合可训练92,873,488；8B随主干hidden size改为4096，connector及LoRA参数量相应增大，不能宣称可训练参数量不变。学习率、rank、batch和步数保持不变。

这仍是类别选项监督的EEG决策SFT，不是开放式通用问答训练。与已有27B相比，参数量、架构、预训练/决策LoRA经历、tokenizer和连接器宽度都不同，属于实用替换对照，不能将差异唯一归因于模型大小。已知TUAB/TUEV候选身份重叠和LEAF上游预训练重叠风险沿用原协议。

运行目录`artifacts/benchmark/leaf_tower_qwen8b_20261124/`；`native_weight_verification.json`保存官方校验，`pipeline.json`保存执行阶段，`train/leaf_infonce_mlp/`保存训练/原test开发曲线数据、transfer audit和checkpoint。绘图文件用`development_test`命名，轴标题也标出original test。尚未有8B最终准确率；4B完整原valid BAcc为42.68%，不可与8B原test开发分数直接作同协议模型对比。

[^qwen8b]: Qwen. Qwen3-VL-8B-Instruct official model repository. https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

## 🎯 问题与假设

目标是用单个新用户的少量标注数据适配已训练EEG→JEV系统，同时保持服务器LM不变。原残差-only两库等权BAcc为50.18%；本轮检验进一步更新Tower是否改善独立session/run预测，而非仅提高支持集拟合。不增加共享源用户训练，不新增对比损失、TTT或LM更新。

## ⚙️ 两组训练路径

两组都使用个人原始EEG→Tower→残差adapter→冻结原connector→冻结JEV；支持集标签与JEV choice logits的交叉熵是唯一损失。梯度穿过冻结LM和连接器，最终预测仍由JEV输出。

| 模块 | joint_lora | joint_full |
| --- | --- | --- |
| LEAF forward Tower | 双分支最后两层attention的Q/V LoRA-r8；alpha8，无LoRA dropout | 全部19,557,632个前向参数 |
| 残差adapter | 更新33,088参数 | 更新33,088参数 |
| 原MLP connector，27,535,872参数 | 冻结 | 冻结 |
| JEV基础权重、已有语言LoRA、choice head | 全冻结 | 全冻结 |
| Tower BN running统计量及所有buffers | 冻结 | 冻结 |
| 总可训练参数 | 65,856 | 19,590,720 |

adapter为256→64→256、GELU、残差上投影零初始化。LoRA为原个人C分支相同的Q/V低秩参数化，不更新K或原投影权重；两组均保留Tower eval模式，仅改变requires_grad，不引入BN适配或dropout差异。全参指实际forward Tower，既有接口已移除不用的重建头。

父模型固定为`artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，原验证选中step1360。每位用户独立重置Tower、LoRA、adapter及优化器；不继承另一人的更新，也不从上一轮个人adapter再加100步。先seed42初始化adapter，再单独重置seed42初始化LoRA，保持adapter初始化跨组一致。每个用户需云端LM前向及输入梯度，不是纯边缘端训练。

## 📋 数据与划分

下表描述父模型处理后窗口与原被试划分，不是个人support/query数量。MI缓存每trial恰好一个4秒窗口。

| 数据集 | 处理后窗口总计（父模型原训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（父模型原训／验／测） |
| --- | ---: | --- | --- | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4类） | 固定跨被试；个人T→E session | 9（5/2/2） |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4类） | 固定跨被试；个人前四→后两imagery run | 109（69/19/21） |

- 正式用户：2a 8、9；PhysioMI 89–109。两组各23人，共46次独立适配。
- 每人按类无放回选择5条，共20条；2a支持来自T session，query为全部E session；PhysioMI支持来自R04/R06/R08/R10，query为全部R12/R14。
- 每组共460条support、1,208条query（2a 576，PhysioMI 632）。支持池剩余样本不训练、不加入query。
- 从已完成`leaf_personal_ce_alltest_20261123`逐条核对支持/query ID、共同基线与100步batch流。
- 复用已校验的原始归一化信号缓存；旧tokens只用于共同不适配对照，联合训练与适配后预测全部重新经过Tower。
- 短测：2a验证用户6、PhysioMI验证用户70，两组各3步、8条query；只核查工程实现，不选超参数。
- 不使用query标签或集合统计更新权重、归一化、选点或早停。2a用record/run/event联合键防止重复trial。

## 🔒 固定预算与审计

| 设置 | 固定值 |
| --- | --- |
| 支持样本／seed | 每人5条/类，共20条；seed42 |
| 步数／选点 | 固定100步，无query选点 |
| Tower与adapter学习率 | 同为1e-4，constant，无新LR组或扫描 |
| 优化器 | AdamW，weight decay0.01 |
| 梯度／batch | 两部分合并global clip1；batch16=micro4×4 |
| 抽样 | 每步类均衡，在已选20条内有放回取样；逐条匹配原baseline |
| GPU | 0/1联合LoRA，2/3联合全参；每组12/11人，无跨用户梯度聚合 |
| 恢复 | 跳过完整用户；未完整用户从父模型重跑，不恢复中途优化器 |

每组先检查两部分非零有限梯度与真实更新，Tower冻结部分及全部buffers哈希不变，JEV/connector/head参数版本不变且没有参数梯度。逐用户重置一致、初始adapter恒等、保存重载tokens/logits完全一致。源码/config/父权重/缓存变化触发停止，不在live job中热改。

资源门槛：四卡各至少60,000MiB空闲，磁盘至少20GiB。所有单元测试和四卡短测通过后才正式运行。每10步保存信号侧`last.pt`及loss，完成保存`adapter.pt`、预测和审计；全参组保存的是完整个人Tower＋adapter，不是小LoRA文件。LM权重不复制。

## 📊 输出、比较与限制

根目录`artifacts/benchmark/leaf_personal_joint_20261124/`。主要比较同一用户/同一query上原残差-only、联合LoRA、联合全参的BAcc。先按用户等权得到每库均值，再计算两库等权均值；另列23用户等权均值、每用户变化及负迁移人数。不能将support Acc作为泛化结果。

全部46次通过审计才生成完整汇总，部分结果不混作完整均值。原baseline来自先前已完成运行，不冒充本轮同步重跑的对照；运行前复核相同父权重、输入、batch和初始预测。仅一个seed、2a两名用户，被试8已观察过；属于探索性对照，非盲测或显著性结论。上游LEAF预训练身份重叠未完全排除，个人query是原test子集，不是完整EEG-FM-Bench。

## ⚡ 推理速度基线（不改变适配协议）

`scripts/benchmark_inference.py`是当前模型的只读诊断工具，不是新增训练方案。固定父step1360和2a被试8 joint_lora checkpoint，单卡batch1，测试0.5/1/2/4/8/16/30秒；原提示词、模型精度、未合并LoRA和10秒上限分块保持不变。每项预热10次，50次CUDA同步计时；完整路径与各模块分别测量。

长片段重复真实4秒trial，短片段取前缀，仅用于计算量/延迟测试；没有据此训练、选模型或评价分类准确率。预处理与当前LEAFSignalAdapter逐值核对，分拆语言主干与原JEVBridge logits核对一致。测速不包括采集等待、模型加载或网络开销，详见[结果页](RESULTS.md)及原始记录。当前缺少Qwen线性注意力快速kernel，未安装依赖或进行代码优化。

同一测速脚本支持`--backbone qwen3vl-8b`：从本地ECG-R1-8B-RL读取内置Qwen3VLTextModel对应的语言权重，复用现有JEVBridge.forward的单次决策读出。小主干未进行EEG对齐，连接器为随机256→4096→4096；仅测速，不替换当前正式方法或报告其分类效果。沿用同一信号侧权重、输入和10/50计时预算；tokenizer差异导致文本token数为72而非JEV的76，需与架构/kernel/LoRA路径差异一起披露。原27B结果保留，比较见[结果页](RESULTS.md)。

`--backbone leaf-native`只测发布的完整LEAF，默认分别测FP32和BF16 autocast；strict加载原checkpoint，不用父模型或个人权重。沿用同一4秒输入及预处理，使用发布MPNet缓存、Motor任务指令level1和四类原型，终点为CPU余弦分类argmax；不计在线文本编码。超过10秒分别跑完整LEAF再按分块时长加权平均类别分数，这是显式测速包装，不当作官方原生长序列能力。没有训练、改变当前个人适配方案或覆盖之前结果。

## 🔗 实现与归档

[配置](../configs/personalization.json)、[模型模块](../eeg_align/personalization.py)、[训练入口](../eeg_align/run_personalization.py)、[控制器](../scripts/run_personalization.py)。此前残差-only归档`34c11a8`，运行源码`513427d`；夜间MEMO/SSF/AdaBN全部完成，保留在隔离worktree及原产物，不恢复其独立实现到当前树。原192次计划的取消标记继续保留。

实验设计技能用于逐用户匹配和隔离数据，资源技能用于容量检查，模型技能用于冻结LM梯度路径，Markdown技能用于同步当前协议；沿用现有本地模型/环境。资源目录未能提供本任务适用模型，不作为模型选择证据。过程参考[^skills]不是方法新颖性来源。

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
