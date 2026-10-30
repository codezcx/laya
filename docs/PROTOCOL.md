# LEAF→JEV 多任务验证性 SFT

_首阶段 2026-10-29；2026-10-30 启动续训。固定划分、一个共享模型。用户同意保留 HMC 子集和 TUAB/TUEV 候选身份重合，先验证方法。_

---

## 🎯 实验定义

输入为原始信号窗口，经 LEAF 的 65 通道适配、任务文本条件化查询，再投影到 JEV 文本塔；JEV 接收问题和候选标签，在末尾 decision 位置输出分类概率。不使用原来的视觉 encoder，也不为每个数据集训练独立分类器。

这是已有基础权重上的**多任务监督微调**，不是从零预训练。测试是数据集内跨被试 holdout；不能称为跨数据集 zero-shot，也不能将本轮结果作为已严格排除泄漏的 benchmark 成绩。

当前实现唯一入口：[训练配置](../configs/benchmark_train.json)、[任务定义](../configs/benchmark_tasks.json)、[控制器](../scripts/run_benchmark.py)。旧 BSR 实现归档于 Git `14c60cf`，不再保留其训练入口或 2b 队列。

## 📊 固定数据划分

使用 `xw1216/EEG-FM-Bench` 提交 `325398d7d057ecc1216fb3510d70c16eb60337cc` 的划分函数和类别定义，但不能称为完全复现论文配置。随机划分使用已记录的本地文件顺序，不冒称拿到了作者未公布的逐人固定名单。[^1]

2026-10-30 补录核查：本地 SEED 选择跨被试模式，偏离论文对应的 `finetune_sub_dependent`；2a、PhysioMI、Workload 的部分窗口数与论文表格不同；HMC 是已接受的子集。SEED-VII/TUEV 另有 `round` 与上游 `floor` 导致的 1 点边界偏移，尚未修复。分项证据和其他模型成绩统一见[结果文档](RESULTS.md#与论文的划分及窗口一致性核查)，原划分和成绩不追溯改写。

唯一划分目录：[multitask_splits_20261029_final](../artifacts/benchmark/multitask_splits_20261029_final/summary.json)。源 summary SHA256 为 `042520e32bf63b304dba955184757adb53db5b661d39f617601ee78c831b1545`。保持原清单不可变；其中早先的 `training_ready=false` 是生成时的审计状态，用户接受风险的授权另记在训练配置，不反向修改原始审计。

| 数据集 | train 分组 | valid 分组 | test 分组 | 窗口 |
| --- | ---: | ---: | ---: | ---: |
| 2a | 5 | 2 | 2 | 4 s |
| PhysioMI | 69 | 19 | 21 | 4 s |
| SEED | 13 | 1 | 1 | 4 s |
| SEED-VII | 16 | 2 | 2 | 15 s |
| TUAB | 2075 | 101 | 152 | 15 s |
| TUEV | 328 | 20 | 22 | 5 s |
| HMC 本地子集 | 75 | 17 | 17 | 30 s |
| Workload | 26 | 5 | 5 | 4 s |
| ADFTD | 64 | 11 | 13 | 10 s |

2a 为被试 1–5 / 6–7 / 8–9，T/E 两个 session 始终同归属。PhysioMI 使用 4、6、8、10、12、14 六个想象 runs，四类分别为左手、右手、双拳、双脚。2b 排除。

| 额外数据集 | 训练缓存样本数 | 本轮验证／测试 |
| --- | ---: | --- |
| BNCI2015001 | 2400 | 无 |
| BNCI2014008 | 33600 | 无 |
| BNCI2014009 | 5760 | 无 |
| CHB_MIT | 26292（原 27436） | 无 |
| SleepEDF | 414961 | 无 |
| AlexMI | 480 | 无 |
| Zhou2016 | 614 | 无；仅现有 session 0 |
| Nakanishi2015 | 1800 | 无 |

这些是已有缓存行数，不是 benchmark 原始文件数。别名重复库 BNCI2014001、EEGMAT、旧 SEED/TUAB PKL 不另加入；额外数据的所有已纳入行只属于 train。

### 训练前信号 QC

首轮第 42 步暴露 CHB_MIT 原始 PKL 全零窗口。新增独立、不可变的 [QC 清单](../artifacts/benchmark/multitask_signal_quality_20261029/summary.json)，逐窗口检查非有限值、全零和全通道时间恒定；规则不使用标签、不拟合阈值。单个缺失通道不触发整窗排除，低幅值但有变化的信号保留。全量 1,053,069 个窗口中，仅 CHB_MIT 的 1,144 个训练窗口全零，其余全部通过这些检查。

原始 PKL、NPY、索引和冻结被试划分不改写；采样池按原始行 ID 排除上述窗口，不重编号，不删除被试或类别。当前有效训练总量为 953,425，valid 49,069、test 49,431 不变。若检查发现无效验证／测试窗口，入口直接阻止启动，不能自动缩小评估分母。全量 QC 只做确定性输入有效性检查，不计算 held-out 模型成绩，不据此选择模型。

每个数组保存 SHA256、大小和修改时间；启动核验固定的 QC 报告与索引哈希及数组 stat。训练前另用实际采样 seed、双卡划分与 microbatch 边界检查全部 2,040 步的信号适配；正式运行仍在异常时停止，不在训练循环中静默跳样本。

## ⚙️ 信号和文本接口

- 原始 benchmark 信号先按原始通道选择连续滤波，再重采样到 LEAF 的 200 Hz。滤波频带默认 0.1–100 Hz（原采样率不足时不另设 100 Hz 低通），2a 为 0.5–45 Hz；工频陷波按数据集取 50/60 Hz。这不是逐字节复现上游默认 256 Hz 预处理。
- 2a 使用 BNCI MAT，trial 起点后 2–6 s 为 cue 后 4 s；所有 288 条/session 均保留，不按标签或 artifact 字段额外筛选。SEED 原 CNT 显式读 int32，并由事件表偏移重算错误的样本长度。
- 其余事件、类别顺序与窗口规则参考固定源码；当前时间转采样点使用 `round` 而非上游 `floor`，不是完全相同的边界实现。TUEV 保留上游重复事件和末尾对齐的完整 5 s 窗口，不把短事件补零成 5 s；SEED-VII 同样保留末尾对齐窗。窗口越界、未知标注逐记录记录，不重新抽被试。
- 额外 PKL 保留已有预处理；Nakanishi 的 829–831 点只补缺失尾部，不裁掉实测点。
- 每个输入 epoch 独立计算 0.1/99.9 百分位裁剪、median/IQR 归一化；不以验证／测试数据拟合统计量。
- 双极通道使用 minimum-norm 电极表示，不能称为恢复真实单极电位；参考电极不虚构空间坐标。映射到 65 通道使用 LEAF 的模板插值，少通道信息不足的限制仍在。
- 完整窗口分成不超过 10 s 的有序片段，每片输出 8 个 256 维 query；4/5/10 s 为 8 tokens，15 s 为 16，30 s 为 24。不截断长窗口。
- 任务描述经冻结 MPNet masked mean pooling + L2 归一化后输入 LEAF。问题和候选文本经 JEV tokenizer 输入文本塔；训练时随机置换选项顺序并对应重映射监督标签。真值、被试 ID 和数据集 ID 不写入 prompt。
- HMC 的标签顺序是 W/R/N1/N2/N3，SleepEDF 是 W/N1/N2/N3/REM；各自映射，不能共用数字标签含义。

## 🔧 本轮训练预算

下表为已完成的首阶段预算；当前追加训练的设置见下方“续训阶段”。

| 项目 | 固定设置 |
| --- | --- |
| 初始化 | 发布 LEAF + 原生 JEV adapter + 新 connector |
| 更新内容 | LEAF query 路径、connector、JEV 原有 LoRA、16 个候选位置的 head 行 |
| 冻结内容 | JEV 主干、MPNet、LEAF 重建／文本输出头、BatchNorm 统计 |
| GPU | 2、3 两卡同步梯度 |
| 总更新数 | 2040；seed=42 |
| 全局 batch | 16；每卡 microbatch=4、累积 2 次 |
| 数据集采样 | 每 17 步随机轮转全部数据集各一次 |
| 数据集内采样 | 类别均衡，再均匀选被试、抽其窗口 |
| LEAF 冻结预热 | 前 102 步；之后更新有效 encoder 路径 |
| 学习率 | connector 1e-4，LoRA 5e-6，head 1e-5，LEAF 1e-6 |
| 调度 | warmup + cosine；每组从开始更新时预热；最小比例 0.02 |
| 稳定性 | bf16 基座、梯度检查点、全局梯度裁剪 1、weight decay 0.01 |
| 可恢复保存 | step 0、每 20 步、冻结/解冻边界 102/103、最终步；含 optimizer/scheduler/各 rank RNG |

每个数据集获得 120 次更新、1920 次样本呈现；大库远未遍历一遍，小库会重复抽样。这是检查方法可行性的 pilot，不宣称充分预训练或完整数据遍历。

## 📈 验证、选模与测试

1. step 0 和每 340 步，在每个 benchmark 固定最多 128 条验证样本上评估；样本选择固定且与模型分数无关。
2. 用 9 个数据集的 balanced accuracy 宏平均选 checkpoint；相同时保留较早 checkpoint，包括 step 0。
3. 首阶段训练结束后，从磁盘重载选中的模型，完整评估 9 个 benchmark 的所有可用验证和测试窗口；额外库不测。当前续训不自动重复此测试。
4. 保存逐样本概率、真值、record ID、被试 ID、每被试结果、混淆矩阵、accuracy 和 balanced accuracy。
5. smoke 只做训练集两步更新（覆盖冻结及解冻），验证非零 encoder 梯度和保存／重载 logits 完全一致；不读测试指标，不把 smoke 权重带入正式训练。

首阶段验证 accuracy / balanced accuracy 日志从 step 0 开始；训练 loss 只记录 step 1–2040，没有 step-0 训练 loss 或验证 loss，不补造这些数值。原始 JSON/JSONL 保留，并由独立 [绘图脚本](../scripts/plot_training_curves.py) 导出 PNG/PDF/SVG 与 CSV。总 loss 展示原值及过去 51 步平均，各任务 loss 展示原值及过去 9 次该任务更新的平均；均须凑满窗口才绘制平滑值。验证只连接实际评估点，若缺失则留断点，不平滑、不混入完整评估结果；纵轴依据观察范围留边并明确标注独立缩放。图表见[结果记录](RESULTS.md)。

恢复训练须通过配置、源码、基础权重、QC 和缓存索引校验；修改运行中代码会触发停止。异常分别写入每个 rank 的故障文件，并更新进度状态；故障时不冒险执行可能死锁的集体保存。首轮没有 `last.pt`，因此修复轮从发布权重重启，而非续接第 41 步。

### 补充完整训练集评估

2026-10-30 按用户要求，用原测试对应的 `train/best.pt`（step 2040）独立推理九个 benchmark 的全部 467,518 个有效训练窗口，不包含八个额外 train-only 库。不重复采样、不改划分、不更新参数、不访问验证／测试信号；因此结果是完整训练划分 Accuracy，而不是训练过程中 batch 指标的平均，也不局限于 2040 步中曾被抽到的样本。

入口为 [evaluate_training_split.py](../scripts/evaluate_training_split.py)，启动命令 `laya/.venv-eeg/bin/python scripts/evaluate_training_split.py --start`。使用原配置 GPU 2、3、microbatch=4、固定选项 variant=0 及原输入适配；校验原训练身份、代码／配置、模型和缓存索引。推理前检查重复预测一致性，前后核对微调参数及全部 buffers 未变。各 rank 每 512 个窗口落盘预测和校验哈希，每库完整结束才汇总 Accuracy、Balanced accuracy、混淆矩阵和每被试指标；不根据部分预测填报完整训练集成绩。

产物单独保存于 `artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/`，不改原训练／测试记录。仅在旧进程已退出且评估未完成时，用同一命令加 `--resume` 继续；已有预测块须通过身份和内容校验，`complete.json` 才表示九库全部完成。原测试集指标不重新计算。

### 续训阶段（2041–4080）

2026-10-30 用户在查看验证曲线后要求继续训练。预算预先定为追加 2040 步（另 120 个完整的 17 任务轮转），不是无限训练至某个满意分数；验证曲线有波动，不能假定每库都未收敛或继续训练必然提高成绩。

- 沿用同一 [train_joint.py](../eeg_align/train_joint.py) 入口和原始训练配置，以 [benchmark_continue.json](../configs/benchmark_continue.json) 明确阶段差异；数据、标签、预处理、采样 seed、batch、更新参数范围均不变，不加入新下载的 HMC 记录。
- 从首阶段 `train/last.pt` 的 step 2040 恢复全部微调参数、buffers、Adam moments 和各 rank RNG；该步也是首阶段 best。父 checkpoint SHA256 为 `3aac8625a801bba73f2b0c815c29ee7ada6169ce1beaf2b0324ce53e0aa26839`。恢复时逐项核验张量指纹，且复现同一验证子集的混淆矩阵后才继续更新。
- 原 cosine 学习率已到初始峰值的 2%。本阶段采用显式低峰值回升：step 2041 保持原末尾 LR，step 2102 达到原峰值的 20%，然后 cosine 衰减至 step 4080 回到原峰值的 2%；不重置优化器，不重新冻结 LEAF。这是修改调度的探索性续训，不冒称原调度无缝延长。
- 四组峰值依次为 connector `2e-5`、JEV LoRA `1e-6`、decision head `2e-6`、LEAF `2e-7`；结束值依次为 `2e-6`、`1e-7`、`2e-7`、`2e-8`。
- 在 step 2040 基线及 2380、2720、3060、3400、3740、4080，评估原来的九库固定验证子集，另记录每库最多 128 条固定训练子集的 Accuracy、Balanced accuracy 和交叉熵；同时新增验证交叉熵。训练子集仅从 QC 通过的 train 行按固定、与标签无关的随机序列选取，保存样本 ID 哈希；不是完整训练集或当步 batch。缺失的首阶段监测点不补造。
- 仍只按九库验证 BAcc 等权平均选择共享 best，包含续训起点 2040，相同时保留较早模型。每 20 步更新可恢复 `last.pt`，每次评估另外保存 `stepXXXXXX.pt`（模型参数及 buffers，不含 optimizer），避免中间模型只剩最后一个。
- 续训阶段在代码入口禁止测试集前向，包括到达 4080 后；不自动重跑完整训练集／测试集。首阶段测试结果已经被查看，因此本阶段须标为探索性，后续若复用同一测试集，不能宣称新的未触碰盲测。

输出为 `artifacts/benchmark/leaf_jev_multitask_continue_20261030/`，与首阶段产物分离但不另建一套模型实现。信号预检覆盖新增全部 2040 步的实际 rank/microbatch 输入，25 项相关测试通过。阶段进度见 [pipeline.json](../artifacts/benchmark/leaf_jev_multitask_continue_20261030/pipeline.json)，训练及监测 JSON 位于其中的 `train/`。本次只增加续训和监测能力，没有改原始数据、表中成绩或已完成的曲线。

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --continuation-config configs/benchmark_continue.json
```

## ⚠️ 已接受但未解决的限制

用户 2026-10-29 明确要求：“先这样训练吧，数据的问题后面再说，我们只是验证试验。”

- HMC 仅本地 109/151 份记录；本轮是子集结果。
- TUAB/TUEV 19 个文件名 ID 重合，5 个角色冲突：aaaaaamc、aaaaabdo、aaaaaeer、aaaaaeuh、aaaaafgz。跨版本真实身份尚未完全核验，本轮不隔离重分。
- 基础权重预训练数据重合未排除；额外 PKL 的原始记录来源审计尚不完整。
- 少通道插值、双极到 minimum-norm 表示、原始输入格式和 200 Hz 适配可能影响表现，结果须分任务报告。
- 原先 2a 微调见过新测试被试，绝不继续它的 checkpoint。

## 📦 产物和启动

信号缓存：`datasets/multitask_20261029/`，每库 `complete.json`、`index.csv` 和分片 NPY。开发或失败缓存保留但不接入训练。训练状态以运行目录中的 `pipeline.json` 与 `train/progress.json` 为准。

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --prepared
```

仅在全部缓存及匹配的 QC 完成时使用 `--prepared`；更换缓存需新建 QC 目录并在配置中固定报告哈希。输出目录默认 `artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/`，拒绝覆盖已有运行。失败首轮 `leaf_jev_multitask_pilot_20261029/` 保留。第二阶段 VideoMAE 全局 token 对照尚未启动。

[^1]: xw1216. EEG-FM-Bench, pinned source. https://github.com/xw1216/EEG-FM-Bench/tree/325398d7d057ecc1216fb3510d70c16eb60337cc

## 🔗 软件过程说明

模型接口核验、资源检查与文档组织参考 Scientific Agent Skills 的相应流程；本轮曲线采用其中 Matplotlib / Scientific Visualization 的原始点保留、平滑披露、颜色冗余编码和输出检查流程。出处已列于下方；这些流程不是 EEG 模型方法或实验结果的依据。[^2]

[^2]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
