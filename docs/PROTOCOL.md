# LEAF→JEV 多任务验证性 SFT

_2026-10-29；固定划分、一个共享模型。用户同意保留 HMC 子集和 TUAB/TUEV 候选身份重合，先验证方法。_

---

## 🎯 实验定义

输入为原始信号窗口，经 LEAF 的 65 通道适配、任务文本条件化查询，再投影到 JEV 文本塔；JEV 接收问题和候选标签，在末尾 decision 位置输出分类概率。不使用原来的视觉 encoder，也不为每个数据集训练独立分类器。

这是已有基础权重上的**多任务监督微调**，不是从零预训练。测试是数据集内跨被试 holdout；不能称为跨数据集 zero-shot，也不能将本轮结果作为已严格排除泄漏的 benchmark 成绩。

当前实现唯一入口：[训练配置](../configs/benchmark_train.json)、[任务定义](../configs/benchmark_tasks.json)、[控制器](../scripts/run_benchmark.py)。旧 BSR 实现归档于 Git `14c60cf`，不再保留其训练入口或 2b 队列。

## 📊 固定数据划分

沿用 `xw1216/EEG-FM-Bench` 提交 `325398d7d057ecc1216fb3510d70c16eb60337cc` 的划分函数与类别、窗口定义。随机划分使用已记录的本地文件顺序，不冒称拿到了作者未公布的逐人固定名单。[^1]

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
| CHB_MIT | 27436 | 无 |
| SleepEDF | 414961 | 无 |
| AlexMI | 480 | 无 |
| Zhou2016 | 614 | 无；仅现有 session 0 |
| Nakanishi2015 | 1800 | 无 |

这些是已有缓存行数，不是 benchmark 原始文件数。别名重复库 BNCI2014001、EEGMAT、旧 SEED/TUAB PKL 不另加入；额外数据的所有已纳入行只属于 train。

## ⚙️ 信号和文本接口

- 原始 benchmark 信号先按原始通道选择连续滤波，再重采样到 LEAF 的 200 Hz。滤波频带默认 0.1–100 Hz（原采样率不足时不另设 100 Hz 低通），2a 为 0.5–45 Hz；工频陷波按数据集取 50/60 Hz。这不是逐字节复现上游默认 256 Hz 预处理。
- 2a 使用 BNCI MAT，trial 起点后 2–6 s 为 cue 后 4 s；所有 288 条/session 均保留，不按标签或 artifact 字段额外筛选。SEED 原 CNT 显式读 int32，并由事件表偏移重算错误的样本长度。
- 其余事件、类别顺序与窗口规则来自固定源码。TUEV 保留上游重复事件和末尾对齐的完整 5 s 窗口，不把短事件补零成 5 s；SEED-VII 同样保留末尾对齐窗。窗口越界、未知标注逐记录记录，不重新抽被试。
- 额外 PKL 保留已有预处理；Nakanishi 的 829–831 点只补缺失尾部，不裁掉实测点。
- 每个输入 epoch 独立计算 0.1/99.9 百分位裁剪、median/IQR 归一化；不以验证／测试数据拟合统计量。
- 双极通道使用 minimum-norm 电极表示，不能称为恢复真实单极电位；参考电极不虚构空间坐标。映射到 65 通道使用 LEAF 的模板插值，少通道信息不足的限制仍在。
- 完整窗口分成不超过 10 s 的有序片段，每片输出 8 个 256 维 query；4/5/10 s 为 8 tokens，15 s 为 16，30 s 为 24。不截断长窗口。
- 任务描述经冻结 MPNet masked mean pooling + L2 归一化后输入 LEAF。问题和候选文本经 JEV tokenizer 输入文本塔；训练时随机置换选项顺序并对应重映射监督标签。真值、被试 ID 和数据集 ID 不写入 prompt。
- HMC 的标签顺序是 W/R/N1/N2/N3，SleepEDF 是 W/N1/N2/N3/REM；各自映射，不能共用数字标签含义。

## 🔧 本轮训练预算

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

每个数据集获得 120 次更新、1920 次样本呈现；大库远未遍历一遍，小库会重复抽样。这是检查方法可行性的 pilot，不宣称充分预训练或完整数据遍历。

## 📈 验证、选模与测试

1. step 0 和每 340 步，在每个 benchmark 固定最多 128 条验证样本上评估；样本选择固定且与模型分数无关。
2. 用 9 个数据集的 balanced accuracy 宏平均选 checkpoint；相同时保留较早 checkpoint，包括 step 0。
3. 训练结束后，从磁盘重载选中的模型，完整评估 9 个 benchmark 的所有可用验证和测试窗口；额外库不测。
4. 保存逐样本概率、真值、record ID、被试 ID、每被试结果、混淆矩阵、accuracy 和 balanced accuracy。
5. smoke 只做训练集两步更新（覆盖冻结及解冻），验证非零 encoder 梯度和保存／重载 logits 完全一致；不读测试指标，不把 smoke 权重带入正式训练。

所有曲线日志从 step 0 开始；当前保留 JSON/JSONL 原始数值，不将日志存在等同于已经绘制曲线。恢复训练须通过配置、源码、基础权重和缓存索引校验；修改运行中代码会触发停止。

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

仅在全部缓存完成时使用 `--prepared`；不带该参数则控制器先准备数据。输出目录默认 `artifacts/benchmark/leaf_jev_multitask_pilot_20261029/`，拒绝覆盖已有运行。第二阶段 VideoMAE 全局 token 对照尚未启动。

[^1]: xw1216. EEG-FM-Bench, pinned source. https://github.com/xw1216/EEG-FM-Bench/tree/325398d7d057ecc1216fb3510d70c16eb60337cc

## 🔗 软件过程说明

模型接口核验、资源检查与文档组织参考 Scientific Agent Skills 的相应流程；这些流程不是 EEG 模型方法或实验结果的依据。[^2]

[^2]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
