# 冻结 FBCSP / REVE 的任务与被试信息探测

这是当前 EEG 表征的独立诊断，不训练解耦器，不改 JEV、LoRA 或 EEG–语言投影器。不占用 GPU；三个数据集各一个 CPU worker，每个限制 4 个计算线程。产物仅写入 `artifacts/feature_probes/`，存在时拒绝覆盖。

## 数据及特征

使用已经存在的源 MI 缓存及其逐试次 manifest，不使用情绪、2a 或 2b：

| 数据集 | 试次 | 原源训练被试 | 原留出被试 | 留出试次 |
|---|---:|---:|---:|---:|
| Cho2017 | 9,480 | 42 | 5 | 1,000 |
| Lee2019_MI | 9,800 | 44 | 5 | 1,000 |
| Dreyer2023 | 18,152 | 67 | 9 | 2,160 |

数据沿用此前缓存的保留规则，并非三个原始数据集的所有被试/试次。本轮不因预测正确与否剔除样本。

- 输入与已有 REVE 缓存匹配：C3/Cz/C4，200 Hz，1–30 Hz，任务 epoch 前 600 点（3 秒），同一 trial 内共同幅度缩放。这里的 epoch 起点按既有数据导出定义，不统一宣称是 trial onset 或 cue 后第几秒。上游滤波可能使用整个已导出的活动 epoch；这是离线诊断，不宣称严格在线因果推理。
- 冻结 REVE 最终层 `12 × 512` tokens 按通道/时间顺序展平到 6144 维，不用 JEV/文本投影器，不用微调后 REVE，不重新加载基模推理。
- FBCSP 使用完全相同的输入信号。频带预注册为 4–8、8–12、12–16、16–20、20–24、24–28、28–30 Hz；各 2 个 CSP 分量，OAS 正则、epoch 协方差、对数功率，共 14 维。频带内使用离线四阶 Butterworth 双向滤波。完整特征及互信息筛选后的 6 维均测；不是旧 4–40 Hz、18 维 FBCSP 成绩的复现。
- 每个数据集仅在下面指定的任务探针训练被试上拟合一套共享 CSP 和 MI 特征选择器，之后冻结；身份探针参与者完全不参与拟合。不能拼接每个人自己拟合的 CSP 输出。
- 数据预处理共同缩放已去除每 trial 绝对幅度尺度；本轮不据此判断原始 EEG 是否包含绝对幅度身份信息。

## 数据划分与防泄漏

任务探针预测左手/右手，而非 MI/情绪类别。原 `split=validation` 被试作为本诊断的固定源留出测试面板。原训练被试按 seed42 再划约 20% 做内部验证，其余训练。所有表示共用同一划分，不在最终面板上选 C、epoch、seed 或架构，不选完再合并验证数据重训。

被试探针只用原留出被试（Cho/Lee 各 5 类，Dreyer 9 类）；同一个人的不同记录分配到训练/验证/测试。此处利用身份面板的训练标签是探针测量步骤，不是给任务分类器做目标适配。两套探针不共享可训练参数。源留出集之前已被其他实验反复查看，因此不是新盲测。

| 数据集 | 身份训练 | 身份验证 | 身份测试 | 结论边界 |
|---|---|---|---|---|
| Lee | session 0 前 70% 的 trial，边界前去 3 条 | session 0 后 30%，边界后去 3 条 | session 1 | 跨会话；内验证同会话，可能选模偏保守/失配 |
| Dreyer | 前 3 runs | 第 4 run | 最后 2 runs | 跨 run；训练和测试部分跨 acquisition/online 条件 |
| Cho | 每个被试×类别按导出 trial 序号前 60%，边界前去 2 条 | 60–80%，两侧各去 2 条 | 最后 20%，边界后去 2 条 | 仅同记录索引块隔离；导出序号按标签分组，不能当真实时间 |

每个数据集额外做左手训练/验证→右手测试，及右手训练/验证→左手测试；从以上固定分区取对应类别，不改变 CSP 拟合。

校验试次路径和信号 SHA256 不跨分区，任务的被试隔离、身份的闭集类别一致、Lee 测试 session 隔离、Dreyer 三组 run 隔离。Cho 不包装为跨会话或可靠时间一致性结论。任一检查失败即停止。

REVE 权重来自预训练基模，未完全排查其预训练数据重叠，不能声称整个表示学习过程未见过这些人。

## 分类器与记录

- 标准化器只在当前探针训练分区拟合。每个被试×任务类别的训练权重相同，避免数量差异主导。
- 线性：StandardScaler + L2 LogisticRegression，C∈{0.001,0.01,0.1,1}，lbfgs、最多 1000 轮、tol=1e-5。按内部验证选 C，保存收敛警告；确定性求解只运行一次。
- 非线性：StandardScaler + Linear(D,128)→GELU→Dropout(0.1)→Linear(128,K)，seeds42/43。AdamW，lr1e-3，weight decay0.01，batch256，最多120 epochs；每5轮验证，25轮未改善停止。两个 seed 均报告，不在测试成绩上挑 seed。
- 任务选模及主指标为被试平均平衡准确率；身份为平衡准确率。另存普通准确率、分被试结果、混淆矩阵、恒定多数类基线。任务随机基准50%，Cho/Lee身份20%，Dreyer身份11.11%。
- 2000次被试 bootstrap 的区间仅为描述性参考；身份分类器与类别集合保持固定，不是“重训到新身份集合”的置信区间。两个随机种子不当作独立被试。
- 准确率是给定探针下的可解码性，不是比特数、信息比例或统计独立性证明。

产物包含源代码快照和哈希、缓存校验、划分、特征、共享 CSP、选参曲线、模型、测试逐样本概率和汇总。启动后如运行依赖代码变化，训练器在后续阶段检查时会拒绝继续，防止静默混用版本。不会修改已有 JEV 训练文件。

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -p test_feature_probes.py -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_feature_probes.py
PYTHONPATH=. laya/.venv-eeg/bin/python -m eeg_probes.train --root artifacts/feature_probes --summarize
```

运行状态：`artifacts/feature_probes/status.json`；持续汇总：`artifacts/feature_probes/RESULTS.md`。每数据集预定 4 种测试×3种表征×（1个线性+2个MLP）=36个读出模型，总计108个；线性内部 C 候选另外记录。

## 方法与工具参考

- [scikit-learn 分组交叉验证与训练内预处理](https://scikit-learn.org/stable/modules/cross_validation.html)。
- [MNE CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html)。
- 本次使用 `scikit-learn` 和 `get-available-resources` 技能组织防泄漏检查与资源限制；参考 Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，检索时最新为2026-09-02修订版。
