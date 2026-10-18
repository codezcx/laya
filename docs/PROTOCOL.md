# 当前协议：配对MI与冻结联合读出

当前方案及必要匹配对照均已完成，结果见[RESULTS.md](RESULTS.md)。本协议区分已停止的接口训练与最终冻结读出，避免将二者的可训练参数混淆。

## 数据和输入

OpenBMI：44名训练被试8800 trials、5名源验证被试1000 trials；每trial包含六道属性题和一道左右手分类题。情绪属性数据保持不变。合计208462训练问答、57100验证问答，详见[数据定义](ATTRIBUTE_DATASET.md)。问题共享同一trial，不能将问答数量当作独立信号数量。

MI使用C3/Cz/C4，连续信号1–30Hz四阶零相位Butterworth滤波并重采样200Hz；提示前[-2.5,-0.5]秒与提示后[0.5,2.5]秒各400点。每段每通道去均值，两段/所有通道共享RMS尺度，无裁剪。每段右侧复制填充160点后独立通过冻结REVE，输出3通道×3patch×512维，两段合计18 tokens。情绪沿用120 tokens。缓存为 `artifacts/jev_paired_qa/tokens/`，42627输入，四个shard及哈希审计。

JEV输入：prefix → 参考段文字 → 9个参考段tokens → 任务段文字 → 9个任务段tokens → 问题/选项/decision。文本先查词嵌入表，信号经LN512→Linear512/5120→GELU→Linear5120/5120；沿序列拼接后输入Qwen因果语言主干。文本显式提供任务，未测试自动任务识别。

## 已停止的接口训练

JEV-27B语言基座和REVE冻结，更新connector与已有语言LoRA；LoRA始终生效。属性题用固定JEV A/B/C决策行，分类用A/B，决策行不训练。初始化来源及SHA256记录在 `artifacts/jev_paired_qa/initialization.json`，权重副本为 `initialization.pt`。

固定映射对照每步全局86条：左右手各16条，其他54个问题/数据集/语义类别组各1条。组内seed42无放回shuffle；分类固定A=左手/B=右手，属性随机排列。loss质量分配：分类25%、MI属性25%、情绪属性50%。四卡22/22/21/21条，microbatch6，AdamW connector 5e-5、LoRA 5e-6，weight decay0.01，梯度裁剪1。计划600步，实际第200步停止，未完成600步，不宣称遍历完整问答集。

第100步的不可变权重 `diagnostics/fixed_step100.pt` 是所有最终原头/联合对照共用checkpoint；已接受3200个不同分类trial的标签。第200步单类塌缩，停止继续优化，保留 `control_fixed_mapping/` 作为匹配控制。稀疏采样和随机选项对照也保留，不是推荐训练路线。

## 最终冻结读出

冻结上述checkpoint所有参数（包括LoRA，但forward仍启用LoRA）。取最后有效decision位置5120维state；REVE按每段、每通道的三个时间patch平均，得到2×3×512=3072维。联合特征为[state, REVE]共8192维，拟合StandardScaler+Ridge双输出one-hot回归；REVE单独3072维是必要对照。

两个标签预算分别是精确重放原采样器所得3200个已见分类trial，以及全部8800个源训练trial。每个预算内按相同35名拟合/9名内部选参被试划分，alpha候选10/1000/100000，仅用源训练内部成绩选择，再用对应预算全部数据重拟合。最终四个头均alpha1000。标准化器只在对应拟合数据上训练，导出时折叠进权重/偏置，float64计算确保分数复现。9名内部被试曾参与冻结JEV的接口训练，不是端到端未见被试。

头文件位于 `diagnostics/fixed_step100_readout/combined/` 和 `reve_export/`，分别命名 `combined_{seen_classification_labels|all_training_labels}_head.json` 与 `reve_{seen_classification_labels|all_training_labels}_head.json`，绑定checkpoint/数据/缓存/权重哈希。分数不经概率校准，不能作为置信概率。

## 外部2b评估

使用B01E至B09E的两个测试反馈会话，共2840 trials、左右手各1420。原250Hz，cue_sample=trial_matlab−1+750；仅保留前三个EEG列，排除EOG。按源预处理规则重采样200Hz，事件舍入误差最多2ms。参考窗包含提示音，称准备期参考，不称纯静息；C3/Cz/C4为双极记录，与OpenBMI参考方式有差异。[官方协议](https://www.bbci.de/competition/iv/desc_2b.pdf)。

外部独立目录 `artifacts/jev_paired_qa/external_2b/` 不进入训练加载器，构建/编码入口为 `scripts/build_external_paired_mi.py`、`scripts/encode_external_paired_mi.py`。已完成原始数据、事件/标签、有限值和哈希核对；编码与源anchor相对L2差0.000358（阈值0.001），受半精度量化/批次计算影响，非bit-exact。

所有头和alpha在本次外部成绩前固定，无目标选参。评估固定A=左手/B=右手的正常信号和被试内打乱信号，联合/REVE×两个预算×源/外部共8个任务，加同checkpoint原头对照。源四个读出4000条预测全部精确复现。2b不能视为完全盲测：[REVE预训练清单](https://arxiv.org/html/2510.21585v1#A2)包含BNCI2014004，历史中也曾查看目标成绩。

## 复现与产物管理

完整联合推理命令见[README](../README.md)。省略 `--external-input` 即源验证；`--readout-head`改为REVE头并加 `--device cpu` 可复现REVE对照；省略readout参数使用原JEV头。各次输出目录必须不存在。

`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/report_readout_evaluation.py`重新核对已保存逐trial预测、总体/被试分数、置换成员、源离线一致性及哈希，生成 `readout_evaluation/summary.json`、`REPORT.md`、`audit_complete.json`。未重新训练或按2b结果改头。

仅代码进入Git，数据、模型、缓存、日志与checkpoint被忽略且保留。旧代码通过Git历史访问；保留当前方法必要控制，不累计独立版本目录。当前没有 `run/` 活跃训练；训练历史位于三个control目录，最终结果统一位于readout_evaluation。
