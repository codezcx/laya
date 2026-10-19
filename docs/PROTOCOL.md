# 当前协议：MI与情绪共同分类监督

当前联合训练同时学习MI左右手分类和SEED/FACED三类情绪。先前完成的MI读出实验仅作为控制，不代表情绪任务完成。

## 当前联合训练配置

在原独立属性文件旁新增 `questions/emotion_class.jsonl`，由 `classification_complete.json`独立绑定原数据/manifest/原始标签哈希；不改旧信号、属性文件或REVE缓存。新增24477条训练、8350条验证情绪分类问答，总训练232939条、验证65450条。标签逐条核对SEED label.mat及FACED events.tsv的binary_label；FACED是刺激赋予标签，不是自评。两个split的被试隔离。

每步MI左右手各16条；SEED与FACED各类别各8条；54个属性组各1条。全局134条，四卡34/34/33/33，microbatch6。分组shuffle，类别均衡采样，但不改变验证集自然分布。MI分类40%、情绪分类40%、MI属性10%、情绪属性10%总loss。300步试验，每50步验证并保存last，最多4次/问答抽样；并非遍历全部问答一轮。

冻结REVE/Qwen基座，已有JEV LoRA始终启用；更新connector（5e-5）、LoRA（5e-6）、共享JEV决策行/偏置（1e-5），其余AdamW配置同下文。头由JEV原决策行初始化，未用REVE旁路。共同从独立initialization.pt恢复connector/LoRA，不沿用仅MI读出头。MI固定A左/B右，情绪固定A负/B中/C正，属性随机选项。不能因此声称任意选项理解。

验证panel：1000条MI分类、SEED/FACED每类各128条（共768情绪窗口）、864条属性，共2632题；smoke每组1条。分别汇报三数据集分类accuracy、BA、混淆矩阵；emotion_classification专指情绪类别，emotion仅指属性。选择分数=0.5×MI分类BA+0.5×情绪两个数据集分类BA的平均，属性不参与选模。结束后用同一best模型做完整65450题源验证及panel被试内信号打乱对照。情绪是3秒窗口级分数，不是独立trial级指标；此次SEED和FACED都是源训练数据，不宣称相互zero-shot。

启动器 `scripts/run_jev.py` 验证输入与初始化，四卡两步smoke检验有限梯度及rank参数一致后才进入正式训练。两阶段均由原初始化启动。日志位于 `artifacts/jev_paired_qa/run/metrics.jsonl`；当前代码提交记录在config，恢复时检查数据、标签补充、缓存、代码和初始化哈希。旧产物保留，不删除忽略的权重/日志。

## 下列为先前MI匹配控制的固定协议

这些控制已完成，结果见[RESULTS.md](RESULTS.md)。下文的冻结头、旧batch和旧选模规则仅描述控制，不是当前联合训练配置。

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

仅代码进入Git，数据、模型、缓存、日志与checkpoint被忽略且保留。旧代码通过Git历史访问；保留当前方法必要控制，不累计独立版本目录。先前训练历史位于三个control目录，先前MI读出结果位于readout_evaluation；当前联合训练使用run目录。

## 属性辅助损失匹配对照（2026-10-19）

用户授权运行同一联合方法的必要对照。参考组为 `artifacts/jev_paired_qa/run`，无辅助损失组为 `artifacts/jev_no_attribute_loss/run`，均300步、seed42、四卡，每50步评估。

唯一训练配方变化是 `--attribute-loss-weight 0`：损失从
`0.4 MI分类 + 0.4 情绪分类 + 0.1 MI属性 + 0.1 情绪属性`
变为 `0.4 MI分类 + 0.4 情绪分类`，不将分类权重重新归一化成0.5。

保留全部134条batch的采样、排序、选项置换、分卡、microbatch和前后向计算；属性样本乘零，不贡献监督梯度。这样维持分类样本及dropout随机数消耗，避免过滤属性样本改变计算顺序。没有节省属性前后向成本，总loss数值不应直接与参考组比较。

两组均从同一4380步初始化加载投影器和现有JEV LoRA，共享决策头均从JEV原始行初始化。REVE与Qwen基座冻结；投影器、LoRA、决策头学习率及优化器不变。MI和情绪任务都保留；SEED、FACED分类分别报告。

验证仍使用同一2632题固定面板，包括未监督的属性题；选模仍只用MI分类BA与情绪数据集宏平均BA各50%。比较同一步数结果（特别是300步）及各自源验证选择的best，最终完整验证与打乱信号对照仍执行。只做单seed探索，不能当作显著性结论。

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py \
  --out artifacts/jev_no_attribute_loss --attribute-loss-weight 0
```

日志：`artifacts/jev_no_attribute_loss/run/metrics.jsonl`；stdout：`run.log`；状态：`status.json`。四卡两步smoke通过后才正式开始。所有旧checkpoint、日志和数据保留。
