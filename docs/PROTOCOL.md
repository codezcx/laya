# 当前协议：配对输入与表示读出对照

固定映射四卡训练已停止并保留于control_fixed_mapping；下文训练配置是该匹配对照的可复现配方。联合读出已在冻结表示上拟合，未重启四卡训练；下一步是固定头的端到端及外部2b验证。

## 数据与信号接口

[数据定义](ATTRIBUTE_DATASET.md)：OpenBMI 8,800 训练 trial、1,000 验证 trial，每 trial 6 道属性题与1道左右手分类题；情绪六类属性题保持不变。总训练208,462题、验证57,100题。同一被试、trial及其全部问答不跨 split。2b不参与本轮训练或选择。

冻结 REVE。MI 的 baseline/task 各为3通道×400点，使用共同幅值归一化，分别右侧复制填充到560点并独立编码；每段9个512维token。JEV输入顺序为 prefix → baseline文字标记 → 9个基线token → task文字标记 → 9个任务token → 问题与选项。两段不是连续时间序列。情绪沿用验证过的120个单窗REVE tokens。

当前自包含缓存 `artifacts/jev_paired_qa/tokens/`，四个shard，共42,627个输入。记录数据集元数据、数组哈希和源身份；禁止误用旧的12-token MI缓存。构建阶段复用情绪缓存的值，训练阶段只读新缓存。文本提示显式告知任务类型，不宣称自动识别任务。

## 模型与初始化

真实 JEV-27B-VL 的语言 decoder，5120维hidden；已有LoRA始终启用并更新，冻结语言基座和REVE。投影器LN512→Linear512/5120→GELU→Linear5120/5120更新。保留固定JEV决策行：属性题A/B/C；左右手题A/B，屏蔽C后计算交叉熵，不生成CoT。

初始化为原源验证选出的4380步投影器与LoRA，已独立保存在 `artifacts/jev_paired_qa/initialization.pt`，原来源与SHA256保存在`initialization.json`。不沿用已停止的旧MI前后半段属性训练权重。优化器重新初始化。

## 四卡训练与采样

四卡全局batch86：左右手分类的每类取16条，其他“问题×数据集×语义答案”组取1条，组内shuffle后无放回遍历，用尽才重新打乱；每步全局打乱。13种问题，共56组。正式600步/51,600次抽样，不声称全部问答遍历一轮；分类每275步覆盖8,800题一遍，600步共19,200条分类监督；每条问答最多曝光3次。seed42。

左右手分类固定A=左手、B=右手；属性题仍独立随机使用6种排列并同步变换标签。先消耗与对照相同的所有随机抽样，再覆盖分类排列为0，保证属性排列和后续RNG一致。训练和主验证使用相同固定分类映射。MI属性六题共占loss25%，左右手分类占25%，情绪属性占50%。因此最终左右手目标不被六道属性题淹没。验证选模同样使用这三个分量的25%/25%/50%权重。

各卡22/22/21/21条，microbatch6；先累积梯度再跨卡平均，等价于全局加权平均。AdamW投影器5e-5、LoRA5e-6，weight_decay0.01，梯度裁剪1.0。每100步覆盖last.pt，每次验证改善覆盖best.pt，不累计逐步checkpoint。smoke仅2步，独立初始化、检查梯度有限与跨卡参数一致；通过后删除smoke权重，只保留审计和指标，再从初始化启动正式训练。

## 验证与可观察指标

正式训练step0、每100步及末步验证。固定panel包含全部1,000条MI分类题，加上每属性题×数据集×答案最多16条，共1,864条。分类报告普通准确率、平衡准确率、混淆矩阵；MI属性和情绪属性分别报告，不能将属性分数称为分类准确率。按数据集和问题宏平均，当前不按被试再平均。

每10步写loss、准确率、分问题准确率、梯度范数、耗时。结束后用源panel选出的best跑完整57,100题验证，并在panel中按问题/数据集打乱信号身份做对照。±1dB固定描述阈值，不按验证或2b调参。属性辅助监督不保证提升跨数据集分类；2b已多次查看且REVE预训练包含它，不宣称全链路盲测。

## 产物与清理

唯一当前运行目录 `artifacts/jev_paired_qa/`；入口 `scripts/run_jev.py`，曲线脚本 `scripts/plot_jev_training.py`。status.json记录阶段，run/metrics.jsonl记录实时指标，run.log记录stdout。断点恢复检查数据/缓存/代码/初始化哈希与采样器及RNG状态。

按用户明确授权删除旧实验日志及checkpoint，保留数据缓存、原始数据、预训练模型和小体积汇总报告；删除清单在cleanup.json。旧checkpoint不可从Git恢复，初始化副本独立保留。本次清理不是删除整个artifacts目录。

## 对照与当前改动

稀疏分类对照保存在 `control_sparse_labels/`。提高分类覆盖后的随机选项对照保存在 `control_random_options/`，第200步全部1000分类题选择B；正常与打乱准确率同为51.2%，交换选项48.8%，语义一致率0%。已停止，完整检查点200步，日志到210步。

当前从同一4380步初始化重新训练。相对覆盖对照，唯一配方变化是分类选项固定；模型、缓存、采样、loss权重、学习率、步数和验证频率不变。该对照测试稳定分类映射是否能改善信号利用，不宣称获得任意选项理解能力。

专用评估器根据checkpoint中的fixed_mi_class选择主评估顺序，另报告交换选项、同被试打乱信号，以及随机选项成绩。不同主评估顺序的结果不能直接视为同一测试条件；在同一选项条件下用正常减打乱差值判断信号利用，保留逐被试和逐trial结果。源验证已多次查看，不是盲测。最终跨数据集目标仍需2b独立评价。

## 外部输入准备

`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/build_external_paired_mi.py`构建独立2b配对输入，默认输出`artifacts/jev_paired_qa/external_2b/`，目录存在时拒绝覆盖。该目录不进入训练加载器。保留两段共享尺度、明确的事件锚点与原始文件哈希；准备期参考含提示音，不能称纯静息。详见[诊断记录](PAIRED_DIAGNOSTICS.md)。

冻结编码命令：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. laya/.venv-eeg/bin/python scripts/encode_external_paired_mi.py`。输出为external_2b/encoded，已存在则拒绝覆盖；使用源缓存anchor核对表示兼容，编码时不使用标签。当前仅准备外部输入，评估结果尚不存在。

外部模型评估复用 `scripts/evaluate_paired_classification.py`，显式传入 `--external-input artifacts/jev_paired_qa/external_2b` 才会读取目标集；默认仍只评估源验证。必须指定不可变checkpoint和新的output目录。入口检查输入/编码哈希及源缓存兼容性，输出逐被试成绩、正常/交换/打乱/随机选项结果和逐trial logits。当前只验证了真实缓存加载及索引，没有执行目标集模型推理；先完成源验证诊断，目标成绩不用于本轮调参。

联合读出命令：`PYTHONPATH=. laya/.venv-eeg/bin/python scripts/probe_decision_state.py --stage fit --output artifacts/jev_paired_qa/diagnostics/fixed_step100_readout --readout-output artifacts/jev_paired_qa/diagnostics/fixed_step100_readout/combined --features combined`。特征顺序固定为语言state5120维，再REVE3072维；只能使用同一冻结100步checkpoint。3,200和8,800标签预算分别训练，REVE单独读出是必要对照。正则只使用训练内被试选择，外部结果不反向指导配置。

## 固定读出推理验证

`--readout-head <head.json>`将评估器的最终分数替换为已导出的冻结读出，描述文件绑定checkpoint、源数据、token审计及权重哈希。联合特征顺序为最后决策state5120维、配对REVE池化3072维。通过hook读取相同决策位置，原语言LoRA保持启用；不改变backbone权重。REVE单独头在CPU上评估，不运行语言模型。输出letter_scores是Ridge分数，不能称为校准概率。

本次预定对照为联合/REVE × 3,200/8,800分类标签预算，分别在源验证和2b上测正常输入与被试内打乱输入；固定A=左手、B=右手。使用不可变100步checkpoint；所有alpha已由源训练内被试确定，禁止根据本次2b成绩再选alpha或改头。四卡分别承担两种预算的源/外部联合推理。只检验固定提示下MI分类，不宣称任意选项理解、属性问答改善或全链路未见2b的盲测。源验证需与已有离线预测逐trial核对，再解释外部成绩。
