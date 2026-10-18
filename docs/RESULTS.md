# 当前阶段：EEG属性问答

源训练346494条、独立源验证75060条问答均已构造并通过校验。十二类问题分别保存，固定1dB容差；训练集合并分布及逐数据集分布见 `datasets/eeg_attribute_qa/statistics.json`。部分题存在明显偏斜，使用语义答案分组采样和选项随机置换。

四卡属性训练进程已启动，训练前先记录初始化检查点的属性基线。实时结果以 `artifacts/jev_attribute_qa/status.json`、`run/metrics.jsonl` 为准。未得到结果前不声称属性读取或最终分类得到改善。

启动代码提交 `f85bfd4`。29项单元检查通过；真实四卡两步smoke中，两次更新后各rank可训练参数哈希一致，loss/梯度有限，峰值分配显存58.73GB/卡。启动器确认smoke成功后进入正式run，smoke权重未作为正式初始化。证据：`artifacts/jev_attribute_qa/smoke.log`、`smoke/complete.json`。

1200步采样重放：108000次抽样、105601条不同问答、47036个不同训练窗口，单问答最多重复3次；不声称完整覆盖346494条问答。审计：`artifacts/jev_attribute_sampling_audit.json`。

正式run已完成step0的2880条源验证panel：MI属性平衡准确率34.03%，情绪属性31.77%，CE1.16743；均为属性题，不是最终类别分数。已确认第1次正式优化器更新完成，训练loss1.19433、梯度有限。后续动态数值以metrics.jsonl为准；初始曲线在 `artifacts/jev_attribute_qa/plots/training_curves.png`，运行绘图脚本可刷新。

## 初始化检查点及五类分类对照

上一轮4380步分类训练、完整源验证和外部评估均已完成。以下用于后续对照，均不是本轮属性成绩：

| 条件 | 源MI平衡准确率 | 源情绪平衡准确率 | 2b普通准确率 | SEED-VII-HNS普通准确率 |
|---|---:|---:|---:|---:|
| 投影器 | 61.73% | 42.44% | 58.94% | 36.89% |
| 投影器＋LoRA | 63.89% | 41.66% | 56.87% | 33.77% |

本轮初始化使用第二行源选出的4380步best.pt。来源：`artifacts/jev_eeg_coverage/runs/{projector,lora}_seed42/full_source_validation.json` 和 `external.json`。源指标按被试、数据集平均；外部栏为普通准确率，不能混用定义。2b/SEED-VII已经被多次查看，且REVE预训练包含2b，不属于全链路未接触盲测。

此前完整实验记录归档于Git `6b42e3d`，原始日志与检查点保持在忽略目录。当前没有自动恢复旧训练或启动外部目标评估。

## 停止旧 MI 属性方案（2026-10-18）

用户要求停止训练并重建含同试次基线的 MI 问答。控制进程及四卡子进程已停止；日志观察到 step 800，最后持久化 last.pt 为 step 700。检查点、日志保留。旧 MI 前后半段比较不是静息/任务比较，不能解释为 ERD；旧结果仅属于旧数据定义。
