# EEG跨数据集实验索引

目标仍未完成：先建立运动想象与情绪的可靠跨数据集证据，再扩展心算/压力。目前是任务已知分类；未证明自动任务识别或未见类别/任务泛化。文本原型未显示优于普通分类头。REVE预训练包含2b，不能称encoder从未见过目标数据集。

|轮次|问题/改动|目前证据|记录|
|---|---|---|---|
|v1/v2|三通道REVE→文本原型；任务mask；去均值|文本未稳定优于随机原型/普通头|[结果](MULTITASK_RESULTS_V1.md)|
|v3|简单统计与冻结REVE源域探针|MI源验证约60–61%，情绪约37–41%|[结果](SOURCE_PROBE_V3_RESULTS.md)|
|v4|情绪通道/频段/幅度消融|62通道相对功率三折源验证53.06%，3通道41.81%|[结果](EMOTION_SOURCE_PROBE_V4_RESULTS.md)|
|v5|60通道情绪+3通道MI共享查询池化|情绪改善、MI下降，整体未改善|[结果](CHANNEL_PROTOTYPES_V5_RESULTS.md)|
|v6|固定坐标空间锚点汇聚|恢复MI源验证；探索性外部结果略改善|[结果](SPATIAL_PROTOTYPES_V6_RESULTS.md)|
|v7|v6三seed完整复验|普通头SEED-VII窗口balanced 40.10±0.71%，2b accuracy54.99±0.28%|[结果](SPATIAL_REPEATS_V7_RESULTS.md)|
|v8|冻结前缀，微调REVE末两层|四组均选step0，无源验证增益|[结果](REVE_TAIL_V8_RESULTS.md)|
|v9|固定均值概率聚合同trial窗口|普通头trial balanced41.74±1.23%；不同评估单位|[结果](TRIAL_AGGREGATION_V9_RESULTS.md)|
|v10|空间对齐频谱与REVE融合|源验证无增益，仅REVE对照复现53.17%|[结果](SPECTRAL_FUSION_V10_RESULTS.md)|
|v11|MI整源留一选模|跨源MI约51–53%，无明显改善|[结果](DOMAIN_VALIDATION_V11_RESULTS.md)|
|v12|完整Cho/Dreyer/Lee源开发池|完整源池审计通过；2b接近59%，情绪未同步改善|[结果](FULL_MI_SOURCES_V12_RESULTS.md)|
|v13|参数近似匹配的分任务读出|情绪外部仍38–39%，未解释主要瓶颈|[结果](TASK_READOUTS_V13_RESULTS.md)|
|v14|FACED作为第二情绪源|分段续传修复后持续获取；四组多源训练已配置，尚待数据完成|[协议](FACED_SOURCE_V14.md)|

## 使用结果的边界

- 所有源选模准确率均不是独立泛化证明。反复源验证开发也可能过拟合，后续需跨源划分验证。
- v6/v7/v9目标结果都是探索性：目标此前已暴露，未按目标选择checkpoint/seed，但不应称新盲测。
- SEED-VII只使用Happy/Neutral/Sad映射，非完整7分类；主指标是按被试macro balanced accuracy。2b报告普通accuracy。
- v9均匀采样窗口遍布trial，不能称3秒实时识别；不可与窗口指标直接排名。
- 当前情绪仅SEED作有标签源，MI为Cho+Dreyer，任务/数据来源高度关联。通道数能泄露任务，不把任务路由结果当生理语义理解。
- EEGMat已准备但未纳入当前训练；新增任务不应掩盖MI/情绪未达标。

## 仓库与产物

配置、代码、协议及结果摘要进Git；模型、原始数据、缓存、逐样本预测留`artifacts/`并被忽略。每轮使用独立输出目录，完成后记录checkpoint哈希；不覆盖旧结果。当前实验分支`experiment/multitask-text-prototypes`，未推送远程。

- v14 complete: [FACED multi-source results](MULTI_EMOTION_V14_RESULTS.md). All four arms reported; emotion transfer remains weak and text prototypes show no gain.

- v15 complete: [common-channel task-free transfer](TASKFREE_V15_RESULTS.md). Five-class unmasked training; high task routing but chance-level emotion transfer, no text-prototype advantage.
