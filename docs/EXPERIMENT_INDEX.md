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
|v14|FACED作为第二情绪源|四组完成；SEED-VII约35–39%，文本原型无优势|[结果](MULTI_EMOTION_V14_RESULTS.md)|
|v15|统一三通道、不提供任务ID|任务路由容易，但留出情绪类别接近随机|[结果](TASKFREE_V15_RESULTS.md)|
|v16|共同30通道频谱跨情绪源|FACED→SEED约44%，反向约38%；三通道接近随机|[结果](EMOTION_TRANSFER_V16_RESULTS.md)|
|v17|同一30通道REVE与频谱融合|八组完成；拼接与文本头均未超过频谱基线|[结果](EMOTION_FUSION_V17_RESULTS.md)|
|v18|冻结模型概率融合|源验证选权重后无迁移收益；保留频谱基线|[结果](EMOTION_LATE_FUSION_V18_RESULTS.md)|
|v19|幅值信号统一平均参考|原参考频谱复现；CAR无一致收益，反向下降|[结果](EMOTION_REFERENCE_V19_RESULTS.md)|
|v20|频段×连续时长对照|45Hz或9秒均未超过30Hz/3秒频谱控制|[结果](EMOTION_CONTEXT_V20_RESULTS.md)|
|v21|REVE末两层跨源同类约束与频谱辅助|四卡训练完成；SEED-VII约41%，仍低于43.72%频谱基线|[结果](EMOTION_ALIGNMENT_V21_RESULTS.md)|
|v22|扩大FACED源被试并匹配验证集|源验证提升1.18–1.43点；SEED-VII普通频谱仍43.72%，无明确迁移收益|[结果](FACED_FULL_V22_RESULTS.md)|
|v23|MI通道相对幅值与整域验证|相对协方差2b准确率69.61%，相关矩阵53.27%；信号基线有收益，尚非REVE/语言模型收益|[结果](MI_COVARIANCE_V23_RESULTS.md)|
|v24|冻结REVE逐通道／共同缩放对照|收敛修复完成；共同缩放2b准确率63.27%，逐通道60.56%，仍低于协方差基线|[结果](MI_REVE_SCALING_V24_RESULTS.md)|
|v25|协方差与REVE概率互补|源选25% REVE融合2b为68.77%，未超过协方差69.61%；不支持融合收益|[结果](MI_FUSION_V25_RESULTS.md)|
|v26|情绪跨通道相对功率信息|SEED-VII平衡准确率43.68%→43.05%；源验证增益未迁移，控制特征精确复现|[结果](EMOTION_POWER_V26_RESULTS.md)|
|v27|情绪监督语义与刺激覆盖审计|FACED源2,744条主观评分完整；当前使用刺激类别，尚未训练评分辅助目标|[审计](EMOTION_SUPERVISION_V27_AUDIT.md)|
|v28|主观评分辅助与未见视频验证|四组完成；真实评分源增益0.26点，低于打乱评分0.55点，不支持个体评分收益|[结果](EMOTION_RATING_V28_RESULTS.md)|
|v29|匹配trial推理上下文的训练单位对照|SEED-VII trial平衡准确率窗口训练46.88%、均值训练45.56%；无外部增益|[结果](EMOTION_TRIAL_V29_RESULTS.md)|

## 使用结果的边界

- 所有源选模准确率均不是独立泛化证明。反复源验证开发也可能过拟合，后续需跨源划分验证。
- v6/v7/v9目标结果都是探索性：目标此前已暴露，未按目标选择checkpoint/seed，但不应称新盲测。
- SEED-VII只使用Happy/Neutral/Sad映射，非完整7分类；主指标是按被试macro balanced accuracy。2b报告普通accuracy。
- v9均匀采样窗口遍布trial，不能称3秒实时识别；不可与窗口指标直接排名。
- v14起情绪源包含SEED与FACED，MI源包含Cho/Dreyer/Lee；各轮具体留出规则见协议。任务与采集来源仍有关联。v15统一三通道后，任务路由仍不能证明生理语义理解。
- EEGMat已准备但未纳入当前训练；新增任务不应掩盖MI/情绪未达标。

## 仓库与产物

配置、代码、协议及结果摘要进Git；模型、原始数据、缓存、逐样本预测留`artifacts/`并被忽略。每轮使用独立输出目录，完成后记录checkpoint哈希；不覆盖旧结果。当前实验分支`experiment/multitask-text-prototypes`，未推送远程。

- v14 complete: [FACED multi-source results](MULTI_EMOTION_V14_RESULTS.md). All four arms reported; emotion transfer remains weak and text prototypes show no gain.

- v15 complete: [common-channel task-free transfer](TASKFREE_V15_RESULTS.md). Five-class unmasked training; high task routing but chance-level emotion transfer, no text-prototype advantage.
