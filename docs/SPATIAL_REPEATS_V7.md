# v7：空间汇聚随机种子复验

冻结v6结构、数据划分与1500step训练预算，追加seed43、44，各训练text_sigmoid与learned_ce。与v6 seed42合并报告，不挑最好seed，不据目标结果修改结构。仅模型初始化和训练采样流变化；被试划分保持不变。

四卡一组一卡，配置分别为`configs/spatial_v7_*_s*.json`。训练记录在对应`artifacts/spatial_v7_*_s*`，进程清单在`artifacts/spatial_repeats_v7/jobs.json`。每组由源验证两任务macro选择checkpoint，完成后记录SHA256。

首先报告源验证逐seed及均值/标准差；这些数值仍是选择指标，不是独立泛化证明。复验目的：判断普通头优势、情绪改善与MI恢复是否依赖单个随机种子。两个任务未建立稳定跨数据集证据前，不加入压力或心算。

四组完成后冻结所有权重，再复用v6的外部特征逐一评估全部新seed。与seed42一起完整报告，不选最佳，不用这些目标结果改超参数。该复验属于已暴露目标上的探索性稳定性检查；不能重新定义为盲测。运行：`python -m eeg_tasks.spatial_repeat_eval`。
