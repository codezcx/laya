# v5：可变通道REVE连接模块

动机：源域消融支持情绪增加通道。原62通道缓存提取被位置数断言中止，REVE位置库缺少CB1、CB2，未生成有效缓存；失败日志保留于`artifacts/emotion_reve62_v5`。修正版显式排除这两个通道，使用其余60通道，仍为1–30Hz、逐窗口逐通道标准化。60通道不等同于先前62通道诊断，不沿用其准确率。

冻结REVE在每通道的时间维池化，得到C×512（情绪C=60，MI C=3）。四个可学习query经4头cross-attention读取有效通道，mask排除padding；REVE已编码电极位置。连接模块自身不另加通道序号位置，输入token重排应不改变结果。readout将4×512映射到512，再映射到离线Qwen文本原型空间或普通7槽分类头。

四组：text_ce、text_sigmoid、random_ce、learned_ce。文本按任务去均值并归一化，随机原型仅对照语义是否必要。不同输出头参数量不同，不声称参数匹配。任务内masked loss，未标注任务不作负样本；跨任务分数不校准。情绪通道数与MI不同，可能强烈泄露任务类型，不能据此声称生理任务识别。

四卡先提取60通道情绪源train/validation，后并行四组训练；MI复用v1冻结缓存。固定原被试划分，seed42，1500step，batch96，lr1e-4，AdamW。每250步按源验证macro task选checkpoint，不在本轮运行源test或外部测试。协议及配置在训练前固定，不根据目标结果筛选。

输出：`artifacts/emotion_reve60_v5`、`artifacts/channel_prototypes_v5`。保留查询池化与padding不变性单测。此次是冻结encoder的接口实验，并非REVE全量微调或Qwen语言推理。
