# v11：整源数据集留一选模

固定v6八个空间锚点、普通头/文本sigmoid、1500step及seed42。双向验证：只用Cho原训练被试训练MI，Dreyer原train+validation作为跨数据集开发验证；反向亦然。SEED仍沿用原train/validation被试，以保持双任务训练。原test及所有外部目标不进入本轮。

这两个MI源此前都参与过模型开发，不能称全新未见的研究证据；对于本轮每个具体模型，heldout数据集没有进入梯度训练。单元测试验证整数据集排除与被试无交集。该设计缩小了训练数据，不能把不同留出方向的分数直接与v6作公平模型排名。

每250step同时计算两种选模依据：

- 跨域：SEED原验证+被留出的MI整数据集开发部分。
- 同域：SEED原验证+实际训练MI数据集的原验证被试。

分别保存best与within_best；比较二者选中step及对应跨域指标。所有指标仍是源开发验证，不作为独立最终成绩。四卡为两个留出方向×两个head。不进行2b/SEED-VII评估。

配置`configs/domain_v11_*.json`；日志`artifacts/domain_validation_v11`，权重在各独立输出目录。
