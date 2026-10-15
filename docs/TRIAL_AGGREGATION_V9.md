# v9：情绪trial评估单位核查

此前SEED-VII每trial均匀取10个不重叠3秒窗口，报告窗口分类。本轮固定规则：对同一dataset/subject/recording/trial内窗口的任务条件softmax概率取均值，再argmax；不调温度、不加权、不挑窗口、不挑seed。保留全部v6/v7文本sigmoid和普通头三个seed。

此为已暴露目标上的探索性统计，不是新模型训练。共6400窗口聚合为640trial，20被试，每人32trial。按被试计算balanced accuracy及bootstrap，不能把窗口数作为独立样本量。

必须单列trial与window结果，不宣称通过更改评估单位改善同一任务。均匀窗口遍布整个trial，需要整段trial，不是3秒实时分类。若不改善，不能继续按目标筛选聚合规则。
