# 多任务文本原型 v1：结果

模型仅训练SEED情绪与Cho/Dreyer运动想象；EEGMat暂未训练。以下为任务已知的分类准确率。

|模型|源验证任务macro|SEED新被试|Cho新被试|Dreyer新被试|Lee未训练数据集子集|
|---|---:|---:|---:|---:|---:|
|text_ce|47.84%|38.22%|57.75%|50.69%|55.75%|
|text_sigmoid|47.86%|40.81%|54.75%|54.17%|58.00%|
|random_ce|48.56%|39.56%|56.00%|50.97%|58.50%|
|learned_ce|49.25%|41.85%|55.50%|51.39%|56.25%|

源验证用于选checkpoint，不能当独立泛化证据。SEED此处是跨被试，不是跨数据集。Lee仅5个新被试、每人每类最多40个trial；不能替代完整Lee或2b评估。random_ce并不使用文本语义，其性能接近文本模型时，不能声称语言理解提升了泛化。

## SEED-VII独立数据集

Happy/Neutral/Sad映射到positive/neutral/negative，非完整7分类。按被试计算balanced accuracy；各类trial数12/8/12，普通accuracy机会基线不等于多数类基线。

|模型|被试macro balanced accuracy|被试bootstrap95CI|
|---|---:|---|
|text_ce|35.12%|33.61–36.70%|
|text_sigmoid|35.53%|33.92–37.48%|
|random_ce|34.71%|33.78–35.67%|
|learned_ce|35.53%|33.85–37.32%|

## 2b完整E评估集

2840 trials；以下CI对应被试macro accuracy，而非pooled accuracy。REVE预训练包含2b，且本项目已有2b评估历史，不能称严格数据集未见。

|模型|Accuracy|被试macro accuracy|被试bootstrap95CI|
|---|---:|---:|---|
|text_ce|53.17%|53.10%|50.03–57.47%|
|text_sigmoid|54.96%|54.87%|50.76–59.90%|
|random_ce|53.66%|53.60%|50.47–57.87%|
|learned_ce|53.84%|53.72%|49.03–59.17%|

## v2：任务内文本原型去均值

保持数据划分不变；seed仅改变模型初始化及采样。先前目标结果已经查看，后续属于探索性迭代。此轮仅比较源验证，不追加SEED-VII/2b筛选。

|配置|源验证macro task accuracy|选中step|
|---|---:|---:|
|multitask_prototypes_v2_text_ce_s42|48.23%|1000|
|multitask_prototypes_v2_text_ce_s43|48.06%|750|
|multitask_prototypes_v2_text_ce_s44|48.42%|500|
|multitask_prototypes_v2_text_sigmoid_s42|47.90%|1000|

## 限制

单seed、少通道、有限窗口。Qwen只离线编码文本，不在EEG推理路径。没有证明未见任务泛化。任务未知联合argmax未校准，不计入成功标准。数据集/设备/参考方式可能造成任务识别捷径。REVE预训练包含2b；其他目标的预训练暴露尚未完全审计。后续迭代仍以源验证选择，不可根据目标成绩反复调参后宣称目标完全未见。
