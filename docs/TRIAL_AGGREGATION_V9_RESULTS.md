# v9：trial聚合核查结果

SEED-VII Happy/Neutral/Sad子集；固定对每trial的10个窗口平均softmax概率。以下trial口径与之前window口径不同。

|模型|seed|trial accuracy|被试macro balanced accuracy|被试bootstrap95CI|
|---|---:|---:|---:|---|
|text_sigmoid|42|37.50%|40.62%|35.90–45.35%|
|text_sigmoid|43|32.34%|38.82%|35.21–42.71%|
|text_sigmoid|44|40.00%|41.39%|37.08–46.18%|
|learned_ce|42|43.13%|43.06%|38.19–48.40%|
|learned_ce|43|38.75%|41.53%|36.88–46.53%|
|learned_ce|44|38.91%|40.63%|36.39–45.42%|

text_sigmoid：三seed均值±样本SD = 40.28±1.32%。

learned_ce：三seed均值±样本SD = 41.74±1.23%。

结论：普通头trial balanced accuracy均值41.74%，对比window口径40.10%，提升有限。未选择seed或聚合规则，未训练任何新权重。不能据此称3秒实时识别改善，也未证明文本语义有额外收益。MI每条输入本来对应独立trial，本轮未改变其评估。

测试：记录路径不同不合并；概率均值计算正确；重复窗口或trial内标签冲突拒绝。三项测试通过。
