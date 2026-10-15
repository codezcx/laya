# v8：REVE末两层适配

v7多seed显示文本头未超过普通头，且两任务外部表现均弱。本轮检验冻结encoder是否限制效果，保持任务、源被试划分、60/3通道及八个空间锚点不变。

缓存冻结前缀：在REVE倒数第二个Transformer block的attention输入处截取完整token序列，保留通道及时间维。情绪每样本240×512，MI为12×512；均为4个时间patch。计算使用fp32，缓存fp16。只缓存源train/validation，不接触源test或目标。

真实checkpoint审计：fp32前缀重放末两层与完整forward最大绝对误差为0；fp16缓存导致相对L2误差约0.00028（60通道）/0.00021（3通道）。反向传播确认只有末两层和head有梯度，末层参数6,280,192。审计脚本`PYTHONPATH=. python scripts/verify_reve_tail.py`；结果存`artifacts/reve_prefix_v8/replay_audit.json`。这是已冻结前缀下的数值近似缓存训练，不是全encoder微调。

四卡四组：

|组|head|encoder学习率|
|---|---|---:|
|frozen_learned|普通分类头|0|
|tail_learned_1e5|普通分类头|1e-5|
|tail_learned_1e6|普通分类头|1e-6|
|tail_text_1e5|文本sigmoid|1e-5|

从v6对应源验证选中的head warm start，REVE末层从原checkpoint初始化；记录起始head哈希。普通头三组起点相同；文本组起点不同，不能把组间差异仅归因于loss。head学习率1e-4，AdamW，seed42，600step，每步情绪32+MI32，两任务loss各权重1/2。task内dataset/class均衡采样。每150步验证，按源验证两任务macro选checkpoint，包含step0，允许选择不更新。此次预算与v6不同，直接比较应使用本轮冻结对照。

只读取源train/validation。新结果不立即用于挑选外部最优，也不扩展压力/心算任务。运行配置`configs/reve_tail_v8.json`；缓存与训练路径分别为`artifacts/reve_prefix_v8`和`artifacts/reve_tail_v8`。
