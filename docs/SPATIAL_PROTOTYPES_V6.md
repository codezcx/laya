# v6：显式空间锚点汇聚

v5情绪改善但MI下降，故检验空间信息是否在查询汇聚中丢失；这是待验证假设，不预设结论。

从REVE位置库读取实际3D坐标，在F3/F4/C3/Cz/C4/P3/P4/Oz八个固定位置汇聚通道token。权重为softmax(-距离平方/(2×0.02²))，不学习；仅对有效通道计算。锚点有固定顺序，保留左右空间信息。任意数量通道均映射为8×512，经LN→Linear→GELU→Dropout→文本空间或普通分类头。不对缺失电极伪造原始波形，但远离实际覆盖区的锚点仍只能插值已有token，不能恢复未采集信号。

60通道情绪和3通道MI缓存、源被试划分、采样策略、1500step、batch96、lr1e-4、seed42与v5保持一致。四卡四组text_ce/text_sigmoid/random_ce/learned_ce。仅源验证选checkpoint，不进行本轮外部测试。参数量与v5不同，比较的是连接结构整体，不是严格参数量匹配。

接口输入只依赖通道名称/坐标与EEG token；任务标签仅用于已知标注任务内loss和指标。当前不同任务的通道配置仍有数据来源捷径，不能声称自动生理任务识别。

配置：`configs/spatial_prototypes_v6.json`。
运行：`python scripts/run_channel_prototypes.py --config configs/spatial_prototypes_v6.json`。
输出：`artifacts/spatial_prototypes_v6`，每组记录具体坐标、通道名及sigma审计。
