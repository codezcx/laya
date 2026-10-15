# v3：源域特征诊断

目的：在继续调整文本接口之前，检验当前三通道信号与冻结REVE特征的线性可分性。此轮是诊断，不是通用模型，也不构成zero-shot结果。

保持v1 manifest及train/validation被试划分，只读取训练与源验证信号。情绪和运动想象分别拟合逻辑回归，所有StandardScaler只在训练数据拟合。样本按dataset/class平衡；C固定网格0.001/0.01/0.1/1/10，仅按源验证被试macro（MI再按dataset平均）选择。不追加外部测试，避免目标反复筛选。

四组输入：

- spectral：Welch功率谱，1–4、4–8、8–13、13–20、20–30Hz半开频带，三通道log power及三组通道差值，30维。
- covariance：逐窗口三通道协方差上三角，6维。由于输入已逐通道标准化，对角线近似常数；此组主要探测通道相关性。
- reve：已有冻结REVE1536维特征。
- reve_spectral：二者拼接，1566维。

每个任务有独立线性模型，与此前共享非线性多任务模型并非只改变一个变量。优势可能来自避免任务间干扰或正则化，不能直接归因于特征。当前频谱也来自已zscore、1–30Hz的数据，不能检验绝对振幅或gamma信息。

运行：`laya/.venv-eeg/bin/python -m eeg_tasks.source_probe --feature spectral`，其余组同理。输出`artifacts/source_probe_v3/<feature>`，拒绝覆盖。CPU线性代数限4线程/进程；此诊断无需占用GPU。记录原manifest哈希、每个C的结果及收敛警告、选中pipeline。

后续决策：若简单频谱明显优于REVE，先检验信号表征与预处理；若REVE线性探针优于此前多任务头，优先检验共享头/采样/优化；若两者均弱，优先用源域更丰富通道/频段做有界消融。任何决定都不能根据反复外部目标测试宣称盲测提升。

方法工作流参考：Scientific Agent Skills（与v1协议同一来源），https://arxiv.org/abs/2609.00065 。
