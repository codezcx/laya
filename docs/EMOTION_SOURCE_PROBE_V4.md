# v4：情绪源域输入消融

在v3结果已知后预先固定四个对照，使用与v1完全相同的SEED训练/验证被试、trial及窗口起点，不读取源测试或SEED-VII。由原始预处理波形提取3秒窗口Welch log band power；1–30Hz或1–45Hz四阶零相位滤波在整个trial上完成。

|组别|通道|窗口方差归一化|频带|
|---|---|---|---|
|3ch_relative_30|C3/Cz/C4|是|1–4/4–8/8–13/13–20/20–30Hz|
|62ch_relative_30|全62通道|是|同上|
|62ch_absolute_30|全62通道|否|同上|
|62ch_absolute_45|全62通道|否|增加30–45Hz|

relative计算log(power)-log(window variance)，对应窗口标准化后的功率，未进行clip15；absolute只表示未逐窗口除以方差，不代表已核验物理单位，也不是原始采集电压。SEED文件已有官方预处理。

固定逻辑回归，StandardScaler仅拟合训练被试。C=0.001/0.01/0.1/1/10，按源验证被试macro accuracy选择。保存完整sweep、收敛警告、特征、行清单和选中pipeline。组间比较主要判断是否需要修订encoder输入，而非证明语言对齐收益。

命令：`laya/.venv-eeg/bin/python -m eeg_tasks.emotion_source_probe`。
输出：`artifacts/emotion_source_probe_v4`；拒绝覆盖。日志：`artifacts/emotion_source_probe_v4.log`；PID另存同名前缀`.pid`。CPU运行无需GPU。

限制：只有3个源验证被试，多个超参数/特征对照会增加源验证过拟合风险；得到改善后需要独立被试或新的源域划分复验，不能立即声称跨数据集有效。空间/幅度特征也可能编码记录条件，而不是情绪生理机制。
