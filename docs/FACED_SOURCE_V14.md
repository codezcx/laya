# v14：第二情绪源FACED接入

目的：此前情绪训练只有SEED，多个读出/encoder适配均未建立稳健跨数据集效果。本轮先验证新源接口，再做SEED+FACED多源训练，不新增压力/心算任务。

## 来源与已完成核查

公开来源：[NEMAR FACED](https://github.com/nemarDatasets/nm000112)，固定revision `4c37c73ece79e68702de2d8afa6b9274811a7cc8`，元数据声明CC-BY-4.0。原论文DOI：[10.1038/s41597-023-02650-w](https://doi.org/10.1038/s41597-023-02650-w)。数据归属、原论文和许可文件随raw缓存保留。下载解析参照官方[EEGDash NEMAR resolver](https://github.com/eegdash/EEGDash/blob/a4dcf9c6b1a82ac4609db56593f360052dfd985d/eegdash/dataset/base.py)，使用匿名公开SHA对象路径，不涉及账号或私有接口。

sub-000真实BDF已下载，468,584,448字节，SHA256与annex指针完全一致。MNE读出32通道、1000Hz、4,881,000采样点；原单位µV，MNE输出V，trial通道std中位数约3.90e-5V。28个合法视频事件，negative12/neutral4/positive12。neutral的emotion_label占位符不当作缺失标签，使用明确的binary_label。

每trial按实际onset/duration读取，重采样200Hz、1–30Hz带通、均匀取10个不重叠3秒窗口，逐通道zscore/clip15，单被试280窗口。每条记录逐一校验采样率、通道、单位、时间边界及数据完整性。初次audit JSON序列化遇到numpy int64，已修复；不完整产物保留在`.incomplete-json-audit`，不参与训练。

## 预定试验集

全123被试按seed1401固定73train/25validation/25test；试验集只从前两者取16train+8validation，cohort_seed1402。保留已做格式QC的sub-000，其余随机选择，不依据模型表现。完整名单记录在`artifacts/faced_source_v14/cohort.json`，创建后不改写。保留test被试不下载、不训练、不选模。

四个并行I/O工作线程下载及预处理，GPU提取另行进行。该24人试验集用于验证多源流程，后续是否扩展完整源池需基于源开发证据；不将小规模试验集重新定义成最终目标。

缓存在`artifacts/faced_source_v14`，不会进Git。下载校验大小与SHA256，保留已完成产物，失败不自动覆盖或无限重试。尚未训练FACED模型，也未新增目标评估。

## 接入接口与预定对照

FACED按自身实际通道名（首个记录含T3/T4/T5/T6和A1/A2）查REVE位置库；已检查首记录所有32名称都有坐标，不虚构位置、不静默丢弃通道。编码后时间池化，再按现有八个空间锚点汇聚为8×512。每分片保留真实通道名、cohort哈希及锚点/sigma。跨分片检查24被试/6720窗口、train/validation无交集、标签合法、标准化信号无重复，未通过不得训练。

四组均用相同SEED+FACED验证被试和完整三个MI源的验证被试：SEED-only、SEED+FACED、FACED-only、SEED+FACED文本sigmoid。只有情绪训练源变化；验证集合不变。相同3000step、batch96、lr1e-4、seed42，冻结REVE，共享读出结构。原SEED-only旧结果的验证集合不同，不能直接当本轮对照。

任务→数据集→类别均衡采样，FACED中性数量少不会直接降低其训练类别权重。主选模仍为两任务macro，任务内先平均数据集/被试。此轮先做源开发验证，不自动运行SEED-VII或2b评估。

控制器依赖完整cohort下载/预处理状态，接着四卡提取空间特征，最后四卡训练。任何阶段失败保留日志并停止后续阶段，不因为观测超时重启正在运行的任务。

## 已验证的传输故障与修复

首个sub-000整文件成功后，首批四个并发记录出现提前EOF：实际字节数未达到annex声明，完整性断言阻止它们进入预处理/训练。旧下载进程及等待控制器在确认失败后受控终止；部分文件、固定cohort、失败记录全部保留，旧控制器目录改为`multi_emotion_v14_controller_attempt1`。

改为4MiB HTTP Range分段续传，每段必须返回206及精确Content-Range，长度正确后才追加；单段最多3次重试。完成后仍验证整文件字节数和SHA256。实测sub-004的0–255字节与已有部分文件完全一致。三个传输测试覆盖正确范围、截断响应、忽略Range/返回错误范围。`--resume`复用既定cohort和revision，不重新抽样被试。

这次修复由已验证的传输失败触发，不是因观察超时重启正常任务。尚无FACED训练结果；控制器依旧要求全部24人完成后才能提取和训练。
