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

## 训练前的选模指标修订

发现FACED每被试类别trial数12/4/12，故在本轮任何训练开始前，四组统一配置`selection_metric=macro_task_balanced_accuracy`：每被试按已标注类别平均召回率→数据集平均→同任务数据集平均→两任务平均。普通accuracy及原macro_task_accuracy仍完整记录。此前实验默认选择指标保持不变。

这项修订避免稀少中性类别在选模时权重偏低；与旧实验accuracy选模口径不同，不将新旧源验证数字直接排名。单测用75%多数类准确率但少数类完全失败的例子，确认balanced指标为50%。任务→数据集→类别训练采样本来已均衡，无需更改其权重。

## 真实输入接口核验

`verify_faced_encoding.py`已使用sub-000真实预处理波形运行REVE：输入2×32×600，encoder输出2×32×4×512，空间汇聚2×8×512；所有通道坐标存在、加载无缺失/额外权重、输出有限。未计算分类预测或使用标签拟合。

`audit_faced_prepared.py`对当时已完成的11人/3080窗口逐条复算标准化信号SHA256，确认无重复、每人120负性/40中性/120正性窗口、28trial×10窗口、与既定被试split一致。其状态明确是partial，不能作为完整cohort通过的证明。全部下载完成后仍需完整检查。

训练程序新增运行时采样质量核验，记录每任务及每dataset/class的采样概率质量。两任务各1/2，任务内数据集均分、数据集内类别均分；任何缺失类别导致质量不符会中止。只记录和核验既有采样规则，没有改变权重。
