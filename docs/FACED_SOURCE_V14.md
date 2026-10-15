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
