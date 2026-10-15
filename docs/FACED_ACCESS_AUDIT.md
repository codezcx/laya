# FACED第二情绪源：初步访问核查

尚未下载EEG波形，尚未纳入训练。当前已获取公开元数据和sub-000事件表，保存在`artifacts/faced_access_audit`。

已核查来源：

- [NEMAR数据仓库](https://github.com/nemarDatasets/nm000112)：FACED，123参与者，28视频，九种情绪。
- [dataset_description.json](https://github.com/nemarDatasets/nm000112/blob/main/dataset_description.json)：声明CC-BY-4.0、BIDS衍生数据、版本1.1.3，引用原数据论文DOI 10.1038/s41597-023-02650-w。
- [EEGDash索引](https://huggingface.co/datasets/EEGDash/nm000112)：这是元数据指针，不包含原始EEG；应通过NEMAR/EEGDash实际数据接口获取。
- [sub-000事件表](https://github.com/nemarDatasets/nm000112/blob/main/sub-000/eeg/sub-000_task-watchingVideoClips_events.tsv)：已实际解析28个带情绪注释的事件，binary_label为negative12、positive12、neutral4；含onset/duration/video_index及自评维度。

重要实际发现：sub-000的neutral事件在emotion_label列使用反斜杠占位，不能仅凭该列枚举排除。应以经核验的binary_label映射negative/neutral/positive，并逐被试验证合法值、视频数、时间边界。需区分刺激目标标签和自评值；本候选计划使用前者，以匹配SEED当前监督含义，不把它解释成真实心理状态的完美测量。

GitHub中的`.bdf`内容只是annex指针，指向468,584,448字节对象，并非下载成功的波形。已重命名为`.annex-pointer.txt`避免误读。下一步应通过公开数据服务下载一个实际记录，校验大小/哈希、MNE读取、通道、单位与采样率，再决定全量源数据处理。

README和EEGDash汇总对不同采样率人数的统计不一致；不得硬编码“所有记录250Hz或1000Hz”。须以每条真实BDF头信息及sidecar为准。官方文档也提示不同记录单位可能不同，需逐文件读取校验。

若成功接入，将明确作为新源开发数据，按被试固定train/validation；SEED-VII仍为已暴露的外部评估，不能称新盲测。REVE是否预训练见过FACED需另行审计，不因新增本轮源数据就声称foundation-level未见。
