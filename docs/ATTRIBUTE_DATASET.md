# 当前属性数据：同试次基线与 MI 配对

训练目录 `datasets/eeg_attribute_qa/`，源验证目录 `datasets/eeg_attribute_qa_validation/`，均 Git ignore。每个问题独立 `questions/*.jsonl`；删除某题无需复制或删除共享信号。`question_catalog.json` 定义问题，`statistics.json` 给出选项分布，`examples.json` 提供完整例子，`complete.json` / `verification.json` 提供校验。

## MI（schema_version=2）

目前仅纳入本地有连续原始记录的 OpenBMI / Lee2019_MI。沿用源 manifest 的 44 名训练、5 名验证被试，共 8,800 / 1,000 trials；每人两 session，仅 EEG_MI_train。Cho2017 / Dreyer2023 当前 task-only 缓存不含配对基线，暂不纳入；不把缓存前半段充当基线。目标数据集 2b 不用于构造/选阈值。

从原始 MAT 的连续 x、事件 t、y_dec 逐 trial 提取 C3/Cz/C4；逐条核对已有 PKL 的事件序号、提示时间、被试、session、run 和类别。遵循已有 MOABB t 索引约定。baseline = 提示前 [-2.5,-0.5] 秒的 fixation，task = 提示后 [0.5,2.5] 秒；两段都是 2 秒。fixation 是准备期参考，不声称已证明为生理静息。重采样后事件取最近采样点，最大偏差 2 ms。

连续三个通道先做 1000 Hz 下四阶 Butterworth 1–30 Hz 零相位滤波，再 polyphase 降采样到 200 Hz，最后切片。保持原始参考方式。两段分别逐通道去均值，然后用两段、三个通道共同的单一 RMS 归一化，不裁幅，不分别标准化。两段分别存 `signals/mi_baseline.npy` 与 `signals/mi_task.npy`，各 `[N,3,400]`。它们不是可直接拼成连续时间的 4 秒信号。

| 文件 | 问题 | 选项 |
|---|---|---|
| mi_change_c3_alpha / mi_change_c4_alpha | C3/C4 μ 功率相对自己的基线如何变化 | A 下降超过1dB；B ±1dB内；C 上升超过1dB |
| mi_change_c3_beta / mi_change_c4_beta | C3/C4 β 功率相对自己的基线如何变化 | 同上 |
| mi_lateralization_alpha / mi_lateralization_beta | C3/C4 哪侧功率变化更负 | A C3更负超过1dB；B 差异±1dB内；C C4更负超过1dB |
| mi_class | 想象左手还是右手 | A 左手；B 右手 |

每 trial 七题。功率由 Welch（200 点 Hann、100 点重叠、200 点 FFT）得到，μ=[8,13)，β=[13,30] Hz；使用绝对频带功率，Δ=10log10(Ptask/Pbaseline)。左右比较用 ΔC3−ΔC4；即使两侧都增强也可比较，不能称其必然存在 ERD。±1dB 是固定描述容差，不是统计显著性或神经生理金标准。分类标签直接来自提示标注，不从功率规则猜测。

input 只有分段信号引用、时间关系、采样率、通道、问题和选项。target、evidence、provenance 和 mi_sources 的标签、原始路径均不进入模型。任务类型在问题中明确给出，不是任务类型 zero-shot 识别训练。选项自然分布保留，不为凑平衡修改真实答案；后续训练需要按问题/选项采样，分类按真实标签监督。生理属性是辅助目标，不保证所有 trial 呈现典型对侧 ERD。

## 情绪

六个 emotion_*.jsonl、emotion_sources.jsonl、signals/emotion.npy 与此前逐字节一致。仍使用 30 通道 3 秒窗的相对频带功率比较；其中前后半段只表示窗内变化，不解释为静息与任务差异。

情绪最终分类另存 `questions/emotion_class.jsonl`：A负性/B中性/C正性。训练24477个窗口（SEED4050、FACED20427），验证8350个窗口（SEED1350、FACED7000）。数据原始不均衡，训练按数据集×类别均衡采样，完整验证保留自然分布。

补充入口 `scripts/build_emotion_classification.py`，原始标签核对SEED label.mat与FACED events.tsv；独立 `classification_complete.json`记录hash、题目、类别分布，不改旧complete.json或信号缓存。训练加载器显式include_emotion_class=True才读入该文件。完整训练14题型、232939条问答，验证65450条。属性标签仍来自功率，分类标签来自数据集，二者不能互相替代。

## 重建与验证

`scripts/build_attribute_dataset.py --install` 会先在 artifacts 暂存、校验新数据，之后仅移除旧 MI 问答/MI 信号，替换索引与元数据；不会删除原始数据、检查点、日志或缓存。此次删除得到用户明确授权。安装日志保存在 `artifacts/mi_paired_rebuild/*_installation.json`。情绪 payload 通过 SHA256 确认未改动。所有 MI 答案由独立 FFT 实现重算；源 train/validation 按被试隔离。

旧单窗 REVE token cache 不含基线，禁止用于新 schema。当前已实现分别编码、baseline/task 文本边界及二/三选项训练，详见 PROTOCOL.md。构建时附带的各数据目录 README 是数据构建时的说明快照。
