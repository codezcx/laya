# EEG 属性选择题：仅源训练集

本次只构造数据，不启动训练，也不改变现有 JEV 模型和训练入口。候选辅助监督用于检验信号可读性；尚未证明可以改善跨数据集分类。

## 位置与拆分

默认目录 `datasets/eeg_attribute_qa/`，整个目录 Git ignore。生成代码与本说明由 Git 管理。使用当前已校验信号缓存，仅导出 `split=train` 的 57,749 个窗口：MI 33,272，情绪 24,477。验证集、2b、SEED-VII 均不导出，不参与阈值选择。验证 manifest 的被试及文件划分互斥。

```
datasets/eeg_attribute_qa/
  signals/mi.npy                  # float32 [33272,3,600]
  signals/emotion.npy             # float32 [24477,30,600]
  mi_sources.jsonl                # 信号索引、原始记录与逐窗 SHA256
  emotion_sources.jsonl
  questions/<question_type>.jsonl # 每个问题独立文件，共12个
  question_catalog.json           # 每题的通道、频段、时间窗口定义
  examples.json                   # 每种问题的一条真实示例
  statistics.json                 # 每题、每数据集的答案分布
  complete.json                   # 来源、代码、产物校验和；完成标记
  README.md
```

所有 `signal_file` 相对数据集根目录解析。信号直接复制自当前缓存，没有重新归一化。每条信号只存一次；每个问题文件引用共享信号。选择某些问题训练时只加载相应 JSONL；移除问题不影响其他问题，但不要删除仍被引用的 signals 文件。可完整移动此目录，不依赖原始缓存读取；sources 中的历史绝对路径仅供溯源。生成器拒绝覆盖已有目录，失败产物保留 `BUILDING.json`，不得训练使用。

## 十二种问题

| 任务 | 类型 | 文件名（questions/ 下） | 比较内容 |
|---|---|---|---|
| MI | 频谱 | mi_spectral_c3.jsonl / mi_spectral_c4.jsonl | C3 或 C4 的 μ 与 β 相对功率 |
| MI | 空间 | mi_spatial_alpha.jsonl / mi_spatial_beta.jsonl | C3 与 C4 的 μ 或 β 相对功率 |
| MI | 时间 | mi_temporal_c3_alpha.jsonl / mi_temporal_c4_alpha.jsonl | C3 或 C4 后半窗与前半窗 μ 相对功率 |
| 情绪 | 频谱 | emotion_spectral_frontal.jsonl / emotion_spectral_posterior.jsonl | 额区或后部区域的 α 与 β 相对功率 |
| 情绪 | 空间 | emotion_spatial_alpha.jsonl / emotion_spatial_beta.jsonl | 额区与后部区域的 α 或 β 相对功率 |
| 情绪 | 时间 | emotion_temporal_frontal_alpha.jsonl / emotion_temporal_frontal_beta.jsonl | 额区后半窗与前半窗 α 或 β 相对功率 |

MI 每题33,272条，情绪每题24,477条，共346,494条问答。这是57,749个窗口的多种监督，不是346,494个独立试次。英文问题保持现有模型接口语言；不输入数据集名、情绪/运动类别或计算所得数值。每题只需三选一，未来接入 JEV 时应限制到 A/B/C，有效 logits 归一化也仅在这三项上进行。

## 可复核的答案

- 200 Hz、原有1–30 Hz滤波、3秒窗口；时间比较使用0–1.5秒和1.5–3秒，均相对于窗口起点，不声称提示起点。
- Welch：Hann窗、200点、100点重叠、FFT200、常数去均值，频率分辨率1 Hz。
- 相对功率为指定频率 bins 的PSD和除以1–30 Hz bins总和。α/μ采用 `[8,13)`，β采用 `[13,30]`；频段比较是积分功率比较，带宽不同，不等价于比较最高谱峰。区域值先逐通道求相对功率，再取通道均值。
- 额区：Fz/F3/F4/F7/F8；后部：Pz/P3/P4/P7/P8/PO3/PO4/Oz/O1/O2。后部是明确的电极集合，不是源定位脑区。
- 差值 `10*log10(first/second)`，除法前将各值下限设为1e-12。
- A：第一项高出超过1 dB；B：差值绝对值不超过1 dB；C：第二项高出超过1 dB。固定1 dB是描述性容差，不是显著性检验、估计置信区间或经过验证的临床阈值。没有用标签或验证集拟合阈值。
- 不声称选项平衡；真实分布写在statistics.json。以后按问题比较多数类基线和平衡准确率，不仅报告总体正确率。

没有使用“左手一定C4抑制”“某频带高一定正性情绪”等标签规则。没有静息基线，因此不生成ERD/ERS；没有30 Hz以上信号，因此不生成γ属性；情绪逐通道标准化丢失的原始绝对功率/DE不作为监督。

## 输入和审计信息必须分离

每条JSONL含 `input`、`target`、`evidence`、`provenance`。模型只接收input中的信号内容、通道/采样元信息、问题与选项；文件路径和索引用于加载，不渲染到prompt。target作为训练标签，evidence是计算依据，provenance用于审计。sources包含原始类别等历史元信息，也不得进入prompt。

构造时未shuffle，便于逐窗核对；未来训练应按独立窗口/被试划分后shuffle，不能将同一窗口的不同问题随机分配到训练与验证。现有源验证集保持未生成状态。通道覆盖可能透露任务身份，本数据不能用来证明自动任务识别能力。

## 重建和检查

在仓库根目录执行：

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python -m unittest discover -s tests -p test_attribute_questions.py -v
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/build_attribute_dataset.py
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/verify_attribute_dataset.py
```

构造前核对原始缓存完整SHA256和manifest；复制信号的逐窗SHA256保存在sources，产物完整SHA256保存在complete.json。没有下载、GPU模型调用、伪造CoT或自动训练。

验证器逐条检查训练覆盖、信号拷贝和答案计算，另对每题24条使用独立FFT实现重算，结果写入 `verification.json`。
