> 历史记录：此处描述之前的公开 Bench/JEV-27B 验证，不是当前 8B 训练入口。

# TUSZ → CerebraGloss 波形图 → JEV / Qwen 监督训练

本目录复用 **CerebraGloss-Bench 的公开绘图格式**，并补上本地 TUSZ EDF 转换、标签构建、训练输入和 LoRA 训练。它不是 CerebraGloss 完整训练配方的复现。

**实际目标 checkpoint 已确认是本地 AAAI 项目的 Qwen3-VL-8B / 32B EEG RL 权重。** 路径、完整性检查、GRPO 来源和 v3 绘图差异见 [LOCAL_CHECKPOINTS.md](LOCAL_CHECKPOINTS.md)。下面已完成的 smoke run 使用原 JEV-27B，尚未适配这些 RL checkpoint；公开 3B 权重不能代替它们。

## 核对到的上游实现

- 仓库：[iewug/CerebraGloss](https://github.com/iewug/CerebraGloss)，固定 commit `0d47617d5502a5260533d6aceb36eb29acf66377`。
- 绘图：[build_img_benchmark.py](../third_party/CerebraGloss/CerebraGlossBench/scripts/build_img_benchmark.py)。输入为已经处理好的 `19 × 2000` NumPy 数组，采样率 200 Hz；每张图对应 10 秒，展示全部 19 个通道。
- 固定顺序：FP1、FP2、F3、F4、C3、C4、P3、P4、O1、O2、F7、F8、T3、T4、T5、T6、FZ、CZ、PZ；标签后缀 `-AV`。
- 黑色波形、负电位向上、通道间距 200 µV、红色 200 µV 标尺、6 × 7.6 英寸、200 DPI JPG。当前环境得到 **1080 × 1280** 像素图片。
- 我们用同一输入对本地绘图与上游函数做 JPG 字节比较，结果完全相同。
- 上游 README 描述了两阶段训练：先用 caption 训练 projector，再用指令数据训练 projector 和语言 decoder；其数据引擎还生成波形描述、选择题和对话。公开快照没有完整的 SFT trainer / 配置，也没有这段 EDF 预处理代码。
- CerebraGloss 自己已把 TUSZ 用于下游发作检测；这里是将这种图像表达接到本地模型上。TUSZ 标注可以监督发作分类，但不能直接充当逐波形解释或推理过程的真值。

绘图改编遵循上游 MIT 许可，Copyright (c) 2026 Gu Wei，完整许可见 [LICENSE](../third_party/CerebraGloss/LICENSE)。下载文件的哈希见 [upstream_manifest.json](../third_party/CerebraGloss/upstream_manifest.json)。

## 本地转换配方和标签

`render.py` 明确实现以下本地配方，不能把这些补充步骤说成作者未公开的原始预处理：

1. 从 EDF 按电极名称提取 19 个共同参考的头皮通道，不使用 EKG、眼电、A1/A2；T7/T8/P7/P8 可映射到 T3/T4/T5/T6。
2. 读取 EDF 物理单位，MNE 输出伏特。缺失、重复或混合参考通道报错，不补零。
3. 整段记录做 **0.5–70 Hz 零相位 FIR 滤波**，再用 MNE polyphase 方法重采样到 200 Hz（含抗混叠处理），最后对这 19 个通道做共同平均参考并转换成 µV。滤波是本项目在人工检查中发现低频漂移后补充的预处理，**不是从上游公开代码确认的作者参数**。不额外陷波，不按窗口去均值或归一化幅度。
4. 从记录起点按 `[0,10), [10,20), …` 无重叠切窗，不足 10 秒的末尾丢弃，再调用一致的绘图函数。
5. 从 `.csv_bi` 读取全通道 TERM 发作区间，合并重叠区间。任意严格大于零的发作重叠定义为 `seiz`，其余为 `bckg`。这是本项目明确选择的窗口标签规则，不宣称复现论文的未公开边界策略。稀疏标注中未列出的时间按非发作处理；空标注文件不自动当成非发作。
6. 默认从训练候选中排除只覆盖部分发作的边界窗，保留 dev / eval 的边界窗，并记录 `seizure_seconds` 和 `boundary`。`--keep-train-boundaries` 可改变训练规则。

**Non-seizure 只表示该时间段没有标注发作，不表示 EEG 正常，也不表示受试者没有癫痫。** 不生成未经标注支持的诊断解释。

训练问题为“这 10 秒 EEG 是否包含发作”，A=Non-seizure，B=Seizure。图片和问题中不出现受试者、源文件名、原始时间位置、发作区间、标签框或答案。模型只读图片内容与固定问题，标签来自单独的监督字段。

## 挂载数据检查

数据根目录：`/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/TUSZ`，本地 AAREADME 标记 v2.0.6。

| 官方集合 | EDF 总数 | 满足本配方的 EDF | 受试者数 | 完整 10 秒窗 | 其中含发作窗 |
|---|---:|---:|---:|---:|---:|
| train | 5,057 | 4,968 | 579 | 314,712 | 13,399 |
| dev | 2,203 | 1,666 | 53 | 88,148 | 2,760 |
| eval | 880 | 880 | 43 | 46,091 | 2,727 |

三个集合受试者交集均为空。625 个文件缺少 FZ/PZ（train 88、dev 537），另有 1 个 train 文件含 `0–0 秒` 的无效发作区间。本版本排除整份记录，逐条原因保存在 inventory 中。排除边界窗后，train 有 10,070 个全发作窗与 301,313 个非发作窗。

这会改变 train/dev 的覆盖范围，不能把这个子集的结果直接说成完整 TUSZ 的结果。eval 的信号没有用于本次训练或调参；本次只读取其标注和通道表用于独立的索引审计。

官方集合保持分离。完整实验应从 train 按受试者再划出内部验证集来调参，dev 用于测试，eval 保留为最终盲测。均衡小样本 pilot 的类别比例与真实数据不同，不用于报告标准基准成绩。

## 文件与命令

在工作区根目录执行，使用已有环境 `laya/.venv-eeg/bin/python`。不需要下载模型或安装新依赖。

```bash
# 全量元数据索引与通道检查；--out 必须是新的空目录。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 laya/.venv-eeg/bin/python -m tusz_sft.prepare \
  --data-root /inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/TUSZ \
  --out artifacts/tusz_cerebragloss/new_index \
  --inventory-only --skip-invalid-recordings

# 生成均衡的工程验证子集：train 128 张，dev 32 张。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 laya/.venv-eeg/bin/python -m tusz_sft.prepare \
  --data-root /inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/TUSZ \
  --index artifacts/tusz_cerebragloss/index_final \
  --out artifacts/tusz_cerebragloss/new_pilot \
  --train-per-class 64 --heldout-per-class 16 --workers 2
```

输出包括 `images/{split}/`、`{split}.manifest.jsonl`（来源、标签、重叠时长、图像哈希）、`{split}.qwen.jsonl`（多模态对话）和 `{split}.jev.jsonl`（决策输入）。完整候选窗索引为 `{split}.windows.jsonl`；索引存在并不代表已经渲染全部图片。

`--full --splits train` 可渲染全部合格训练窗。全量数据有明显类别不均衡；正式训练前应制定并记录采样/类别权重策略，不能把 pilot 的类别比例或两步 loss 当成泛化结果。CLI 的 `--train-per-class` 可以生成更大的均衡训练子集。

```bash
# JEV：继续现有语言部分和输出层 LoRA，视觉塔冻结。
# 这是 2 次优化器更新的链路验证，不是完整训练。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 laya/.venv-eeg/bin/python -m tusz_sft.train_legacy \
  --backend jev --model laya/JEV-27B-VL \
  --train-data artifacts/tusz_cerebragloss/examples_filtered/train.jev.jsonl \
  --out artifacts/tusz_cerebragloss/new_smoke \
  --steps 2 --grad-accum 2

# 无需加载模型权重即可检查 Qwen 图像 token 和 assistant 标签掩码。
laya/.venv-eeg/bin/python -m tusz_sft.train_legacy \
  --backend qwen \
  --model /inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/cere/CerebraGloss-3B \
  --train-data artifacts/tusz_cerebragloss/pilot_filtered/train.qwen.jsonl \
  --out artifacts/tusz_cerebragloss/new_qwen_input_check --check-input-only
```

本地 `CerebraGloss-3B` 是已经微调的模型，**不能当成原生 Qwen2.5-VL 基线**。上面的第二条命令只使用其 processor；如做原生 Qwen SFT，应传入相应的原生模型目录。`--backend qwen` 的梯度更新路径在本次交付中未跑实际训练，只验证了输入构造。

`train.py` 的训练边界：

- JEV 使用现有 A/B verbalizer 对应分数，加原决策 bias 后计算二类交叉熵；不采用原通用任务 temperature 来假定 TUSZ 已校准。这是决策监督微调。
- Qwen 使用语言 decoder 的新 LoRA；只监督 assistant 的答案及结束标记，图片与问题 token 不进入 loss。这是生成式 SFT。
- 微批量为 1，可用梯度累积；BF16 主干、梯度检查点、AdamW、梯度裁剪。默认保留接近原图的 1,500,000 像素预算。
- 训练只接受 `split=train` 的行；新 adapter、processor、输入审计、loss/梯度/显存日志写入独立目录，不改源模型。这个简洁入口按固定步数训练，不自动调参或选择最优 checkpoint，也不保存优化器以供精确断点续训。

## 样例与验证

当前应使用 `artifacts/tusz_cerebragloss/pilot_filtered/` 和 `examples_filtered/`（配方 v2）。后者包含同一个训练 EDF 的 10–20、20–30、40–50、50–60 秒四个片段。其发作标注为 36.8868–237.2101 秒，因此前两张为非发作，后两张为发作。样例图片本身不带这些文字。固定 200 µV 刻度下，大幅度活动仍可能跨越相邻通道；不为使图片更整齐而改变单个窗口的尺度。

- [非发作样例，10–20 秒](../artifacts/tusz_cerebragloss/examples_filtered/images/train/24e5bb3c736f3a6bdac22500.jpg)
- [发作样例，50–60 秒](../artifacts/tusz_cerebragloss/examples_filtered/images/train/38098b35f7ab1c0e0c3e75c1.jpg)
- [来源和标签](../artifacts/tusz_cerebragloss/examples_filtered/train.manifest.jsonl)
- [全库审计](../artifacts/tusz_cerebragloss/index_final/inventory.json)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 laya/.venv-eeg/bin/python \
  -m unittest discover -s tests -p 'test_tusz_sft.py' -v
```

检查覆盖：发作区间合并与精确窗口边界、稀疏标注、异常区间拒绝、物理单位与平均参考、通道顺序/缺失/重复、与上游 JPG 完全一致。

运行环境：Python 3.10、MNE 1.12.1、NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.9、PyTorch 2.11.0+cu128、Transformers 5.13.1、PEFT 0.19.1。代码使用本地文件和本地 GPU，不调用外部视觉模型服务。

## 本次实际运行结果（2026-10-14）

- 配方 v2 的 `pilot_filtered/` 已生成 128 张 train 图（79 位受试者，64/64 两类）和 32 张 dev 图（19 位受试者，16/16 两类）。两者受试者不重叠；dev 有 4 个边界窗。所有 JPG 哈希、尺寸、文件可读性以及两种格式中的标签映射均验证通过。
- `examples_filtered/` 的 4 张训练图用于 JEV 链路验证：2 次优化器更新，每步梯度累积 2，实际覆盖 4 个样本。两步 loss 分别为 0.99157、0.71689，梯度有限，检查的 LoRA B 参数最大改变量约 `2.00e-5`，峰值已分配显存 57.11 GiB。每步使用不同样本，不能用这两个 loss 推断收敛或性能提升。
- 权重位于 [jev_smoke_filtered/adapter](../artifacts/tusz_cerebragloss/jev_smoke_filtered/adapter)，详细记录见 [training_summary.json](../artifacts/tusz_cerebragloss/jev_smoke_filtered/training_summary.json)。训练没有改写原 JEV。该 adapter 是训练流程验证产物，不应当作正式分类器。
- Qwen processor 已验证其 1,898-token 输入和 `A/B + im_end` 的 assistant 监督目标；未对 Qwen 分支执行梯度更新。
- 尚未渲染全量图像、进行全量 SFT、选择阈值或报告分类准确率 / AUROC / 发作事件指标。
