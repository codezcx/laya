# REVE-only：linear probe 与全量监督微调

本实验响应“在 Cho2017、Lee2019_MI、Dreyer2023 联合训练，并复用上一轮划分，在内部留出被试和 2a/2b 上测试”的请求。
不使用 Qwen、视觉编码器、语言提示、连接层或 LoRA。这里 SFT 指 EEG 标签监督的全量分类微调，不是生成文本的指令微调。

当前状态（2026-10-15）：用户已授权执行现有本地模型代码；12 项 CPU 回归/结果审计测试及两组真实 REVE GPU smoke 均通过。
全部 manifest SHA256 与上一轮一致，路径/信号/被试跨 split 检查通过。两组训练和全部测试均已完成。
LP 训练 26 轮、选择第 11 轮；FT 训练 7 轮、选择第 2 轮；两组都按预设 patience 早停。
结果：[完整对比与限制](../artifacts/reve_only_multi/train_20261015_1404/RESULTS.md)。
运行目录：`artifacts/reve_only_multi/train_20261015_1404`。
Smoke 记录：`artifacts/reve_only_multi/smoke_{linear_probe,full_sft}_20261015_1406`（目录后缀仅为标识，实际运行在 14:04 UTC 前完成）。
首次 smoke 仅因 transformers 加载审计中的 set 无法 JSON 序列化而失败，修复日志序列化并新增测试后通过；没有改模型权重或数据划分。

## 固定数据与比较口径

直接复用 `artifacts/reve_mi_multi/manifests_v2`，启动前/完成后验证与上一轮保存的 SHA256 完全一致；不重新划分。

| 数据集 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| Cho2017 | 8,480 | 1,000 | 1,040 |
| Lee2019_MI | 8,800 | 1,000 | 1,000 |
| Dreyer2023 | 15,992 | 2,160 | 2,160 |
| 合计 | 33,272 | 4,160 | 4,200 |

各训练来源内按被试隔离。2a 仅左右手 E 会话 1,296 条，2b 仅 E 会话 2,840 条；两者不参与优化、早停或模型选择。
所有评估都是强制左右手二分类、覆盖率 100%，报告普通/平衡准确率、各数据集指标、混淆矩阵和逐样本概率。

继续使用 C3/Cz/C4、200 Hz、逐试次逐通道 z-score/clip15，以及缓存中原生 3/4/5 秒窗口（2b 为 4.5 秒）。
同一批中不同真实长度分别进入 REVE；只在试次末尾复制最后一个值补到完整 patch，不把批量补零送入 encoder。

## 结构及两组训练

`EEG → REVE-base → B×3×patches×512 → 每通道时间平均 → 1536 → Linear(1536,2)`。
保留三个通道的身份，不把左右通道一起平均。两组池化/头结构完全相同；分类头没有额外 MLP 或可训练归一化。

- **Linear probe**：原预训练 REVE 冻结并始终 eval，预计算 train/validation 特征，仅训练线性头。
- **Full SFT**：独立从同一原预训练 REVE 开始，线性头同种子初始化，解冻所有参与 encoder forward 的参数；不从 linear probe 最佳 checkpoint 热启动。
- 导出 REVE 的 `cls_query_token` 仅供可选 attention pooling 使用，本实验不调用该路径，所以该 512 参数保持冻结。除此之外，全量微调不冻结任何有效 encoder 参数。
- 两组均 FP32，AdamW，batch 64，weight decay 0.01，gradient clip 1；采用与上一轮相同的 dataset×class 均衡有放回采样。
- LP：头学习率 1e-3，最多 100 epochs，验证宏平均准确率连续 15 epochs 不改善则早停。
- FT：encoder 学习率 1e-5、头 1e-3，最多 10 epochs，patience 5。
- 每轮只按三个验证集准确率的算术平均选择 best；结束后从磁盘重新加载 best，再评估 test、2a、2b。测试数据不用于学习率搜索。

这是预先固定的首轮超参数，不是最优参数声明；LP/FT 训练预算与上一轮 REVE→Qwen 不完全一致，不应归因于单一结构因素。
均值池化适配不同持续时间，但会损失时间顺序，不能据此认定 REVE 的能力上限；也不是官方完整实验配方的逐项复现。

## 运行与产物

先执行不加载模型的检查：

```sh
laya/.venv-eeg/bin/python -m unittest discover -s tests -p 'test_reve_only.py'
laya/.venv-eeg/bin/python -m eeg_align.train_reve_only --mode linear_probe --output unused --audit-only
```

本轮已获用户授权执行现有本地 `brain-bzh/reve-base` / `brain-bzh/reve-positions` 自定义代码，两组 synthetic smoke 已通过。重跑命令：

```sh
# 仅在用户确认执行本地自定义模型代码后运行。
bash scripts/run_reve_only_multi.sh artifacts/reve_only_multi/UNIQUE_RUN --allow-local-reve-code
```

默认两个实验分别使用可见 GPU 0、1，可通过 REVE_PROBE_GPU / REVE_FULL_GPU 改变。没有多卡通信、没有上传/下载权重。
已有运行目录拒绝覆盖。每组包含 config、source/checkpoint/manifest hashes、loading audit、gradient check、逐轮验证日志、
best/last safetensors、selected validation、三个测试的 metrics/predictions 和 status。LP 还保存训练/验证特征及元数据。
权重仅留本地，不提交或推送。

## 已继承限制

- 2b 是双极导联，使用 C3/Cz/C4 的标称坐标不代表与其他数据集的参考方案完全一致。
- 部分上游 pickle 来源字段是推断/目录回退，尚未从原始记录完整重建；当前为探索性实验。
- 未完成 REVE 预训练样本与这些数据集的重叠核验，“留出被试/外部”只保证本轮监督微调的隔离。
- 保留原先 Dreyer 被试 13/14 隔离，不修改原始数据。单个种子结果不代表统计显著性。

## 方法及工具参考

- REVE 作者代码及模型：https://github.com/elouayas/reve_eeg ，https://huggingface.co/brain-bzh/reve-base 。
- 本轮使用 get-available-resources 规划资源，使用 hugging-science / transformers 核对本地加载、冻结和 checkpoint 行为。
  技能目录中未找到 EEG 专项条目，模型依据转向 REVE 官方资源。
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
  https://doi.org/10.48550/arXiv.2609.00065 （2026-10-15 核验，最新记录 v2；工具流程引用，不是实验有效性的证据）。
