# LEAF 13 库部分复现协议

_2026-11-24；当前实现见 leaf_repro/，固定官方提交 08ee6577ca4e9517536e73ba3053e676c028a379。_

## 🎯 复现范围

本轮复现原生 LEAF 的多任务指令训练与直接推理，不是 Tower→JEV，也不是从零复现全部无监督预训练或逐数据集监督微调。官方模型/损失来自本地固定版本的 `LEAF.py`、`b_instruct_tuning.py`；只增加本机路径、分片读取、日志、断点与检查，不修改上游仓库。[^native]

初始化为官方 `leaf-v1.0-pretrain.ckpt`，SHA256 `bd5c29ddc047d0765fe72cbd4ee5e95b2bba37dd8632c2bfd31185fc010163ee`。不使用已接受下游标签训练的 instruct checkpoint。MPNet 文本缓存直接复用作者仓库；不在线更新文本编码器。

## ⚙️ 联合训练

将 13 库 train 样本合并后统一 shuffle；一份模型、一套优化器、混合 batch，不是每个库单独训练。按样本数自然采样，不额外均衡数据集。

| 项目 | 设置 |
| --- | --- |
| EEG 网络 | 双分支 Tower＋FiLM＋Q-Former，全量训练实际前向参数 |
| Q-Former | 8 queries、4 layers、256 维；拼接后 Dropout(0.3)＋Linear(2048,768) |
| 损失 | 归一化 EEG 向量与 23 个全局唯一冻结文本原型的 cosine logits，交叉熵，无额外温度 |
| 文本条件 | 10% Default 指令；独立 90% 概率混合所有候选类别文本的归一化均值，不用真实标签构造指令 |
| 输入 | 200 Hz、65 通道；训练随机补零/裁剪至 2000 点，验证右补零/从头裁剪 |
| 优化器 | AdamW，weight decay 0.001 |
| LR | tokenizer 0.001，其余 0.0002；2 epoch warmup，cosine multiplier 1→0.1 |
| 预算 | 100 epochs，seed42，bf16-mixed |
| Batch | 官方脚本的每卡 256，四卡总 1024；没有额外 LR scaling |
| 验证 | 每 10 epoch，完整 valid、全局原型 CE/Acc，额外记录每库 loss/全局 Acc |
| 保存 | 每 epoch 保存可续训状态；每 10 epoch 导出原生 state_dict；保存最低 valid loss 点 |
| 测试 | 训练完成后，固定 final epoch 与 valid-loss 选中点；不据 test 再挑模型 |

公开仓库入口默认四卡、每卡256，当前YAML warmup为2epochs；论文v3表6明确写global batch512、warmup1epoch。本轮按公开代码默认执行global1024/warmup2，二者并不完全一致，属于已记录的部分复现差异，不声称严格重现论文数值。额外最低验证 loss checkpoint 是本地记录选择，final epoch 结果同时报告，避免隐去此差异。

训练和验证使用全局原型分类；原生 direct inference 则限定到当前数据集候选类别。直接推理不补成10秒，按作者脚本使用原生4秒/10秒窗口，分别报告 Default、Task、Task+Target 三种指令层次。完整 test 使用单卡顺序加载，不含 DDP 补齐的重复样本。训练期分布式 valid 沿用官方 DistributedSampler，最多补齐3个样本，日志不是逐库独立测试分数。

## 📊 数据与划分

以下是本轮目标名单；窗口数在全量预处理后由 `datasets/leaf13/manifest.json` 实测。数量不从历史 EEG-FM-Bench 表搬用。被试总数为库内唯一人数；train 与 valid 的被试集合重叠，因此三个被试数不能相加求总数。[^data]

| 数据集 | 本轮处理后窗口：总（训/验/测） | 任务（分类数） | 本轮划分协议 | 被试：唯一总数（训/验/测） |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 处理中，待清单核验 | 运动想象（4） | A01–A07 中每第5样本验证；A08–A09测试 | 9（7/7/2） |
| OpenBMI-MI | 处理中，待清单核验 | 运动想象（2） | 前42人训验池，后12人测试 | 54（42/42/12） |
| SHU-MI | 处理中，待清单核验 | 运动想象（2） | 前20人训验池，后5人测试 | 25（20/20/5） |
| HighGamma | 处理中，待清单核验 | 左/右运动解码（2） | 前10人训验池，后4人测试；去掉feet | 14（10/10/4） |
| Cho2017 | 处理中，待清单核验 | 运动想象（2） | 排除32/46/49，剩余前40/后9人 | 49（40/40/9） |
| Shin2017A | 处理中，待清单核验 | 运动想象（2） | 排除S05，剩余前22/后6人 | 28（22/22/6） |
| PhysioNet | 处理中，待清单核验 | T1/T2运动执行与想象（2） | 前80人训验池，后29人测试；R03–R14 | 109（80/80/29） |
| FACED | 处理中，待清单核验 | 情绪（9） | 000–099训验池，100–122测试 | 123（100/100/23） |
| SEED | 处理中，待清单核验 | 情绪（3） | 每人trial 1–9/10–12/13–15 | 15（15/15/15） |
| SEED-IV | 处理中，待清单核验 | 情绪（4） | 每人每session trial 1–16/17–20/21–24 | 15（15/15/15） |
| SEED-VII | 处理中，待清单核验 | 情绪（7） | 每人每session trial 1–10/11–15/16–20 | 20（20/20/20） |
| OpenBMI-SSVEP | 处理中，待清单核验 | SSVEP（4） | 前42人训验池，后12人测试；仅公开train blocks | 54（42/42/12） |
| Workload | 处理中，待清单核验 | 负荷（2） | 前32人训验池，后4人测试 | 36（32/32/4） |

所有“训验池”都先按官方被试/样本顺序拼接，再取从index0开始每5个样本中的一个为valid，其余train；不是每名被试重新从0计数。

## ⚠️ 已知差异与限制

- 缺少 SEED-V、BCIC-Upperlimb、BCIC-Speech；下载结果不会中途加入已启动的13库训练。
- Cho2017/HighGamma/SSVEP 的 MOABB 缓存已经有200Hz、0.1–75Hz与60Hz notch处理，再执行LEAF的0.3–40Hz/尺度归一化；不能恢复滤掉的信息，不能宣称与作者原NPZ逐样本相同。逐trial源路径、run、标签另存到shard旁的sources.json。
- SSVEP现有两session公开train共10,800条，未包含公开test blocks。与后续个人/跨被试划分的“test”不是同一个概念。
- PhysioNet本地缺少的执行运动runs从PhysioNet公开源补齐，再调用官方R03–R14处理器。官方代码将T1/T2统一记为Left/Right，但其中也包含双拳/双脚及实际运动；不应称为纯左右手想象。
- FACED按代码的100/23被试划分，不按论文表格可能存在的边界笔误。SEED采用当前官方修正后的Negative/Neutral/Positive顺序；SHU严格采用当前tasks.yaml的Right/Left映射，后续对原始标注语义仍应单独核验。
- 复用官方按被试或按被试/划分估计的robust scaling，包含同一评估组内未标注信号统计；不是单trial在线归一化。保留原始short-window滤波与大量插值通道的警告，不擅自改变方法。
- 13库的留出测试仍来自指令训练涉及的同名数据集，不是13个全新数据集的zero-shot；跨数据集zero-shot应额外用未参与指令训练的Dreyer2023、Weibo2014等独立数据评估。[^heldout]
- 官方主结果还包含每库单独的Tower＋MLP监督微调；本轮未运行该阶段，不能把direct-inference结果与其最佳监督微调结果等同。

## 🔧 工程与软件

四卡短测从单被试/库的训练样本进行3步；正式阶段重新加载无监督预训练Tower与随机Q-Former。原作者前向与优化器保留；只将数据存储改为按被试HDF5分片，避免每个rank重复加载整库。过程有源码SHA256、模型SHA256、完整划分indices、有限值检查、梯度审计、每库CSV日志、完整模型/优化器恢复。

Lightning 2.6.4，PyTorch 2.11.0+cu128；依赖原有MNE 1.12.1及h5py 3.16.0。使用训练框架、资源检查、科学模型检索与文档技能辅助容量检查、原始权重核验及协议记录；这些是工程流程，不是模型创新。[^skills]

[^native]: LEAF官方指令训练说明及固定代码。https://leaf-bci.github.io/docs/instruction-tuning.html ; https://github.com/Jiang-Muyun/LEAF/tree/08ee6577ca4e9517536e73ba3053e676c028a379
[^data]: LEAF官方数据处理与划分。https://leaf-bci.github.io/docs/datasets/instruction-tuning-datasets.html
[^heldout]: LEAF官方独立数据集评测。https://leaf-bci.github.io/docs/datasets/held-out.html
[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
