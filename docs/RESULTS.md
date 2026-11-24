# LEAF Tower→JEV 当前实验与历史对照

归档前完成更新（2026-11-24 15:29:46 UTC）：原生8B完成2040步，选中step2040。完整原test开发集61,375个窗口，九库等权BAcc 43.6316%、Acc 44.1258%；未执行原valid。原test参与选点，结果不是独立测试。产物保留在`artifacts/benchmark/leaf_tower_qwen8b_20261124/`。随后按用户要求切换至13库原生LEAF部分复现，旧实现通过Git归档。

_更新：2026-11-24。当前换为原生8B共享SFT；4B与个人联合适配已完成，见[协议](PROTOCOL.md)。_

本轮用途约定已落实到共享决策训练：原test进行开发评估和checkpoint选择，原valid待方案确定后再评估，见[协议](PROTOCOL.md)。以下4B等历史结果仍按各自原始协议解读，不追溯改名。原test选点成绩标为开发结果；原valid已有被观察和用于选点的历史，不将其冒充全新盲测。

## 📋 原生 Qwen3-VL-8B 指令微调

换用官方8B语言权重，四个safetensors分片及配置/tokenizer/index均通过官方哈希校验。原InfoNCE step1700 Tower、17库训练、2040步、340步connector-only、global batch16和全部LR保持不变；新connector为256→4096→4096，语言LoRA-r32，choice交叉熵，不增加蒸馏或第二轮对比学习。

输出`artifacts/benchmark/leaf_tower_qwen8b_20261124/`。124项自动测试通过，含开发用途门禁、抽样ID稳定性、计数恢复和实际曲线导出；2040个计划batch输入预检通过（98.5秒）。四卡17步短测完成，梯度、rank权重一致、输入分片、encoder更新、checkpoint精确重载全部通过，短测只评估train。2026-11-24 14:57:19 UTC控制器292730已进入正式阶段，训练进程429384，运行源码`896056a`。[执行阶段](../artifacts/benchmark/leaf_tower_qwen8b_20261124/pipeline.json)、[短测审计](../artifacts/benchmark/leaf_tower_qwen8b_20261124/smoke/leaf_infonce_mlp/complete.json)。

实际语言基础参数7,568,405,504，总可训练124,751,632；原生A–P行来自`lm_head.weight`，不是4B的tied embedding。训练中每340步在九库固定原test子集（每库最多512）选点，最后评估完整原test；本轮不运行原valid。尚无最终准确率；结果属于开发/探索性结果，与下方4B原valid成绩不能直接比较。4B实现已归档`ad4c054`；没有新建独立版本训练器。

启动核查时已完成正式13/2040步，四rank初始权重与短测的初始状态完全匹配。此时仅connector有梯度，语言LoRA/head/Tower的LR和梯度为0，符合前340步预热；不是仅创建队列。step0原test开发子集九库BAcc为29.50%，这是未训练连接器的起点，不是最终结果；原valid模型评估次数为0。固定ID和用途见本轮`test_selection_ids.json`及`test_step000000_subset.json`。模型/资源/绘图技能用于官方权重核验、四卡检查与开发标签审计，过程引用沿用[协议](PROTOCOL.md)。

## 📋 原生 Qwen3-VL-4B 指令微调控制

原生4B从本地复制并核对官方权重SHA256；不是此前测速的ECG-R1-8B派生模型。复用InfoNCE step1700 Tower，使用相同17库、2040步、global batch16及原LR；新连接器和语言LoRA/决策读出，单次forward交叉熵训练。没有27B教师蒸馏，也没有用test选点。[配置](../configs/language_backbones.json)、[完整口径](PROTOCOL.md)。

输出根目录`artifacts/benchmark/leaf_tower_qwen4b_20261124/`。119项测试全部通过；真实GPU前向/反向检查确认输入、connector和LoRA具有有限非零梯度，语言基础权重没有梯度。实际语言主干4,022,468,096参数，联合阶段总可训练92,873,488参数。

4B已全部完成2040步，原valid固定子集选中step2040，子集九库平均BAcc为43.26%；完整原valid九库61,327窗口平均BAcc为42.68%，平均Acc为44.28%。27B历史同一完整valid IDs的BAcc为47.83%，4B低5.15个百分点；选中步数、架构、初始化和接口容量不同，不能唯一归因于参数量。[完成审计](../artifacts/benchmark/leaf_tower_qwen4b_20261124/train/leaf_infonce_mlp/complete.json)、[完整valid指标](../artifacts/benchmark/leaf_tower_qwen4b_20261124/train/leaf_infonce_mlp/valid_step002040_full.json)。

全部2040个计划batch预检通过（131.8秒），四卡17库短测、正式rank权重一致性、encoder更新、checkpoint精确重载及初始哈希匹配均通过。正式训练加最终完整valid记录耗时1797.3秒，未进行原test前向。[完成阶段](../artifacts/benchmark/leaf_tower_qwen4b_20261124/pipeline.json)。

## 📋 已完成：联合更新 adapter 与 Tower

两组各23人，GPU 0/1为联合LoRA-r8，GPU 2/3为联合全参。每人20条support、seed42、100步；Tower和adapter同为LR1e-4，原AdamW/wd0.01/global clip1/batch16全部不变。共享父模型仍为step1360，JEV和原connector全冻结，BN统计量不更新。每位用户重新初始化，不从上一轮个人adapter再叠加训练。

2026-11-24 11:39:00 UTC，46次正式个人适配全部完成并通过审计。两库等权BAcc：残差-only 50.18%、联合LoRA 50.44%（+0.26个百分点）、联合全参49.62%（−0.56个百分点）；单seed探索性结果，不作显著提升结论。[完整汇总](../artifacts/benchmark/leaf_personal_joint_20261124/summary.md)、[完成审计](../artifacts/benchmark/leaf_personal_joint_20261124/complete.json)。

训练启动前109项自动测试通过，真实Tower参数核对：LoRA联合65,856参数，全参联合19,590,720参数。四卡各做原验证用户3步短测，全部通过两部分非零梯度/真实更新、冻结参数/buffers、共同基线/批次及保存重载审计；两个配对用户的初始adapter梯度范数也一致。[短测审计](../artifacts/benchmark/leaf_personal_joint_20261124/smoke_verified.json)。旧C分支的单独LoRA结果不是本轮联合训练结果。

2026-11-24 08:54:08 UTC，控制器84713自动进入正式阶段，GPU 0/1/2/3 worker PID为151498/151503/151508/151511，承担12/11/12/11人。运行源码为`98d257d`。短测每组三步约28秒，GPU峰值约49.5–49.8GiB；据短测及历史基线估计整轮约3小时，实际以正式训练速度为准。

08:55:35–36 UTC四个正式worker均完成第1步更新：被试8两组初始CE均1.11288208，被试9两组均0.99153653；不是仅创建队列。以上为启动记录，当前均已完成。

首次08:41启动在权重加载期间进程退出，无完成短测、无正式更新、无Python报错；系统OOM计数为0，退出原因未确定。08:51以相同源码和配置恢复，旧日志保留；没有修改LR或训练预算。

根目录`artifacts/benchmark/leaf_personal_joint_20261124/`。实际状态见[pipeline.json](../artifacts/benchmark/leaf_personal_joint_20261124/pipeline.json)，每10步loss和信号侧checkpoint均保留；完整汇总来自全部46次通过审计的适配。

## ⚡ 已完成：不同 EEG 时长的推理基线

2026-11-24，单张H100 80GB，batch=1；共享InfoNCE→JEV step1360，加2a被试8的joint_lora个人权重。原精度、LoRA和分块策略不变；eval＋inference_mode，无量化、合并LoRA或编译。固定76个文本tokens，每0.5秒EEG产生2个tokens。每个长度/阶段预热10次，再测50次；表内为CUDA同步后的墙钟中位数，P95为完整路径第95百分位。

| EEG 时长（秒） | Tower（ms） | JEV语言主干（ms） | 完整计算路径（ms） | 完整路径P95（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 8.17 | 259.03 | 269.05 | 273.36 |
| 1 | 8.96 | 259.76 | 268.77 | 274.65 |
| 2 | 8.64 | 256.55 | 268.13 | 272.00 |
| 4 | 8.69 | 255.12 | 269.49 | 274.32 |
| 8 | 8.46 | 261.33 | 274.27 | 279.53 |
| 16 | 17.54 | 266.21 | 291.31 | 297.01 |
| 30 | 25.83 | 277.16 | 314.61 | 320.33 |

完整路径从已滤波/重采样到200Hz、驻留CPU内存的22通道EEG开始，包含归一化、映射65通道、分块、H2D、Tower、个人adapter、connector、JEV和概率回CPU；不含采集等待、EDF读取、上游滤波/重采样、模型加载、网络与排队。各模块独立计时，不能用中位数简单相加替代完整路径。峰值PyTorch allocated显存48.81–48.85GiB。

0.5–4秒使用同一4秒trial前缀，8–30秒通过重复该trial构造形状测试，不是真实连续MI记录；只报告速度，不报告这些长度的准确率。16秒按8＋8秒、30秒按10＋10＋10秒编码后拼接。短时长之间约1–6ms差异不能解释为稳定的算法优劣。

观察：4秒Tower仅约8.7ms，JEV语言主干约255ms；30秒总计算延迟比4秒约增17%，主瓶颈在JEV。环境核对发现Qwen的`causal_conv1d`和FLA gated-delta快速算子不可用，`is_fast_path_available=False`，回退普通PyTorch实现。因此这是当前实现基线，不是H100最佳部署速度；本次未安装依赖或改变模型，尚不能声称补齐kernel的实际收益。

[完整分解与口径](../artifacts/benchmark/inference_latency_20261124/summary.md)、[50次原始计时](../artifacts/benchmark/inference_latency_20261124/results.json)、[环境核对](../artifacts/benchmark/inference_latency_20261124/environment_audit.json)、[测速脚本](../scripts/benchmark_inference.py)。资源/模型/文档技能分别用于容量检查、无梯度推理和报告边界；过程来源见[协议中的技能引用](PROTOCOL.md)，不作为加速方法贡献。

### 本地 Qwen3-VL-8B 语言主干速度对照

直接复用`third_party/ECG-R1/ECG-R1-8B-RL`，只加载Qwen3-VL语言主干（7,567,324,160参数）和16个词表行，不加载视觉/ECG分支，不新增下载或训练。这是ECG-R1微调后的Qwen3-VL-8B派生权重，不是小号JEV。其隐藏维度4096，测速采用随机初始化256→4096→4096连接器，不能作为EEG效果结果。

| EEG 时长（秒） | JEV-27B完整计算（ms） | 本地8B完整计算（ms） | 完整路径加速比 | 8B完整路径P95（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 269.05 | 31.29 | 8.60× | 33.16 |
| 1 | 268.77 | 32.15 | 8.36× | 32.47 |
| 2 | 268.13 | 32.01 | 8.38× | 33.47 |
| 4 | 269.49 | 33.09 | 8.14× | 34.24 |
| 8 | 274.27 | 34.50 | 7.95× | 36.16 |
| 16 | 291.31 | 46.88 | 6.21× | 61.20 |
| 30 | 314.61 | 59.35 | 5.30× | 61.57 |

两组均单H100、batch1、预热10次/测量50次；同一LEAF＋个人LoRA/adapter权重、同一输入及预处理，计时定义与上节相同。8B语言主干约20.1–20.5ms，峰值allocated显存约14.28–14.30GiB。源码和权重不在测速期间变动；113项自动测试通过。

不是参数量的严格消融：27B是缺快速kernel的Qwen3.5混合注意力＋原语言LoRA，8B为Qwen3-VL标准注意力；相同提示词经原生tokenizer得到76/72个文本tokens，连接器维度也不同。只说明当前两套执行路径的实测差异，不说明分类效果保持或所有8B模型都会同样快。

[8B完整记录及模型来源](../artifacts/benchmark/inference_latency_qwen8b_20261124/summary.md)、[原始计时](../artifacts/benchmark/inference_latency_qwen8b_20261124/results.json)、[比较审计](../artifacts/benchmark/inference_latency_qwen8b_20261124/comparison.json)。过程技能引用已保留在报告中。

### 原生 LEAF 发布版速度对照

直接加载`leaf-v1.0-instruct-mpnet-base.ckpt`完整LEAF，不加载我们训练的父模型/个人adapter。发布模型共26.424M参数，推理有效参数25.752M＝Tower 19.558M＋指令Q-Former 6.194M，不含离线MPNet。原生直接推理以缓存文本向量条件化FiLM/Q-Former，输出768维向量与类别原型做余弦匹配，不是把EEG送入自回归语言模型。[上游推理接口](https://github.com/Jiang-Muyun/LEAF/blob/main/c_direct_inference.py)

相同H100、batch1、输入和预处理，预热10次/测量50次；单位ms，完整路径墙钟中位数。前两轮JEV/8B数据保留，未重跑。

| EEG时长（秒） | 原生LEAF FP32（ms） | 原生LEAF BF16 autocast（ms） | Tower→本地8B（ms） | Tower→JEV-27B（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 7.85 | 10.82 | 31.29 | 269.05 |
| 1 | 8.73 | 11.70 | 32.15 | 268.77 |
| 2 | 9.23 | 11.26 | 32.01 | 268.13 |
| 4 | 9.81 | 12.62 | 33.09 | 269.49 |
| 8 | 10.76 | 14.12 | 34.50 | 274.27 |
| 16 | 20.94 | 27.43 | 46.88 | 291.31 |
| 30 | 33.74 | 42.78 | 59.35 | 314.61 |

FP32为本地上游推理默认，BF16只作计算精度控制，未验证分类精度变化。4秒完整LEAF前向7.91ms，含预处理/传输/CPU匹配共9.81ms；峰值allocated 0.158GiB，不含离线文本编码器。MPNet文本向量已缓存，不把该耗时说成“在线跑完整语言模型”。比较中的8B仍未做EEG对齐；原27B缺部分快速kernel，系统能力/参数/权重/读出不同，不是等价能力对照。

LEAF单次输入上限10秒；16/30秒用8+8或10+10+10逐块运行完整LEAF，再按时长平均类别余弦分数，这是本次测速wrapper而非上游原生超长推理。长输入重复同一4秒trial，仅作形状测速；没有进行准确率评估或训练。114项自动测试、14组GPU测速及上游源码哈希核对通过，四卡已释放。

[完整LEAF报告](../artifacts/benchmark/inference_latency_leaf_native_20261124/summary.md)、[原始计时](../artifacts/benchmark/inference_latency_leaf_native_20261124/results.json)、[比较审计](../artifacts/benchmark/inference_latency_leaf_native_20261124/comparison.json)。

## 📊 已完成：23名留出用户残差-only基线

2026-11-23 17:56:59 UTC全部完成，运行源码`513427d`。每人20条support、seed42、100步，仅更新33,088参数；Tower/JEV/连接器冻结。下表原划分计数与个人support/query不同。

| 数据集 | 处理后窗口总计（父模型原训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（父模型原训／验／测） | 本轮用户 | Support/query | 不适配BAcc（%） | 残差adapter BAcc（%） | 提升（百分点） | 提升/下降/持平人数 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4类） | 固定跨被试；个人T→E | 9（5/2/2） | 2 | 40/576 | 50.87 | 57.81 | +6.94 | 2/0/0 |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4类） | 固定跨被试；个人前四→后两run | 109（69/19/21） | 21 | 420/632 | 38.08 | 42.54 | +4.46 | 14/7/0 |

两库等权44.48%→50.18%，+5.70个百分点；23用户等权39.19%→43.87%，+4.68个百分点。两种权重不是同一指标。被试8复现64.93%，被试9为50.69%。[完整汇总](../artifacts/benchmark/leaf_personal_ce_alltest_20261123/summary.md)、[机器记录](../artifacts/benchmark/leaf_personal_ce_alltest_20261123/summary.json)。

## 📊 已完成：夜间方法控制

MEMO、SSF、AdaBN均23/23完成，控制器于2026-11-23 20:57:10 UTC结束，无遗留队列。两库等权BAcc：MEMO 49.80%，SSF 47.03%，AdaBN＋残差CE 48.99%。原残差-only为50.18%，未观察到这些方法超过它。

MEMO逐query无标签更新，严格同单trial batch对照为50.12%（而非原batch8的50.18%，有bf16批大小数值差异），相对该匹配对照下降0.32个百分点；不把差异都归为算法。SSF有63,488参数，容量不等同残差adapter；MEMO/AdaBN为TTA，不冒称原始联合自监督TTT。

实现保留在隔离worktree `../eegjev-worktrees/personal-method-controls`（实现`460d31e`）。[完整方法结果](../artifacts/benchmark/leaf_personal_methods_20261123/summary.json)、[完成记录](../artifacts/benchmark/leaf_personal_methods_20261123/complete.json)；原产物不删除。

## 📦 已归档：G/H＋FiLM前向适配

该轮已于2026-11-23 16:00:33 UTC完成，不再运行。冻结Tower与全参Tower两组分别离线训练共享G/H/FiLM，两组均340步、验证选中170步；不是个人100步微调。2a被试8前向适配BAcc分别53.125%、52.0833%，未超过同用户不适配53.8194%。[完整结果](../artifacts/benchmark/leaf_amortized_film_20261123/summary.json)、[完成记录](../artifacts/benchmark/leaf_amortized_film_20261123/complete.json)。

实现归档到Git `36b33d0`，输出全部保留。当前不继续74名源被试共享训练；历史LM LoRA对照不恢复，Tower LoRA只用于当前个人联合训练组。

## 🧪 历史：单被试 JEV CE 与 LM LoRA 对照已完成

按用户要求，原192次完整个人评估缩为 BCIC-IV-2a 被试8、每类5条、seed42的四组初步验证；不增加其他被试、trial数量或seed。四组均已训练100步，使用T session的20条support，评估E session的全部288条query。信号LR1e-4、LM LR5e-6、有效batch16及其余超参数没有改变，connector/head均冻结。

共同不适配基线为JEV BAcc 53.82%。下表都是同一被试、同一批support/query；query四类各72条，所以本例query Acc与BAcc相同，不是全数据集成绩。

| 信号侧 | LM更新 | support Acc（%，20条） | query BAcc（%，288条） |
| --- | --- | ---: | ---: |
| B：输出残差adapter | 冻结 | 70.00 | 64.93 |
| B：输出残差adapter | 既有语言LoRA-r32 | 100.00 | 59.38 |
| C：Tower Q/V LoRA-r8 | 冻结 | 60.00 | 59.03 |
| C：Tower Q/V LoRA-r8 | 既有语言LoRA-r32 | 100.00 | 59.03 |

本例更新LM提高了support拟合，但相对同CE的冻结LM组，B的query BAcc低5.56个百分点、C持平。仅一个被试和一个seed，不能推断跨被试效果、统计显著性或“27B太大”的原因。旧InfoNCE对照必须筛被试8、每类5条、seed42，不能直接对比上一轮三seed/多被试均值。

运行实现 `f42fc00`，102项自动测试和8次验证被试短测已通过。四组100步训练实测733–772秒，均通过共同初始预测、support/query/batch IDs、冻结/更新、保存重载审计。[四组结果及审计](../artifacts/benchmark/leaf_personalization_jev_ce_20261123/pilot_summary.json)、[短测审计](../artifacts/benchmark/leaf_personalization_jev_ce_20261123/smoke_verified.json)。

2026-11-23 14:12:56 UTC，原控制器767389及四个worker 849831/849835/849838/849843已退出，四张GPU释放。停止时后续10-shot已进入训练但未完成，不计入结果；原始manifest/config、完成权重、预测及日志全部保留，没有删除产物。`cancelled.json` 阻止 `--start`/`--resume` 再执行；原pipeline的 `failed_or_stopped` 表示用户主动取消剩余任务，不是这四组训练失败。全192次未完成，不生成虚假的完整汇总。[取消记录](../artifacts/benchmark/leaf_personalization_jev_ce_20261123/cancelled.json)、[运行状态](../artifacts/benchmark/leaf_personalization_jev_ce_20261123/pipeline.json)。

## 📊 已完成：本地 InfoNCE 个人校准

已固定 B（输出残差 adapter，33,088 参数）与 C（Tower 最后两层双分支 Q/V LoRA-r8，32,768 参数），均只更新个人信号侧小模块，冻结 JEV/connector/文本读出。2a 每类 5/10/20 trial，PhysioMI 每类 5/10；3 个采样种子。23 名留出目标被试，共 288 次独立适配，包含共同的 0-shot 控制。协议和源代码预检完成后由四卡真实短测门控正式运行。

2026-11-23 13:25 UTC 四卡验证被试短测全部通过，正式四 worker 已启动：GPU 0/1 为 B，GPU 2/3 为 C，每 worker 72 次适配。99 项自动测试通过；短测实测 zero-init 输出误差为 0、LoRA 有非零梯度，确认 B/C 的 support/query/batch IDs 与共同基线一致、基础 Tower/文本读出/JEV 未更新、小 checkpoint 重载输出完全一致。[短测审计](../artifacts/benchmark/leaf_personalization_20261123/smoke_verified.json)、[运行状态](../artifacts/benchmark/leaf_personalization_20261123/pipeline.json)。实现提交 `36493bb`。

2026-11-23 13:32:35 UTC，288 次适配全部完成，144 个 B/C 配对审计通过，四卡正常释放。[完整汇总](../artifacts/benchmark/leaf_personalization_20261123/summary.md)。2a 上 B 的直接文本 BAcc 为 60.76/59.61/60.19%，对应每类 5/10/20 trial，冻结 JEV 为 52.26/52.43/51.68%；各自不适配基线为 49.13/50.87%。PhysioMI 上 B 的文本为 42.51/42.20%、JEV 为 38.01/38.18%；C 的文本为 42.07/42.44%、JEV 为 38.54/38.75%，对应每类 5/10 trial，各自基线为 41.81/38.08%。先按种子、再按被试等权平均；2a 仅 2 人，不宣称显著。文本投影头曾用源 MI 训练被试额外拟合 1000 步，JEV 没有等量专项训练，因此这是接口兼容性实验，不是等预算模型能力比较。

这些是带标签 support 的个人跨 session/run 分数，不是下方九库原完整测试分数，也不等同于第一阶段 51.39%。

## 📋 已完成的共享模型与选择

2026-11-23 用户选择保留 InfoNCE，不再执行另一组后续实验。InfoNCE→JEV 已完成 2040 步，验证选中 step 1360；完整九库验证 61,327 窗，等权 Acc 50.01%、BAcc 47.83%。[完整验证](../artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/valid_step002040_full.json)、[完成审计](../artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/complete.json)、[训练/验证曲线](../artifacts/benchmark/leaf_tower_contrastive_20261123/figures/leaf_infonce_mlp/training_validation_curves.png)。

逐库验证样本 ID 哈希与两个对照完全一致：

| 方法 | 完整验证 Acc（%） | 完整验证 BAcc（%） |
| --- | ---: | ---: |
| 原完整 LEAF 编码路径冻结→JEV | 49.61 | 46.52 |
| 纯 Tower→MLP/JEV，无新增对比 | 46.32 | 43.41 |
| InfoNCE 对齐的纯 Tower→MLP/JEV | 50.01 | 47.83 |

相对原完整 LEAF 组 BAcc +1.31 个百分点，相对纯 Tower 无对比组 +4.42 个百分点；单 seed 且有额外预训练预算，不能宣称统计显著或完全归因于对比损失。测试集未运行。

两组独立对比学习阶段也已完成：InfoNCE 直接文本匹配完整验证 BAcc 51.39%、Acc 52.65%（best 1700）；Sigmoid 48.52%/49.92%（best 1360）。这些不是 JEV 成绩；当前 InfoNCE 接 JEV 后的 BAcc 仍比直接文本匹配低 3.57 个百分点。

取消操作：先关闭控制器 818578，后向四个已核验的 Sigmoid smoke worker 发送 SIGTERM；该组在 step 7 保存 `last.pt` 并退出，正式 `train/leaf_sigmoid_mlp` 目录未创建。无其他实验被中断，四张 GPU 释放；没有删除任何权重、日志或数据。[取消记录](../artifacts/benchmark/leaf_tower_contrastive_20261123/cancelled.json) 保存用户决定，入口阻止误用 `--resume` 重新启动。原启动配置/日志仍作为历史运行证据保留。

## 🔍 启动与检查记录

运行 `leaf_tower_contrastive_20261123` 原计划多正例 InfoNCE、平衡多正例 Sigmoid 两组，第一阶段只用 LEAF Tower 和固定类别文本向量，第二阶段仅迁移 Tower 到同一 MLP/JEV。2026-11-23 09:56:19 UTC 两组正式对比训练启动（GPU 0/1，PID 910815/910986，控制器 818578）；当前状态以上方完成与取消记录及 [pipeline.json](../artifacts/benchmark/leaf_tower_contrastive_20261123/pipeline.json) 为准。对齐阶段 encoder LR 1e-5/global batch 32 沿用此前对比实验；下游仍 encoder LR 1e-6/global batch 16、2040 步，与下列无对比基线一致。

89 项测试通过；CPU 预检覆盖 65,280 次训练样本呈现。两组真实 GPU 17 步 smoke 均通过，验证 Tower 更新、冻结文本、保存重载；每组约 36.6 秒（不含加载）。正式初始化与 smoke 的初始化哈希一致，两种损失的初始 Tower/投影/文本/buffers 哈希也一致。每组 Tower 19,557,632 参数、对比投影 788,480 参数，InfoNCE scale 1 参数、Sigmoid scale/bias 2 参数。第一轮短测发现整数 JSON 0 导致 Sigmoid bias dtype 错误，修复后重跑全部检查，数值和超参数未改；失败产物保留在 `artifacts/benchmark/leaf_tower_contrastive_20261123_failed_smoke/`，未进入正式训练。运行代码提交 `d1b5e74`。

原自动计划是在两组对比训练及完整验证结束后，依次启动各自四卡下游 smoke 和 2040 步正式 JEV SFT。InfoNCE 已完成，Sigmoid 后续按上述用户决定停止。测试集全程未运行。

## 📊 无新增对比学习的 Tower 基线

运行名 `leaf_tower_jev_20261123`，arm `leaf_tower_mlp`。代码与 73 项单元测试已通过，含四进程梯度等价性、原始 Tower 输出一致性、无文本模块、分块/有效 token 掩码、冻结划分及测试集禁用检查。

2026-11-23 09:29:26 UTC 已完成：2040 次更新，best step 2040；完整九库验证 61,327 窗，等权 Acc 46.32%、BAcc 43.41%。LEAF Tower 确实更新、四 rank 权重一致、全局 batch 划分一致、磁盘重载 logits 精确一致，测试前向为 0。[完整结果](../artifacts/benchmark/leaf_tower_jev_20261123/train/leaf_tower_mlp/valid_step002040_full.json)、[完成审计](../artifacts/benchmark/leaf_tower_jev_20261123/train/leaf_tower_mlp/complete.json)、[曲线](../artifacts/benchmark/leaf_tower_jev_20261123/figures/leaf_tower_mlp/training_validation_curves.png)。

运行源码提交 `d23b5e8`。初始参数量：Tower 前向 19,557,632；connector 27,535,872；JEV LoRA 217,579,520；choice head 81,936。峰值分配显存约 54.6 GiB/rank；正式训练与完整验证在模型载入后用时 6,561.8 秒（约 1 小时 49 分）。无对比学习组作为后续文本对齐的基线保留，不覆盖其权重和结果。

- 四张 H100 80GB；global batch 16 = 每 rank 4。
- 2040 步；前 340 步仅连接器，之后 LEAF Tower 全参＋JEV 已有 LoRA/head；JEV base 不更新。
- 学习率维持 connector 1e-4、JEV LoRA 5e-6、head 1e-5、encoder 1e-6。
- 同一现有 17 库训练/九库验证划分；SEED 被试内，HMC 完整 151 份。测试前向为 0。
- 新模型不包含原 LEAF 文本条件化 Q-Former，不能直接继承其下游分类表现。

同一完整验证窗口 ID 对照：完整 LEAF 编码路径冻结为 BAcc 46.52%，VideoMAE InfoNCE/Q-Former 全参为 44.15%、LoRA-r64 为 44.28%，本基线为 43.41%。这些结构不是单因素消融。历史 50.59% 是另一协议的完整测试，不能混为同口径验证。

## 📊 已完成 VideoMAE 控制

以下仅为同一 SEED/HMC151 协议下的历史完整验证参考，61,327 窗、九库等权平均，全部测试前向为 0。与新 Tower-only 不是严格单因素消融。

| 方法　　　　　　　　　| encoder 更新 | best step | 完整验证 Acc（%） | 完整验证 BAcc（%） |
| -----------------------| --------------| ----------:| ------------------:| -------------------:|
| InfoNCE＋CLS→JEV　　　| 冻结　　　　 | 1700      | 46.37　　　　　　 | 43.91　　　　　　　|
| InfoNCE＋Q-Former→JEV | 冻结　　　　 | 1360      | 47.28　　　　　　 | 43.94　　　　　　　|
| Sigmoid＋CLS→JEV　　　| 冻结　　　　 | 1020      | 44.47　　　　　　 | 43.27　　　　　　　|
| Sigmoid＋Q-Former→JEV | 冻结　　　　 | 680       | 47.61　　　　　　 | 45.56　　　　　　　|
| InfoNCE＋Q-Former→JEV | 全参　　　　 | 1700      | 47.79　　　　　　 | 44.15　　　　　　　|
| InfoNCE＋Q-Former→JEV | LoRA-r64　　 | 1700      | 46.47　　　　　　 | 44.28　　　　　　　|

LoRA 相对全参 BAcc +0.13 个百分点、Acc -1.32 个百分点，单 seed 不宣称可靠提升。两者已核对相同初始化、学习率、2040 个全局 batch 和验证样本 ID；LoRA 基础 VideoMAE 未变、低秩参数已更新。[五组完成记录](../artifacts/benchmark/aligned_videomae_jev_20261102/pipeline.json)、[LoRA 完成记录](../artifacts/benchmark/aligned_videomae_jev_lora64_20261102/train/infonce_qformer_lora64/complete.json)。

父阶段直接文本对齐（没有 JEV）的完整验证 BAcc 为 InfoNCE 47.49%、Sigmoid 46.55%；不等于接 JEV 的初始化成绩。其独立实现/配置/详细报告已归档到 `6247e46`，产物不删除。

## 📊 历史 LEAF 完整训练与测试对照

来源为已完成的 2040 步 LEAF→JEV 共享模型，按固定验证128子集选中 step 2040。完整测试 49,431 窗；完整九库训练评估 467,518 窗。训练评估仅做推理，不更新权重、不重复测试。[原训练完成记录](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train/complete.json)、[完整训练评估](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval/complete.json)。

2026-11-02 已逐库核对“LEAF→JEV 完整训练 Acc”列：它是该轮 step 2040 checkpoint 在各库完整训练集上的 Accuracy，不是验证准确率、训练 batch 准确率或最近冻结 LEAF 的每库 128 窗训练子集成绩。完整训练集只评估了这一个 checkpoint，因此本表不是随 step 变化的曲线。

表内窗口与被试数描述历史 LEAF 试验，不代表论文 REVE baseline 的具体窗口或被试名单；连续切出的窗口不是独立原始 trial。保留 REVE 与历史数值，按仓库文档规范补回单独的划分协议列：这张表中的历史 LEAF 与冻结 VideoMAE 均使用 SEED 跨被试 13/1/1；除 HMC 外八库沿用同一划分。REVE 与 LEAF 测试列为测试 BAcc，LEAF 训练列为训练 Acc，VideoMAE 列为验证 BAcc，不能直接相减或排名。

VideoMAE 列现使用 `bsr_jev_readouts_hmc151_20261031/train/cls`：发布 BSR-VideoMAE encoder 全程冻结，训练 CLS 读出、连接器及按阶段开放的 JEV LoRA／head。训练 2040 步，由固定验证子集选中 step 1700，再评估完整九库验证集 55,978 窗。完成记录确认 `frozen_bsr_verified=true`、`test_evaluations=0`。这是历史冻结＋CLS 组，不是解冻后 31.41% 的组，也不是本轮纯信号 LEAF Tower 组。[配置](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/cls/config.json)、[完整验证结果](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/cls/valid_step002040_full.json)、[完成记录](../artifacts/benchmark/bsr_jev_readouts_hmc151_20261031/train/cls/complete.json)。

| 数据集　　　 | 处理后窗口数：总计（训／验／测，历史 LEAF） | 任务种类（分类数）　　　　| 划分协议（历史 LEAF）　　| 被试数：总计（训／验／测，历史 LEAF） | REVE（baseline） | LEAF→JEV 完整训练 Acc（%） | LEAF→JEV 完整测试 BAcc（%） | VideoMAE→JEV 冻结（CLS）：完整验证 BAcc（%） |
| --------------| --------------------------------------------:| ---------------------------| --------------------------| --------------------------------------:| -----------------:| ---------------------------:| ----------------------------:| ---------------------------------------------:|
| BCIC-IV-2a　 | 5,184（2,880/1,152/1,152）　　　　　　　　　| 运动想象（4 类）　　　　　| 固定跨被试　　　　　　　 | 9（5/2/2）　　　　　　　　　　　　　　| 41.87　　　　　　| 45.94　　　　　　　　　　　| 46.18　　　　　　　　　　　 | 25.00　　　　　　　　　　　　　　　　　　　　|
| PhysioMI　　 | 9,837（6,210/1,734/1,893）　　　　　　　　　| 运动想象（4 类）　　　　　| 固定跨被试　　　　　　　 | 109（69/19/21）　　　　　　　　　　　 | 58.82　　　　　　| 37.13　　　　　　　　　　　| 33.73　　　　　　　　　　　 | 24.77　　　　　　　　　　　　　　　　　　　　|
| SEED　　　　 | 37,890（32,838/2,526/2,526）　　　　　　　　| 情绪识别（3 类）　　　　　| 随机跨被试　　　　　　　 | 15（13/1/1）　　　　　　　　　　　　　| 72.62　　　　　　| 71.54　　　　　　　　　　　| 69.00　　　　　　　　　　　 | 51.81　　　　　　　　　　　　　　　　　　　　|
| SEED-VII　　 | 19,420（15,536/1,942/1,942）　　　　　　　　| 情绪识别（7 类）　　　　　| 随机跨被试　　　　　　　 | 20（16/2/2）　　　　　　　　　　　　　| 27.62　　　　　　| 41.56　　　　　　　　　　　| 36.18　　　　　　　　　　　 | 20.84　　　　　　　　　　　　　　　　　　　　|
| TUAB　　　　 | 272,320（247,728/12,315/12,277）　　　　　　| EEG 异常检测（2 类）　　　| 原训练＋评估集按被试分层 | 2,328（2,075/101/152）　　　　　　　　| 81.18　　　　　　| 65.76　　　　　　　　　　　| 63.68　　　　　　　　　　　 | 74.03　　　　　　　　　　　　　　　　　　　　|
| TUEV　　　　 | 113,353（87,834/12,473/13,046）　　　　　　 | EEG 事件分类（6 类）　　　| 按被试分层　　　　　　　 | 370（328/20/22）　　　　　　　　　　　| 74.27　　　　　　| 50.36　　　　　　　　　　　| 39.55　　　　　　　　　　　 | 42.52　　　　　　　　　　　　　　　　　　　　|
| HMC　　　　　| 98,867（68,215/15,512/15,140）　　　　　　　| 睡眠分期（5 类）　　　　　| 子集随机跨被试　　　　　 | 109（75/17/17）　　　　　　　　　　　 | 71.93　　　　　　| 56.68　　　　　　　　　　　| 53.92　　　　　　　　　　　 | 52.47‡　　　　　　　　　　　　　　　　　　　 |
| Workload　　 | 2,134（1,534/300/300）　　　　　　　　　　　| 认知负荷识别（2 类）　　　| 随机跨被试　　　　　　　 | 36（26/5/5）　　　　　　　　　　　　　| 74.67　　　　　　| 84.68　　　　　　　　　　　| 71.78　　　　　　　　　　　 | 48.89　　　　　　　　　　　　　　　　　　　　|
| ADFTD　　　　| 7,013（4,743/1,115/1,155）　　　　　　　　　| AD／FTD／正常识别（3 类） | 按被试分层　　　　　　　 | 88（64/11/13）　　　　　　　　　　　　| 52.12　　　　　　| 55.03　　　　　　　　　　　| 41.34　　　　　　　　　　　 | 34.88　　　　　　　　　　　　　　　　　　　　|
| 九库等权平均 | —　　　　　　　　　　　　　　　　　　　　　 | —　　　　　　　　　　　　 | —　　　　　　　　　　　　| —　　　　　　　　　　　　　　　　　　 | —　　　　　　　　| 56.52　　　　　　　　　　　| 50.59　　　　　　　　　　　 | 41.69　　　　　　　　　　　　　　　　　　　　|

历史 LEAF 九库等权平均：完整训练 Acc 56.52%、BAcc 56.74%；完整测试 Acc 51.71%、BAcc 50.59%。REVE 数字来自此前核对的 EEG-FM-Bench 论文 Table 2，单位为测试 BAcc（%）；当前未重新训练 REVE。[论文对应仓库](https://github.com/xw1216/EEG-FM-Bench)。

‡ HMC 的 VideoMAE 冻结组使用完整 151 份记录，不对应主表历史 LEAF 的 109 份子集；其他八库沿用原划分。下表的测试窗口是预留集合数量，不表示已评估测试集。

| 数据集 | 处理后窗口数：总计（训／验／测，VideoMAE） | 任务种类（分类数） | VideoMAE 冻结组划分方式 | 被试数：总计（训／验／测，VideoMAE） |
| --- | ---: | --- | --- | ---: |
| HMC 完整数据 | 136,925（92,454/22,421/22,050） | 睡眠分期（5 类） | 完整 151 份记录，固定随机跨被试 | 151（103/24/24） |

划分与窗口证据：[划分摘要](../artifacts/benchmark/multitask_splits_hmc151_20261031/summary.json)、[SEED 缓存](../datasets/multitask_hmc151_20261031/seed/complete.json)、[HMC 缓存](../datasets/multitask_hmc151_20261031/hmc/complete.json)。VideoMAE 冻结＋CLS 组九库等权完整验证 Acc 为 39.94%、BAcc 为 41.69%，不是测试成绩；与历史 LEAF 仍不是严格单因素消融。

其他八个库只用于训练，不参加验证／测试。以下是 QC 前缓存数量；CHB_MIT 中 1,144 个全零／无效训练窗口被固定 QC 排除，其他数量不变。

| 数据集　　　　| 缓存窗口总数（训／验／测） | 任务种类（分类数）　　 | 划分方式　　　　| 被试总数（训／验／测） |
| ---------------| ---------------------------:| ------------------------| -----------------| -----------------------:|
| BNCI2015001　 | 2,400（2,400/0/0）　　　　 | 运动想象（2 类）　　　 | 全部 train-only | 12（12/0/0）　　　　　 |
| BNCI2014008　 | 33,600（33,600/0/0）　　　 | P300（2 类）　　　　　 | 全部 train-only | 8（8/0/0）　　　　　　 |
| BNCI2014009　 | 5,760（5,760/0/0）　　　　 | P300（2 类）　　　　　 | 全部 train-only | 10（10/0/0）　　　　　 |
| CHB_MIT　　　 | 27,436（27,436/0/0）　　　 | 癫痫检测（2 类）　　　 | 全部 train-only | 23（23/0/0）　　　　　 |
| SleepEDF　　　| 414,961（414,961/0/0）　　 | 睡眠分期（5 类）　　　 | 全部 train-only | 78（78/0/0）　　　　　 |
| AlexMI　　　　| 480（480/0/0）　　　　　　 | 运动想象／静息（3 类） | 全部 train-only | 8（8/0/0）　　　　　　 |
| Zhou2016　　　| 614（614/0/0）　　　　　　 | 运动想象（3 类）　　　 | 全部 train-only | 4（4/0/0）　　　　　　 |
| Nakanishi2015 | 1,800（1,800/0/0）　　　　 | SSVEP（12 类）　　　　 | 全部 train-only | 10（10/0/0）　　　　　 |

## ⚠️ 保留的划分、数据与来源问题

- 2a 被试 1–5/6–7/8–9 与仓库固定名单一致；我们训练窗 2,880，论文 2,784。PhysioMI 被试 1–69/70–88/89–109 与代码一致，但我们的测试窗 1,893，论文 1,803。具体数量差异仍待定位。
- SEED 我们按跨被试 13/1/1，论文是被试内 trial 9/3/3，不能把分数当成相同设置。Workload 的窗口数我们 1,534/300/300，论文 1,537/300/297，也非完全一致。
- SEED-VII、TUAB、TUEV、ADFTD 已对同一本地记录重算官方划分函数：被试归属一致、三份窗口数一致；但作者未公开那次运行的完整名单，因此不能证明与论文逐条一致。
- 四库逐窗核查共 412,106 窗：TUAB 272,320 窗和 ADFTD 7,013 窗边界一致；SEED-VII 7,491/19,420 窗、TUEV 7,781/113,353 窗起点相差 1 个 200 Hz 采样点，源于本地 round 与官方 floor。标签、长度及逐记录窗口数一致。缓存未重建，不声称 5 ms 差异一定无影响。
- 历史 HMC 成绩使用 109/151 份记录，75/17/17。当前完整 HMC 151 份记录已补齐，采用新的 103/24/24 分组；原始测试分母及成绩不改写。当前三份窗口数与论文仍不同，作者具体随机被试名单未确认。
- TUAB/TUEV 19 个候选共享文件名被试标识，5 个存在跨库训练／测试角色冲突，精确身份映射未解决。原用户已同意先做验证性实验，不应宣称严格无泄漏成绩。
- SEED 原 CNT 头部异常已在缓存生成时处理；PhysioMI S104R08 第 12 个零起始事件越界，未生成该窗口，但保留记录及被试。上述旧事实的详细证据保存在 Git `741fe34` 的旧 RESULTS 和原逐记录 JSON。
- BSR 预训练来源与目标库重叠未完整排除；本轮不是跨数据集 zero-shot，也不是 EEG-FM-Bench 论文完全复现。

## 📈 保留的历史曲线与当前日志

历史 LEAF step 0–2040 图件未删除：[loss 和验证总图](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_validation_curves.png)、[各库验证](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/validation_by_dataset.png)、[各库 loss](../artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/figures/training_loss_by_dataset.png)。完整训练集准确率只测过 step 2040，不把训练 batch 曲线称为逐 checkpoint 的完整训练准确率。

本轮指标保存在 `train/leaf_tower_mlp/metrics.jsonl`、`valid_step*_subset.json`、`train_step*_subset.json`；完整验证在 `valid_step002040_full.json` 中通过 `selected_step` 标明实际加载的最佳权重。失败或尚未完成不填写准确率。代码变更、历史研究过程和旧方案恢复方式见 [ARCHIVE.md](ARCHIVE.md)。
