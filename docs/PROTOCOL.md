# LEAF Tower → MedJev 固定协议

日期：2026-11-26。目的：在现有EEG数据协议下验证较小、裁剪语言层的直接决策模型。此轮不做元学习，不做个人支持集适配，不把医学图像结果当作EEG结果。

## 1. 模型与迁移边界

- 基础模型：用户本地Qwen/Qwen3.5-4B，不加载Karl28/MedJev-4B医学适配权重。
- EEG：发布的instruction-tuned LEAF Tower，原checkpoint SHA256 `f9ca9cabbd734d2691fa6748adf3e78056f895d4031f7edc481e7d7b6822f3fa`；只取信号Tower，移除未参与forward的重建头。
- 连接器：LayerNorm(256) → Linear(256,2560) → GELU → Linear(2560,2560)，新初始化，保留全部Tower tokens。
- 语言层：保留第1–23层，删除第24–31层；第32层仅复用input norm、Q/K及Q/K norm。原最终语言norm、词表输出和视觉路径不参与模型。
- 读出：决策marker的Q和候选marker的K做RoPE后的逐头缩放点积，跨头均值；加上共享MLP([decision, option, decision×option])。MLP宽256，输出层零初始化。
- 文本：原任务instruction/question/class wording和相同label-independent选项排列；增加question-end、option-end及decision三个标记。用Qwen chat envelope包裹，EEG tokens插入user消息。不把正确答案写进输入。
- 语言与读出Q/K的LoRA：r16、alpha32、dropout0，目标q/k/v/o和MLP gate/up/down线性层，与上游目标名一致。普通LoRA实现与PEFT数值匹配。GatedDeltaNet的in_proj不额外加LoRA。
- 三个marker向量、残差MLP、三组读出norm全参训练；原始语言权重冻结。没有医学视觉merger，因为EEG连接器替代了它。
- 本轮总参数量以真实加载审计为准；不能把论文包含视觉模块的3.55B直接当成EEG版本参数量。

上游固定为 `zfkarl/MedJev@386c19c963f2a1a32ea7210e8b30c55ca3c5303b`。批量化读出与上游单样本公式对照；不改上游文件。PyTorch SDPA用于本地兼容，不要求另装FlashAttention，不安装新的驱动或改变现有共享环境。

## 2. 数据成员和评估角色

继续使用当前13库 `datasets/leaf13/manifest.json`，SHA256 `60647f3ae8a5fb9cd286a6958cc758d4a394a6c2f748ec8c9379e908400c9f2a`。

数据为BCIC-IV-2a、OpenBMI MI、ShanghaiU MI、HighGamma MI、Cho2017、Shin2017A、PhysioNet MI、FACED、SEED3、SEED4、SEED7、OpenBMI SSVEP、Workload。沿用每库原始标签顺序、样本ID、65通道200Hz HDF5窗口及已有归一化，不重新滤波/缩放/插值。本轮不是更早的18库/9库benchmark协议，也不声称所有13库都是同一种跨被试划分。

train用于更新；原test为development/selection，固定subset每库最多512条，step0、每340步和终点监控。按13库等权BAcc选最早达到最大值的checkpoint，训练结束在完整原test报告。原valid不forward，且它历史上已被观察；开发分数不是独立盲测。train监控每库128条。保存实际样本ID、混淆矩阵、各被试结果及split-role ledger。

## 3. 固定优化配方

| 项目 | 本轮 |
| --- | --- |
| 总更新步数 | 2040 |
| 全局有效batch | 16 |
| 设备 | 4×H100，四卡同步梯度；每卡4条 |
| 新连接器warmup阶段 | 前340步，其他参数不更新 |
| Connector峰值LR | 1e-4 |
| 语言/读出QK LoRA峰值LR | 5e-6 |
| 决策头、marker、读出norm峰值LR | 1e-5 |
| LEAF Tower峰值LR | 1e-6 |
| Warmup / cosine下限 | 62步 / 峰值的0.02 |
| AdamW weight decay / clip | 0.01 / 1.0 |
| Seed | 42 |
| 恢复checkpoint | 每20步与阶段边界 |
| 任务采样 | 每13步覆盖13库；类均衡、被试均衡，沿用原算法 |

Tower第二阶段全参更新，BatchNorm running buffers和dropout保持原eval策略。stage1不对Tower/LM/head产生梯度。stage2同时更新，不改变数据成员。本轮沿用我们的配方而非上游174,360医学图像/1epoch/lr1e-4配方，因此属于架构迁移，不是完整MedJev论文复现。

## 4. 运行前检查和故障行为

检查下载控制文件、safetensors声明长度、所有分片实际大小、索引键集合及SHA256。分片SHA256与2026-11-26从官方HuggingFace LFS元数据读取并固定到配置的两个哈希核对；不匹配则停止。权重未齐则等待，最多24小时，不覆盖用户下载。

CPU测试覆盖作者读出公式/梯度、PEFT等价LoRA、特殊marker、标签独立性、padding与真实小型Qwen前反向、数据和评估隔离。之后仅使用train样本做真实Qwen+Tower前反向验证，再执行四卡13步smoke，包括两训练阶段、梯度同步、保存/恢复一致性。smoke不使用valid/test，也不将smoke权重用于正式训练。

正式训练保留全部梯度/学习率/任务loss、train监控、development曲线与step0记录。live源码或权重变化立即拒绝继续；不自动改LR、batch、步数或resume身份。

## 5. 来源与软件说明

- Zhang et al. (2026), [MedJev: A System One Vision-Language Decision Model for Fast and Effective Medical Image Understanding](https://arxiv.org/abs/2611.20992)，[源码](https://github.com/zfkarl/MedJev)。
- LEAF发布代码/权重：本地既有配置与checkpoint哈希，信号预处理与13库manifest不变。
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。模型接入、资源检查和本地验证工作流使用相关skills；2026-11-26核对当前v2元数据。
