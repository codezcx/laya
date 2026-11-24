# 个人残差 adapter＋LEAF Tower 联合适配协议

_2026-11-24：相对完成的23人残差-only对照，仅改变Tower参数更新方式。_

## 🎯 问题与假设

目标是用单个新用户的少量标注数据适配已训练EEG→JEV系统，同时保持服务器LM不变。原残差-only两库等权BAcc为50.18%；本轮检验进一步更新Tower是否改善独立session/run预测，而非仅提高支持集拟合。不增加共享源用户训练，不新增对比损失、TTT或LM更新。

## ⚙️ 两组训练路径

两组都使用个人原始EEG→Tower→残差adapter→冻结原connector→冻结JEV；支持集标签与JEV choice logits的交叉熵是唯一损失。梯度穿过冻结LM和连接器，最终预测仍由JEV输出。

| 模块 | joint_lora | joint_full |
| --- | --- | --- |
| LEAF forward Tower | 双分支最后两层attention的Q/V LoRA-r8；alpha8，无LoRA dropout | 全部19,557,632个前向参数 |
| 残差adapter | 更新33,088参数 | 更新33,088参数 |
| 原MLP connector，27,535,872参数 | 冻结 | 冻结 |
| JEV基础权重、已有语言LoRA、choice head | 全冻结 | 全冻结 |
| Tower BN running统计量及所有buffers | 冻结 | 冻结 |
| 总可训练参数 | 65,856 | 19,590,720 |

adapter为256→64→256、GELU、残差上投影零初始化。LoRA为原个人C分支相同的Q/V低秩参数化，不更新K或原投影权重；两组均保留Tower eval模式，仅改变requires_grad，不引入BN适配或dropout差异。全参指实际forward Tower，既有接口已移除不用的重建头。

父模型固定为`artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，原验证选中step1360。每位用户独立重置Tower、LoRA、adapter及优化器；不继承另一人的更新，也不从上一轮个人adapter再加100步。先seed42初始化adapter，再单独重置seed42初始化LoRA，保持adapter初始化跨组一致。每个用户需云端LM前向及输入梯度，不是纯边缘端训练。

## 📋 数据与划分

下表描述父模型处理后窗口与原被试划分，不是个人support/query数量。MI缓存每trial恰好一个4秒窗口。

| 数据集 | 处理后窗口总计（父模型原训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（父模型原训／验／测） |
| --- | ---: | --- | --- | ---: |
| BCIC-IV-2a | 5,184（2,880/1,152/1,152） | 运动想象（4类） | 固定跨被试；个人T→E session | 9（5/2/2） |
| PhysioMI | 9,837（6,210/1,734/1,893） | 运动想象（4类） | 固定跨被试；个人前四→后两imagery run | 109（69/19/21） |

- 正式用户：2a 8、9；PhysioMI 89–109。两组各23人，共46次独立适配。
- 每人按类无放回选择5条，共20条；2a支持来自T session，query为全部E session；PhysioMI支持来自R04/R06/R08/R10，query为全部R12/R14。
- 每组共460条support、1,208条query（2a 576，PhysioMI 632）。支持池剩余样本不训练、不加入query。
- 从已完成`leaf_personal_ce_alltest_20261123`逐条核对支持/query ID、共同基线与100步batch流。
- 复用已校验的原始归一化信号缓存；旧tokens只用于共同不适配对照，联合训练与适配后预测全部重新经过Tower。
- 短测：2a验证用户6、PhysioMI验证用户70，两组各3步、8条query；只核查工程实现，不选超参数。
- 不使用query标签或集合统计更新权重、归一化、选点或早停。2a用record/run/event联合键防止重复trial。

## 🔒 固定预算与审计

| 设置 | 固定值 |
| --- | --- |
| 支持样本／seed | 每人5条/类，共20条；seed42 |
| 步数／选点 | 固定100步，无query选点 |
| Tower与adapter学习率 | 同为1e-4，constant，无新LR组或扫描 |
| 优化器 | AdamW，weight decay0.01 |
| 梯度／batch | 两部分合并global clip1；batch16=micro4×4 |
| 抽样 | 每步类均衡，在已选20条内有放回取样；逐条匹配原baseline |
| GPU | 0/1联合LoRA，2/3联合全参；每组12/11人，无跨用户梯度聚合 |
| 恢复 | 跳过完整用户；未完整用户从父模型重跑，不恢复中途优化器 |

每组先检查两部分非零有限梯度与真实更新，Tower冻结部分及全部buffers哈希不变，JEV/connector/head参数版本不变且没有参数梯度。逐用户重置一致、初始adapter恒等、保存重载tokens/logits完全一致。源码/config/父权重/缓存变化触发停止，不在live job中热改。

资源门槛：四卡各至少60,000MiB空闲，磁盘至少20GiB。所有单元测试和四卡短测通过后才正式运行。每10步保存信号侧`last.pt`及loss，完成保存`adapter.pt`、预测和审计；全参组保存的是完整个人Tower＋adapter，不是小LoRA文件。LM权重不复制。

## 📊 输出、比较与限制

根目录`artifacts/benchmark/leaf_personal_joint_20261124/`。主要比较同一用户/同一query上原残差-only、联合LoRA、联合全参的BAcc。先按用户等权得到每库均值，再计算两库等权均值；另列23用户等权均值、每用户变化及负迁移人数。不能将support Acc作为泛化结果。

全部46次通过审计才生成完整汇总，部分结果不混作完整均值。原baseline来自先前已完成运行，不冒充本轮同步重跑的对照；运行前复核相同父权重、输入、batch和初始预测。仅一个seed、2a两名用户，被试8已观察过；属于探索性对照，非盲测或显著性结论。上游LEAF预训练身份重叠未完全排除，个人query是原test子集，不是完整EEG-FM-Bench。

## 🔗 实现与归档

[配置](../configs/personalization.json)、[模型模块](../eeg_align/personalization.py)、[训练入口](../eeg_align/run_personalization.py)、[控制器](../scripts/run_personalization.py)。此前残差-only归档`34c11a8`，运行源码`513427d`；夜间MEMO/SSF/AdaBN全部完成，保留在隔离worktree及原产物，不恢复其独立实现到当前树。原192次计划的取消标记继续保留。

实验设计技能用于逐用户匹配和隔离数据，资源技能用于容量检查，模型技能用于冻结LM梯度路径，Markdown技能用于同步当前协议；沿用现有本地模型/环境。资源目录未能提供本任务适用模型，不作为模型选择证据。过程参考[^skills]不是方法新颖性来源。

[^skills]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
