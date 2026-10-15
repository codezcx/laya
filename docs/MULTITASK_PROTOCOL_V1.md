# 多任务 EEG 文本原型实验 v1

2026-10-15。用户目标：持续迭代运动想象与情绪跨数据集zero-shot，之后扩展心算/压力。第一轮是有界可复现基线，不承诺达到目标。

## 数据与划分

统一C3/Cz/C4、200Hz、3秒、1–30Hz四阶零相位带通、逐窗口逐通道zscore/clip15。SEED使用预处理波形而非DE特征，每trial均匀取10个不重叠窗口；先按被试9/3/3划分，所有session同split。EEGMat只准备数据，尚不训练；匹配两种状态的前60秒，按被试划分。原MI被试划分保持不变，Cho+Dreyer为源，Lee完全不训练；每被试每类每split最多40trial，固定seed44。所有数据读取与重复窗口检查留档。

3通道和相同窗口减少显式采集指纹，但不能消除设备、原预处理、参考方式的差异；不把task识别高分当生理语义证据。3通道及30Hz上限会限制情绪信息，后续仅依据源验证做通道/频段消融。SEED的电影身份在被试间共享，所以跨被试分数不是跨刺激泛化。

## 模型与训练

REVE冻结，输出按通道时间池化形成1536维；共享LN→Linear1536→512→GELU→Dropout，再投影到4096维与L2归一化文本原型比较。文本由当前CerebraGloss Qwen语言骨干离线编码：任务+类别完整文本、最后层token均值、L2归一化。它不是专用检索embedding模型；需要随机原型对照。

仅情绪三类与MI两类进入训练。四卡四组：text_ce（已知任务内原型CE），text_sigmoid（仅已标注任务内的逐类BCE，SigLIP风格，非完整SigLIP复现），random_ce（固定随机归一化原型），learned_ce（普通可学习分类头）。共享前级结构、相同初始化与样本流。1500step，batch192，AdamW lr3e-4，seed42。按task→dataset→class均衡采样。每250step在独立源验证被试上评估，按task宏平均（任务内dataset平均）选checkpoint，平分取早。

其他任务未标注类别不作为负样本；loss mask已单测。分类头对照与原型头输出参数量不同，不是参数量匹配实验。没有自动LoRA、没有语言生成、没有目标数据自适应。

## 评估

主指标：任务给定时分类准确率、被试macro、balanced accuracy、按被试bootstrap95CI。任务未知联合argmax是未校准诊断，不作达标指标。信号打乱为反事实对照。原型语义是否有增益，以random/learned对照评估。

源test及整数据集Lee只能在checkpoint冻结后评估；初始协议将MI最终2b暂时封存；v1 checkpoint冻结后已进行完整2840 trial评估，详见结果文档。情绪额外目标候选为SEED-VII（标签需经官方表核验，Happy→positive，Neutral→neutral，Sad→negative子集；不是7分类），作为跨数据集外部评估，不能混进训练/调参。其相同研究机构与可能共享刺激等限制需要记录。已知REVE预训练包含2b；其他数据集暴露须继续审计，当前仅声称下游源数据训练。

在用户未给更高标准前，暂定可行性标准：两个任务的独立跨数据集被试CI下界高于机会水平，且不劣于encoder-only对照；单次或单seed成功不足以称稳定通用模型。失败也不能通过反复目标集调参把目标改成开发集而仍称严格zero-shot。

## 数据来源

- SEED说明：https://bcmi.sjtu.edu.cn/home/seed/seed.html
- SEED-VII说明：https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html
- EEGMat标签规则：https://physionet.org/content/eegmat/1.0.0/README.txt
- 实验设计技能方法辅助：Scientific Agent Skills, https://arxiv.org/abs/2609.00065
