# LEAF Tower → MedJev：当前状态与结果

2026-11-26。当前主线为13库共享监督训练，Tower、两层连接器及语言LoRA统一r16/alpha32；不做元学习。

## 已完成

- MedJev固定版本源码、MIT许可和来源信息已纳入Git，不含模型权重、数据、环境。
- 接入23层Qwen语言主干、源第32层Q/K读出、零初始化残差MLP和LEAF信号Tower。
- 批量读出前向及反向与作者单样本实现一致；LoRA与PEFT一致；真实小型Qwen验证EEG拼接、padding和梯度。
- 前一全参配置31项自动测试及真实4B+Tower前反向检查通过，不代替本轮LoRA检查。新增统一rank、MHA融合QKV/out_proj梯度、冻结范围及随机连接器基础权重保存/恢复测试。
- 本轮LoRA配置37项自动测试全部通过，包含CPU和GPU实际计算；冷启动总耗时157.59秒。
- Qwen3.5-4B两分片官方SHA256校验通过，738个张量与索引一致。接入基线已归档并推送Git `bc434ad`。
- 训练数据/评估成员与原13库配置保持一致；原test为开发评估，原valid不forward。
- 旧个人LoRA位置/rank和MAML实现归档在Git `c46748c` / `dffd139`；所有ignored运行产物保留。

## 尚未得到的新结果

尚无本轮LoRA版MedJev EEG准确率。权重已齐，先验证本轮真实模型和四卡train-only smoke，再从全新初始化执行正式训练，不复用smoke权重。

连接器基础矩阵为随机初始化且冻结；本轮是其低秩修正训练，不是对已训练连接器做个人LoRA。没有额外全参warmup，也没有更改学习率、数据划分、batch或2040步预算。

控制器只在权重完整后执行真实模型检查、四卡smoke及正式训练。状态入口：

- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/pipeline.json`
- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/controller.log`
- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/weight_audit.json`
- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/model_preflight.json`
- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/train/metrics.jsonl`
- `artifacts/benchmark/leaf13_medjev4b_lora16_20261126/train/complete.json`

其中尚不存在的文件表示相应阶段尚未完成，不是0分。真实模型和四卡smoke未通过之前不声称适配训练成功。正式结果产生后按[协议](PROTOCOL.md)报告完整原test开发BAcc，不与历史四人个人适配均值混比。
