# LEAF Tower → MedJev：当前状态与结果

2026-11-26。当前主线为13库共享监督训练，非元学习。

## 已完成

- MedJev固定版本源码、MIT许可和来源信息已纳入Git，不含模型权重、数据、环境。
- 接入23层Qwen语言主干、源第32层Q/K读出、零初始化残差MLP和LEAF信号Tower。
- 批量读出前向及反向与作者单样本实现一致；LoRA与PEFT一致；真实小型Qwen验证EEG拼接、padding和梯度。
- 31项自动测试通过，包括GPU/bfloat16小型Qwen的完整前反向，以及无下载标记但文件截断时的拒绝加载。小模型测试不代替完整权重的实际检查。
- 训练数据/评估成员与原13库配置保持一致；原test为开发评估，原valid不forward。
- 旧个人LoRA位置/rank和MAML实现归档在Git `c46748c` / `dffd139`；所有ignored运行产物保留。

## 尚未得到的新结果

尚无MedJev EEG准确率。正式权重目前未通过完整性检查，不能把“目录存在”或“aria2标记消失”视为下载完成。两个分片声明长度分别为5,329,398,688和3,990,429,408 bytes；初次检查实际只有约10MB和4MB，safetensors拒绝加载。随后文件重新预分配到完整长度，但aria2下载标记仍存在；以pipeline状态及官方SHA256核对结果为准。

控制器只在权重完整后执行真实模型检查、四卡smoke及正式训练。状态入口：

- `artifacts/benchmark/leaf13_medjev4b_20261126/pipeline.json`
- `artifacts/benchmark/leaf13_medjev4b_20261126/controller.log`
- `artifacts/benchmark/leaf13_medjev4b_20261126/weight_audit.json`
- `artifacts/benchmark/leaf13_medjev4b_20261126/model_preflight.json`
- `artifacts/benchmark/leaf13_medjev4b_20261126/train/metrics.jsonl`
- `artifacts/benchmark/leaf13_medjev4b_20261126/train/complete.json`

其中尚不存在的文件表示相应阶段尚未完成，不是0分。真实模型和四卡smoke未通过之前不声称适配训练成功。正式结果产生后按[协议](PROTOCOL.md)报告完整原test开发BAcc，不与历史四人个人适配均值混比。
