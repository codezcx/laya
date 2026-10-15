# REVE → Qwen 阶段诊断协议

2026-10-15，用户授权排查，并已授权执行现有本地 REVE 自定义代码。只冻结读取既有最佳 REVE→Qwen checkpoint，训练独立诊断探针；不改原模型或原始划分。

## 事先固定的比较

- 沿用 `artifacts/reve_mi_multi/train_4gpu_20261015_121912/lora/best`，不是 JEV-27B。
- 四个位置：REVE token（1536）、connector token（12288）、Qwen 最后一层 EEG token（12288）、Qwen 最后有效输入位置（4096）。前三者都每通道沿时间取均值再拼接；最后一个与原分类头的输入位置一致。
- 所有模型参数冻结，eval，已有 forward 上安装只读 hook。使用原 FP32 REVE / BF16 Qwen 路径；独立检查原 head logits 可被抓取的 answer state 精确重建。
- 使用原 manifest，运行前后核对 SHA256、被试和样本隔离。训练 33272、验证 4160、内部测试 4200、外部 2a 1296、2b 2840；所有任务为左右手二分类。
- 按 manifest 顺序提取，4 张 GPU 分片，完整覆盖不重复；测试特征可提前缓存，但不用于任何探针拟合、标准化、正则化选择或停止决策。
- 所有探针使用训练集逐特征均值/标准差（std < 1e-6 时取 1）；相同 dataset×class 均衡样本权重。
- 标量 logistic 线性读出，含 bias。FP64 full-batch L-BFGS、strong-Wolfe，最多 300 次迭代；记录最终梯度、目标函数及是否达到迭代上限。
- 目标：均衡加权平均 BCE + λ/2 × 权重平方和，bias 不惩罚。固定 λ = [1e-4, 1e-3, 1e-2, 0.1, 1]。
- 每个位置只按验证集三个数据集准确率宏平均选 λ，平局按验证 log loss；保存 selection 后才计算测试指标。
- 除原分类头重跑外，四组都是新拟合的诊断读出，不能称作原方法的新准确率，也不是最终新模型。

## 解释限制

不同位置的维度不同，统一训练规则不等于完全统一容量；时间均值也可能丢失可非线性读取的信息。线性探针能检验线性可读性，不能证明所有信息被消除。现有 connector/Qwen 已用源域训练，不能把后续新探针当作端到端重训的公平架构排名。

BF16 在不同 batch size 下可能有数值差异，原模型重跑结果会与历史指标同时报告。预训练重叠与上游来源限制继承原协议；这是查看过 2b 结果之后进行的探索性诊断，不是新的盲测。

### 数值收敛复核（第一轮之后添加）

首轮 300 次迭代中 connector 的选中探针最大梯度为 2.28e-4、达到上限，部分低正则候选也未充分收敛。
因此固定原五档 λ 和原选择规则，对所有四个位置统一使用最多 2000 次迭代重算，输出到 `converged/`。
这是由优化器诊断触发的数值复核；不根据 2b 分数增加参数候选、不改变选择规则。首轮输出完整保留。

## 运行

```sh
bash scripts/run_stage_diagnostic.sh artifacts/reve_stage_diagnostic/run_20261015_1444 --allow-local-reve-code
```

## 流程参考

资源检查、hugging-science 和 transformers 技能用于本地加载、冻结及提取核验；科学资源目录没有 EEG 条目，不下载替代模型。
Kassis T, Agarwal V, He Y, Patel D, Brueckner AM (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 （2026-10-15 核验最新记录 v2；流程引用，不是模型有效性的证据）。
