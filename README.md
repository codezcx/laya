# 运动想象 → REVE → JEV：语义预训练与选项绑定

当前进行两个匹配对照，每组两张 GPU。GPU0–1 用左右手交叉熵短训 REVE，GPU2–3 用冻结 Qwen3-Embedding-0.6B 的任务＋类别文本进行双向多正样本对比预训练。两组从同一 REVE-base 初始化，各训练一轮、使用相同训练顺序和优化器配置；源验证集选择最佳 encoder。随后分别冻结 encoder，接入同样初始化的 JEV，随机交换 A/B 左右手选项并同步交换监督标签。

训练49,193条；被试隔离验证12,508条；2b的6,520条只在最终 JEV 训练结束后测试。使用全部已认证 EEG 通道与完整 trial。原始数据与现有 checkpoint/log 不删除；Git 历史保留旧代码。

- [当前协议](docs/PROTOCOL.md)
- [结果与运行位置](docs/RESULTS.md)
- [数据审计](docs/MI_DATASETS.md)

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_mi_preconditioning.py
```

入口拒绝覆盖既有输出。每组自动依次执行短预训练和 JEV 训练，两组使用独立 GPU/端口。启动前已检查两阶段梯度、选项映射和最长输入 microbatch。JEV 全局 batch64，每卡 microbatch4，训练三轮，每200步和每轮结束做双顺序验证，按两种顺序的源数据集宏平均 BA 选模型。

- 管理状态：`artifacts/mi_preconditioning_pipeline.json`
- REVE：`artifacts/mi_precondition_{ce,contrast}/metrics.jsonl`
- JEV：`artifacts/mi_jev_precondition_{ce,contrast}/metrics.jsonl`

适配后的 REVE 特征首次读取时在线计算，随后缓存到主机内存；不会误用原始冻结 REVE 的 token 数值。独立旧实验与原始 token 缓存仍保留，用于匹配对照和数据清单校验。
