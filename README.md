# LEAF Tower → JEV：跨任务个人适配

_2026-11-25；只比较连接器LoRA与正确标签ICL，复用旧InfoNCE parent，不重训共享模型。_

## 📋 当前实验

2a全部9人；SEED、HMC、Nakanishi SSVEP各4人。每人100条支持样本、两个配对方法，共21个用户×任务、42份结果。2a和SSVEP按试次，SEED/HMC按窗口，不能混称100个独立试次。

四卡各常驻一份冻结JEV，动态领取独立用户；同一用户依次跑两个方法，每次重新初始化个人参数。LoRA保持rank4、100步、batch16/micro4、lr1e-4；ICL只做前向。不修改共享Tower、原连接器权重或JEV，不合并不同人的support。

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.plan
laya/.venv-eeg/bin/python -m personal.run --start
```

- [当前协议及样本隔离](docs/PROTOCOL.md)
- [运行与结果](docs/RESULTS.md)
- [配置](configs/personal13.json)
- [任务文本](configs/personal_tasks.json)
- [历史归档](docs/ARCHIVE.md)

输出：`artifacts/benchmark/leaf_personal_multitask_20261125/`。`plan.json`固定support/query ID、原split、源试次/时间隔离及共享训练暴露；`summary.json`按数据集、暴露组、原split分别汇总，不把已见与未见被试混成泛化分数。12小时是上限，有限队列完成即退出。

## ⚠️ 结果解释

2a A01–A05及SSVEP的query来自原共享训练池，是用户授权的探索，不是独立测试。SEED的人已见，但query是原test试次。HMC和2a A06–A09未进入本次共享训练池；历史选点/评估和LEAF发布权重暴露仍需披露。

上一轮四人九方法36份结果完整归档Git `8bc8e94`；所有原结果和权重保持不变。当前删除已淘汰的其他个人模块实现，仅保留两种方法与必要控制/测试。共享`jev_train/`和数据基础依赖保留；不恢复历史独立训练框架。

实验设计检查用于同人配对、源试次/时间隔离和暴露分组；资源及Transformers检查用于同环境加载和长上下文实测；Markdown技能用于协议与结果记录。相关技能引用见协议。
