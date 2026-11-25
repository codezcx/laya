# LEAF Tower → JEV：每人100条个人适配

_2026-11-25；用户确认复用旧InfoNCE＋LEAF Tower→JEV checkpoint，不再使用13库混合场次的共享权重。_

## 📋 当前运行

A06–A09每人从完整T场次取100条（每类25条）做个人适配，E场次完整288条仅评估。原共享训练仅使用A01–A05，A06/A07为原valid，A08/A09为原test。保留原split成员和名称，两组结果分开汇总；历史验证选点及测试观察不构成新的盲测。

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m personal.plan
laya/.venv-eeg/bin/python -m personal.run --start
```

当前parent：`artifacts/benchmark/leaf_tower_contrastive_20261123/train/leaf_infonce_mlp/best.pt`，step1360。个人输出：`artifacts/benchmark/leaf_infonce_personal_2a100_20261125/`。四卡各常驻一个冻结模型，每人优先跑完整9种方法/对照；完成即退出，12小时仅为上限。

- [当前协议与边界](docs/PROTOCOL.md)
- [运行状态与结果](docs/RESULTS.md)
- [当前配置](configs/personal13.json)
- [历史归档](docs/ARCHIVE.md)

## ⚙️ 不变的训练关系

LEAF信号Tower、原连接器、JEV全部共享参数冻结；只更新每人的残差adapter、soft prompt、连接器增量LoRA或KV prefix。保留CE、KD/组合及两种ICL对照。100步、batch16/micro4、lr1e-4、seed42不变；未额外对比训练、未重训共享模型。固定step100评估，不用个人query选点。

恢复旧22通道200Hz缓存、逐trial裁剪/median-IQR归一化、65通道插值和旧任务文本；不读13库已归一化HDF5。A08/A09强制检查不适配预测复现旧基线，失败则停止。代码`personal/`只保留本轮协议；`jev_train/`用于已有模型加载/决策等共享实现，13库共享训练不再启动。

## ⚠️ 比较边界与归档

主要比较同一人的同一批288条query上，不适配与适配后的Acc/BAcc。旧20条CE实验使用类别均衡batch和固定选项，本轮沿用最近9方法框架的随机抽样和选项重排，不能把旧20条与本轮100条的差值全部归因于样本量。

原58条队列已按用户要求停止，4份residual结果和未完成断点保留，不自动恢复；其源码/配置归档Git `9955da9`。13库57.14%是不同协议的共享开发分数，不作为本轮个人适配基线。数据、权重、日志和环境未删除。

实验设计技能用于按被试/场次隔离及同人配对，资源和Transformers技能用于四卡与旧权重加载检查，Markdown技能用于记录当前协议和实际运行状态。
