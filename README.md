# TUAB → REVE → JEV：正常／异常脑电分类

当前唯一训练任务是TUAB记录级正常／异常二分类；没有运动想象、情绪或属性辅助损失。19通道、200 Hz，每条记录8个10秒窗口，冻结REVE提取每窗口209个512维token，经投影器与Qwen语言嵌入拼接，JEV输出A正常／B异常。聚合8个窗口的logits后计算一次记录级交叉熵。

使用原始本地JEV适配器与决策行，投影器重新初始化，不加载之前MI/情绪训练权重。冻结REVE和Qwen基座，更新投影器、JEV LoRA及两行决策头。

- [当前协议](docs/PROTOCOL.md)
- [当前结果与状态](docs/RESULTS.md)

```sh
PYTHONPATH=. laya/.venv-eeg/bin/python scripts/run_jev.py
```

入口按顺序完成数据预处理、四卡REVE缓存、简单分类头对照、四卡两步检查、5个epoch正式训练和最终测试。数据和输出目录已存在时拒绝覆盖。原始数据：`/inspire/dataset/tuab/v0/v3.0.1/edf`。生成数据：`datasets/tuab/`。日志和checkpoint：`artifacts/tuab_jev/`。

主要文件：`status.json`（运行阶段）、`prepare.log`、`encode_0.log`至`encode_3.log`、`smoke.log`、`run.log`、`run/metrics.jsonl`、`run/progress.json`。每步记录训练loss和记录级准确率；每50步及每个epoch末完整验证。模型选择用验证集平衡准确率，最终单独报告普通准确率、正常召回率、异常召回率和AUROC。

旧联合实验代码保存在Git历史`f83edcb`。旧数据、日志和checkpoint保留原路径，不作为清理对象。上一轮无属性损失实验按用户切换TUAB的要求停止，最后保存step100；记录在`artifacts/jev_no_attribute_loss/stopped_for_tuab.json`。这次任务与样本定义不同，不能用TUAB与MI的准确率差异单独证明MI数据不足。

验证代码：`PYTHONPATH=. laya/.venv-eeg/bin/python -m pytest tests -q`。
