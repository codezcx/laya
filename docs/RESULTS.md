# LEAF 原生13库复现状态

_更新：2026-11-24；本轮为部分复现，尚无正式准确率。_

## 📋 当前运行

运行目录：`artifacts/benchmark/leaf_native13_20261124/`。

- 官方预训练Tower权重校验通过，严格加载成功；原生LEAF总参数26,424,208。
- 13库各一名被试的预处理短测通过；6项自动测试通过。
- 四卡3步训练短测通过，Tower与Q-Former梯度有限且非零。短测不是正式训练，分数不作为结果。
- 控制器PID2567580已启动全量551个库内被试处理任务。当前阶段以[pipeline.json](../artifacts/benchmark/leaf_native13_20261124/pipeline.json)及[预处理进度](../datasets/leaf13/prepare_progress.json)为准；完成后自动重新初始化并训练100epochs。
- 输出每库训练loss、验证loss与global-prototype accuracy；完整测试尚未执行。

## 📦 补充数据

BCIC-Upperlimb官方Track4清单49文件，总5,811,008,369字节，已开始下载并按OSF提供的SHA256核验；[下载进度](../artifacts/benchmark/leaf_native13_20261124/downloads/track4/status.json)。官方来源为[2020 BCI竞赛数据页](https://brain.korea.ac.kr/bci2020/competition.php)。

BCIC-Speech的官方Track3当前只提供113字节readme，说明数据正在复核、后续重新开放；这不等于EEG已下载。现有第三方旧镜像尚未采用。SEED-V需通过[官方申请页](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/)取得授权链接，尚未开始下载。

## 📚 历史对照

原生Qwen8B＋Tower实验已经完成：完整原test开发集61,375窗口，九库等权BAcc43.63%，不是独立测试。本轮数据、划分、模型与损失均不同，不能直接横比来归因语言主干效果。

旧实现/全部历史结果见Git `5611b49`，旧产物未删除。当前[协议](PROTOCOL.md)与[归档](ARCHIVE.md)说明复现边界及恢复方式。
