# LEAF 13库：JEV训练状态与原生参照

_更新：2026-11-24 18:23 UTC；100 epochs 与两个 checkpoint 的13库完整测试均已完成。_

## 📋 当前JEV运行

用户确认使用JEV-27B。运行目录：`artifacts/benchmark/leaf13_jev27b_20261124/`，代码`jev_train/`，配置[jev13_train.json](../configs/jev13_train.json)。13库预处理manifest与native LEAF完全共用；初始化、预算与评估角色差异见[协议](PROTOCOL.md)。

输入审计通过：144,553/47,286/53,363个原train/valid/test窗口，13库标签顺序核对无误。17项单元测试通过。四卡短测和正式训练准备启动；尚无本轮JEV指标。原test仅作为开发池，不视为独立测试。

## 📊 已完成的原生LEAF参照

运行目录：`artifacts/benchmark/leaf_native13_20261124/`。

- 官方预训练Tower权重校验通过，严格加载成功；原生LEAF总参数26,424,208。
- 13库全部551个库内被试预处理任务完成，0个错误；10项自动测试通过。
- 四卡3步训练短测通过，Tower与Q-Former梯度有限且非零。短测不是正式训练，分数不作为结果。
- 首次控制器因PhysioNet下载超时导致63项预处理失败而退出；补齐数据后，于17:23 UTC恢复流水线，复用已完成的488项，补跑63项成功。未修改训练源码、学习率、batch size、训练预算或划分。
- 控制器PID809268已正常结束，流水线状态为complete。正式训练100 epochs、14,200步、3336.98秒。最终全局23类验证Acc为51.10%，不是逐库测试BAcc。见[pipeline.json](../artifacts/benchmark/leaf_native13_20261124/pipeline.json)和[训练进度](../artifacts/benchmark/leaf_native13_20261124/train/progress.json)。
- 完整清单包含144,553个训练窗口、47,286个验证窗口、53,363个测试窗口；每epoch142步，计划100epochs。详细计数与划分以[manifest.json](../datasets/leaf13/manifest.json)和[初始化记录](../artifacts/benchmark/leaf_native13_20261124/train/initial_state.json)为准。
- 输出每库训练loss、验证loss与global-prototype accuracy。最低验证loss选中第50轮；第100轮也完整测试，两者均包含13库×3种指令，共39条结果和对应逐样本预测。每种设置共53,363个测试窗口，未缺库、未缺预测文件。

## 📊 完整直接推理结果

各数据集限定自身候选类别，计算BAcc后13库等权平均。未逐库FT，也不根据test重新选checkpoint。

| Checkpoint | Default BAcc（%） | Task BAcc（%） | Task+Targets BAcc（%） |
| --- | ---: | ---: | ---: |
| 第50轮，最低valid loss | 45.49 | 47.51 | 49.59 |
| 第100轮，最终模型 | 44.94 | 46.62 | 48.85 |

来源：[选中checkpoint完整测试](../artifacts/benchmark/leaf_native13_20261124/test_selected.json)、[最终checkpoint完整测试](../artifacts/benchmark/leaf_native13_20261124/test_final.json)。与论文16库50.04%的数据组合和部分预处理不同，不视为严格复现或可直接排名。

## 📦 补充数据

PhysioNet本轮所需R03–R14已补齐：109名被试，共1,308个EDF文件；逐文件EDF头与记录长度核验通过。24路预下载正常结束；这是结构/长度检查，不等同于官方SHA256校验。[下载状态](../artifacts/benchmark/leaf_native13_20261124/downloads/physionet_prefetch/status.json)。

BCIC-Upperlimb官方Track4清单49文件，总5,811,008,369字节；下载因网络中断失败，目前2个文件完成、47个失败，完整文件按OSF提供的SHA256核验。该库不属于本轮13库，不阻塞训练；[下载进度](../artifacts/benchmark/leaf_native13_20261124/downloads/track4/status.json)。官方来源为[2020 BCI竞赛数据页](https://brain.korea.ac.kr/bci2020/competition.php)。

BCIC-Speech的官方Track3当前只提供113字节readme，说明数据正在复核、后续重新开放；这不等于EEG已下载。现有第三方旧镜像尚未采用。SEED-V需通过[官方申请页](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/)取得授权链接，尚未开始下载。

## 📚 历史对照

原生Qwen8B＋Tower实验已经完成：完整原test开发集61,375窗口，九库等权BAcc43.63%，不是独立测试。本轮数据、划分、模型与损失均不同，不能直接横比来归因语言主干效果。

旧实现/全部历史结果见Git `5611b49`，旧产物未删除。当前[协议](PROTOCOL.md)与[归档](ARCHIVE.md)说明复现边界及恢复方式。
