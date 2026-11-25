# LEAF 13库：JEV训练状态与原生参照

_更新：2026-11-25；原生LEAF与共享JEV已完成，旧少样本队列已停止；当前启动2a每人58条支持样本实验。_

## 📋 当前JEV运行

用户确认使用JEV-27B。运行目录：`artifacts/benchmark/leaf13_jev27b_20261124/`，代码`jev_train/`，配置[jev13_train.json](../configs/jev13_train.json)。13库预处理manifest与native LEAF完全共用；初始化、预算与评估角色差异见[协议](PROTOCOL.md)。

输入审计通过：144,553/47,286/53,363个原train/valid/test窗口，13库标签顺序核对无误。最初17项单元测试通过。原控制器PID1453443；13步四卡真实短测完成，训练分支梯度/四卡同步/样本分片/checkpoint重载检查全部通过，短测访问train208条，valid/test均为0。峰值已分配显存约54.57GiB/卡。

节点更换后从完整step1200断点继续，恢复优化器与四卡RNG；共享训练已完成2040步，选中step2040，完整checkpoint重载核验通过。原共享代码/配置哈希没有改变，global batch16及学习率预算未改。正式初始化与短测起点哈希一致，短测权重未继承。step0原test开发子集共6,232条，13库等权BAcc37.21%；最终同子集56.89%，完整原test 53,363窗口的13库等权BAcc57.14%、Acc57.48%。原test参与选点，因此是开发分数而非独立测试。共享训练的原valid访问计数为0；后续个人适配访问valid不追溯改变此事实。

完整结果：[step2040全量原test](../artifacts/benchmark/leaf13_jev27b_20261124/train/test_step002040_full.json)、[完成与重载核验](../artifacts/benchmark/leaf13_jev27b_20261124/train/complete.json)。57.14%与下方原生LEAF的49.59%在模型、初始化、优化与选点方式上均有差异，不能解释成单一模块带来7.55个百分点的收益。

实时状态：[pipeline](../artifacts/benchmark/leaf13_jev27b_20261124/pipeline.json)、[训练进度](../artifacts/benchmark/leaf13_jev27b_20261124/train/progress.json)、[短测完成审计](../artifacts/benchmark/leaf13_jev27b_20261124/smoke/complete.json)、[step0开发基线](../artifacts/benchmark/leaf13_jev27b_20261124/train/test_step000000_subset.json)。

| 可训练组 | 参数量 |
| --- | ---: |
| LEAF信号Tower实际forward | 19,557,632 |
| 两层MLP连接器 | 27,535,872 |
| JEV已有LoRA | 217,579,520 |
| 16行choice权重与偏置 | 81,936 |

前340步仅连接器更新，之后四组联合训练；原始27B语言主干权重保持冻结。

## 👤 个人适配队列

旧队列`leaf13_personal_20261124`已按用户要求停止，未重新挂起。已保存245份方法结果与26个完整用户×数据集方法组：2a 7组、OpenBMI-MI 8组、PhysioNet 8组、SSVEP 3组；部分完成checkpoint与日志保留。旧[队列](../artifacts/benchmark/leaf13_personal_20261124/queue.json)与[汇总](../artifacts/benchmark/leaf13_personal_20261124/summary.json)保持原协议，代码/配置归档Git `d3b6616`。

当前按用户“2a使用58条训练看看”于2026-11-25 06:56 UTC启动新队列，控制器PID2648675，四卡worker为2653276–2653279。27项单元测试和真实27B梯度/冻结/重载检查均通过，四卡正式residual适配均已保存step10/100、梯度有限且非零。仅A01/A02/A03/A07各有完整58条原valid T-session支持样本；每人沿用旧E-session query，数量分别58/57/57/58，不足58条的A04/A05/A06跳过。4人的不适配query logits已与原20条实验核对，最大绝对差均为0。当前设置、类别计数和固定训练参数见[协议](PROTOCOL.md)。尚无58条适配完成分数，后续进度以实时文件为准。

实时文件：[队列状态](../artifacts/benchmark/leaf13_personal_2a58_20261125/queue.json)、[支持与评估清单](../artifacts/benchmark/leaf13_personal_2a58_20261125/plan.json)、[控制器日志](../artifacts/benchmark/leaf13_personal_2a58_20261125/controller.log)。随后生成`smoke.json`、每卡`progress_gpuN.json`与`summary.json`。新汇总包含同人、同query的不适配/20条/58条比较；方法间只比较相同用户完整组。个人support反传、query不反传且不选点，原test不用于本轮个人适配；所有结论仍是见过被试的开发探索。

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
