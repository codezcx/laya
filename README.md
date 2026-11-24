# LEAF Tower → 语言决策主干

当前评估约定（2026-11-24）：原`train`只用于训练；原`test`用于开发过程评估、选择步数和比较方案；原`valid`待用户确定方案后再评估。划分名单和文件标签不交换，原test成绩明确标作开发/探索性结果。原valid已在历史实验中使用，不能再宣称为从未看过的盲测。新共享决策训练已实施此规则；已完成的4B保持原valid协议，不追溯改名。见[评估策略](configs/evaluation_policy.json)和[协议](docs/PROTOCOL.md)。

当前使用同一训练流程换成原生Qwen3-VL-8B-Instruct，复用InfoNCE对齐后的LEAF Tower、17库共享指令数据和原训练预算。只加载语言塔，新训练256→4096→4096连接器、语言LoRA-r32和A–P决策读出；不搬运27B JEV/ECG-R1权重，不进行教师蒸馏。8B尚无正式准确率结果；已完成4B完整原valid九库平均BAcc为42.68%，详情见[结果页](docs/RESULTS.md)。

原生8B从本地`CerebraGloss/DataEngine/models/Qwen3-VL-8B-Instruct`复制至`third_party/Qwen3-VL-8B-Instruct`，四份权重SHA256及配置/tokenizer Git blob哈希与官方仓库匹配。权重保持Git忽略，来源和固定revision见[配置](configs/language_backbones.json)。模型、资源、绘图和文档技能用于严格加载、容量检查和评估标签核对，引用保留在[协议](docs/PROTOCOL.md)。

```bash
laya/.venv-eeg/bin/python scripts/run_benchmark.py --start --backbone qwen3vl8b --arm leaf_infonce_mlp
```

输出`artifacts/benchmark/leaf_tower_qwen8b_20261124/`；先权重核验、全训练输入预检和四卡短测，再从相同初始化正式训练2040步。step0和每340步用九库固定原test子集选点（每库最多512），最后跑完整原test；不执行原valid前向。LR、batch、数据划分不变。8B开发成绩与4B原valid成绩不是同一评价口径，不能直接据此声称模型更好。4B实现归档在Git `ad4c054`。

## 📋 已完成：个人联合适配对照

每位用户用自己的20条标注样本，联合更新残差adapter与Tower；比较Tower LoRA和Tower全参两种方式。JEV与原连接器全冻结，最终仍由JEV分类，不采用文本相似度读出。以下个人协议与新共享SFT对照分开记录。

| 组别 | Tower更新 | adapter更新 | 个人可训练参数 | GPU |
| --- | --- | --- | ---: | --- |
| joint_lora | 双分支最后两层Q/V LoRA-r8，32,768参数 | 33,088参数 | 65,856 | 0/1 |
| joint_full | forward Tower全参，19,557,632参数 | 33,088参数 | 19,590,720 | 2/3 |

两组各23名原测试用户：2a 8、9；PhysioMI 89–109。每人每类5条，seed42，100步；AdamW LR1e-4、wd0.01、global clip1，有效batch16=micro4×4。Tower和adapter使用相同LR，不扫描参数。BN统计量及dropout策略不变。

逐用户重置到共享InfoNCE→Tower→JEV best step1360，加全新恒等残差adapter；不从已经训练100步的个人adapter继续叠加训练。两卡分担同一组的12/11名用户，不合并用户梯度，不做源用户共享重训。

## ⚙️ 运行与产物

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python scripts/run_personalization.py --start
# 意外中断后，跳过完整用户，未完整用户从父模型重跑：
laya/.venv-eeg/bin/python scripts/run_personalization.py --start --resume
```

根目录：`artifacts/benchmark/leaf_personal_joint_20261124/`。输入核对及四卡验证用户3步短测通过后，自动进入46次正式个人训练。

2026-11-24 11:39:00 UTC，46次正式个人适配已全部完成并通过审计。运行实现`98d257d`，每组两卡分别处理12/11人；两库等权BAcc：残差-only 50.18%、联合LoRA 50.44%、联合全参49.62%。见[完整汇总](artifacts/benchmark/leaf_personal_joint_20261124/summary.md)。

`pipeline.json`和`worker_*.json`记录实际状态；`episodes/*/support_history.json`记录loss；`last.pt`每10步保存信号侧权重，`adapter.pt`保存完成的联合权重。没有LM权重副本，不承诺中途优化器精确续训。已生成`summary.md/json`，同时列原残差-only匹配对照。当前运行状态见[结果页](docs/RESULTS.md)。

## 📊 已完成对照与边界

原残差-only23人基线已完成：2a BAcc 50.87%→57.81%，PhysioMI 38.08%→42.54%；两库等权44.48%→50.18%（+5.70个百分点）。MEMO/SSF/AdaBN三组也已完成，产物和历史实现保留，但不与本轮混作一个方法。

个人适配使用授权support标签，query只评估；2a T→E session，PhysioMI前四→后两imagery run。这不是zero-shot，也不是原benchmark整份test评测。仅单seed、2a仅两名目标用户，不预设联合更新一定更好。

- [当前协议](docs/PROTOCOL.md)
- [结果与历史对照](docs/RESULTS.md)
- [不同 EEG 时长的推理延迟](artifacts/benchmark/inference_latency_20261124/summary.md)：单H100、batch1，0.5–30秒完整计算延迟约268–315ms；当前Qwen快速kernel缺失，未做部署优化。
- [本地Qwen3-VL-8B主干速度对照](artifacts/benchmark/inference_latency_qwen8b_20261124/summary.md)：同一信号侧，4秒输入约33.09ms（当前27B约269.49ms）；小模型连接器未训练，只比较速度，不报告EEG准确率。
- [Git归档](docs/ARCHIVE.md)
- [原生LEAF完整推理测速](artifacts/benchmark/inference_latency_leaf_native_20261124/summary.md)：发布权重＋原Q-Former＋缓存文本余弦分类；4秒FP32约9.81ms，BF16 autocast约12.62ms。没有在线运行MPNet或Qwen，不是等价LM能力对照。

原残差-only入口归档在`34c11a8`（运行实现`513427d`）；新两组是同一个人适配方案的匹配控制。数据、权重、日志、环境及用户未提交的ECG/专利/图件等均保留。
