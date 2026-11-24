# LEAF Tower＋个人残差 adapter → 冻结 Qwen-JEV

当前实验：每位用户用自己的20条标注样本，联合更新残差adapter与Tower；比较Tower LoRA和Tower全参两种方式。JEV与原连接器全冻结，最终仍由JEV分类，不采用文本相似度读出。

## 📋 两组匹配实验

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

`pipeline.json`和`worker_*.json`记录实际状态；`episodes/*/support_history.json`记录loss；`last.pt`每10步保存信号侧权重，`adapter.pt`保存完成的联合权重。没有LM权重副本，不承诺中途优化器精确续训。全部46次完成并审计通过后生成`summary.md/json`，同时列原残差-only匹配对照。当前运行状态见[结果页](docs/RESULTS.md)。

## 📊 已完成对照与边界

原残差-only23人基线已完成：2a BAcc 50.87%→57.81%，PhysioMI 38.08%→42.54%；两库等权44.48%→50.18%（+5.70个百分点）。MEMO/SSF/AdaBN三组也已完成，产物和历史实现保留，但不与本轮混作一个方法。

个人适配使用授权support标签，query只评估；2a T→E session，PhysioMI前四→后两imagery run。这不是zero-shot，也不是原benchmark整份test评测。仅单seed、2a仅两名目标用户，不预设联合更新一定更好。

- [当前协议](docs/PROTOCOL.md)
- [结果与历史对照](docs/RESULTS.md)
- [Git归档](docs/ARCHIVE.md)

原残差-only入口归档在`34c11a8`（运行实现`513427d`）；新两组是同一个人适配方案的匹配控制。数据、权重、日志、环境及用户未提交的ECG/专利/图件等均保留。
