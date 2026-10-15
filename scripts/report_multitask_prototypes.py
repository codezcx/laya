import json
from pathlib import Path
r=Path('artifacts/multitask_prototypes_v1')
lines=['# 多任务文本原型 v1：结果','', '模型仅训练SEED情绪与Cho/Dreyer运动想象；EEGMat暂未训练。以下为任务已知的分类准确率。','', '|模型|源验证任务macro|SEED新被试|Cho新被试|Dreyer新被试|Lee未训练数据集子集|','|---|---:|---:|---:|---:|---:|']
for arm in ['text_ce','text_sigmoid','random_ce','learned_ce']:
 b=json.loads((r/arm/'best_metrics.json').read_text());e=json.loads((r/arm/'evaluation.json').read_text());vals=[b['macro_task_accuracy']]+[e['test']['tasks'][d]['accuracy'] for d in ['SEED','Cho2017','Dreyer2023']]+[e['external_lee']['accuracy']]
 lines.append('|'+arm+'|'+'|'.join(f'{v*100:.2f}%' for v in vals)+'|')
lines +=['','源验证用于选checkpoint，不能当独立泛化证据。SEED此处是跨被试，不是跨数据集。Lee仅5个新被试、每人每类最多40个trial；不能替代完整Lee或2b评估。random_ce并不使用文本语义，其性能接近文本模型时，不能声称语言理解提升了泛化。','']
p=r/'external_seed_vii/results.json'
if p.exists():
 d=json.loads(p.read_text())['results'];lines+=['## SEED-VII独立数据集','', 'Happy/Neutral/Sad映射到positive/neutral/negative，非完整7分类。按被试计算balanced accuracy；各类trial数12/8/12，普通accuracy机会基线不等于多数类基线。','', '|模型|被试macro balanced accuracy|被试bootstrap95CI|','|---|---:|---|']
 for arm,v in d.items():
  n=v['normal'];ci=n['subject_balanced_95ci'];lines.append(f"|{arm}|{n['subject_macro_balanced_accuracy']*100:.2f}%|{ci[0]*100:.2f}–{ci[1]*100:.2f}%|")
else:lines+=['SEED-VII外部评估进行中，尚无结果。']
p=r/'external_2b/results.json'
if p.exists():
 lines+=['','## 2b完整E评估集','', '2840 trials；以下CI对应被试macro accuracy，而非pooled accuracy。REVE预训练包含2b，且本项目已有2b评估历史，不能称严格数据集未见。','', '|模型|Accuracy|被试macro accuracy|被试bootstrap95CI|','|---|---:|---:|---|']
 for arm,v in json.loads(p.read_text())['results'].items():
  n=v['tasks']['BNCI2014_004'];ci=n['subject_95ci']
  lines.append(f"|{arm}|{v['accuracy']*100:.2f}%|{n['subject_macro_accuracy']*100:.2f}%|{ci[0]*100:.2f}–{ci[1]*100:.2f}%|")
lines+=['','## v2：任务内文本原型去均值','', '保持数据划分不变；seed仅改变模型初始化及采样。先前目标结果已经查看，后续属于探索性迭代。此轮仅比较源验证，不追加SEED-VII/2b筛选。','', '|配置|源验证macro task accuracy|选中step|','|---|---:|---:|']
for cfg_path in sorted(Path('configs').glob('multitask_prototypes_v2_*.json')):
 cfg=json.loads(cfg_path.read_text());p=Path(cfg['output'])/cfg['arms'][0]/'best_metrics.json'
 if p.exists():
  b=json.loads(p.read_text());lines.append(f"|{cfg_path.stem}|{b['macro_task_accuracy']*100:.2f}%|{b['step']}|")
lines+=['','## 限制','', '单seed、少通道、有限窗口。Qwen只离线编码文本，不在EEG推理路径。没有证明未见任务泛化。任务未知联合argmax未校准，不计入成功标准。数据集/设备/参考方式可能造成任务识别捷径。REVE预训练包含2b；其他目标的预训练暴露尚未完全审计。后续迭代仍以源验证选择，不可根据目标成绩反复调参后宣称目标完全未见。']
(r/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
