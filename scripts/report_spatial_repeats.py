import json
from pathlib import Path
import numpy as np
root=Path('artifacts/spatial_repeats_v7');external=json.loads((root/'external_results.json').read_text())['results'];old=json.loads(Path('artifacts/spatial_prototypes_v6/external_results.json').read_text())['results']
lines=['# v7：三种随机种子的完整复验','','任务给定分类；SEED-VII限Happy/Neutral/Sad。源验证只用于checkpoint选择。目标此前已暴露，以下是探索性复验，不是新的盲测。','','|模型|seed|源验证两任务macro|SEED-VII被试balanced accuracy|2b accuracy|','|---|---:|---:|---:|---:|'];aggregates={}
for arm in ['text_sigmoid','learned_ce']:
 values=[]
 for seed in [42,43,44]:
  name=f'spatial_v7_{arm}_s{seed}';p=Path('artifacts/spatial_prototypes_v6')/arm if seed==42 else Path('artifacts')/name/arm;b=json.loads((p/'best_metrics.json').read_text())
  e=old['seed_vii'][arm] if seed==42 else external['seed_vii'][name];m=old['2b'][arm] if seed==42 else external['2b'][name];v=[b['macro_task_accuracy'],e['subject_macro_balanced_accuracy'],m['accuracy']];values.append(v);lines.append(f'|{arm}|{seed}|'+ '|'.join(f'{x*100:.2f}%' for x in v)+'|')
 values=np.array(values)*100;aggregates[arm]={'mean':values.mean(0).tolist(),'sample_std':values.std(0,ddof=1).tolist()}
lines+=['','|模型|源验证均值±SD|SEED-VII均值±SD|2b均值±SD|','|---|---:|---:|---:|']
for arm,d in aggregates.items():lines.append('|'+arm+'|'+'|'.join(f'{m:.2f}±{s:.2f}%' for m,s in zip(d['mean'],d['sample_std']))+'|')
lines+=['','SD是三次初始化/采样的样本标准差，不是被试置信区间。所有seed共用相同数据划分，不证明跨划分稳健性；没有选择最好seed或报告目标调参后的最佳分数。普通头与文本头的差异不能视为语言理解的因果证据。']
text='\n'.join(lines)+'\n';(root/'RESULTS.md').write_text(text);Path('docs/SPATIAL_REPEATS_V7_RESULTS.md').write_text(text);print(text)
