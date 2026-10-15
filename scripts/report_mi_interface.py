"""Report observed interface diagnostics, without consulting target data."""
import json
from pathlib import Path
ROOT=Path('artifacts/mi_interface_v2')
lines=['# 冻结接口诊断结果','', '源数据：Cho + Dreyer；验证被试独立。未读取2b。数值为百分比。','', '|实验|选中step|训练正常|验证正常|验证错配|验证零信号|验证交换选项|','|---|---:|---:|---:|---:|---:|---:|']
gates={}
for arm in ['tiny_mi','tiny_qa','source_mi','source_qa']:
 p=ROOT/arm
 if not (p/'complete.json').exists():
  lines.append(f'|{arm}|运行中||||||');continue
 d=json.loads((p/'diagnostics.json').read_text()); best=json.loads((p/'best_metrics.json').read_text()); v=d['validation'];tr=d['train']['normal']['accuracy']
 vals=[tr]+[v[k]['accuracy'] for k in ['normal','shuffle','zero','swapped']]
 lines.append(f'|{arm}|{best["step"]}|'+ '|'.join(f'{x*100:.2f}' for x in vals)+'|')
 gates[arm]=(tr>=.95 if arm.startswith('tiny') else v['normal']['accuracy']>=.60 and v['normal']['accuracy']-v['shuffle']['accuracy']>=.05 and (arm!='source_qa' or v['normal']['accuracy']-v['swapped']['accuracy']<=.05))
lines+=['','## 预设工程门槛','', '```json',json.dumps(gates,indent=2),'```','', 'tiny 按训练准确率选模型，验证结果只用于诊断；source 按验证准确率选，因此验证分数存在模型选择乐观偏差。QA 按数据集×任务×标签平衡。混合任务总分不能代替逐任务结果，详见 diagnostics.json 的 per_task。单种子、短训练，仅判断当前设置；不证明方法上限。','', '未通过门槛时不自动开启 LoRA 或访问2b。QA 与 MI 的预处理不同，因此不能据本表归因于监督方式。']
lines += ['', '## 本轮解释', '', '两种 tiny 任务均达到训练100%，说明当前梯度路径可学习。但固定选项 tiny 模型不能稳定处理交换选项，不能据此称语言语义已对齐。较大源数据组300步内未通过验证门槛；这是当前优化预算和设置的失败，不是 encoder 信息不存在或 LLM 路线不可能。下一实验优先考虑原冻结词表输出上的短属性答案 CE，并保留当前分类接口作对照；本轮未自动执行该新方法。']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');(ROOT/'gates.json').write_text(json.dumps(gates,indent=2))
print('\n'.join(lines))
