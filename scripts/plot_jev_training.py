"""Plot a read-only snapshot of current JEV logs; never modify running training code."""
import json,math
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/jev_eeg_coverage/runs');OUT=Path('artifacts/jev_eeg_coverage/plots')
ARMS=[('projector_seed42','Projector only (seed 42)'),('lora_seed42','LoRA + projector (seed 42)')]

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6));summary={};snapshot={}
    colors=['#555555','#0072B2','#56B4E9','#D55E00']
    for (name,label),color in zip(ARMS,colors):
        records=[]
        path=ROOT/name/'metrics.jsonl'
        for line in path.read_text().splitlines() if path.exists() else []:
            try:records.append(json.loads(line))
            except json.JSONDecodeError:continue  # Last record may still be writing.
        snapshot[name]=records
        loss=[r for r in records if 'loss' in r];val=[r for r in records if 'full_source_validation' in r]
        axes[0].plot([r['step'] for r in loss],[r['loss'] for r in loss],color=color,label=label,lw=1.7)
        for ax,key in zip(axes[1:],['MI','emotion']):
            ax.plot([r['step'] for r in val],[100*r['full_source_validation'][key] for r in val],color=color,lw=1.7,marker='o',ms=4)
        summary[name]=dict(latest_loss=loss[-1] if loss else None,latest_validation=val[-1] if val else None,
                           selected=json.loads((ROOT/name/'best.json').read_text()) if (ROOT/name/'best.json').exists() else None)
        for filename in ['full_source_validation.json','external.json']:
            if (ROOT/name/filename).exists():summary[name][filename]=json.loads((ROOT/name/filename).read_text())
    axes[0].axvspan(0,50,color='#eeeeee',zorder=-1)
    axes[0].axhline(.5*(math.log(2)+math.log(3)),color='#888888',ls=':',lw=1)
    axes[0].text(4300,.5*(math.log(2)+math.log(3))+.07,'Task-only uniform CE = 0.896',ha='right',fontsize=8)
    axes[0].set(title='Training cross-entropy (logged batches)',ylabel='Loss',ylim=(0,None))
    for ax,chance,title in zip(axes[1:],[50,100/3],['MI: full source validation','Emotion: full source validation']):
        ax.axhline(chance,color='#888888',ls=':',lw=1)
        ax.set(title=title,ylabel='Subject / dataset macro balanced accuracy (%)',ylim=(0,100))
    for ax in axes:
        ax.set_xlabel('Optimizer step');ax.set_xlim(0,4380);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Actual JEV-27B-VL: pretrained LoRA active in every arm',fontsize=14)
    fig.tight_layout(rect=[0,.13,1,.93]);fig.savefig(OUT/'training_curves.png',dpi=180);fig.savefig(OUT/'training_curves.pdf');plt.close(fig)
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),validation_note='Complete source validation, 12510 windows, at coverage-round endpoints; not external performance',
                runs=summary)
    (OUT/'metrics_snapshot.json').write_text(json.dumps(output,indent=2))
    (OUT/'logs_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    for name,r in summary.items():
        loss=r['latest_loss'];val=r['latest_validation'];best=r['selected']
        print(name,'step',loss['step'] if loss else None,'loss',loss['loss'] if loss else None,
              'source_metrics',val['full_source_validation'] if val else 'pending first coverage round',
              'selected',best['step'] if best else None)

if __name__=='__main__':main()
