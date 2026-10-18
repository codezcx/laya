"""Read-only loss and attribute-validation plot for the current run."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('artifacts/jev_paired_qa/run');out=root.parent/'plots'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for line in (root/'metrics.jsonl').read_text().splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:continue
    loss=[r for r in records if 'loss' in r]
    val=[r for r in records if 'validation' in r]
    config=json.loads((root/'config.json').read_text())
    masses=config['task_loss_mass']
    uniform_ce=(masses['mi_classification']*math.log(2)+
                (masses['mi_attributes']+masses['emotion_attributes'])*math.log(3))
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot([r['step'] for r in loss],[r['loss'] for r in loss],label='Weighted training CE')
    ax[0].plot([r['step'] for r in val],[r['validation']['loss'] for r in val],marker='o',label='Unweighted panel CE')
    ax[0].axhline(uniform_ce,color='gray',ls='--',label=f'Uniform training CE ({uniform_ce:.4f})')
    if val:
        panel=val[-1]['validation']['questions']
        panel_uniform=sum(q['n']*math.log(2 if name=='mi_class' else 3)
                          for name,q in panel.items())/sum(q['n'] for q in panel.values())
        ax[0].axhline(panel_uniform,color='gray',ls=':',label=f'Uniform panel CE ({panel_uniform:.4f})')
    ax[0].set(title='Training and validation use different loss weights',ylabel='Cross-entropy')
    for key in ['MI','MI_classification','emotion']:
        ax[1].plot([r['step'] for r in val],[100*r['validation'][key] for r in val],marker='o',label=key)
    ax[1].axhline(100/3,color='gray',ls=':',label='3-choice chance')
    ax[1].axhline(50,color='gray',ls='--',label='MI classification chance')
    ax[1].set(title='MI attributes / MI classification / emotion attributes',ylabel='Macro balanced accuracy (%)',ylim=(0,100))
    for a in ax:a.set_xlabel('Optimizer step');a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    (out/'snapshot.json').write_text(json.dumps(dict(last_training=loss[-1] if loss else None,
        last_validation=val[-1] if val else None,uniform_training_ce=uniform_ce),indent=2))
    print(out/'training_curves.png')


if __name__=='__main__':main()
