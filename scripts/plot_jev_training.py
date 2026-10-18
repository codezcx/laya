"""Read-only loss and attribute-validation plot for the current run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('artifacts/jev_attribute_qa/run');out=root.parent/'plots'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for line in (root/'metrics.jsonl').read_text().splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:continue
    loss=[r for r in records if 'loss' in r]
    val=[r for r in records if 'validation' in r]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot([r['step'] for r in loss],[r['loss'] for r in loss],label='Weighted training CE')
    ax[0].plot([r['step'] for r in val],[r['validation']['loss'] for r in val],marker='o',label='Panel CE')
    ax[0].set(title='Attribute question loss',ylabel='Cross-entropy')
    for key in ['MI','emotion']:
        ax[1].plot([r['step'] for r in val],[100*r['validation'][key] for r in val],marker='o',label=key+' attributes')
    ax[1].axhline(100/3,color='gray',ls=':')
    ax[1].set(title='Source validation attributes (not class labels)',ylabel='Macro balanced accuracy (%)',ylim=(0,100))
    for a in ax:a.set_xlabel('Optimizer step');a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    (out/'snapshot.json').write_text(json.dumps(dict(last_training=loss[-1] if loss else None,
        last_validation=val[-1] if val else None),indent=2))
    print(out/'training_curves.png')


if __name__=='__main__':main()
