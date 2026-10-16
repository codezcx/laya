"""Replay the completed JEV sampler exactly; count actual exposures, not corpus size."""
import json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('artifacts/jev_eeg/runs');MANIFEST=Path('artifacts/multitask_bridge_v30/source_cache/rows.jsonl')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    rows=[json.loads(s) for s in MANIFEST.read_text().splitlines()];groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':groups[(r['dataset'],r['class_id'])].append(i)
    total=sum(map(len,groups.values()));assert total==57749 and len(groups)==12
    results={}
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or not (folder/'config.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text());best=json.loads((folder/'best.json').read_text())['step']
        assert cfg['manifest_sha256']==sha(MANIFEST) and cfg['batch_size']==12
        # This audit applies only to the unchanged sampler implementation.
        for p,h in cfg['code_sha256'].items():assert sha(p)==h,(p,'source changed; audit historical code in a worktree')
        rng=np.random.default_rng(cfg['seed']);draws=[];snapshots={}
        for step in range(1,cfg['steps']+1):
            draws.extend(int(rng.choice(groups[k])) for k in sorted(groups))
            if step not in {cfg['warmup_steps'],best,cfg['steps']}:continue
            counts=Counter(draws);unique=set(counts);by_dataset={}
            for ds in sorted({r['dataset'] for r in rows}):
                possible=[i for key,ids in groups.items() if key[0]==ds for i in ids]
                seen=[i for i in unique if rows[i]['dataset']==ds]
                by_dataset[ds]=dict(available=len(possible),draws=sum(counts[i] for i in seen),unique=len(seen),coverage=len(seen)/len(possible))
            snapshots[str(step)]=dict(draws=len(draws),unique=len(unique),coverage=len(unique)/total,
                exposures_divided_by_corpus_size=len(draws)/total,repeated_draws=len(draws)-len(unique),by_dataset=by_dataset)
        results[folder.name]=dict(seed=cfg['seed'],selected_step=best,available_train_examples=total,snapshots=snapshots)
    (ROOT/'sampling_coverage_audit.json').write_text(json.dumps(dict(manifest_sha256=sha(MANIFEST),
        method='Exact NumPy default_rng replay of unchanged training sampler; no GPU work or target access.',runs=results),indent=2))
    for name,r in results.items():
        end=r['snapshots']['500'];selected=r['snapshots'][str(r['selected_step'])]
        print(name,'end_draws',end['draws'],'end_unique',end['unique'],'coverage',round(100*end['coverage'],2),
              'selected_step',r['selected_step'],'selected_unique',selected['unique'],'selected_coverage',round(100*selected['coverage'],2))

if __name__=='__main__':main()
