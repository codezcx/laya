"""Read-only full scan of remaining left/right exports; never promotes candidates automatically."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

ROOT=Path('/inspire/dataset/moabb-processed/v1/EEG-processed')
OUT=Path('artifacts/mi_dataset_audit')


def inspect(item):
    ds,path=item
    try:
        r=pickle.load(path.open('rb'));names={str(k).upper():k for k in r['eeg']}
        missing=[c for c in ['C3','CZ','C4'] if c not in names]
        channels=['C3','CZ','C4'] if not missing else list(names)
        x=np.stack([np.asarray(r['eeg'][names[c]],dtype=np.float32) for c in channels])
        std=x.std(-1);valid=bool(np.isfinite(x).all())
        return dict(dataset=ds,path=str(path),subject=str(r['subject']),session=str(r['session']),run=str(r['run']),
            label=r['label'],samples=x.shape[1],freq=r['freq'],unit=r['unit'],missing=missing,
            finite=valid,flat=bool((std<1e-8).any()),median_channel_std=float(np.median(std)) if valid else None,
            max_abs=float(np.max(abs(x))) if valid else None,
            schema=r.get('schema_version'),label_matches=r['label']==path.stem.rsplit('_label-',1)[-1],
            hash=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),hash_channels=channels,
            raw_name=r.get('raw_name'),start_time=r.get('start_time'),end_time=r.get('end_time'),
            input_scale_to_volts=r.get('input_scale_to_volts'),original_freq=r.get('original_freq'))
    except Exception as e:return dict(dataset=ds,path=str(path),error=f'{type(e).__name__}: {e}')


def main():
    OUT.mkdir(exist_ok=True)
    cfg=json.loads(Path('configs/mi_data.json').read_text());inv=json.loads(Path('datasets/mi_index/inventory.json').read_text())
    names=[r['dataset'] for r in inv if {'left_hand','right_hand'}<=r['labels'].keys() and r['dataset'] not in cfg['source_datasets']+cfg['heldout_datasets']]
    # Include known ME and alias directories: verify cross-dataset duplicate claims too.
    seen={};dups=[];summaries={}
    for path in Path('datasets/mi_index').glob('*.jsonl'):
        if path.stem in ['train','validation','heldout']:continue
        for line in path.read_text().splitlines():
            r=json.loads(line);seen[(('C3','CZ','C4'),r['signal_sha256'])]=r
    with ThreadPoolExecutor(max_workers=8) as pool, (OUT/'remaining_records.jsonl').open('x') as output:
        for ds in names:
            paths=sorted(p for p in (ROOT/ds).rglob('*.pkl') if p.stem.rsplit('_label-',1)[-1] in ['left_hand','right_hand'])
            records=[]
            for r in pool.map(inspect,((ds,p) for p in paths)):
                output.write(json.dumps(r)+'\n');records.append(r)
                if 'error'in r:continue
                key=(tuple(r['hash_channels']),r['hash'])
                if key in seen:
                    old=seen[key];oldlabel=old.get('label_name',old.get('label'))
                    dups.append(dict(dataset=ds,path=r['path'],duplicate_of=old['path'],other_dataset=old['dataset'],conflicting_label=r['label']!=oldlabel))
                else:seen[key]=r
            ok=[r for r in records if 'error'not in r];values=[r['median_channel_std'] for r in ok if r['finite']]
            summaries[ds]=dict(n=len(records),subjects=len({r['subject'] for r in ok}),labels=dict(Counter(r['label'] for r in ok)),
                errors=[r for r in records if 'error'in r],missing_channels=sum(bool(r['missing']) for r in ok),
                nonfinite=sum(not r['finite'] for r in ok),flat=sum(r['flat'] for r in ok),
                unit_counts=dict(Counter(r['unit'] for r in ok)),frequency_counts=dict(Counter(r['freq'] for r in ok)),
                samples=dict(Counter(r['samples'] for r in ok)),std_quantiles=np.quantile(values,[0,.01,.5,.99,1]).tolist() if values else [],
                label_mismatch=sum(not r['label_matches'] for r in ok),schema_counts=dict(Counter(r['schema'] for r in ok)),
                source_examples=sorted({str(r['raw_name']) for r in ok})[:3])
            print(ds,'n',len(ok),'missing',summaries[ds]['missing_channels'],'std',summaries[ds]['std_quantiles'],flush=True)
    report=dict(datasets=summaries,duplicates=dups,n_scanned=sum(v['n'] for v in summaries.values()),scope='Remaining left/right labelled export directories including aliases and execution controls; no training')
    (OUT/'remaining_audit.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['n_scanned'],'duplicates',len(dups),flush=True)


if __name__=='__main__':main()
