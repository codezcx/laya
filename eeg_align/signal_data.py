"""Subject-disjoint manifests and online EEG loading for joint LEAF/JEV SFT."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .benchmark_data import catalog,to_electrodes,resample,segment

ROOT=Path('datasets/benchmark/signals')


def cross_subject_groups(subjects,dataset,fold=0,seed=42):
    subjects=sorted(set(map(str,subjects)),key=lambda s:float(s))
    if len(subjects)<3:raise ValueError('Need separate train, validation and test subjects')
    if dataset in {'TUAB','SleepEDF'}:
        if not 0<=fold<10:raise ValueError('10-fold index must be in [0,9]')
        groups=np.array_split(np.random.default_rng(seed).permutation(subjects),10)
        test=set(groups[fold]);validation=set(groups[(fold+1)%10])
        protocol='subject_10fold_with_separate_validation_group'
    else:
        if not 0<=fold<len(subjects):raise ValueError('LOSO index out of range')
        test={subjects[fold]};validation={subjects[(fold+1)%len(subjects)]}
        protocol='leave_one_subject_out_with_separate_validation_subject'
    mapping={s:'test' if s in test else 'validation' if s in validation else 'train' for s in subjects}
    assert set(mapping.values())=={'train','validation','test'}
    return mapping,protocol


def make_manifest(fold=0):
    from .leaf_runtime import digest
    out=ROOT/f'fold{fold}';out.mkdir(exist_ok=False)
    rows={s:[] for s in ['train','validation','test']};summary={}
    for tid,task in enumerate(catalog()):
        ds=task['task_id'];folder=ROOT/ds;meta=json.loads((folder/'complete.json').read_text())
        assert meta['catalog_sha256']==digest('configs/benchmark_tasks.json')
        rr=[json.loads(v) for v in (folder/'rows.jsonl').read_text().splitlines()]
        mapping,protocol=cross_subject_groups([r['subject'] for r in rr],ds,fold)
        summary[ds]=dict(protocol=protocol,subjects=mapping,counts={},source_manifest_sha256=digest(folder/'complete.json'))
        for r in rr:r['split']=mapping[r['subject']];rows[r['split']].append(r)
        for split in rows:
            selected=[r for r in rr if r['split']==split]
            counts=Counter(r['label'] for r in selected)
            if set(counts)!=set(range(len(task['classes']))):raise ValueError((ds,split,counts))
            summary[ds]['counts'][split]=dict(n=len(selected),labels=dict(counts))
    for split,rr in rows.items():
        keys={(r['dataset'],r['subject']) for r in rr}
        for other,oo in rows.items():
            if other!=split:assert not keys & {(r['dataset'],r['subject']) for r in oo}
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (out/'complete.json').write_text(json.dumps(dict(fold=fold,datasets=summary,counts={s:len(r) for s,r in rows.items()},files_sha256={s:digest(out/f'{s}.jsonl') for s in rows}),indent=2))
    return out


def load_rows(split,fold=0,validation_cap=None):
    from .leaf_runtime import digest
    folder=ROOT/f'fold{fold}';done=json.loads((folder/'complete.json').read_text())
    assert done['files_sha256'][split]==digest(folder/f'{split}.jsonl')
    rows=[json.loads(v) for v in (folder/f'{split}.jsonl').read_text().splitlines()]
    if validation_cap:
        selected=[];rng=np.random.default_rng(42)
        for tid,task in enumerate(catalog()):
            rr=[r for r in rows if r['task']==tid]
            if len(rr)<=validation_cap:selected.extend(rr);continue
            for label in range(len(task['classes'])):
                group=[r for r in rr if r['label']==label]
                indices=rng.choice(len(group),min(len(group),max(1,validation_cap//len(task['classes']))),replace=False)
                selected.extend(group[i] for i in sorted(indices))
        rows=selected
    return rows


class SignalStore:
    def __init__(self):
        from .leaf_runtime import UP
        from leaf_datasets.electrode_unifier import ElectrodeUnifier
        self.arrays={};self.meta={};self.spatial={};self.keep={}
        unifier=ElectrodeUnifier('LEAF-ch65');aliases={k.upper():k for k in unifier.reverse_lookup}
        for tid,task in enumerate(catalog()):
            ds=task['task_id'];folder=ROOT/ds
            self.meta[tid]=json.loads((folder/'complete.json').read_text());meta=self.meta[tid]
            self.arrays[tid]=np.load(folder/'signals.npy',mmap_mode='r')
            first=meta['first_valid_index'];_,names,_=to_electrodes(self.arrays[tid][first:first+1],meta['channels'])
            keep=[i for i,c in enumerate(names) if c.upper() not in {'A1','A2'}];self.keep[tid]=keep
            canonical=[aliases[names[i].upper()] for i in keep]
            self.spatial[tid]=unifier.map_to_template(np.eye(len(keep),dtype=np.float32)[None],canonical)[0]

    def batch(self,rows):
        grouped=defaultdict(list)
        for i,r in enumerate(rows):grouped[r['task']].append(i)
        result=[]
        for tid,indices in grouped.items():
            meta=self.meta[tid];source=[rows[i]['source_index'] for i in indices]
            raw=np.array(self.arrays[tid][source],dtype=np.float32)
            xx,_,_=to_electrodes(raw,meta['channels']);xx=resample(xx[:,self.keep[tid]],meta['sampling_rate'])
            limits=np.percentile(xx,[.1,25,50,75,99.9],axis=(1,2));scale=limits[3]-limits[1]
            fallback=scale<=0;scale[fallback]=xx[fallback].std(axis=(1,2))
            if (scale<=0).any() or not np.isfinite(xx).all():raise ValueError('Invalid online signal batch')
            xx=(np.clip(xx,limits[0,:,None,None],limits[4,:,None,None])-limits[2,:,None,None])/scale[:,None,None]
            xx=np.einsum('ec,bct->bet',self.spatial[tid],xx).astype(np.float32)
            if meta['dataset']=='SEED':xx=np.clip(xx,-10,10)
            pieces=[np.pad(v,((0,0),(0,0),(0,(-v.shape[-1])%100))) for v in segment(xx)]
            result.append((tid,indices,pieces))
        return result
