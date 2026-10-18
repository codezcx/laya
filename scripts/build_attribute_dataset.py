"""Replace obsolete MI QA with same-trial OpenBMI pairs; preserve emotion bytes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
from eeg_tasks.attribute_questions import specifications, mi_specifications, paired_normalize, paired_answers, MI_CHANNELS

RAW = Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542')
MANIFEST = Path('artifacts/multitask_bridge_v30/source_cache/rows.jsonl')
ROOTS = {'train': Path('datasets/eeg_attribute_qa'), 'validation': Path('datasets/eeg_attribute_qa_validation')}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def emotion_files(root):
    return [root/'signals/emotion.npy', root/'emotion_sources.jsonl', *sorted((root/'questions').glob('emotion_*.jsonl'))]


def build(args):
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    selected = [(i,r) for i,r in enumerate(rows) if r['dataset']=='Lee2019_MI']
    assert len(selected)==9800
    subjects = [{r['subject'] for _,r in selected if r['split']==s} for s in ROOTS]
    assert not subjects[0] & subjects[1]
    args.stage.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for i,r in selected: groups[int(r['subject']),int(r['session'])+1].append((i,r))
    arrays, sources, handles, stats, examples, preserved, counters = {}, {}, {}, {}, {}, {}, Counter()
    specs = mi_specifications()
    try:
        for split, old in ROOTS.items():
            root=args.stage/split
            (root/'questions').mkdir(parents=True); (root/'signals').mkdir()
            preserved[split]={str(p.relative_to(old)):digest(p) for p in emotion_files(old)}
            for name in preserved[split]:
                os.link(old/name, root/name)  # read-only reuse, never mutate emotion payloads
            old_stats=json.loads((old/'statistics.json').read_text())
            stats[split]={k:v for k,v in old_stats.items() if k.startswith('emotion_')}
            examples[split]={k:v for k,v in json.loads((old/'examples.json').read_text()).items() if k.startswith('emotion_')}
            n=sum(r['split']==split for _,r in selected)
            arrays[split]=[np.lib.format.open_memmap(root/f'signals/mi_{segment}.npy',mode='w+',dtype='float32',shape=(n,3,400)) for segment in ('baseline','task')]
            sources[split]=(root/'mi_sources.jsonl').open('w')
            handles[split]={s['id']:(root/'questions'/f'{s["id"]}.jsonl').open('w') for s in specs}
            for s in specs: stats[split][s['id']]={'answers':Counter(),'datasets':{'Lee2019_MI':Counter()}}
        for group_index, ((subject,session), members) in enumerate(sorted(groups.items())):
            path=args.raw/f'session{session}'/f's{subject}'/f'sess{session:02d}_subj{subject:02d}_EEG_MI.mat'
            raw_sha=digest(path)
            raw=loadmat(path,variable_names=['EEG_MI_train'],squeeze_me=True,struct_as_record=False)['EEG_MI_train']
            assert int(raw.fs)==1000
            classes={int(k):str(v) for k,v in raw.__dict__['class']}
            assert classes=={1:'right',2:'left'}
            channels=list(raw.chan)
            x=np.asarray(raw.x[:,[channels.index(c) for c in MI_CHANNELS]],dtype=np.float64).T
            assert np.isfinite(x).all()
            # Process continuous data before slicing, avoiding artificial join/filter edges.
            x=sosfiltfilt(butter(4,[1,30],btype='bandpass',fs=1000,output='sos'),x,axis=-1)
            x=resample_poly(x,1,5,axis=-1)
            events=np.asarray(raw.t,dtype=np.int64)
            assert len(events)==100 and np.all(np.diff(events)>6500)
            event_seen=set()
            for source_index,row in members:
                meta=pickle.load(open(row['path'],'rb'))
                ordinal=int(meta['event_index'])
                assert ordinal==int(meta['trial'])
                assert ordinal not in event_seen; event_seen.add(ordinal)
                cue=int(events[ordinal]); label=0 if int(raw.y_dec[ordinal])==2 else 1
                assert meta['run']=='1train' and label==row['label']
                assert meta['label']==('left_hand' if label==0 else 'right_hand')
                assert abs(float(meta['annotation_onset'])-cue/1000)<1e-9
                assert meta['subject']==subject and int(meta['session'])==session-1
                center=int(round(cue/5))
                bounds=[[center-500,center-100],[center+100,center+500]]
                assert bounds[0][0]>=0 and bounds[1][1]<=x.shape[-1]
                pair,scale=paired_normalize(*(x[:,a:b] for a,b in bounds))
                split=row['split']; index=counters[split]; counters[split]+=1
                for array,segment in zip(arrays[split],pair): array[index]=segment
                sid=f'source-{source_index:06d}'
                provenance=dict(dataset='Lee2019_MI',subject=row['subject'],source_row_index=source_index)
                src=dict(sample_id=sid,signal_index=index,source_row_index=source_index,source_metadata=row,
                    raw_file=str(path.resolve()),raw_sha256=raw_sha,raw_structure='EEG_MI_train',
                    event_index=ordinal,legacy_trial_event_index=int(meta['trial_event_index']),cue_sample_1000hz=cue,event_convention='MOABB t used directly as raw sample index',
                    slice_bounds_200hz=bounds,shared_scale_uv=scale,label=label,
                    pair_sha256=hashlib.sha256(pair.tobytes()).hexdigest())
                sources[split].write(json.dumps(src)+'\n')
                answers=paired_answers(pair,label)
                for spec in specs:
                    answer,evidence=answers[spec['id']]
                    item=dict(id=f'{sid}:{spec["id"]}',sample_id=sid,split=split,question_type=spec['id'],
                        input=dict(segments=[dict(role=role,signal_file=f'signals/mi_{role}.npy',signal_index=index,
                            cue_relative_seconds=window) for role,window in [('baseline',[-2.5,-.5]),('task',[.5,2.5])]],
                            sample_rate=200,channels=MI_CHANNELS,question=spec['question'],options=spec['options']),
                        target=dict(answer=answer),evidence=evidence,provenance=provenance)
                    handles[split][spec['id']].write(json.dumps(item,ensure_ascii=False)+'\n')
                    stats[split][spec['id']]['answers'][answer]+=1
                    stats[split][spec['id']]['datasets']['Lee2019_MI'][answer]+=1
                    examples[split].setdefault(spec['id'],item)
            assert len(event_seen)==100
            print(f'{group_index+1}/{len(groups)} raw sessions; pairs {dict(counters)}',flush=True)
            del raw,x
    finally:
        for f in sources.values():f.close()
        for group in handles.values():
            for f in group.values():f.close()
        for group in arrays.values():
            for a in group:a.flush()
    for split,old in ROOTS.items():
        root=args.stage/split
        write_json(root/'question_catalog.json',specifications())
        write_json(root/'statistics.json',stats[split]); write_json(root/'examples.json',examples[split])
        shutil.copyfile('docs/ATTRIBUTE_DATASET.md',root/'README.md')
        assert all(digest(root/name)==sha for name,sha in preserved[split].items())
        emotion_n=sum(1 for _ in (root/'emotion_sources.jsonl').open())
        write_json(root/'complete.json',dict(schema_version=2,source_splits=[split],
            source_manifest=str(MANIFEST.resolve()),source_manifest_sha256=digest(MANIFEST),
            source_windows=counters[split]+emotion_n,mi_pairs=counters[split],emotion_windows=emotion_n,
            question_records=sum(sum(s['answers'].values()) for s in stats[split].values()),
            files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
            preserved_emotion_sha256=preserved[split],threshold_db=1.,threshold_fitting='none; descriptive tolerance, not significance',
            mi_preprocessing='Native reference, continuous 1–30 Hz fourth-order zero-phase Butterworth at 1000 Hz; polyphase resample to 200 Hz; per-segment/channel demean; one RMS over both segments/all channels; no clipping',
            mi_spectrum='Absolute Welch PSD sum; Hann200 overlap100 nfft200; alpha [8,13), beta [13,30]; delta=10log10(task/baseline)',
            cue_rounding_max_seconds=.002,baseline='Same-trial pre-cue fixation; not independently certified physiological rest',
            excluded_mi_datasets=['Cho2017','Dreyer2023'],
            code_sha256={p:digest(p) for p in ['scripts/build_attribute_dataset.py','eeg_tasks/attribute_questions.py']},
            input_contract='Only input enters model; target/evidence/provenance/source metadata excluded'))
        from scripts.verify_attribute_dataset import verify
        verify(root)
    if args.install:
        # The user explicitly authorized removal of obsolete MI QA. No raw/cache/checkpoint deletion.
        for split,root in ROOTS.items():
            stage=args.stage/split
            assert all(digest(root/name)==sha for name,sha in preserved[split].items())
            removed=[]
            (root/'BUILDING.json').write_text('{"reason":"installing verified paired MI"}\n')
            for p in [*sorted((root/'questions').glob('mi_*.jsonl')), *sorted((root/'signals').glob('mi*.npy')),root/'mi_sources.jsonl']:
                if p.exists():removed.append(str(p));p.unlink()
            for p in sorted(stage.rglob('*')):
                if p.is_file() and str(p.relative_to(stage)) not in preserved[split]:
                    os.replace(p,root/p.relative_to(stage))
            (root/'BUILDING.json').unlink()
            verify(root)
            write_json(args.stage/f'{split}_installation.json',dict(removed_obsolete_mi=removed,preserved_emotion_sha256=preserved[split]))
            shutil.rmtree(stage) # only staging hardlinks/empty dirs remain
        print('Installed verified paired MI; obsolete MI QA removed; emotion payloads unchanged.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=RAW)
    p.add_argument('--stage',type=Path,default=Path('artifacts/mi_paired_rebuild'))
    p.add_argument('--install',action='store_true')
    build(p.parse_args())
