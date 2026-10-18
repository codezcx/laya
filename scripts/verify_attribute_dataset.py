"""Verify hashes, coverage, paired provenance and independently recompute QA labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.build_attribute_dataset import digest, write_json


def fft_power(x):
    """Direct FFT implementation, independent of scipy.signal.welch."""
    x=np.asarray(x,dtype=np.float64)
    spectra=[]
    for start in range(0,x.shape[-1]-199,100):
        segment=x[...,start:start+200]
        segment=(segment-segment.mean(-1,keepdims=True))*np.hanning(201)[:-1]
        p=np.abs(np.fft.rfft(segment,axis=-1))**2
        p[...,1:-1]*=2
        spectra.append(p/(200*np.sum(np.hanning(201)[:-1]**2)))
    return np.mean(spectra,axis=0)


def verify(root):
    root=Path(root); complete=json.loads((root/'complete.json').read_text())
    assert complete['schema_version']==2
    assert not (root/'BUILDING.json').exists()
    for name,sha in complete['files'].items():assert digest(root/name)==sha,name
    for name,sha in complete['preserved_emotion_sha256'].items():assert digest(root/name)==sha,name
    assert digest(complete['source_manifest'])==complete['source_manifest_sha256']
    original=[json.loads(s) for s in Path(complete['source_manifest']).read_text().splitlines()]
    split,=complete['source_splits']
    expected={i for i,r in enumerate(original) if r['split']==split and r['dataset']=='Lee2019_MI'}
    sources=[json.loads(s) for s in (root/'mi_sources.jsonl').read_text().splitlines()]
    assert {s['source_row_index'] for s in sources}==expected
    assert len(sources)==len(expected)==complete['mi_pairs']
    baseline=np.load(root/'signals/mi_baseline.npy',mmap_mode='r')
    task=np.load(root/'signals/mi_task.npy',mmap_mode='r')
    assert baseline.shape==task.shape==(len(sources),3,400)
    for i,s in enumerate(sources):
        assert s['signal_index']==i and s['source_metadata']==original[s['source_row_index']]
        center=int(round(s['cue_sample_1000hz']/5))
        assert s['slice_bounds_200hz']==[[center-500,center-100],[center+100,center+500]]
        pair=np.stack([baseline[i],task[i]])
        assert np.isfinite(pair).all() and abs(float(np.mean(pair**2))-1)<1e-6
        assert np.max(np.abs(pair.mean(-1)))<1e-6
        assert hashlib.sha256(pair.tobytes()).hexdigest()==s['pair_sha256']
    # Batch all MI spectra; independent FFT labels for every trial/question.
    p0,p1=fft_power(baseline),fft_power(task)
    band_slices={'alpha':slice(8,13),'beta':slice(13,31)}
    delta={b:10*np.log10(p1[...,sl].sum(-1)/p0[...,sl].sum(-1)) for b,sl in band_slices.items()}
    emotion=np.load(root/'signals/emotion.npy',mmap_mode='r')
    catalog=json.loads((root/'question_catalog.json').read_text())
    stats=json.loads((root/'statistics.json').read_text())
    total=0; emotion_recomputed=0
    for spec in catalog:
        seen=set();counts=Counter()
        with (root/'questions'/f'{spec["id"]}.jsonl').open() as f:
            for j,line in enumerate(f):
                item=json.loads(line); inp=item['input']; answer=item['target']['answer']
                assert item['split']==split and inp['question']==spec['question']
                assert item['id'] not in seen;seen.add(item['id']);counts[answer]+=1
                if spec['task']=='mi':
                    assert inp['options']==spec['options']
                    assert set(inp)=={'segments','sample_rate','channels','question','options'}
                    assert inp['sample_rate']==200 and inp['channels']==['C3','Cz','C4']
                    i=inp['segments'][0]['signal_index'];assert i==j
                    for seg,role,window in zip(inp['segments'],['baseline','task'],[[-2.5,-.5],[.5,2.5]]):
                        assert seg==dict(role=role,signal_file=f'signals/mi_{role}.npy',signal_index=i,cue_relative_seconds=window)
                    assert item['sample_id']==sources[i]['sample_id']
                    assert item['provenance']['source_row_index']==sources[i]['source_row_index']
                    if spec['family']=='classification':expected_answer='AB'[sources[i]['label']]
                    else:
                        d=delta[spec['band']][i]
                        value=float(d[['C3','Cz','C4'].index(spec['channel'])]) if spec['family']=='baseline_change' else float(d[0]-d[2])
                        np.testing.assert_allclose(value,item['evidence']['decision_value_db'],rtol=1e-8,atol=1e-8)
                        expected_answer='A' if value < -1 else 'C' if value > 1 else 'B'
                    assert answer==expected_answer
                elif j%997==0: # emotion payloads hash-identical; retain independent spot checks
                    x=emotion[inp['signal_index']];channels=inp['channels']
                    values=[]
                    for side in [spec['first'],spec['second']]:
                        window={'full':slice(None),'early':slice(0,300),'late':slice(300,600)}[side['window']]
                        p=fft_power(x[[channels.index(c) for c in side['channels']],window])
                        values.append((p[...,band_slices[side['band']]].sum(-1)/p[...,1:31].sum(-1)).mean())
                    db=10*np.log10(max(values[0],1e-12)/max(values[1],1e-12))
                    assert answer==('A' if db>1 else 'C' if db < -1 else 'B')
                    emotion_recomputed+=1
        assert counts==Counter(stats[spec['id']]['answers'])
        assert len(seen)==(len(sources) if spec['task']=='mi' else len(emotion))
        total+=len(seen)
    assert total==complete['question_records']
    write_json(root/'verification.json',dict(status='passed',complete_sha256=digest(root/'complete.json'),
        mi_pairs=len(sources),mi_questions_recomputed=len(sources)*7,
        emotion_payloads_hash_unchanged=True,emotion_questions_recomputed=emotion_recomputed,
        question_records=total))
    print(f'PASS {root}: {len(sources)} paired MI trials; {total} QA records',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);verify(p.parse_args().root)
