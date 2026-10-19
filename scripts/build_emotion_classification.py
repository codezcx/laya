"""Add audited valence questions without changing existing signals or cache identities."""
import json
from collections import Counter
from pathlib import Path
from scipy.io import loadmat
from eeg_align.cache import ROOT, digest
from eeg_tasks.faced_events import read_table, video_events

SPEC = dict(id='emotion_class', task='emotion', family='classification',
    question='This EEG segment belongs to an emotion recognition task. Which emotional valence category does the recording represent?',
    options={'A':'Negative emotional valence.', 'B':'Neutral emotional valence.', 'C':'Positive emotional valence.'})


def main():
    manifest=[json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
    label_cache={};subjects={};reports={}
    roots={'train':Path('datasets/eeg_attribute_qa'),'validation':Path('datasets/eeg_attribute_qa_validation')}
    for split,root in roots.items():
        if (root/'classification_complete.json').exists() or (root/'questions/emotion_class.jsonl').exists():
            raise FileExistsError('Classification supplement already exists; refusing overwrite')
    for split,root in roots.items():
        sources=[json.loads(s) for s in (root/'emotion_sources.jsonl').read_text().splitlines()]
        template=json.loads(next((root/'questions/emotion_spectral_frontal.jsonl').open()))['input']
        output=[];counts=Counter();seen=set();subjects[split]=set();raw_hashes={}
        for src in sources:
            i=src['source_row_index'];r=manifest[i]
            assert i not in seen and r==src['source_metadata'] and r['split']==split
            seen.add(i);ds=r['dataset'];subjects[split].add((ds,r['subject']))
            if ds=='SEED':
                path=Path(r['path']).parent/'label.mat'
                if str(path) not in label_cache:
                    label_cache[str(path)]=loadmat(path)['label'].reshape(-1).astype(int)+1
                label=int(label_cache[str(path)][int(r['trial'])-1])
            elif ds=='FACED-NEMAR':
                path=Path(r['path']).with_name(Path(r['path']).name.replace('_eeg.bdf','_events.tsv'))
                if str(path) not in label_cache:
                    label_cache[str(path)]={e['video_index']:['negative','neutral','positive'].index(e['binary_label']) for e in video_events(read_table(path))}
                label=label_cache[str(path)][r['trial']]
            else:raise ValueError(ds)
            assert label==r['class_id'] and label in (0,1,2)
            if str(path) not in raw_hashes:raw_hashes[str(path)]=digest(path)
            inp={k:template[k] for k in ['sample_rate','channels']}
            inp.update(signal_file=src['signal_file'],signal_index=src['signal_index'],question=SPEC['question'],options=SPEC['options'])
            output.append(dict(id=src['sample_id']+':emotion_class',sample_id=src['sample_id'],split=split,
                question_type='emotion_class',input=inp,target=dict(answer='ABC'[label]),
                evidence=dict(label_source='dataset stimulus-assigned valence; not an inferred spectral label or self-report'),
                provenance=dict(dataset=ds,subject=r['subject'],source_row_index=i)))
            counts[ds,'ABC'[label]]+=1
        if split=='validation':assert not subjects['train'] & subjects['validation']
        target=root/'questions/emotion_class.jsonl'
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
        report=dict(schema_version=1,split=split,n=len(output),specification=SPEC,
            base_sha256=digest(root/'complete.json'),source_manifest_sha256=digest(ROOT/'rows.jsonl'),
            sources_sha256=digest(root/'emotion_sources.jsonl'),question_sha256=digest(target),
            raw_label_sha256=raw_hashes,code_sha256=digest(__file__),
            counts={ds:{a:counts[ds,a] for a in 'ABC'} for ds in sorted({k[0] for k in counts})},
            input_contract='Only shared EEG tokens and question/options enter model; no label, dataset or subject metadata',
            status='verified_against_original_label_files')
        (root/'classification_complete.json').write_text(json.dumps(report,indent=2)+'\n')
        reports[split]=dict(n=len(output),counts=report['counts'])
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()
