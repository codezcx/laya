"""Cache frozen task-conditioned LEAF queries, preserving all benchmark epochs."""
import argparse,json,sys,pickle,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_align.benchmark_data import catalog,load_pickle,to_electrodes,resample,segment,subject_split
from eeg_align.leaf_data import CONFIG,UP,digest
from LEAF import LEAF
from load_config import build_model_config,load_yaml
from leaf_datasets.electrode_unifier import ElectrodeUnifier
OUT=ROOT/'datasets/benchmark'


def read_dataset(task):
    source=Path(task['source'])
    if source.is_file():
        d=load_pickle(source)
        return d.eeg_data,np.asarray(d.labels,dtype=int),np.asarray(d.subject_ids).astype(str),d.channel_names,float(d.sampling_rate)
    signals=[];labels=[];subjects=[];names=None
    options=[c['option'] for c in task['classes']]
    for path in sorted(source.rglob('*.pkl')):
        d=pickle.load(path.open('rb'));ch=list(d['eeg'])
        if names is None:names=ch
        assert ch==names and int(d['freq'])==200
        signals.append(np.stack([d['eeg'][c] for c in names]));labels.append(options.index(f"{float(d['label']):.2f} Hz"));subjects.append(str(d['subject']))
    # Exported Nakanishi epochs differ by +/-1 sample (829/830/831).
    # Retain all observed samples and pad only the missing tail, never crop.
    lengths=[v.shape[-1] for v in signals];longest=max(lengths)
    if longest-min(lengths)>2:raise ValueError(f'Unexpected Nakanishi duration range: {min(lengths)}..{longest}')
    signals=[np.pad(v,((0,0),(0,longest-v.shape[-1]))) for v in signals]
    return np.stack(signals),np.asarray(labels),np.asarray(subjects),names,200.


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');p.add_argument('--gpu',type=int,default=0);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(a.gpu)
    model=LEAF(build_model_config(load_yaml(UP/'configs/LEAF_mpnet.yaml')))
    model.load_state_dict(torch.load(CONFIG['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.cuda().eval().requires_grad_(False)
    text_info=json.loads((ROOT/'artifacts/benchmark/instructions.json').read_text());text=np.load(ROOT/'artifacts/benchmark/instructions.npy')
    unifier=ElectrodeUnifier('LEAF-ch65');aliases={k.upper():k for k in unifier.reverse_lookup}
    tasks=catalog();OUT.mkdir(parents=True,exist_ok=True)
    for tid,task in enumerate(tasks):
        ds=task['task_id']
        if a.datasets and ds not in a.datasets:continue
        dest=OUT/ds;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            done=json.loads((dest/'complete.json').read_text());assert done['catalog_sha256']==digest(ROOT/'configs/benchmark_tasks.json');continue
        start=time.monotonic();x,y,subjects,names,fs=read_dataset(task)
        groups=np.array(['01' if ds=='CHB_MIT' and s=='21' else s for s in subjects])
        splits=subject_split(groups)
        # Determine matrix once. Missing bipolar measurements vary per epoch and
        # are handled before this spatial interpolation.
        probe,raw_names,_=to_electrodes(x[:1],names)
        keep=[i for i,c in enumerate(raw_names) if c.upper() not in {'A1','A2'}]
        canonical=[aliases[raw_names[i].upper()] for i in keep]
        spatial=unifier.map_to_template(np.eye(len(keep),dtype=np.float32)[None],canonical)[0]
        samples=int(np.ceil(x.shape[-1]*200/fs));chunks=int(np.ceil(samples/2000));tokens=8*chunks
        features=np.lib.format.open_memmap(dest/'queries.npy',mode='w+',dtype=np.float16,shape=(len(x),tokens,256))
        inst=torch.from_numpy(text[text_info['texts'].index(task['instruction'])]).cuda()[None]
        accepted=np.ones(len(x),dtype=bool);excluded=[]
        report=dict(nakanishi_tail_padding='0-2 samples to 831 at 200Hz; all original samples retained' if ds=='Nakanishi2015' else None,iqr_fallback_count=0,dataset=ds,source=task['source'],n=len(x),original_channels=names,excluded_reference_electrodes=[c for c in raw_names if c.upper() in {'A1','A2'}],sampling_rate=fs,tokens=tokens,subjects=splits,split_policy='70/15/15 subject holdout seed42, NOT official LOSO/fewshot; chb01/chb21 grouped',catalog_sha256=digest(ROOT/'configs/benchmark_tasks.json'),checkpoint_sha256=digest(CONFIG['checkpoint']),normalization='Per complete epoch robust clipping and median/IQR; labels and other epochs never used',max_bipolar_residual=0.,missing_lead_instances=0)
        with (dest/'rows.jsonl').open('w') as rows:
            for offset in range(0,len(x),a.batch):
                end=min(len(x),offset+a.batch);raw=x[offset:end];good=np.isfinite(raw).all(axis=(1,2)) & np.any(raw!=0,axis=(1,2))
                accepted[offset:end]=good
                excluded.extend(dict(source_index=offset+int(j),reason='all_zero_or_nonfinite') for j in np.flatnonzero(~good))
                ids=np.arange(offset,end)[good]
                if not len(ids):continue
                xx,_,audit=to_electrodes(raw[good],names)
                report['max_bipolar_residual']=max(report['max_bipolar_residual'],audit.get('relative_residual',0.))
                report['missing_lead_instances']+=audit.get('missing_lead_instances',0)
                xx=resample(xx[:,keep],fs)
                limits=np.percentile(xx,[.1,25,50,75,99.9],axis=(1,2));iqr=limits[3]-limits[1]
                fallback=iqr<=0;report['iqr_fallback_count']+=int(fallback.sum())
                # Sparse nonzero signal / gaps can have zero IQR despite genuine
                # variance. Use per-epoch standard deviation only in that case.
                iqr[fallback]=xx[fallback].std(axis=(1,2))
                if not np.isfinite(xx).all() or (iqr<=0).any():raise ValueError(f'{ds} nonfinite/constant epochs at {offset}')
                xx=(np.clip(xx,limits[0,:,None,None],limits[4,:,None,None])-limits[2,:,None,None])/iqr[:,None,None]
                xx=np.einsum('ec,bct->bet',spatial,xx).astype(np.float32)
                if ds=='SEED':xx=np.clip(xx,-10,10)
                qq=[]
                with torch.inference_mode():
                    for piece in segment(xx):
                        # Conv tokenizer consumes 100-sample patches. Zero-pad the
                        # final patch so short P300 tails are not silently dropped.
                        padding=(-piece.shape[-1])%100
                        if padding:piece=np.pad(piece,((0,0),(0,0),(0,padding)))
                        _,_,history=model.forward_with_q(torch.from_numpy(piece.copy()).cuda(),inst.expand(len(piece),-1))
                        q=model.qformer.final_norm(history[-1]);assert torch.isfinite(q).all();qq.append(q.cpu().numpy())
                features[ids]=np.concatenate(qq,axis=1)
                for i in ids:
                    row=dict(id=f'{ds}-{i}',parent_id=f'{ds}-{i}',task=tid,dataset=ds,subject=str(groups[i]),original_subject=str(subjects[i]),label=int(y[i]),source_index=int(i),split=splits[groups[i]],tokens=tokens)
                    rows.write(json.dumps(row)+'\n')
                (dest/'progress.json').write_text(json.dumps(dict(encoded=end,total=len(x),seconds=time.monotonic()-start)))
                if offset==0 or end==len(x) or end%1024==0:print(ds,end,len(x),time.monotonic()-start,flush=True)
        features.flush();del features
        report['excluded']=excluded;report['valid_n']=int(accepted.sum())
        report['counts']={split:{'n':int(sum(splits[g]==split for g in groups[accepted])), 'labels':{str(c):int(sum((splits[g]==split and label==c) for g,label in zip(groups[accepted],y[accepted]))) for c in np.unique(y)}} for split in ['train','validation','test']}
        assert all(all(n>0 for n in r['labels'].values()) for r in report['counts'].values()),report['counts']
        report['files_sha256']={n:digest(dest/n) for n in ['queries.npy','rows.jsonl']};report['seconds']=time.monotonic()-start
        (dest/'complete.json').write_text(json.dumps(report,indent=2));print('COMPLETE',ds,report['counts'],flush=True)
        del x,y

if __name__=='__main__':main()
