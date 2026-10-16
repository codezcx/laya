"""Frozen paired evaluation on already-exposed 2b and SEED-VII targets."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel, AutoTokenizer
from eeg_tasks.data import preprocess, normalized_window
from eeg_tasks.emotion_transfer_probe import CHANNELS, picks, readrows
from eeg_tasks.external_emotion import SOURCE, labels as emotion_labels
from eeg_tasks.mi_reve_scaling_v24 import scale_window
from .cache_multitask_v30 import digest

ROOT = Path('artifacts/multitask_bridge_v30')
OUT = ROOT/'external'


def metrics(rows,pred):
    y=np.array([r['class_id'] for r in rows]);pred=np.asarray(pred);people={}
    assert pred.shape==y.shape
    classes=np.unique(y)
    for s in sorted({r['subject'] for r in rows}):
        ii=np.array([i for i,r in enumerate(rows) if r['subject']==s]);yy=y[ii];pp=pred[ii]
        assert np.array_equal(np.unique(yy),classes)
        people[s]=dict(accuracy=float(np.mean(pp==yy)),balanced_accuracy=float(np.mean([np.mean(pp[yy==c]==c) for c in classes])))
    values=np.array([v['balanced_accuracy'] for v in people.values()])
    boot=np.random.default_rng(2026).choice(values,(10000,len(values)),replace=True).mean(1)
    return dict(n=len(rows),accuracy=float(np.mean(pred==y)),subject_accuracy=float(np.mean([v['accuracy'] for v in people.values()])),
        subject_balanced_accuracy=float(values.mean()),subject_balanced_95ci=np.quantile(boot,[.025,.975]).tolist(),per_subject=people)


def frozen(root=ROOT):
    root=Path(root)
    record = json.loads((root/'source_freeze.json').read_text())
    for arm, files in record['arms'].items():
        for name,h in files.items(): assert digest(root/arm/name)==h
    return digest(root/'source_freeze.json')


@torch.no_grad()
def extract(shard):
    freeze = frozen(); out = OUT/str(shard); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    model,info = AutoModel.from_pretrained(cfg['reve_checkpoint'],local_files_only=True,trust_remote_code=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model = model.cuda().eval().requires_grad_(False)
    bank = AutoModel.from_pretrained(cfg['positions_checkpoint'],local_files_only=True,trust_remote_code=True)
    positions = {n:bank(ch).detach().cuda() for n,ch in [(3,['C3','Cz','C4']),(30,CHANNELS)]}
    arrays = []; rows = []; lengths = []
    def encode(signals,rr):
        for start in range(0,len(signals),32):
            x = torch.from_numpy(np.stack(signals[start:start+32])).cuda()
            tail = (-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap)
            z = model(torch.nn.functional.pad(x,(0,tail),mode='replicate'),positions[x.shape[1]][None].expand(len(x),-1,-1))
            assert z.shape==(len(x),x.shape[1],4,512) and torch.isfinite(z).all()
            z = z.flatten(1,2).cpu().numpy().astype(np.float16)
            padded = np.zeros((len(z),120,512),np.float16); padded[:,:z.shape[1]]=z
            arrays.append(padded);lengths.extend([z.shape[1]]*len(z))
        rows.extend(rr)
    manifest = Path('artifacts/reve_mi_multi/manifests_v2/external_2b.jsonl')
    mi = readrows(manifest)
    assert len(mi)==len({r['path'] for r in mi})==2840
    source = readrows(ROOT/'source_cache/rows.jsonl')
    assert not {r['path'] for r in source}&{r['path'] for r in mi}
    selected = mi[shard::4]
    for begin in range(0,len(selected),32):
        signals=[]; rr=[]
        for row in selected[begin:begin+32]:
            with open(row['path'],'rb') as f: item=pickle.load(f)
            assert item['freq']==200 and cfg['labels'].index(item['label'])==row['label']
            x=preprocess(np.stack([item['eeg'][c] for c in ['C3','CZ','C4']]),200,{'bandpass':[1,30]})[:,:600]
            signals.append(scale_window(x,'common_scale'))
            rr.append(dict(row,dataset='2b',class_id=row['label']+3,task=1))
        encode(signals,rr)
    old = Path('artifacts/emotion_alignment_v21/external_seed_vii')
    emotion = readrows(old/f'rows_{shard}.jsonl')
    lab=emotion_labels();mapping={'Sad':0,'Neutral':1,'Happy':2}
    assert all(mapping[lab[int(r['trial'])]]==r['class_id'] for r in emotion)
    assert len(emotion)==1600 and {int(r['subject']) for r in emotion}==set(range(shard+1,21,4))
    assert not {r['path'] for r in source}&{r['path'] for r in emotion}
    names=[s.split()[-1] for s in (SOURCE/'channel_62_pos.locs').read_text().splitlines() if s.strip()]
    cp=picks(names)
    for subject in sorted({r['subject'] for r in emotion},key=int):
        rr=[r for r in emotion if r['subject']==subject]
        data=loadmat(rr[0]['path'],variable_names=list({r['trial'] for r in rr}));filtered={};signals=[]
        for r in rr:
            trial=r['trial']
            if trial not in filtered: filtered[trial]=preprocess(data[trial][cp],200,{'bandpass':[1,30]})
            s=r['start_sample'];signals.append(normalized_window(filtered[trial][:,s:s+600]))
        encode(signals,rr);print(shard,subject,len(rows),flush=True)
    np.save(out/'tokens.npy',np.concatenate(arrays));np.save(out/'lengths.npy',np.asarray(lengths))
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert len(rows)==2310
    (out/'complete.json').write_text(json.dumps(dict(n=len(rows),freeze_sha256=freeze,
        hashes={n:digest(out/n) for n in ['tokens.npy','lengths.npy','rows.jsonl']},
        target_manifests={str(p):digest(p) for p in [manifest,old/f'rows_{shard}.jsonl']},
        preprocessing='same as source: MI common scale; emotion per-channel; 200Hz 1-30Hz 3s'),indent=2))


@torch.no_grad()
def evaluate(checkpoint_root=ROOT, output=None):
    from .model import load_text_only
    from .multitask_bridge import MultitaskBridge,PREFIX,SUFFIX
    from .signal_control_v30 import SignalControl
    checkpoint_root=Path(checkpoint_root)
    freeze=frozen(checkpoint_root);cache_freeze=frozen();torch.set_num_threads(4)
    output=Path(output) if output is not None else checkpoint_root/'external'
    if output.resolve()==OUT.resolve() and checkpoint_root.resolve()!=ROOT.resolve():
        raise ValueError('New checkpoints require separate evaluation output')
    if (output/'results.json').exists(): raise FileExistsError(output/'results.json')
    # Preserve old run naming while allowing the current single trainer's arm names.
    names={'signal_control':'signal_control','qwen_frozen':'qwen_frozen'} if checkpoint_root.resolve()==ROOT.resolve() else {'signal_control':'signal','qwen_frozen':'qwen'}
    record=json.loads((checkpoint_root/'source_freeze.json').read_text())
    assert set(record['arms'])==set(names.values())
    for directory in names.values():
        run=json.loads((checkpoint_root/directory/'run.json').read_text())
        done=json.loads((checkpoint_root/directory/'complete.json').read_text())
        assert run['source_manifest_sha256']==digest(ROOT/'source_cache/rows.jsonl')
        assert run['cache_audit_sha256']==digest(ROOT/'source_cache/verified.json')
        assert done['best_sha256']==digest(checkpoint_root/directory/'best.pt')
    output.mkdir(parents=True,exist_ok=True)
    arrays=[];lens=[];rows=[]
    for shard in range(4):
        p=OUT/str(shard);done=json.loads((p/'complete.json').read_text());assert done['freeze_sha256']==cache_freeze
        for name,h in done['hashes'].items():assert digest(p/name)==h
        arrays.append(np.load(p/'tokens.npy'));lens.append(np.load(p/'lengths.npy'));rows+=readrows(p/'rows.jsonl')
    x=np.concatenate(arrays);lengths=np.concatenate(lens)
    assert len(rows)==9240 and x.shape==(9240,120,512) and np.isfinite(x).all()
    assert np.array_equal(lengths,[12 if r['class_id']>=3 else 120 for r in rows])
    assert len({(r['path'],r.get('trial'),r.get('start_sample')) for r in rows})==len(rows)
    source=readrows(ROOT/'source_cache/rows.jsonl');assert not {r['path'] for r in source}&{r['path'] for r in rows}
    cfg=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());results={};predictions={}
    for arm in ['signal_control','qwen_frozen']:
        if arm=='signal_control':model=SignalControl().cuda()
        else:
            tok=AutoTokenizer.from_pretrained(cfg['qwen_checkpoint'],local_files_only=True)
            ids=[tok.encode(s,add_special_tokens=False) for s in 'ABCDE'];assert all(len(i)==1 for i in ids)
            text,head,_=load_text_only(cfg['qwen_checkpoint'],[i[0] for i in ids],'cuda:0',torch.bfloat16)
            enc=lambda s:torch.tensor(tok.encode(s,add_special_tokens=False),device='cuda')
            model=MultitaskBridge(text,head,enc(PREFIX),enc(SUFFIX)).cuda()
        model.connector.load_state_dict(torch.load(checkpoint_root/names[arm]/'best.pt',weights_only=True,map_location='cuda'));model.eval();logits=[]
        for start in range(0,len(rows),8):
            ll=lengths[start:start+8];k=int(max(ll));tokens=torch.from_numpy(x[start:start+8,:k].astype(np.float32)).cuda()
            valid=torch.arange(k,device='cuda')[None]<torch.tensor(ll,device='cuda')[:,None]
            logits.append(model(tokens,valid)['logits'].cpu().numpy())
        z=np.concatenate(logits);assert np.isfinite(z).all();np.save(output/f'{arm}_logits.npy',z)
        predictions[arm]=z.argmax(1);results[arm]={}
        for ds in ['2b','SEED-VII-HNS']:
            ii=np.array([i for i,r in enumerate(rows) if r['dataset']==ds]);rr=[rows[i] for i in ii];zz=z[ii];yy=np.array([r['class_id'] for r in rr]);pred=zz.argmax(1)
            m=metrics(rr,pred);m['task_routing_accuracy']=float(np.mean((pred>=3)==(yy>=3)))
            oracle=zz.copy()
            if ds=='2b':oracle[:,:3]=-np.inf
            else:oracle[:,3:]=-np.inf
            m['oracle_task']=metrics(rr,oracle.argmax(1))
            shuffled=pred.copy();rng=np.random.default_rng(2026)
            for subject in sorted({r['subject'] for r in rr}):
                jj=np.array([j for j,r in enumerate(rr) if r['subject']==subject]);shuffled[jj]=pred[rng.permutation(jj)]
            m['within_subject_shuffle']=metrics(rr,shuffled);results[arm][ds]=m
        del model
        if arm=='qwen_frozen':del text
        torch.cuda.empty_cache()
    paired={}
    for ds in ['2b','SEED-VII-HNS']:
        key='accuracy' if ds=='2b' else 'balanced_accuracy'
        a=results['qwen_frozen'][ds]['per_subject'];b=results['signal_control'][ds]['per_subject']
        delta=np.array([a[s][key]-b[s][key] for s in sorted(a)]);boot=np.random.default_rng(2026).choice(delta,(10000,len(delta)),replace=True).mean(1)
        paired[ds]=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    assert frozen(checkpoint_root)==freeze and frozen()==cache_freeze
    (output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'results.json').write_text(json.dumps(dict(freeze_sha256=freeze,cache_freeze_sha256=cache_freeze,checkpoint_root=str(checkpoint_root),results=results,paired_subject_delta=paired,
        note='Repeated-target exploratory evaluation. REVE pretraining includes 2b. No target fitting/selection.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--evaluate',action='store_true')
    p.add_argument('--checkpoint-root',type=Path,default=ROOT);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.evaluate:evaluate(a.checkpoint_root,a.output)
    elif a.shard is not None:extract(a.shard)
    else:p.error('Choose stage')
