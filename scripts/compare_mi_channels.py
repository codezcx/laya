"""Paired 2a 22-vs-3 EEG channel ablation with an unchanged held-out probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from eeg_align.linear_probe import probe_data,spatial_pool
from eeg_align.mi_training_data import config,preprocess,digest,write_json,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,default=Path('artifacts/mi_reve_probe_holdout_2a'));p.add_argument('--out',type=Path,default=Path('artifacts/mi_reve_2a_channel_comparison'));a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    saved=torch.load(a.probe/'best.pt',map_location='cpu',weights_only=False)
    assert saved['config']['heldout_dataset']=='BNCI2014_001'
    assert saved['config']['cache_sha256']==digest(Path(config()['cache'])/'complete.json')
    assert saved['config']['code_sha256']==digest('eeg_align/linear_probe.py')
    data=probe_data('heldout','BNCI2014_001');rows=data.rows
    old=json.loads((a.probe/'test_2a.json').read_text());old_pred={r['id']:r for r in old['predictions']}
    assert len(rows)==2592 and {r['id'] for r in rows}==set(old_pred)
    ec=json.loads(Path('configs/encoder.json').read_text())
    model,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval().requires_grad_(False)
    bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False)
    head=torch.nn.Linear(1536,2).cuda().eval().requires_grad_(False);head.load_state_dict(saved['head'])
    mean=saved['mean'].cuda();std=saved['std'].cuda()
    predictions={22:[],3:[]}
    with torch.no_grad(),ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];items=list(pool.map(preprocess,batch))
            names=items[0][1];assert len(names)==22 and all(i[1]==names and i[0].shape==(22,920) for i in items)
            picks=[next(i for i,n in enumerate(names) if n.upper()==q) for q in ['C3','CZ','C4']]
            x=torch.from_numpy(np.stack([i[0] for i in items])).cuda()
            for count,indices in [(22,list(range(22))),(3,picks)]:
                selected_names=[names[i] for i in indices];positions=bank(selected_names).cuda();assert positions.shape==(count,3)
                # Re-run the encoder for BOTH arms; do not select C3/Cz/C4
                # from features already contextualized by 22-channel attention.
                z=model(x[:,indices],positions[None].expand(len(batch),-1,-1)).flatten(1,2)
                # Match the frozen cache's float16 storage used to train the head.
                z=z.half().float()
                f=torch.stack([spatial_pool(t,positions,selected_names) for t in z])
                logits=head((f-mean)/std);prob=logits.softmax(-1)[:,1].cpu().tolist()
                for r,v in zip(batch,prob):
                    assert r['label']==old_pred[r['id']]['label']
                    predictions[count].append(dict(id=r['id'],dataset=r['dataset'],subject=r['subject'],label=r['label'],probability_right=v))
            if start%320==0:print('PAIRED',start,len(rows),flush=True)
    full=metrics(predictions[22]);reduced=metrics(predictions[3])
    delta=max(abs(r['probability_right']-old_pred[r['id']]['probability_right']) for r in predictions[22])
    agreement=sum((r['probability_right']>.5)==(old_pred[r['id']]['probability_right']>.5) for r in predictions[22])
    assert delta<1e-4 and agreement==len(rows),'Full-channel control failed to reproduce original predictions'
    y=np.array([r['label'] for r in predictions[22]])
    p22=np.array([r['probability_right'] for r in predictions[22]])>.5;p3=np.array([r['probability_right'] for r in predictions[3]])>.5
    paired=dict(both_correct=int(((p22==y)&(p3==y)).sum()),only_22_correct=int(((p22==y)&(p3!=y)).sum()),only_3_correct=int(((p22!=y)&(p3==y)).sum()),both_wrong=int(((p22!=y)&(p3!=y)).sum()))
    result=dict(n=len(rows),checkpoint=str(a.probe/'best.pt'),checkpoint_sha256=digest(a.probe/'best.pt'),selected_epoch=saved['epoch'],
        fixed='same trial, full duration, preprocessing, electrode reference, frozen REVE, spatial readout, scaler, linear weights and threshold0.5',
        changed='EEG input channel set before REVE: all22 versus C3/Cz/C4',
        full_control_probability_max_difference=delta,full_control_prediction_agreement=agreement,
        full_22=full,reduced_3=reduced,accuracy_difference_3_minus_22=reduced['accuracy']-full['accuracy'],paired_correctness=paired,
        code_sha256=digest(__file__),encoder_config=ec)
    write_json(a.out/'comparison.json',result)
    for n in [22,3]:write_json(a.out/f'predictions_{n}ch.json',predictions[n])
    write_json(a.out/'complete.json',dict(status='passed',comparison_sha256=digest(a.out/'comparison.json')))
    print('RESULT',json.dumps({k:result[k] for k in ['n','accuracy_difference_3_minus_22','paired_correctness','full_control_probability_max_difference']}),flush=True)
    print('ACCURACY',full['accuracy'],reduced['accuracy'],flush=True)

if __name__=='__main__':main()
