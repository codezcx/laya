"""Persistent frozen JEV per GPU; independent reset for every user/arm/support size."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from jev_train.checkpoint import seed_rng, tensor_fingerprint, capture_rng, restore_rng
from jev_train.data import classification_metrics
from .common import CONFIG, OUT, ROOT, check_rows, identity, write_json
from .data import TrialStore, SignalBatch, source_group
from .initialization import load_initialization
from .model import PersonalServer
from .modules import UserParameters


def atomic_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,tmp); tmp.replace(path)


def cpu_state(user):
    return {n:p.detach().cpu().clone() for n,p in user.state_dict().items()}


def metrics(logits,rows,classes):
    result=classification_metrics([r['label'] for r in rows],logits.softmax(-1).numpy(),classes)
    result['ce']=float(F.cross_entropy(logits.float(),torch.tensor([r['label'] for r in rows])))
    return result


def make_user(server,arm):
    return UserParameters(CONFIG,arm=arm,dim=server.model.text.config.hidden_size).cuda()


def optimizer_for(user,server):
    groups=[dict(params=list(user.parameters()),lr=CONFIG['learning_rate'])]
    if server.tower_enabled:
        groups.append(dict(params=server.tower_parameters(),lr=CONFIG['tower_learning_rate']))
    optimizer=torch.optim.AdamW(groups,weight_decay=CONFIG['weight_decay'])
    ids={id(p) for p in list(user.parameters())+server.tower_parameters()}
    if ids!={id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer parameter mismatch')
    forbidden={id(p) for n,p in server.model.named_parameters() if n not in server.personal_names}
    if ids & forbidden:
        raise RuntimeError('Optimizer includes a frozen shared weight')
    return optimizer


def backward_batch(server,user,inputs,rows,task,step):
    seed_rng(CONFIG['seed']+1009*step)
    user.zero_grad(set_to_none=True); server.model.eeg_encoder.zero_grad(set_to_none=True)
    total=0.
    for offset in range(0,len(rows),CONFIG['microbatch']):
        part=rows[offset:offset+CONFIG['microbatch']]
        logits=server.forward(inputs[offset:offset+len(part)],part,task,user,step=step,training=True)
        loss=F.cross_entropy(logits,torch.tensor([r['label'] for r in part],device='cuda'))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite support CE')
        weight=len(part)/len(rows); (loss*weight).backward(); total+=float(loss.detach())*weight
    if any(p.grad is None or not p.grad.isfinite().all() for p in list(user.parameters())+server.tower_parameters()):
        raise RuntimeError('Missing/nonfinite LoRA gradient')
    return total


def update(user,optimizer):
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['clip_grad_norm'],error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def encoded(server,store,rows,directory,stamp):
    path=directory/'features.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['stamp']!=stamp or saved['ids']!=[r['id'] for r in rows]:
            raise RuntimeError('Frozen feature cache identity changed')
        return saved['tokens']
    tokens=server.encode(store,rows)
    atomic_save(dict(stamp=stamp,ids=[r['id'] for r in rows],tokens=tokens),path)
    return tokens


def prepare_person(server,store,case,directory,stamp,progress):
    server.configure_arm('last_only')
    byid={r['id']:r for r in store.rows[case['task']]}
    support=[byid[i] for i in case['support_ids']]; query=[byid[i] for i in case['query_ids']]
    check_rows(support,case,gradient=True); check_rows(query,case)
    task=store.tasks[case['task']]
    folder=directory/'cases'/case['key']
    progress('encoding_target',case=case['key'])
    tokens=encoded(server,store,support+query,folder,stamp)
    baseline=server.predict(tokens[len(support):],query,task)
    prior_path=ROOT/CONFIG['paired_reference']/'cases'/case['key']/'connector_lora'/'predictions.npz'
    with np.load(prior_path) as prior:
        if prior['query_ids'].tolist()!=case['query_ids']:
            raise RuntimeError('Historical query IDs changed')
        if not torch.equal(torch.from_numpy(prior['baseline_logits']).argmax(-1),baseline.argmax(-1)):
            raise RuntimeError('Historical baseline predictions not reproduced')
    write_json(folder/'baseline.json',dict(case=case,stamp=stamp,
        metrics=metrics(baseline,query,case['classes']),historical_predictions_identical=True))
    return dict(support=support,query=query,task=task,tokens=tokens,baseline=baseline,
                base_metric=metrics(baseline,query,case['classes']),signals=None)


def smoke(server,inputs,support,case,task,arm,directory):
    rows=support[:2]; check_rows(rows,case,gradient=True)
    baseline=server.predict(inputs[:2],rows,task)
    user=make_user(server,arm)
    torch.testing.assert_close(server.predict(inputs[:2],rows,task,user),baseline,rtol=0,atol=0)
    original=tensor_fingerprint(user.state_dict()); original_tower=tensor_fingerprint(server.tower_state())
    optimizer=optimizer_for(user,server)
    loss=backward_batch(server,user,inputs[:2],rows,task,1); norm=update(user,optimizer)
    if norm<=0 or tensor_fingerprint(user.state_dict())==original:
        raise RuntimeError('No real connector update')
    changed=tensor_fingerprint(server.tower_state())!=original_tower
    if changed!=server.tower_enabled:
        raise RuntimeError('Tower LoRA update does not match assigned arm')
    before=server.predict(inputs[:2],rows,task,user)
    restored=make_user(server,arm); restored.load_state_dict(cpu_state(user),strict=True)
    tower_state=server.tower_state(); server.load_tower(server.initial_tower); server.load_tower(tower_state)
    torch.testing.assert_close(before,server.predict(inputs[:2],rows,task,restored),rtol=0,atol=0)
    server.assert_frozen()
    write_json(directory/f'smoke_{arm}.json',dict(passed=True,arm=arm,support_only=True,
        identity_exact=True,real_update=True,all_base_weights_and_buffers_frozen=True,
        reload_exact=True,tower_lora_updated=changed,trainable_parameters=sum(p.numel() for p in list(user.parameters())+server.tower_parameters()),
        support_ce=loss,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))


def run_episode(server,store,plan,base_case,episode,cache,directory,deadline,stamp,progress,smoked):
    arm=episode['arm']; n=episode['support_total']; case=dict(base_case)
    case['support_ids']=plan['support_sets'][case['key']][str(n)]
    case['support_sha256']=hashlib.sha256('\n'.join(case['support_ids']).encode()).hexdigest()
    index={r['id']:i for i,r in enumerate(cache['support'])}
    selected=[index[i] for i in case['support_ids']]
    support=[cache['support'][i] for i in selected]; query=cache['query']; task=cache['task']
    case['support_class_counts']=dict(Counter(str(r['label']) for r in support))
    case['support_records']=sorted({r['record'] for r in support})
    case['support_original_groups']=len({source_group(r) for r in support})
    check_rows(support,case,gradient=True); check_rows(query,case)
    folder=directory/'cases'/case['key']/f'{arm}_n{n}'
    curve_dir=folder/'curves'; curve_dir.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result['stamp']!=stamp or result['episode']!=episode:
            raise RuntimeError('Completed episode identity mismatch')
        return True
    server.configure_arm(arm)
    if server.tower_enabled:
        if cache['signals'] is None:
            progress('preparing_online_signals',case=case['key'],episode=episode['key'])
            cache['signals']=SignalBatch.prepare(store,cache['support']+query)
        all_inputs=cache['signals']
    else:
        all_inputs=cache['tokens']
    st=all_inputs[selected]; qt=all_inputs[len(cache['support']):]
    if arm not in smoked:
        progress('real_gradient_smoke',episode=episode['key'],personal_arm=arm)
        smoke(server,st,support,case,task,arm,directory)
        smoked.add(arm); server.configure_arm(arm)
    init=plan['initializations'][arm]
    user=make_user(server,arm); user.load_state_dict(load_initialization(init),strict=True)
    optimizer=optimizer_for(user,server)
    step,history=0,[]; state_path=folder/'last.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        if saved['stamp']!=stamp or saved['episode']!=episode:
            raise RuntimeError('Personal resume identity mismatch')
        user.load_state_dict(saved['user'],strict=True)
        if server.tower_enabled: server.load_tower(saved['tower'])
        optimizer.load_state_dict(saved['optimizer']); step,history=saved['step'],saved['history']
    schedule=plan['schedules'][episode['schedule']]

    def save():
        server.assert_frozen()
        atomic_save(dict(stamp=stamp,episode=episode,case=case,user=cpu_state(user),
            tower=server.tower_state() if server.tower_enabled else None,
            optimizer=optimizer.state_dict(),step=step,history=history,initialization=init),state_path)

    def evaluate():
        if step not in CONFIG['eval_steps']: return
        path=curve_dir/f'step{step:06d}.json'
        sha=tensor_fingerprint(user.state_dict()); tower_sha=tensor_fingerprint(server.tower_state())
        if path.exists():
            old=json.loads(path.read_text())
            if (old['stamp']!=stamp or old['episode']!=episode or old['step']!=step or
                    old['user_sha256']!=sha or old['tower_sha256']!=tower_sha):
                raise RuntimeError('Saved curve point differs from resumed state')
            return
        progress('evaluating_curve',episode=episode['key'],personal_arm=arm,step=step)
        started=time.time(); rng=capture_rng()
        try:
            after=server.predict(qt,query,task,user); support_logits=server.predict(st,support,task,user)
        finally:
            restore_rng(rng)
        if tensor_fingerprint(user.state_dict())!=sha or tensor_fingerprint(server.tower_state())!=tower_sha:
            raise RuntimeError('Evaluation changed personal parameters')
        server.assert_frozen()
        if step==0:
            torch.testing.assert_close(after,cache['baseline'],rtol=.01,atol=.02)
            if not torch.equal(after.argmax(-1),cache['baseline'].argmax(-1)):
                raise RuntimeError('Zero-delta baseline predictions changed')
        atomic_save(dict(user=cpu_state(user),tower=server.tower_state() if server.tower_enabled else None),
                    curve_dir/f'step{step:06d}.pt')
        pred_path=curve_dir/f'step{step:06d}.npz'; tmp=pred_path.with_suffix('.npz.tmp')
        with tmp.open('wb') as handle:
            np.savez_compressed(handle,query_ids=np.asarray(case['query_ids']),support_ids=np.asarray(case['support_ids']),
                labels=np.asarray([r['label'] for r in query]),support_labels=np.asarray([r['label'] for r in support]),
                baseline_logits=cache['baseline'].numpy(),logits=after.numpy(),support_logits=support_logits.numpy())
        tmp.replace(pred_path)
        metric=metrics(after,query,case['classes']); seen={i for b in schedule[:step] for i in b['indices']}
        point=dict(arm=arm,episode=episode,case=case,stamp=stamp,step=step,baseline=cache['base_metric'],query=metric,
            support=metrics(support_logits,support,case['classes']),support_samples_used=len(seen),support_n=n,
            equivalent_epochs=step*CONFIG['batch_size']/n,user_sha256=sha,tower_sha256=tower_sha,
            initialization=init,seconds=time.time()-started,query_used_for_gradients=False,query_used_for_selection=False,
            independent_test_estimate=False)
        write_json(path,point)
        progress('curve_complete',episode=episode['key'],personal_arm=arm,step=step,
                 query_bacc=metric['balanced_accuracy'],eval_seconds=point['seconds'])

    for earlier in CONFIG['eval_steps']:
        if earlier<step and not (curve_dir/f'step{earlier:06d}.json').exists():
            raise RuntimeError('Resume would skip a prescribed earlier curve point')
    save(); evaluate()
    for batch in schedule[step:]:
        if time.time()>=deadline:
            save(); return False
        if batch['step']%10==1 and identity()!=stamp['source_hashes']:
            raise RuntimeError('Live source changed')
        started=time.time(); ix=batch['indices']; rows=[support[i] for i in ix]
        if [r['id'] for r in rows]!=batch['ids'] or batch['step']!=step+1:
            raise RuntimeError('Matched training schedule changed')
        check_rows(rows,case,gradient=True)
        loss=backward_batch(server,user,st[ix],rows,task,batch['step']); norm=update(user,optimizer)
        step=batch['step']; history.append(dict(step=step,ce=loss,grad_norm=norm,lr=CONFIG['learning_rate'],
            seconds=time.time()-started,sample_ids=batch['ids']))
        if step%10==0 or step in CONFIG['eval_steps']:
            save(); progress('personal_adaptation',episode=episode['key'],personal_arm=arm,
                             **{k:v for k,v in history[-1].items() if k!='sample_ids'})
        evaluate()
    points=[json.loads((curve_dir/f'step{s:06d}.json').read_text()) for s in CONFIG['eval_steps']]
    if points[-1]['support_samples_used']!=n: raise RuntimeError('Not all support samples used')
    server.assert_frozen()
    atomic_save(dict(user=cpu_state(user),tower=server.tower_state() if server.tower_enabled else None),folder/'personal.pt')
    write_json(result_path,dict(status='complete',arm=arm,episode=episode,case=case,stamp=stamp,
        support_n=n,query_n=len(query),baseline=cache['base_metric'],adapted=points[-1]['query'],
        support=points[-1]['support'],steps=step,history=history,initialization=init,
        trainable_parameters=sum(p.numel() for p in list(user.parameters())+server.tower_parameters()),
        all_base_weights_and_buffers_frozen=True,tower_lora_updated=server.tower_enabled,
        query_used_for_gradients=False,query_used_for_selection=False,independent_test_estimate=False))
    progress('personal_complete',episode=episode['key'],personal_arm=arm,bacc=points[-1]['query']['balanced_accuracy'])
    return True


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',type=int,required=True); args=parser.parse_args()
    torch.set_num_threads(3); torch.backends.mha.set_fastpath_enabled(False); torch.cuda.set_device(0)
    plan=json.loads((OUT/'plan.json').read_text()); queue=json.loads((OUT/'queue.json').read_text())
    job=plan['jobs'][args.worker]; directory=OUT/'jobs'/job['key']; directory.mkdir(parents=True,exist_ok=True)
    hashes=identity()
    if plan['source_hashes']!=hashes or queue['source_hashes']!=hashes: raise RuntimeError('Plan/queue identity mismatch')
    stamp=dict(source_hashes=hashes,parent_sha256=queue['parent_sha256'])
    def progress(state,**kw):
        report=dict(state=state,job=job['key'],time=time.time(),pid=os.getpid(),**kw)
        write_json(OUT/f'progress_gpu{args.worker}.json',report); print(json.dumps(report),flush=True)
    progress('loading_frozen_parent'); server=PersonalServer(queue['parent_sha256']); store=TrialStore()
    if json.loads(json.dumps(store.stats))!=plan['source_array_stats']: raise RuntimeError('EEG array identities changed')
    write_json(directory/'tower_lora_manifest.json',server.tower_lora_manifest)
    episodes={e['key']:e for e in plan['episodes']}; cases={c['key']:c for c in plan['cases']}
    cache={}; smoked=set()
    for key in job['episodes']:
        if time.time()>=queue['deadline']: progress('budget_exhausted'); return
        ep=episodes[key]; case=cases[ep['case_key']]
        if case['key'] not in cache: cache[case['key']]=prepare_person(server,store,case,directory,stamp,progress)
        if not run_episode(server,store,plan,case,ep,cache[case['key']],directory,queue['deadline'],stamp,progress,smoked):
            progress('budget_exhausted'); return
    write_json(directory/'complete.json',dict(job=job,stamp=stamp,time=time.time()))
    progress('worker_done')


if __name__=='__main__': main()
