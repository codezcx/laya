"""One frozen 27B server per GPU; independent per-person connector LoRA and ICL."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from jev_train.checkpoint import seed_rng, tensor_fingerprint
from jev_train.data import classification_metrics
from .data import TrialStore
from .common import CONFIG, OUT, check_rows, identity, write_json
from .model import PersonalServer
from .modules import UserParameters


def atomic_save(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def metrics(logits,rows,classes):
    return classification_metrics([r['label'] for r in rows],logits.softmax(-1).numpy(),classes)


def make_user(server,seed):
    seed_rng(seed)
    return UserParameters(CONFIG,server.model.text.config.hidden_size).cuda()


def assert_optimizer(optimizer,user,server):
    ids={id(p) for p in user.parameters()}
    if ids!={id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer contains parameters outside personal LoRA')
    if ids & {id(p) for p in server.model.parameters()}:
        raise RuntimeError('Optimizer contains shared parameters')


def optimizer_for(user,server):
    optimizer=torch.optim.AdamW(user.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    assert_optimizer(optimizer,user,server)
    return optimizer


def smoke(server,store,cases,directory):
    """Real gradients plus full100-example ICL contexts for every task, support-only."""
    reports={}
    for case in cases:
        task=store.tasks[case['task']]
        byid={r['id']:r for r in store.rows[case['task']]}
        support=[byid[i] for i in case['support_ids']]
        check_rows(support,case,gradient=True)
        all_tokens=server.encode(store,support)
        rows=[support[0],support[-1]]
        tokens=all_tokens[[0,-1]]
        baseline=server.predict(tokens,rows,task)
        with torch.no_grad():
            from jev_train.task_options import make_variant
            suffix=[torch.tensor(server.tokenizer.encode(make_variant(task,r['label'],r['id'],0,CONFIG['seed'],0)['suffix'],
                add_special_tokens=False),device='cuda') for r in rows]
            server.model.eval()
            legacy=server.model(tokens.cuda(),torch.ones(tokens.shape[:2],dtype=torch.bool,device='cuda'),
                suffix_ids=suffix,num_choices=case['classes'])['logits'].cpu()
            torch.testing.assert_close(baseline,legacy,atol=0,rtol=0)
        user=make_user(server,CONFIG['seed'])
        optimizer=optimizer_for(user,server)
        initial=tensor_fingerprint(user.state_dict())
        logits=server.forward(tokens,rows,task,user,step=1,training=True)
        loss=F.cross_entropy(logits,torch.tensor([r['label'] for r in rows],device='cuda'))
        loss.backward()
        grads=[p.grad for p in user.parameters() if p.grad is not None]
        if not grads or not all(g.isfinite().all() for g in grads) or not any(g.abs().max()>0 for g in grads):
            raise RuntimeError('Invalid personal gradient')
        torch.nn.utils.clip_grad_norm_(user.parameters(),CONFIG['clip_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        if tensor_fingerprint(user.state_dict())==initial:
            raise RuntimeError('Personal parameters did not update')
        before=server.predict(tokens,rows,task,user)
        saved={n:t.detach().cpu().clone() for n,t in user.state_dict().items()}
        restored=make_user(server,CONFIG['seed']+1)
        restored.load_state_dict(saved,strict=True)
        after=server.predict(tokens,rows,task,restored)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        del user,restored,optimizer,loss,logits
        demos,audit=server.demonstrations(all_tokens,support,task)
        context_bound=len(demos)+all_tokens.shape[1]+1024
        if context_bound>server.model.text.config.max_position_embeddings:
            raise RuntimeError('ICL context exceeds language model capacity; no silent truncation')
        torch.cuda.reset_peak_memory_stats()
        icl=server.predict(tokens,rows,task,demonstrations=demos)
        if not icl.isfinite().all():
            raise RuntimeError('Nonfinite full-support ICL logits')
        server.assert_frozen()
        reports[case['dataset']]=dict(case=case['key'],support_n=len(support),support_only=True,
            base_forward_exact=True,nonzero_finite_gradient=True,personal_update=True,reload_exact=True,
            signal_tokens=all_tokens.shape[1],demonstration_tokens=len(demos),
            icl_full_support_finite=True,icl_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        del all_tokens,demos,icl
        torch.cuda.empty_cache()
    write_json(directory/'smoke.json',dict(passed=True,shared_model_frozen=True,tasks=reports))


def run_case(server,store,case,deadline,hashes,parent_sha,worker):
    directory=OUT/'cases'/case['key']
    directory.mkdir(parents=True,exist_ok=True)
    index={r['id']:r for r in store.rows[case['task']]}
    support=[index[i] for i in case['support_ids']]
    query=[index[i] for i in case['query_ids']]
    check_rows(support,case,gradient=True)
    check_rows(query,case)
    task=store.tasks[case['task']]
    def progress(state,**kw):
        write_json(OUT/f'progress_gpu{worker}.json',dict(state=state,case=case['key'],time=time.time(),pid=os.getpid(),**kw))
    progress('encoding')
    tokens=server.encode(store,support+query)
    st,qt=tokens[:len(support)],tokens[len(support):]
    target=torch.tensor([r['label'] for r in support],device='cuda')
    baseline_path=directory/'baseline.pt'
    if baseline_path.exists():
        baseline=torch.load(baseline_path,map_location='cpu',weights_only=False)
        if baseline['parent_sha256']!=parent_sha or baseline['case']!=case:
            raise RuntimeError('Baseline cohort/parent identity changed')
    else:
        progress('baseline_evaluation')
        baseline=dict(parent_sha256=parent_sha,case=case,support_logits=server.predict(st,support,task),
                      query_logits=server.predict(qt,query,task))
        atomic_save(baseline,baseline_path)
    audit={}
    if case['historical_prediction']:
        old=torch.load(case['historical_prediction'],map_location='cpu',weights_only=False)
        if old['ids']!=case['query_ids'] or old['labels']!=[r['label'] for r in query]:
            raise RuntimeError('Historical baseline sample identity changed')
        old_logits=torch.as_tensor(old['baseline']['jev'])
        new_logits=baseline['query_logits']
        audit=dict(historical_baseline_predictions_identical=torch.equal(old_logits.argmax(-1),new_logits.argmax(-1)),
                   historical_baseline_logits_exact=torch.equal(old_logits,new_logits),
                   historical_baseline_max_logit_difference=float((old_logits-new_logits).abs().max()),eval_batch=CONFIG['eval_batch'])
        write_json(directory/'baseline_audit.json',audit)
        if not audit['historical_baseline_predictions_identical']:
            raise RuntimeError('Historical baseline not reproduced')
    base_metric=metrics(baseline['query_logits'],query,case['classes'])
    for arm in CONFIG['arms']:
        arm_dir=directory/arm
        arm_dir.mkdir(exist_ok=True)
        if (arm_dir/'result.json').exists():
            saved=json.loads((arm_dir/'result.json').read_text())
            if saved['source_hashes']!=hashes or saved['parent_sha256']!=parent_sha or saved['case']!=case:
                raise RuntimeError('Finished result identity changed')
            continue
        if time.time()>=deadline:
            return False
        if identity()!=hashes:
            raise RuntimeError('Live source changed')
        started=time.time()
        user=optimizer=init_logits=support_after=None
        history,step=[],0
        info=dict(audit)
        if arm=='icl':
            progress('evaluating_icl',arm=arm,query_n=len(query),support_n=len(support))
            demos,details=server.demonstrations(st,support,task)
            info.update(details,demonstration_tokens=len(demos))
            after=server.predict(qt,query,task,demonstrations=demos)
            parameter_count=0
            del demos
        else:
            user=make_user(server,CONFIG['seed']+case['order']*101)
            parameter_count=sum(p.numel() for p in user.parameters())
            optimizer=optimizer_for(user,server)
            state_path=arm_dir/'last.pt'
            if state_path.exists():
                saved=torch.load(state_path,map_location='cpu',weights_only=False)
                if saved['parent_sha256']!=parent_sha or saved['source_hashes']!=hashes or saved['case']!=case:
                    raise RuntimeError('Personal resume identity mismatch')
                user.load_state_dict(saved['user'],strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                step,history,init_logits=saved['step'],saved['history'],saved['initial_logits']
                del saved
            else:
                init_logits=server.predict(qt,query,task,user)
                torch.testing.assert_close(init_logits,baseline['query_logits'],atol=0,rtol=0)

            def save():
                server.assert_frozen()
                atomic_save(dict(user={n:p.detach().cpu() for n,p in user.state_dict().items()},
                    optimizer=optimizer.state_dict(),step=step,history=history,initial_logits=init_logits,
                    parent_sha256=parent_sha,source_hashes=hashes,case=case),state_path)
            progress('adapting',arm=arm,step=step,steps=CONFIG['steps'])
            for step in range(step+1,CONFIG['steps']+1):
                if time.time()>=deadline:
                    step-=1
                    save()
                    return False
                rng=np.random.default_rng(np.random.SeedSequence([CONFIG['seed'],case['order'],step]))
                selected=rng.integers(len(support),size=CONFIG['batch_size']).tolist()
                check_rows([support[i] for i in selected],case,gradient=True)
                optimizer.zero_grad(set_to_none=True)
                total=0.
                for offset in range(0,len(selected),CONFIG['microbatch']):
                    ix=selected[offset:offset+CONFIG['microbatch']]
                    logits=server.forward(st[ix],[support[i] for i in ix],task,user,step=step,training=True)
                    loss=F.cross_entropy(logits,target[ix])
                    weight=len(ix)/CONFIG['batch_size']
                    (loss*weight).backward()
                    total+=float(loss.detach())*weight
                norm=torch.nn.utils.clip_grad_norm_(user.parameters(),CONFIG['clip_grad_norm'],error_if_nonfinite=True)
                if step==1 and float(norm)<=0:
                    raise RuntimeError('No personal gradients')
                optimizer.step()
                history.append(dict(step=step,loss=total,ce=total,grad_norm=float(norm),lr=CONFIG['learning_rate']))
                if step%10==0 or step==CONFIG['steps']:
                    save()
                    progress('adapting',arm=arm,steps=CONFIG['steps'],**history[-1])
            progress('evaluating_adapted',arm=arm,step=step)
            after=server.predict(qt,query,task,user)
            support_after=server.predict(st,support,task,user)
            atomic_save({n:v.detach().cpu() for n,v in user.state_dict().items()},arm_dir/'user.pt')
        server.assert_frozen()
        metric=metrics(after,query,case['classes'])
        np.savez_compressed(arm_dir/'predictions.npz',query_ids=np.asarray(case['query_ids']),
            labels=np.asarray([r['label'] for r in query]),baseline_logits=baseline['query_logits'].numpy(),
            logits=after.numpy(),initial_logits=np.empty((0,case['classes'])) if init_logits is None else init_logits.numpy())
        write_json(arm_dir/'result.json',dict(status='complete',arm=arm,case=case,parent_sha256=parent_sha,
            source_hashes=hashes,selected_parent_step=server.parent_step,steps=step,
            trainable_parameters=parameter_count,support_n=len(support),query_n=len(query),
            baseline=base_metric,adapted=metric,initial=None if init_logits is None else metrics(init_logits,query,case['classes']),
            support=None if support_after is None else metrics(support_after,support,case['classes']),
            delta_pp=100*(metric['balanced_accuracy']-base_metric['balanced_accuracy']),shared_weights_frozen=True,
            query_used_for_gradients=False,query_used_for_selection=False,independent_test_estimate=False,
            role=CONFIG['role'],seconds=time.time()-started,history=history,audit=info))
        print(json.dumps(dict(case=case['key'],arm=arm,bacc=metric['balanced_accuracy'],seconds=time.time()-started)),flush=True)
        del user,optimizer,after,support_after
        torch.cuda.empty_cache()
    write_json(directory/'complete.json',dict(case=case['key'],arms=CONFIG['arms'],parent_sha256=parent_sha))
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_device(0)
    hashes=identity()
    plan=json.loads((OUT/'plan.json').read_text())
    queue=json.loads((OUT/'queue.json').read_text())
    if plan['source_hashes']!=hashes or queue['source_hashes']!=hashes:
        raise RuntimeError('Plan/queue identity mismatch')
    write_json(OUT/f'progress_gpu{args.worker}.json',dict(state='loading_frozen_parent',pid=os.getpid(),time=time.time()))
    server=PersonalServer(queue['parent_sha256'])
    store=TrialStore()
    if json.loads(json.dumps(store.stats))!=plan['source_array_stats']:
        raise RuntimeError('Input identities changed after planning')
    smoke_path=OUT/'smoke.json'
    write_json(OUT/f'progress_gpu{args.worker}.json',dict(state='smoke_or_waiting_for_smoke',pid=os.getpid(),time=time.time()))
    if args.worker==0 and not smoke_path.exists():
        representatives=[next(c for c in plan['cases'] if c['dataset']==n) for n in CONFIG['datasets']]
        smoke(server,store,representatives,OUT)
    while not smoke_path.exists():
        if time.time()>=queue['deadline']:
            return
        time.sleep(5)
    if not json.loads(smoke_path.read_text())['passed']:
        raise RuntimeError('Smoke failed')
    for case in plan['cases']:
        if time.time()>=queue['deadline']:
            break
        directory=OUT/'cases'/case['key']
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'complete.json').exists():
            continue
        with (directory/'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (directory/'complete.json').exists():
                continue
            if not run_case(server,store,case,queue['deadline'],hashes,queue['parent_sha256'],args.worker):
                break
    write_json(OUT/f'progress_gpu{args.worker}.json',dict(state='worker_done',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
