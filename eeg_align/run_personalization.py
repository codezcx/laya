"""Episodic amortized adaptation: source-only optimization, no target gradients."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.distributed as dist

from .benchmark_data import classification_metrics
from .checkpoint import capture_rng, restore_rng, restore_model, seed_rng, tensor_fingerprint
from .contrastive_protocol import load_text_bank
from .distributed import atomic_save, average_gradients, rank_rows
from .leaf_data import LEAFSignalAdapter
from .multitask_data import TrialStore, write_json
from .personalization import (CONFIG, DATA, ROOT, AmortizedFiLM, assert_episode,
    fixed_server_versions, ids_hash, load_encoder, load_parent, make_manifest, module_hash, source_identity)
from .runtime import digest
from .task_options import make_variant


def report(path, **values):
    values['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    write_json(path,values)
    print(json.dumps(values,allow_nan=False),flush=True)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    hashes = source_identity()
    if (root/'prepared.json').exists():
        proof = json.loads((root/'prepared.json').read_text())
        if proof['code_sha256'] != hashes:
            raise ValueError('Prepared source identity differs')
        for path,sha in proof['artifacts_sha256'].items():
            if digest(path)!=sha:
                raise ValueError(f'Changed prepared artifact: {path}')
        return
    store = TrialStore()
    manifest = make_manifest(store)
    saved = load_parent()
    if saved['identity']['cache_indexes'] != manifest['cache_indexes']:
        raise ValueError('Parent and episodes use different data')
    # Assert the test support/query IDs match the previously authorized pilot.
    old = json.loads((Path(CONFIG['prior_pilot'])/'manifest.json').read_text())
    for c in manifest['cohorts']:
        if c['split']=='test':
            prior = next(x for x in old['cases'] if x['dataset']==c['dataset'] and x['subject']==c['subject']
                         and x['shots_per_class']==CONFIG['shots_per_class'] and x['seed']==CONFIG['seed'])
            if ([r['id'] for r in c['support']] != prior['support_ids']
                    or [r['id'] for r in c['query']] != prior['query_ids']):
                raise ValueError('Exploratory test cohort differs from historical pilot')
    write_json(root/'manifest.json',manifest)
    bank,vectors = load_text_bank()
    semantic = {}
    for name in CONFIG['datasets']:
        spec = bank['banks'][name]
        semantic[name] = torch.from_numpy(np.stack([
            vectors[np.array(spec['text_indices'])[np.array(spec['class_labels'])==c]].mean(0)
            for c in range(len(spec['semantic_ids']))]))
    atomic_save(semantic,root/'label_vectors.pt')
    adapter = LEAFSignalAdapter()
    (root/'subjects').mkdir(exist_ok=True)
    artifacts = {str(root/n):digest(root/n) for n in ('manifest.json','label_vectors.pt')}
    counts = defaultdict(lambda:dict(train=0,valid=0,test=0,subjects=defaultdict(int)))
    for index,c in enumerate(manifest['cohorts']):
        rows = c['support_pool']+c['query']
        chunks = []
        for start in range(0,len(rows),32):
            part = rows[start:start+32]
            groups = adapter.batch(part)
            if len(groups)!=1 or groups[0][1]!=list(range(len(part))) or len(groups[0][2])!=1:
                raise ValueError('Pilot requires one equal-length segment per MI trial')
            chunks.append(torch.from_numpy(groups[0][2][0]))
        x = torch.cat(chunks)
        if x.shape[1:]!=(65,800) or not torch.isfinite(x).all():
            raise ValueError('Expected four-second 65-electrode normalized input')
        path = root/'subjects'/f'{c["key"]}.pt'
        atomic_save(dict(rows=rows,x=x),path)
        artifacts[str(path)] = digest(path)
        counts[c['dataset']][c['split']] += len(rows)
        counts[c['dataset']]['subjects'][c['split']] += 1
        report(root/'prepare_progress.json',status='preparing',completed=index+1,total=len(manifest['cohorts']))
    if source_identity()!=hashes:
        raise RuntimeError('Code changed during preparation')
    proof = dict(code_sha256=hashes,artifacts_sha256=artifacts,
        parent_checkpoint_sha256=digest(Path(CONFIG['parent'])/'best.pt'),parent_step=saved['step'],
        counts=dict(counts),preprocessing=adapter.audit,original_splits_modified=False,
        query_used_for_preprocessing=False,semantic_labels='Fixed cached MPNet captions, mean over templates; no text similarity decision',
        episodes=len(manifest['episodes']),episode_stream_sha256=digest(root/'manifest.json'))
    write_json(root/'prepared.json',proof)
    report(root/'prepare_progress.json',status='complete',counts=proof['counts'],episodes=proof['episodes'])


def guard_factory(root):
    hashes = source_identity()
    proof = json.loads((root/'prepared.json').read_text())
    if proof['code_sha256']!=hashes:
        raise ValueError('Prepared source differs')
    for path,sha in proof['artifacts_sha256'].items():
        if digest(path)!=sha:
            raise ValueError(f'Prepared input changed: {path}')
    parent = Path(CONFIG['parent'])
    if digest(parent/'best.pt')!=proof['parent_checkpoint_sha256']:
        raise ValueError('Parent weights changed')
    files = [Path(p) for p in proof['artifacts_sha256']]+[root/'prepared.json']
    files += [parent/n for n in ('best.pt','complete.json','config.json')]
    identity = json.loads((parent/'config.json').read_text())
    for name,expected in identity['jev_shard_stats'].items():
        path = Path('laya/JEV-27B-VL')/name
        if [path.stat().st_size,path.stat().st_mtime_ns]!=expected:
            raise ValueError('Native JEV differs from parent')
        files.append(path)
    stats = {str(p):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in files}
    def guard():
        if (root/'cancelled.json').exists() or hashes!=source_identity():
            raise RuntimeError('Run cancelled or live source changed')
        if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ctime_ns)!=s for p,s in stats.items()):
            raise RuntimeError('Live input/checkpoint identity changed')
    return proof,hashes,guard


def load_server(saved,encoder):
    from .jev import JEVBridge, load_jev
    text,weight,bias,tokenizer = load_jev('cuda')
    server = JEVBridge(text,weight,bias,tokenizer,True,True,head_choices=16,input_dim=256)
    encoder.requires_grad_(True)
    server.add_module('eeg_encoder',encoder)
    restore_model(server,saved)
    server.requires_grad_(False).eval()
    server.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if any(isinstance(m,torch.nn.Dropout) and m.p for m in server.text.modules()):
        raise ValueError('Unexpected language dropout')
    return server,tokenizer


@lru_cache(maxsize=80)
def subject_cache(root,key):
    value = torch.load(Path(root)/'subjects'/f'{key}.pt',map_location='cpu',weights_only=False,mmap=True)
    value['lookup'] = {r['id']:i for i,r in enumerate(value['rows'])}
    return value


def inputs(root,key,ids):
    value = subject_cache(str(root),key)
    indices = [value['lookup'][i] for i in ids]
    return value['x'][indices].cuda(),torch.tensor([int(value['rows'][i]['label']) for i in indices],device='cuda')


def suffix_for(task,tokenizer):
    variant = make_variant(task,0,'label_independent',0,DATA['seed'],0)
    return torch.tensor(tokenizer.encode(variant['suffix'],add_special_tokens=False),device='cuda')


def make_context(root,cohort,support_ids,encoder,adapter,semantics,train_tower):
    x,y = inputs(root,cohort['key'],support_ids)
    with nullcontext() if train_tower else torch.no_grad():
        z = encoder(x)
    return adapter.summarize(z,semantics[cohort['dataset']][y])


@torch.no_grad()
def evaluate(root,cohorts,encoder,adapter,server,suffixes,semantics,rank,world,mode='correct',donors=None,cap=None):
    server.eval()
    results,predictions = {},{}
    for c in cohorts:
        context_cohort = c if mode!='wrong_user' else donors[c['dataset']]
        u = make_context(root,context_cohort,[r['id'] for r in context_cohort['support']],encoder,adapter,semantics,False)
        if mode=='zero_context':
            u = torch.zeros_like(u)
        rows = c['query'] if cap is None else c['query'][:cap]
        local = []
        indices = list(range(rank,len(rows),world))
        for offset in range(0,len(indices),CONFIG['eval_microbatch']):
            ix = indices[offset:offset+CONFIG['eval_microbatch']]
            x,_ = inputs(root,c['key'],[rows[i]['id'] for i in ix])
            z = encoder(x)
            if mode!='bypass':
                z = adapter(z,u)
            valid = torch.ones(z.shape[:2],dtype=torch.bool,device='cuda')
            scores = server(z,valid,suffix_ids=[suffixes[c['dataset']]]*len(z),num_choices=c['classes'])['logits']
            local.extend((i,v) for i,v in zip(ix,scores.cpu().tolist()))
        parts = [None]*world
        dist.all_gather_object(parts,local)
        merged = sorted(v for part in parts for v in part)
        if [i for i,_ in merged]!=list(range(len(rows))):
            raise RuntimeError('Missing/duplicate distributed evaluation row')
        logits = np.asarray([v for _,v in merged],dtype=np.float32)
        labels = [int(r['label']) for r in rows]
        results[c['key']] = classification_metrics(labels,logits,c['classes'])
        predictions[c['key']] = dict(ids=[r['id'] for r in rows],labels=labels,logits=torch.from_numpy(logits),context=u.cpu())
    grouped = defaultdict(list)
    for c in cohorts:
        grouped[c['dataset']].append(results[c['key']]['balanced_accuracy'])
    return dict(subjects=results,macro_bacc=float(np.mean([np.mean(v) for v in grouped.values()])),mode=mode),predictions


def train(root,arm,smoke=False,resume=False):
    rank,world = int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    if world!=2:
        raise ValueError('Matched pilot uses exactly two GPUs per arm')
    output = root/('smoke' if smoke else 'train')/arm
    output.mkdir(parents=True,exist_ok=True)
    stop = False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    proof,hashes,guard = guard_factory(root)
    manifest = json.loads((root/'manifest.json').read_text())
    cohorts = {c['key']:c for c in manifest['cohorts']}
    val = [c for c in cohorts.values() if c['split']=='valid']
    test = [c for c in cohorts.values() if c['split']=='test']
    donors = {name:next(c for c in cohorts.values() if c['dataset']==name and c['split']=='train') for name in CONFIG['datasets']}
    saved = load_parent()
    encoder = load_encoder(saved)
    if rank==0:
        report(output/'progress.json',status='loading_jev',arm=arm,smoke=smoke)
    server,tokenizer = load_server(saved,encoder)
    del saved
    train_tower = arm=='tower_full'
    encoder.set_adaptation_enabled(train_tower)
    seed_rng(CONFIG['seed'])
    adapter = AmortizedFiLM(label_width=CONFIG['label_width'],context_width=CONFIG['context_width'],
                           bottleneck=CONFIG['adapter_width']).cuda()
    semantics = {k:v.cuda() for k,v in torch.load(root/'label_vectors.pt',weights_only=True).items()}
    from .benchmark_data import catalog
    suffixes = {t['task_id']:suffix_for(t,tokenizer) for t in catalog() if t['task_id'] in CONFIG['datasets']}
    groups = [dict(params=list(adapter.parameters()),lr=CONFIG['adapter_learning_rate'],name='adapter')]
    if train_tower:
        groups.append(dict(params=list(encoder.parameters()),lr=CONFIG['tower_learning_rate'],name='tower'))
    optimizer = torch.optim.AdamW(groups,weight_decay=CONFIG['weight_decay'])
    parameters = [p for group in groups for p in group['params']]
    if {id(p) for p in parameters} & {id(p) for n,p in server.named_parameters() if not n.startswith('eeg_encoder.')}:
        raise RuntimeError('Frozen server entered optimizer')
    initial_encoder,initial_adapter = module_hash(encoder),module_hash(adapter)
    encoder_buffers = tensor_fingerprint(dict(encoder.named_buffers()))
    server_versions = fixed_server_versions(server)
    server_buffers = tensor_fingerprint({n:b for n,b in server.named_buffers() if not n.startswith('eeg_encoder.')})
    total = CONFIG['smoke_steps'] if smoke else CONFIG['steps']
    initial_modules = {n:module_hash(getattr(adapter,n)) for n in ('label_projection','element','context','condition','down','up')}
    best,selected,start_step,history = -1.,0,0,[]
    def freeze_audit():
        if fixed_server_versions(server)!=server_versions:
            raise RuntimeError('Frozen LM/connector/head parameter changed')
        if any(p.requires_grad or p.grad is not None for n,p in server.named_parameters() if not n.startswith('eeg_encoder.')):
            raise RuntimeError('Frozen server received parameter gradient')
        if tensor_fingerprint({n:b for n,b in server.named_buffers() if not n.startswith('eeg_encoder.')})!=server_buffers:
            raise RuntimeError('Frozen server buffers changed')
        if tensor_fingerprint(dict(encoder.named_buffers()))!=encoder_buffers:
            raise RuntimeError('Tower BatchNorm/other buffers changed')
        if not train_tower and module_hash(encoder)!=initial_encoder:
            raise RuntimeError('Frozen Tower changed')
    def checkpoint(step,path):
        state = dict(step=step,arm=arm,smoke=smoke,code_sha256=hashes,parent_sha256=proof['parent_checkpoint_sha256'],
            encoder={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()},
            adapter={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
            optimizer=optimizer.state_dict(),rng=capture_rng(),best=best,selected_step=selected,history=history)
        if rank==0:
            atomic_save(state,path)
        dist.barrier()
    def restore(path,with_optimizer=False):
        state = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if (state['code_sha256']!=hashes or state['arm']!=arm or state['smoke']!=smoke
                or state['parent_sha256']!=proof['parent_checkpoint_sha256']):
            raise ValueError('Resume identity mismatch')
        encoder.load_state_dict(state['encoder'],strict=True)
        adapter.load_state_dict(state['adapter'],strict=True)
        if with_optimizer:
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'])
        return state
    cap = 8 if smoke else None
    if resume and (output/'last.pt').exists():
        state=restore(output/'last.pt',True)
        start_step,best,selected,history = state['step'],state['best'],state['selected_step'],state['history']
        del state
        baseline=json.loads((output/'baseline.json').read_text())
    else:
        baseline,_ = evaluate(root,val,encoder,adapter,server,suffixes,semantics,rank,world,mode='bypass',cap=cap)
        identity,_ = evaluate(root,val,encoder,adapter,server,suffixes,semantics,rank,world,cap=cap)
        if identity['subjects']!=baseline['subjects']:
            raise RuntimeError('Zero-up initialization changed baseline')
        best,selected = baseline['macro_bacc'],0
        if rank==0:
            write_json(output/'baseline.json',baseline)
            write_json(output/'valid_step000000.json',identity)
        checkpoint(0,output/'best.pt')
        checkpoint(0,output/'last.pt')
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,total+1):
        guard()
        if stop:
            checkpoint(step-1,output/'last.pt')
            raise RuntimeError('Worker requested stop')
        # Smoke uses source episodes too; held-out subjects only receive forward passes.
        episode=manifest['episodes'][step-1]
        c=cohorts[episode['key']]
        assert_episode(c,episode,training=True)
        local=rank_rows(episode['query_ids'],rank,world)
        server.text.train()  # Checkpoint backward to inputs; all LM weights still frozen.
        encoder.eval()
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.
        for offset in range(0,len(local),CONFIG['microbatch']):
            part=local[offset:offset+CONFIG['microbatch']]
            u=make_context(root,c,episode['support_ids'],encoder,adapter,semantics,train_tower)
            x,y=inputs(root,c['key'],part)
            with nullcontext() if train_tower else torch.no_grad():
                z=encoder(x)
            z=adapter(z,u)
            valid=torch.ones(z.shape[:2],dtype=torch.bool,device='cuda')
            loss=server(z,valid,labels=y,suffix_ids=[suffixes[c['dataset']]]*len(z),num_choices=c['classes'])['loss']
            (loss*len(part)/len(local)).backward()
            loss_sum+=float(loss.detach())*len(part)/len(local)
        average_gradients(parameters,world)
        norms={g['name']:float(torch.nn.utils.clip_grad_norm_(g['params'],CONFIG['clip_grad_norm'],error_if_nonfinite=True)) for g in groups}
        optimizer.step()
        value=torch.tensor(loss_sum,device='cuda')
        dist.all_reduce(value)
        history.append(dict(step=step,dataset=c['dataset'],subject=c['subject'],query_ce=float(value/world),grad_norm=norms))
        if step==1 or step%10==0 or step==total:
            freeze_audit()
            if rank==0:
                write_json(output/'history.json',history)
                report(output/'progress.json',status='training',arm=arm,smoke=smoke,step=step,total=total,
                    loss=history[-1]['query_ce'],best_validation_bacc=best,selected_step=selected,
                    elapsed_seconds=time.monotonic()-started,seconds_per_step=(time.monotonic()-started)/(step-start_step),
                    trainable_parameters=sum(p.numel() for p in parameters),grad_norm=norms)
        if step%CONFIG['validation_every']==0 or step==total:
            valid_result,_=evaluate(root,val,encoder,adapter,server,suffixes,semantics,rank,world,cap=cap)
            if rank==0:
                write_json(output/f'valid_step{step:06d}.json',valid_result)
            if valid_result['macro_bacc']>best:
                best,selected=valid_result['macro_bacc'],step
                checkpoint(step,output/'best.pt')
        if step%CONFIG['checkpoint_every']==0 or step==total:
            checkpoint(step,output/'last.pt')
    freeze_audit()
    encoder_updated=module_hash(encoder)!=initial_encoder
    modules_updated={n:module_hash(getattr(adapter,n))!=h for n,h in initial_modules.items()}
    if encoder_updated!=train_tower or not all(modules_updated.values()):
        raise RuntimeError(f'Unexpected updates: tower={encoder_updated}, modules={modules_updated}')
    final_hashes=(module_hash(encoder),module_hash(adapter))
    peers=[None]*world
    dist.all_gather_object(peers,final_hashes)
    if len(set(peers))!=1:
        raise RuntimeError('Rank model divergence')
    restore(output/'last.pt')
    if (module_hash(encoder),module_hash(adapter))!=final_hashes:
        raise RuntimeError('Checkpoint round trip changed weights')
    # Verify actual predictions across a reload, not just parameter names.
    probe,pred_before=evaluate(root,val[:1],encoder,adapter,server,suffixes,semantics,rank,world,cap=8)
    restore(output/'last.pt')
    _,pred_after=evaluate(root,val[:1],encoder,adapter,server,suffixes,semantics,rank,world,cap=8)
    if any(not torch.equal(pred_before[k]['logits'],pred_after[k]['logits']) for k in pred_before):
        raise RuntimeError('Checkpoint reload changed logits')
    result=dict(arm=arm,smoke=smoke,updates=total,code_sha256=hashes,
        episode_stream_sha256=proof['episode_stream_sha256'],initial_encoder_sha256=initial_encoder,
        initial_adapter_sha256=initial_adapter,encoder_updated=encoder_updated,modules_updated=modules_updated,
        frozen_server_verified=True,frozen_tower_buffers_verified=True,rank_weights_identical=True,
        checkpoint_reload_verified=True,baseline=baseline,selected_step=selected,best_validation_bacc=best,
        train_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        tower_parameters=sum(p.numel() for p in encoder.parameters()),adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        test_gradient_steps=0,test_used_for_selection=False)
    restore(output/'best.pt')
    if not smoke:
        before_target=(module_hash(encoder),module_hash(adapter),fixed_server_versions(server))
        target_results={}
        for mode in ('correct','wrong_user','zero_context','bypass'):
            metrics,pred=evaluate(root,test,encoder,adapter,server,suffixes,semantics,rank,world,mode=mode,donors=donors)
            target_results[mode]=metrics
            if rank==0:
                atomic_save(pred,output/f'test_{mode}_predictions.pt')
                write_json(output/f'test_{mode}.json',metrics)
        if (module_hash(encoder),module_hash(adapter),fixed_server_versions(server))!=before_target:
            raise RuntimeError('Target evaluation changed weights')
        result['test']=target_results
        result['target_forward_only_verified']=True
    freeze_audit()
    if rank==0:
        write_json(output/'complete.json',result)
        report(output/'progress.json',status='complete',**result)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--arm',choices=CONFIG['arms'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare(ROOT)
    else:
        train(ROOT,args.arm,args.smoke,args.resume)


if __name__=='__main__':
    main()
