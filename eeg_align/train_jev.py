"""Four-rank EEG attribute QA; frozen REVE, active/updatable JEV LoRA."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .cache import ROOT, digest
from .paired_cache import PairedTokenCache, ROOT as TOKEN_ROOT
from .jev import load_jev, JEVBridge, CHECKPOINT
from .sampling import BalancedCoverageSampler
from .distributed import average_gradients, atomic_save
from .attribute_data import AttributeData, QuestionTokens, semantic_predictions, summarize, draw_permutations, mask_choices, training_weights


def cpu_trainable(model, names):
    params = dict(model.named_parameters())
    return {n:params[n].detach().cpu().clone() for n in names}


def restore_trainable(model, state, names):
    if set(state) != set(names):raise ValueError('Trainable parameter set changed')
    model.load_state_dict(state, strict=False)


def rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']);np.random.set_state(state['numpy'])


@torch.no_grad()
def validate(model, cache, data, questions, ids, rank, world, microbatch=6, shuffle=False):
    model.eval()
    ids = np.asarray(ids, dtype=np.int64)
    # A deterministic, target-independent choice order, unchanged across checkpoints.
    permutations = draw_permutations(data,data.records,np.random.default_rng(20261018))
    signal_ids = data.records[:,0].copy()
    if shuffle:
        rng = np.random.default_rng(1701)
        # Preserve dataset and question, scramble EEG identity within each stratum.
        for q, ds in sorted(set(map(tuple, data.records[ids][:,[1,3]]))):
            selected = ids[(data.records[ids,1] == q) & (data.records[ids,3] == ds)]
            signal_ids[selected] = rng.permutation(signal_ids[selected])
    mine = ids[rank::world]; predictions = []; losses = []; letters_correct = []
    started = time.monotonic()
    for start in range(0, len(mine), microbatch):
        ii = mine[start:start+microbatch]
        z, mask, segments = cache.batch(signal_ids[ii], 'cuda')
        suffix, labels = questions.batch(data.records[ii], permutations[ii])
        logits = model(z, mask, suffix_ids=suffix, num_choices=3, segment_splits=segments)['logits']
        logits = mask_choices(logits,data.choice_counts[data.records[ii,1]])
        letter = logits.argmax(1).cpu().numpy()
        predictions.extend(semantic_predictions(letter, permutations[ii],data.choice_counts[data.records[ii,1]]).tolist())
        losses.extend(F.cross_entropy(logits, labels, reduction='none').cpu().tolist())
        letters_correct.extend((letter == labels.cpu().numpy()).tolist())
        if rank == 0 and start % 600 == 0:
            print('QA validation', 'shuffled' if shuffle else 'matched', start, '/', len(mine), flush=True)
    parts = [None]*world
    dist.all_gather_object(parts, (mine.tolist(), predictions, losses, letters_correct))
    combined = {i:(p,l,c) for ii,pp,ll,cc in parts for i,p,l,c in zip(ii,pp,ll,cc)}
    if len(combined) != len(ids):raise ValueError('Distributed validation coverage error')
    result = summarize(data, ids, np.array([combined[int(i)][0] for i in ids]))
    result.update(loss=float(np.mean([v[1] for v in combined.values()])),
                  letter_accuracy=float(np.mean([v[2] for v in combined.values()])),
                  seconds=time.monotonic()-started, shuffled_signal=shuffle)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path, default=Path('artifacts/jev_paired_qa/initialization.pt'))
    p.add_argument('--train-data', type=Path, default=Path('datasets/eeg_attribute_qa'))
    p.add_argument('--validation-data', type=Path, default=Path('datasets/eeg_attribute_qa_validation'))
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    if rank == 0:
        if args.resume:
            assert (args.out/'last.pt').exists() and not (args.out/'complete.json').exists()
        else:args.out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    train = AttributeData(args.train_data, 'train')
    val = AttributeData(args.validation_data, 'validation')
    assert train.specs == val.specs
    assert not set(train.records[:,0]) & set(val.records[:,0])
    assert len(train.groups) == 56 and len(val.groups) == 56
    batch_size=len(train.groups);nq=len(train.specs);question_weights=training_weights(train)
    steps = 2 if args.smoke else args.steps
    # No example can recur more than four times in this bounded run.
    assert steps <= 4 * min(map(len, train.groups.values()))
    sampler = BalancedCoverageSampler(train.groups, per_group=1, seed=args.seed)
    panel = val.panel(per_group=1 if args.smoke else 16)
    cache = PairedTokenCache()
    text, w, b, tok = load_jev()
    model = JEVBridge(text, w, b, tok, train_lora=True)
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    saved = torch.load(args.init, map_location='cpu', weights_only=False)
    assert saved['config']['train_lora'] and not saved['config']['smoke']
    assert saved['config']['manifest_sha256'] == digest(ROOT/'rows.jsonl')
    restore_trainable(model, saved['trainable'], names)
    init_step = saved['step'];del saved
    questions = QuestionTokens(tok, train.specs)
    opt = torch.optim.AdamW([
        dict(params=list(model.connector.parameters()), lr=5e-5),
        dict(params=[p for p in model.text.parameters() if p.requires_grad], lr=5e-6)], weight_decay=.01)
    paths = ['eeg_align/train_jev.py','eeg_align/jev.py','eeg_align/attribute_data.py',
             'eeg_align/paired_cache.py','eeg_align/sampling.py','eeg_align/distributed.py','eeg_tasks/attribute_questions.py']
    config = dict(stage='paired_attribute_qa', seed=args.seed, steps=steps, world_size=world, batch_size=batch_size,
        microbatch=6, learning_rates=dict(projector=5e-5,lora=5e-6),
        train_lora=True, train_encoder=False, jev_lora_always_active=True, smoke=args.smoke,
        init=str(args.init.resolve()), init_step=init_step, init_sha256=digest(args.init),
        checkpoint=str(CHECKPOINT.resolve()), trainable_names=names,
        train_data_sha256=train.root_hash, validation_data_sha256=val.root_hash,
        manifest_sha256=digest(ROOT/'rows.jsonl'), validation_ids=panel.tolist(),
        validation_n=len(panel), eval_every=args.eval_every,
        selection='fixed source-subject validation panel; task/question/dataset macro semantic balanced accuracy',
        sampler='one draw per question x dataset x semantic answer; without replacement within group until exhausted',
        max_repeats=4, task_loss_mass=dict(mi_attributes=.25,mi_classification=.25,emotion_attributes=.5),
        question_weights=question_weights.tolist(),token_audit_sha256=digest(TOKEN_ROOT/'verified.json'),
        code_sha256={f:digest(f) for f in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    best = -1.;prior_seconds = 0.;start_step = 0
    if args.resume:
        saved = torch.load(args.out/'last.pt', map_location='cpu', weights_only=False)
        for k,v in config.items():
            if k != 'git_commit' and saved['config'][k] != v:raise ValueError(f'Resume mismatch: {k}')
        config = saved['config']
        restore_trainable(model, saved['trainable'], names);opt.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler']);restore_rng(saved['rng_states'][rank])
        best = saved['best_score'];prior_seconds = saved['elapsed_seconds'];start_step = saved['step']
        assert sampler.step == start_step
        del saved
    elif rank == 0:
        (args.out/'config.json').write_text(json.dumps(config,indent=2))
    dist.barrier();began = time.monotonic()
    log = (args.out/'metrics.jsonl').open('a', buffering=1) if rank == 0 else None

    def evaluate(step):
        nonlocal best
        result = validate(model,cache,val,questions,panel,rank,world)
        if rank == 0:
            log.write(json.dumps(dict(step=step,validation=result))+'\n')
            (args.out/f'validation_{step:04d}.json').write_text(json.dumps(result,indent=2))
            print('VALIDATION',step,'MI attributes',result['MI'],'MI classification',result['MI_classification'],'emotion',result['emotion'],'loss',result['loss'],flush=True)
        if result['selection_score'] > best:
            best = result['selection_score']
            if rank == 0:
                atomic_save(dict(trainable=cpu_trainable(model,names),encoder=None,config=config,
                                 step=step,validation=result),args.out/'best.pt')
                (args.out/'best.json').write_text(json.dumps(dict(step=step,validation=result),indent=2))
        return result

    if not args.resume:
        evaluate(0)
    for step in range(start_step+1, steps+1):
        model.train();opt.zero_grad(set_to_none=True)
        global_ids = sampler.next_batch().reshape(-1)
        rng = np.random.default_rng(args.seed+step*1009)
        rng.shuffle(global_ids)
        permutations = draw_permutations(train,train.records[global_ids],rng)
        ids = global_ids[rank::world];perms = permutations[rank::world]
        loss_sum = 0.;correct_sum = 0.; semantic_correct = np.zeros(nq); semantic_count = np.zeros(nq)
        for start in range(0,len(ids),6):
            ii = ids[start:start+6];records = train.records[ii]
            z,mask,segments = cache.batch(records[:,0],'cuda')
            suffix, labels = questions.batch(records,perms[start:start+6])
            logits = model(z,mask,suffix_ids=suffix,num_choices=3,segment_splits=segments)['logits']
            logits = mask_choices(logits,train.choice_counts[records[:,1]])
            per_sample = F.cross_entropy(logits,labels,reduction='none')
            weights = torch.tensor(question_weights[records[:,1]],device='cuda')
            # Exact global weighted mean after cross-rank gradient averaging.
            loss = (per_sample*weights).sum()*world/len(global_ids)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();loss_sum += loss.item()
            correct = (logits.argmax(1)==labels).detach().cpu().numpy()
            correct_sum += correct.sum()
            np.add.at(semantic_correct,records[:,1],correct)
            np.add.at(semantic_count,records[:,1],1)
        active = [p for group in opt.param_groups for p in group['params']]
        average_gradients(active,world)
        grad = torch.nn.utils.clip_grad_norm_(active,1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        stats = torch.tensor([loss_sum/world,correct_sum,*semantic_correct,*semantic_count],device='cuda')
        dist.all_reduce(stats)
        if args.smoke:
            h = hashlib.sha256()
            for v in cpu_trainable(model,names).values():h.update(v.numpy().tobytes())
            hashes = [None]*world;dist.all_gather_object(hashes,h.hexdigest())
            assert len(set(hashes)) == 1, 'Rank parameters diverged'
            if rank == 0:print('Four-rank parameter synchronization verified',step,flush=True)
        if rank == 0 and (step == 1 or step%10 == 0 or args.smoke):
            item = dict(step=step,loss=float(stats[0]),accuracy=float(stats[1])/batch_size,
                question_accuracy={s['id']:float(stats[2+q]/stats[2+nq+q]) for q,s in enumerate(train.specs)},
                gradient_norm=float(grad),elapsed_seconds=prior_seconds+time.monotonic()-began,
                draws=step*batch_size,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
        endpoint = step == steps
        if step%args.eval_every == 0 or endpoint:evaluate(step)
        if step%100 == 0 or endpoint or step == args.stop_after or args.smoke:
            states = [None]*world;dist.all_gather_object(states,rng_state())
            if rank == 0:
                atomic_save(dict(trainable=cpu_trainable(model,names),optimizer=opt.state_dict(),
                    sampler=sampler.state_dict(),config=config,step=step,best_score=best,
                    rng_states=states,elapsed_seconds=prior_seconds+time.monotonic()-began),args.out/'last.pt')
                (args.out/'progress.json').write_text(json.dumps(dict(step=step,steps=steps),indent=2))
            dist.barrier()
        if step == args.stop_after and not endpoint:
            dist.destroy_process_group();return
    # Final held-out attributes, never external MI/emotion task labels.
    if not args.smoke:
        saved = torch.load(args.out/'best.pt',map_location='cpu',weights_only=False)
        restore_trainable(model,saved['trainable'],names);selected_step=saved['step'];del saved
        full = validate(model,cache,val,questions,np.arange(len(val.records)),rank,world)
        shuffled = validate(model,cache,val,questions,panel,rank,world,shuffle=True)
        if rank == 0:
            (args.out/'full_validation.json').write_text(json.dumps(dict(step=selected_step,validation=full),indent=2))
            (args.out/'shuffled_validation.json').write_text(json.dumps(dict(step=selected_step,validation=shuffled),indent=2))
    if rank == 0:
        (args.out/'complete.json').write_text(json.dumps(dict(steps=steps,smoke=args.smoke,
            best_sha256=digest(args.out/'best.pt'),elapsed_seconds=prior_seconds+time.monotonic()-began),indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':main()
