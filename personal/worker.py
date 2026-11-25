"""One frozen server per GPU: source initialization learning, then matched users."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from jev_train.checkpoint import seed_rng, tensor_fingerprint
from jev_train.data import classification_metrics
from .common import CONFIG, OUT, ROOT, check_rows, check_source_rows, identity, write_json
from .data import TrialStore
from .meta import clone_initialization, first_order_gradients
from .model import PersonalServer
from .modules import UserParameters


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_state(user):
    return {n: p.detach().cpu().clone() for n, p in user.state_dict().items()}


def metrics(logits, rows, classes):
    return classification_metrics([r['label'] for r in rows], logits.softmax(-1).numpy(), classes)


def make_user(server, seed):
    seed_rng(seed)
    return UserParameters(CONFIG, server.model.text.config.hidden_size).cuda()


def optimizer_for(user, server, offline=False):
    lr = CONFIG['offline']['learning_rate'] if offline else CONFIG['learning_rate']
    optimizer = torch.optim.AdamW(user.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])
    ids = {id(p) for p in user.parameters()}
    if ids != {id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer parameter mismatch')
    if ids & {id(p) for p in server.model.parameters()}:
        raise RuntimeError('Optimizer includes frozen shared weights')
    return optimizer


def backward_batch(server, user, tokens, rows, task, step):
    # Explicit per-batch RNG makes source arms and resumed checkpoints comparable.
    seed_rng(CONFIG['seed'] + 1009 * step)
    user.zero_grad(set_to_none=True)
    total = 0.
    for offset in range(0, len(rows), CONFIG['microbatch']):
        part = rows[offset:offset + CONFIG['microbatch']]
        logits = server.forward(tokens[offset:offset + len(part)], part, task, user, step=step, training=True)
        target = torch.tensor([r['label'] for r in part], device='cuda')
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite CE')
        weight = len(part) / len(rows)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    if any(p.grad is None or not p.grad.isfinite().all() for p in user.parameters()):
        raise RuntimeError('Missing/nonfinite personal gradient')
    return total


def update(user, optimizer):
    norm = torch.nn.utils.clip_grad_norm_(user.parameters(), CONFIG['clip_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def encoded(server, store, rows, directory, stamp):
    path = directory / 'features.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['stamp'] != stamp or saved['ids'] != [r['id'] for r in rows]:
            raise RuntimeError('Frozen feature cache identity changed')
        return saved['tokens']
    tokens = server.encode(store, rows)
    atomic_save(dict(stamp=stamp, ids=[r['id'] for r in rows], tokens=tokens), path)
    return tokens


def smoke(server, store, case, directory):
    byid = {r['id']: r for r in store.rows[case['task']]}
    rows = [byid[i] for i in case['support_ids'][:2]]
    check_source_rows(rows, case, 'support')
    tokens = server.encode(store, rows)
    task = store.tasks[case['task']]
    baseline = server.predict(tokens, rows, task)
    initial = make_user(server, CONFIG['seed'])
    torch.testing.assert_close(server.predict(tokens, rows, task, initial), baseline, rtol=0, atol=0)
    original = tensor_fingerprint(initial.state_dict())
    adapted = clone_initialization(initial)
    inner = optimizer_for(adapted, server)
    loss = backward_batch(server, adapted, tokens, rows, task, 1)
    norm = update(adapted, inner)
    if norm <= 0 or tensor_fingerprint(adapted.state_dict()) == original:
        raise RuntimeError('No real personal update')
    if tensor_fingerprint(initial.state_dict()) != original:
        raise RuntimeError('Inner loop mutated meta initialization')
    # Real outer gradient must use a disjoint source-train query batch.
    qrows = [byid[i] for i in case['query_ids'][:2]]
    check_source_rows(qrows, case, 'query')
    qt = server.encode(store, qrows)
    qloss = backward_batch(server, adapted, qt, qrows, task, 2)
    outer = optimizer_for(initial, server, offline=True)
    first_order_gradients(initial, adapted)
    update(initial, outer)
    if tensor_fingerprint(initial.state_dict()) == original:
        raise RuntimeError('No meta-initialization update')
    before = server.predict(tokens, rows, task, initial)
    restored = make_user(server, CONFIG['seed'] + 1)
    restored.load_state_dict(cpu_state(initial), strict=True)
    torch.testing.assert_close(before, server.predict(tokens, rows, task, restored), rtol=0, atol=0)
    server.assert_frozen()
    write_json(directory / 'smoke.json', dict(passed=True, source_train_only=True, identity_exact=True,
        inner_isolated=True, real_outer_query_gradient=True, frozen_shared_weights=True, reload_exact=True,
        signal_tokens=tokens.shape[1], trainable_parameters=sum(p.numel() for p in initial.parameters()),
        support_ce=loss, query_ce=qloss, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))


def offline_train(server, store, plan, job, directory, deadline, stamp, progress):
    sources = {c['key']: c for c in plan['sources'] if c['dataset'] == job['dataset']}
    packs = {}
    for key, case in sources.items():
        progress('encoding_source', source=key)
        byid = {r['id']: r for r in store.rows[case['task']]}
        s = [byid[i] for i in case['support_ids']]
        q = [byid[i] for i in case['query_ids']]
        check_source_rows(s, case, 'support')
        check_source_rows(q, case, 'query')
        tokens = encoded(server, store, s + q, directory / 'sources' / key, stamp)
        packs[key] = dict(case=case, support=s, query=q, support_tokens=tokens[:len(s)], query_tokens=tokens[len(s):])
    initial = make_user(server, CONFIG['seed'])
    starting_sha = tensor_fingerprint(initial.state_dict())
    outer = optimizer_for(initial, server, offline=True)
    state_path = directory / 'offline_last.pt'
    done, history = 0, []
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['stamp'] != stamp or saved['job'] != job:
            raise RuntimeError('Offline resume identity mismatch')
        initial.load_state_dict(saved['user'], strict=True)
        outer.load_state_dict(saved['optimizer'])
        done, history = saved['episode'], saved['history']

    def save():
        server.assert_frozen()
        atomic_save(dict(stamp=stamp, job=job, user=cpu_state(initial), optimizer=outer.state_dict(),
                         episode=done, history=history, starting_sha=starting_sha), state_path)

    for episode in plan['schedules'][job['dataset']][done:]:
        if time.time() >= deadline:
            save()
            return None
        started = time.time()
        if identity() != stamp['source_hashes']:
            raise RuntimeError('Live source changed')
        pack = packs[episode['source']]
        case = pack['case']
        task = store.tasks[case['task']]
        if job['arm'] == 'fomaml':
            adapted = clone_initialization(initial)
            inner = optimizer_for(adapted, server)
            pre_inner_sha = tensor_fingerprint(initial.state_dict())
        else:
            adapted, inner = initial, outer
        losses, norms = [], []
        for batch in episode['batches']:
            part, ix = batch['part'], batch['indices']
            rows = [pack[part][i] for i in ix]
            if [r['id'] for r in rows] != batch['ids']:
                raise RuntimeError('Matched source batch changed')
            check_source_rows(rows, case, part)
            loss = backward_batch(server, adapted, pack[part + '_tokens'][ix], rows, task, batch['prompt_step'])
            if job['arm'] == 'fomaml' and part == 'query':
                if tensor_fingerprint(initial.state_dict()) != pre_inner_sha:
                    raise RuntimeError('Inner loop changed meta initialization')
                outer.zero_grad(set_to_none=True)
                first_order_gradients(initial, adapted)
                norm = update(initial, outer)
            else:
                norm = update(adapted, inner)
            losses.append(loss)
            norms.append(norm)
        done = episode['episode']
        history.append(dict(episode=done, source=case['key'], support_ce=float(np.mean(losses[:-1])),
            query_ce=losses[-1], batch_losses=losses, grad_norms=norms, seconds=time.time()-started,
            gradient_batches=done*(CONFIG['offline']['inner_steps']+1),
            sample_presentations=done*(CONFIG['offline']['inner_steps']+1)*CONFIG['batch_size']))
        save()
        progress('offline_training', **history[-1])
        del adapted, inner
    atomic_save(cpu_state(initial), directory / 'initialization.pt')
    write_json(directory / 'offline_complete.json', dict(episodes=done, job=job, stamp=stamp,
        trainable_parameters=sum(p.numel() for p in initial.parameters()), history=history,
        starting_sha=starting_sha, final_sha=tensor_fingerprint(initial.state_dict()),
        query_means_source_train=True, target_users_used=False,
        outer_optimizer_steps=done if job['arm']=='fomaml' else done*(CONFIG['offline']['inner_steps']+1)))
    return cpu_state(initial)


def run_person(server, store, case, initialization, job, directory, deadline, stamp, progress):
    byid = {r['id']: r for r in store.rows[case['task']]}
    support = [byid[i] for i in case['support_ids']]
    query = [byid[i] for i in case['query_ids']]
    check_rows(support, case, gradient=True)
    check_rows(query, case)
    task = store.tasks[case['task']]
    directory = directory / 'cases' / case['key']
    progress('encoding_target', case=case['key'])
    tokens = encoded(server, store, support + query, directory, stamp)
    st, qt = tokens[:len(support)], tokens[len(support):]
    baseline = server.predict(qt, query, task)
    prior_path = ROOT / CONFIG['paired_reference'] / 'cases' / case['key'] / 'connector_lora' / 'predictions.npz'
    with np.load(prior_path) as prior:
        if prior['query_ids'].tolist() != case['query_ids']:
            raise RuntimeError('Historical target query IDs changed')
        historical = torch.from_numpy(prior['baseline_logits'])
        if not torch.equal(historical.argmax(-1), baseline.argmax(-1)):
            raise RuntimeError('Historical no-adaptation baseline not reproduced')
    # Run a fresh ordinary personal LoRA as the no-offline control, not copied old scores.
    for arm in ('ordinary_lora', job['arm']):
        arm_dir = directory / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        result_path = arm_dir / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['stamp'] != stamp or result['case'] != case:
                raise RuntimeError('Completed result identity mismatch')
            continue
        user = make_user(server, CONFIG['seed'] + case['order'] * 101)
        if arm != 'ordinary_lora':
            user.load_state_dict(initialization, strict=True)
        optimizer = optimizer_for(user, server)
        initial_logits = server.predict(qt, query, task, user)
        step, history = 0, []
        state_path = arm_dir / 'last.pt'
        if state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['stamp'] != stamp or saved['case'] != case or saved['arm'] != arm:
                raise RuntimeError('Personal resume identity mismatch')
            user.load_state_dict(saved['user'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step, history, initial_logits = saved['step'], saved['history'], saved['initial_logits']

        def save():
            server.assert_frozen()
            atomic_save(dict(stamp=stamp, case=case, arm=arm, user=cpu_state(user),
                optimizer=optimizer.state_dict(), step=step, history=history, initial_logits=initial_logits), state_path)

        for next_step in range(step + 1, CONFIG['steps'] + 1):
            if time.time() >= deadline:
                save()
                return False
            started = time.time()
            rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], case['order'], next_step]))
            ix = rng.integers(len(support), size=CONFIG['batch_size']).tolist()
            rows = [support[i] for i in ix]
            check_rows(rows, case, gradient=True)
            loss = backward_batch(server, user, st[ix], rows, task, next_step)
            norm = update(user, optimizer)
            step = next_step
            history.append(dict(step=step, ce=loss, grad_norm=norm, lr=CONFIG['learning_rate'], seconds=time.time()-started))
            if step % 10 == 0 or step == CONFIG['steps']:
                save()
                progress('personal_adaptation', case=case['key'], personal_arm=arm, **history[-1])
        progress('evaluating_personal', case=case['key'], personal_arm=arm)
        after = server.predict(qt, query, task, user)
        support_logits = server.predict(st, support, task, user)
        seen = set()
        for s in range(1, step + 1):
            rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], case['order'], s]))
            seen.update(rng.integers(len(support), size=CONFIG['batch_size']).tolist())
        if len(seen) != len(support):
            raise RuntimeError('Not all personal support samples were used')
        server.assert_frozen()
        atomic_save(cpu_state(user), arm_dir / 'user.pt')
        np.savez_compressed(arm_dir / 'predictions.npz', query_ids=np.asarray(case['query_ids']),
            labels=np.asarray([r['label'] for r in query]), baseline_logits=baseline.numpy(),
            initial_logits=initial_logits.numpy(), logits=after.numpy())
        base_metric, metric = metrics(baseline, query, case['classes']), metrics(after, query, case['classes'])
        write_json(result_path, dict(status='complete', arm=arm, job=job, case=case, stamp=stamp,
            support_n=len(support), support_samples_used=len(seen), query_n=len(query), baseline=base_metric, adapted=metric,
            initial=metrics(initial_logits, query, case['classes']),
            support=metrics(support_logits, support, case['classes']),
            delta_pp=100*(metric['balanced_accuracy']-base_metric['balanced_accuracy']),
            steps=step, history=history, trainable_parameters=sum(p.numel() for p in user.parameters()),
            shared_weights_frozen=True, query_used_for_gradients=False, query_used_for_selection=False,
            independent_test_estimate=False, historical_baseline_predictions_identical=True))
        progress('personal_complete', case=case['key'], personal_arm=arm, bacc=metric['balanced_accuracy'])
        del user, optimizer
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_device(0)
    job = CONFIG['jobs'][args.worker]
    directory = OUT / 'jobs' / f"{job['dataset']}_{job['arm']}"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = identity()
    plan = json.loads((OUT / 'plan.json').read_text())
    queue = json.loads((OUT / 'queue.json').read_text())
    if plan['source_hashes'] != hashes or queue['source_hashes'] != hashes:
        raise RuntimeError('Plan/queue identity mismatch')
    stamp = dict(source_hashes=hashes, parent_sha256=queue['parent_sha256'])
    def progress(state, **kw):
        report = dict(state=state, job=job, time=time.time(), pid=os.getpid(), **kw)
        write_json(OUT / f'progress_gpu{args.worker}.json', report)
        print(json.dumps(report), flush=True)
    progress('loading_frozen_parent')
    server = PersonalServer(queue['parent_sha256'])
    store = TrialStore()
    if json.loads(json.dumps(store.stats)) != plan['source_array_stats']:
        raise RuntimeError('Input array identities changed')
    source = next(c for c in plan['sources'] if c['dataset'] == job['dataset'])
    progress('real_gradient_smoke')
    smoke(server, store, source, directory)
    initialization = offline_train(server, store, plan, job, directory, queue['deadline'], stamp, progress)
    if initialization is None:
        progress('budget_exhausted')
        return
    for case in plan['cases']:
        if case['dataset'] == job['dataset']:
            if not run_person(server, store, case, initialization, job, directory, queue['deadline'], stamp, progress):
                progress('budget_exhausted')
                return
    write_json(directory / 'complete.json', dict(job=job, stamp=stamp, time=time.time()))
    progress('worker_done')


if __name__ == '__main__':
    main()
