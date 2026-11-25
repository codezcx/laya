"""One user per GPU; matched connector-only and full-Tower personal adaptation."""
import argparse
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
from .data import TrialStore, SignalBatch
from .initialization import load_initialization
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
    result = classification_metrics([r['label'] for r in rows], logits.softmax(-1).numpy(), classes)
    result['ce'] = float(F.cross_entropy(logits.float(), torch.tensor([r['label'] for r in rows])))
    return result


def make_user(server, seed):
    seed_rng(seed)
    return UserParameters(CONFIG, server.model.text.config.hidden_size).cuda()


def optimizer_for(user, server):
    lr = CONFIG['learning_rate']
    groups = [dict(params=list(user.parameters()), lr=lr)]
    if server.tower_enabled:
        groups.append(dict(params=server.tower_parameters(), lr=CONFIG['tower_learning_rate']))
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    ids = {id(p) for p in list(user.parameters()) + server.tower_parameters()}
    if ids != {id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer parameter mismatch')
    if ids & {id(p) for n, p in server.model.named_parameters() if not n.startswith('eeg_encoder.')}:
        raise RuntimeError('Optimizer includes frozen shared weights')
    return optimizer


def backward_batch(server, user, tokens, rows, task, step):
    # Explicit per-batch RNG makes source arms and resumed checkpoints comparable.
    seed_rng(CONFIG['seed'] + 1009 * step)
    user.zero_grad(set_to_none=True)
    server.model.eeg_encoder.zero_grad(set_to_none=True)
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
    if any(p.grad is None or not p.grad.isfinite().all()
           for p in list(user.parameters()) + server.tower_parameters()):
        raise RuntimeError('Missing/nonfinite personal gradient')
    return total


def update(user, optimizer):
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG['clip_grad_norm'], error_if_nonfinite=True)
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


def smoke(server, tokens, support, case, task, directory):
    rows = support[:2]
    check_rows(rows, case, gradient=True)
    baseline = server.predict(tokens[:2], rows, task)
    user = make_user(server, CONFIG['initialization_seed'])
    torch.testing.assert_close(server.predict(tokens[:2], rows, task, user), baseline, rtol=0, atol=0)
    original = tensor_fingerprint(user.state_dict())
    original_tower = tensor_fingerprint(server.model.eeg_encoder.state_dict())
    optimizer = optimizer_for(user, server)
    loss = backward_batch(server, user, tokens[:2], rows, task, 1)
    norm = update(user, optimizer)
    if norm <= 0 or tensor_fingerprint(user.state_dict()) == original:
        raise RuntimeError('No real personal update')
    changed = tensor_fingerprint(server.model.eeg_encoder.state_dict()) != original_tower
    if changed != server.tower_enabled:
        raise RuntimeError('Tower update does not match the assigned arm')
    before = server.predict(tokens[:2], rows, task, user)
    restored = make_user(server, CONFIG['initialization_seed'])
    restored.load_state_dict(cpu_state(user), strict=True)
    if server.tower_enabled:
        tower_state = server.tower_state()
        server.load_tower(server.initial_tower)
        server.load_tower(tower_state)
    torch.testing.assert_close(before, server.predict(tokens[:2], rows, task, restored), rtol=0, atol=0)
    server.assert_frozen()
    write_json(directory / 'smoke.json', dict(passed=True, support_only=True,
        identity_exact=True, real_update=True, frozen_jev_and_original_connector=True, reload_exact=True,
        tower_updated=changed,
        trainable_parameters=sum(p.numel() for p in list(user.parameters()) + server.tower_parameters()),
        support_ce=loss, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))


def run_person(server, store, plan, case, job, directory, deadline, stamp, progress):
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
    base_metric = metrics(baseline, query, case['classes'])
    cached_st, cached_qt = st, qt
    prepared = None
    # Start the new Tower arm first; the already-completed historical control is
    # still rerun with identical inputs and all arms remain independent resets.
    for arm in reversed(CONFIG['arms']):
        arm_dir = directory / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        curve_dir = arm_dir / 'curves'
        curve_dir.mkdir(exist_ok=True)
        result_path = arm_dir / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['stamp'] != stamp or result['case'] != case:
                raise RuntimeError('Completed result identity mismatch')
            continue
        server.configure_arm(arm)
        if arm == 'tower_full':
            if prepared is None:
                progress('preparing_online_tower_signals', case=case['key'])
                prepared = SignalBatch.prepare(store, support + query)
            st, qt = prepared[:len(support)], prepared[len(support):]
        else:
            st, qt = cached_st, cached_qt
        progress('real_gradient_smoke', case=case['key'], personal_arm=arm)
        smoke(server, st, support, case, task, arm_dir)
        server.configure_arm(arm)  # The smoke update must never become initialization.
        init_record = plan['initializations'][case['dataset']][arm]
        user = make_user(server, CONFIG['initialization_seed'])
        user.load_state_dict(load_initialization(init_record), strict=True)
        optimizer = optimizer_for(user, server)
        step, history = 0, []
        state_path = arm_dir / 'last.pt'
        if state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['stamp'] != stamp or saved['case'] != case or saved['arm'] != arm:
                raise RuntimeError('Personal resume identity mismatch')
            user.load_state_dict(saved['user'], strict=True)
            if server.tower_enabled:
                server.load_tower(saved['tower'])
            optimizer.load_state_dict(saved['optimizer'])
            step, history = saved['step'], saved['history']

        def save():
            server.assert_frozen()
            atomic_save(dict(stamp=stamp, case=case, arm=arm, user=cpu_state(user),
                tower=server.tower_state() if server.tower_enabled else None,
                optimizer=optimizer.state_dict(), step=step, history=history,
                initialization=init_record), state_path)

        def evaluate():
            if step not in CONFIG['eval_steps']:
                return
            path = curve_dir / f'step{step:06d}.json'
            state_sha = tensor_fingerprint(user.state_dict())
            tower_sha = tensor_fingerprint(server.model.eeg_encoder.state_dict())
            if path.exists():
                point = json.loads(path.read_text())
                if (point['stamp'] != stamp or point['case'] != case or point['arm'] != arm or
                        point['step'] != step or point['user_sha256'] != state_sha or
                        point['tower_sha256'] != tower_sha):
                    raise RuntimeError('Saved curve point differs from resumed state')
                return
            progress('evaluating_curve', case=case['key'], personal_arm=arm, step=step)
            started = time.time()
            rng = capture_rng()
            try:
                after = server.predict(qt, query, task, user)
                support_logits = server.predict(st, support, task, user)
            finally:
                restore_rng(rng)
            if tensor_fingerprint(user.state_dict()) != state_sha:
                raise RuntimeError('Evaluation changed personal parameters')
            if tensor_fingerprint(server.model.eeg_encoder.state_dict()) != tower_sha:
                raise RuntimeError('Evaluation changed Tower parameters/buffers')
            server.assert_frozen()
            if step == 0 and arm == 'ordinary_lora':
                torch.testing.assert_close(after, baseline, rtol=0, atol=0)
            if step == 0 and arm == 'tower_full':
                # Online microbatch kernels can differ slightly from the encoding cache.
                torch.testing.assert_close(after, baseline, rtol=0.01, atol=0.02)
                if not torch.equal(after.argmax(-1), baseline.argmax(-1)):
                    raise RuntimeError('Online Tower changed step-zero predictions')
            weight_path = curve_dir / f'step{step:06d}.pt'
            atomic_save(dict(user=cpu_state(user),
                             tower=server.tower_state() if server.tower_enabled else None), weight_path)
            pred_path = curve_dir / f'step{step:06d}.npz'
            tmp = pred_path.with_suffix('.npz.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, query_ids=np.asarray(case['query_ids']),
                    support_ids=np.asarray(case['support_ids']),
                    labels=np.asarray([r['label'] for r in query]),
                    support_labels=np.asarray([r['label'] for r in support]),
                    baseline_logits=baseline.numpy(), logits=after.numpy(), support_logits=support_logits.numpy())
            tmp.replace(pred_path)
            metric = metrics(after, query, case['classes'])
            seen = {i for b in plan['schedules'][case['key']][:step] for i in b['indices']}
            point = dict(arm=arm, case=case, stamp=stamp, step=step, baseline=base_metric, query=metric,
                support=metrics(support_logits, support, case['classes']),
                support_samples_used=len(seen), user_sha256=state_sha, tower_sha256=tower_sha,
                initialization=init_record,
                seconds=time.time()-started, query_used_for_gradients=False,
                query_used_for_selection=False, independent_test_estimate=False)
            # JSON is the commit marker: weights and predictions are already durable.
            write_json(path, point)
            progress('curve_complete', case=case['key'], personal_arm=arm, step=step,
                query_bacc=metric['balanced_accuracy'], eval_seconds=point['seconds'])

        for earlier in CONFIG['eval_steps']:
            if earlier < step and not (curve_dir / f'step{earlier:06d}.json').exists():
                raise RuntimeError('Resume would skip a prescribed earlier curve point')
        save()
        evaluate()  # completes a point interrupted after its training checkpoint
        for batch in plan['schedules'][case['key']][step:]:
            if time.time() >= deadline:
                save()
                return False
            if batch['step'] % 10 == 1 and identity() != stamp['source_hashes']:
                raise RuntimeError('Live personal source changed')
            started = time.time()
            ix = batch['indices']
            rows = [support[i] for i in ix]
            if [r['id'] for r in rows] != batch['ids'] or batch['step'] != step + 1:
                raise RuntimeError('Matched personal training schedule changed')
            check_rows(rows, case, gradient=True)
            loss = backward_batch(server, user, st[ix], rows, task, batch['step'])
            norm = update(user, optimizer)
            step = batch['step']
            history.append(dict(step=step, ce=loss, grad_norm=norm, lr=CONFIG['learning_rate'],
                seconds=time.time()-started, sample_ids=batch['ids']))
            if step % 10 == 0 or step in CONFIG['eval_steps']:
                save()  # before evaluation, so interruption cannot skip a point
                progress('personal_adaptation', case=case['key'], personal_arm=arm,
                         **{k:v for k,v in history[-1].items() if k != 'sample_ids'})
            evaluate()
        points = [json.loads((curve_dir / f'step{s:06d}.json').read_text()) for s in CONFIG['eval_steps']]
        if points[-1]['support_samples_used'] != len(support):
            raise RuntimeError('Not all personal support samples were used')
        server.assert_frozen()
        atomic_save(cpu_state(user), arm_dir / 'user.pt')
        if server.tower_enabled:
            atomic_save(server.tower_state(), arm_dir / 'tower.pt')
        metric = points[-1]['query']
        write_json(result_path, dict(status='complete', arm=arm, job=job, case=case, stamp=stamp,
            support_n=len(support), support_samples_used=points[-1]['support_samples_used'],
            query_n=len(query), baseline=base_metric, adapted=metric, initial=points[0]['query'],
            support=points[-1]['support'], delta_pp=100*(metric['balanced_accuracy']-base_metric['balanced_accuracy']),
            steps=step, eval_steps=CONFIG['eval_steps'], history=history, initialization=init_record,
            trainable_parameters=sum(p.numel() for p in list(user.parameters()) + server.tower_parameters()),
            jev_and_original_connector_frozen=True, tower_updated=server.tower_enabled,
            query_used_for_gradients=False, query_used_for_selection=False,
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
    directory = OUT / 'jobs' / job['key']
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
    case = next(c for c in plan['cases'] if c['key'] == job['key'])
    if not run_person(server, store, plan, case, job, directory, queue['deadline'], stamp, progress):
        progress('budget_exhausted')
        return
    write_json(directory / 'complete.json', dict(job=job, stamp=stamp, time=time.time()))
    progress('worker_done')


if __name__ == '__main__':
    main()
