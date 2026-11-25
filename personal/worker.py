"""One persistent frozen JEV per GPU, with exact per-user reset and resumable jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from jev_train.checkpoint import seed_rng, tensor_fingerprint
from jev_train.data import classification_metrics
from .data import TrialStore
from .common import CONFIG, OUT, ROOT, check_rows, identity, write_json
from .model import PersonalServer
from .modules import UserParameters, support_loss


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def metrics(logits, rows, classes):
    return classification_metrics([r['label'] for r in rows], logits.softmax(-1).numpy(), classes)


def make_user(arm, server, seed):
    seed_rng(seed)
    cfg = server.model.text.config
    return UserParameters(arm, server.prompt_init, CONFIG, cfg.num_key_value_heads, cfg.head_dim).cuda()


def assert_optimizer(optimizer, user, server):
    ids = {id(p) for p in user.parameters()}
    if ids != {id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer does not exclusively contain the personal parameters')
    if ids & {id(p) for p in server.model.parameters()}:
        raise RuntimeError('Personal optimizer contains shared parameters')


def smoke(server, store, case, directory):
    """Real 27B checks, support-only: every new arm, gradients, exact reload."""
    index = {r['id']: r for r in store.rows[case['task']]}
    rows = [index[case['support_ids'][0]], index[case['support_ids'][-1]]]
    check_rows(rows, case, gradient=True)
    tokens = server.encode(store, rows)
    task = store.tasks[case['task']]
    baseline = server.predict(tokens, rows, task)
    # The new no-adaptation forward must reproduce the original shared forward.
    with torch.no_grad():
        from jev_train.task_options import make_variant
        suffix = [torch.tensor(server.tokenizer.encode(make_variant(task, r['label'], r['id'], 0, CONFIG['seed'], 0)['suffix'],
                  add_special_tokens=False), device='cuda') for r in rows]
        server.model.eval()
        legacy = server.model(tokens.cuda(), torch.ones(tokens.shape[:2], dtype=torch.bool, device='cuda'),
                              suffix_ids=suffix, num_choices=len(task['classes']))['logits'].cpu()
        torch.testing.assert_close(baseline, legacy, atol=0, rtol=0)
    reports = {}
    for arm in CONFIG['arms']:
        if arm.startswith('icl'):
            demos, audit = server.demonstrations(tokens, rows, task, arm == 'icl_shuffled')
            logits = server.predict(tokens, rows, task, demonstrations=demos)
            if not logits.isfinite().all():
                raise RuntimeError('Nonfinite ICL smoke logits')
            reports[arm] = dict(finite=True, sequence_tokens=len(demos), **audit)
            continue
        user = make_user(arm, server, CONFIG['seed'])
        optimizer = torch.optim.AdamW(user.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        assert_optimizer(optimizer, user, server)
        initial = tensor_fingerprint(user.state_dict())
        logits = server.forward(tokens, rows, task, user, step=1, training=True)
        labels = torch.tensor([r['label'] for r in rows], device='cuda')
        loss, _, _ = support_loss(logits, labels)
        loss.backward()
        grads = [p.grad for p in user.parameters() if p.grad is not None]
        if not grads or not all(g.isfinite().all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
            raise RuntimeError(f'Invalid personal gradient: {arm}')
        torch.nn.utils.clip_grad_norm_(user.parameters(), CONFIG['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        if tensor_fingerprint(user.state_dict()) == initial:
            raise RuntimeError(f'Personal parameters did not update: {arm}')
        before = server.predict(tokens, rows, task, user)
        saved = {n: t.detach().cpu().clone() for n, t in user.state_dict().items()}
        restored = make_user(arm, server, CONFIG['seed'] + 1)
        restored.load_state_dict(saved, strict=True)
        after = server.predict(tokens, rows, task, restored)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        server.activate(None)
        server.assert_frozen()
        reports[arm] = dict(parameters=sum(p.numel() for p in user.parameters()),
                            nonzero_finite_gradient=True, changed=True, reload_exact=True)
        del user, restored, optimizer, loss, logits
    server.activate(None)
    server.assert_frozen()
    write_json(directory / 'smoke.json', dict(passed=True, support_only=True, samples=[r['id'] for r in rows],
               original_split=case['original_split'], base_forward_exact=True, shared_model_frozen=True, arms=reports))
    torch.cuda.empty_cache()


def run_case(server, store, case, deadline, hashes, parent_sha, worker):
    directory = OUT / 'cases' / case['key']
    directory.mkdir(parents=True, exist_ok=True)
    rows_by_id = {r['id']: r for r in store.rows[case['task']]}
    support = [rows_by_id[i] for i in case['support_ids']]
    query = [rows_by_id[i] for i in case['query_ids']]
    check_rows(support, case, gradient=True)
    check_rows(query, case)
    task = store.tasks[case['task']]
    tokens = server.encode(store, support + query)
    st, qt = tokens[:len(support)], tokens[len(support):]
    target = torch.tensor([r['label'] for r in support], device='cuda')
    cache_path = directory / 'baseline.pt'
    if cache_path.exists():
        baseline = torch.load(cache_path, map_location='cpu', weights_only=False)
        if baseline['parent_sha256'] != parent_sha or baseline['case'] != case:
            raise RuntimeError('Saved baseline belongs to another cohort or parent')
    else:
        predictions = [server.predict(st, support, task, variant=v) for v in CONFIG['teacher_variants']]
        teacher = torch.stack([p.softmax(-1) for p in predictions]).mean(0)
        correct = torch.stack([p.argmax(-1) == target.cpu() for p in predictions]).all(0)
        baseline = dict(parent_sha256=parent_sha, case=case, support_logits=predictions[0],
            query_logits=server.predict(qt, query, task), teacher=teacher, teacher_correct=correct)
        atomic_save(baseline, cache_path)
    baseline_audit = {}
    if case.get('historical_prediction'):
        old = torch.load(case['historical_prediction'], map_location='cpu', weights_only=False)
        if old['ids'] != case['query_ids'] or old['labels'] != [r['label'] for r in query]:
            raise RuntimeError('Historical baseline sample identity changed')
        old_logits = torch.as_tensor(old['baseline']['jev'])
        new_logits = baseline['query_logits']
        identical = torch.equal(old_logits.argmax(-1), new_logits.argmax(-1))
        baseline_audit = dict(historical_baseline_predictions_identical=identical,
            historical_baseline_logits_exact=torch.equal(old_logits, new_logits),
            historical_baseline_max_logit_difference=float((old_logits-new_logits).abs().max()),
            historical_baseline_mismatches=int((old_logits.argmax(-1)!=new_logits.argmax(-1)).sum()),
            eval_batch=CONFIG['eval_batch'])
        write_json(directory / 'baseline_audit.json', baseline_audit)
        if not identical:
            raise RuntimeError('Historical no-adaptation predictions not reproduced; stop before training')
    teacher = baseline['teacher'].cuda()
    teacher_correct = baseline['teacher_correct'].cuda()
    base_metric = metrics(baseline['query_logits'], query, case['classes'])
    for arm in CONFIG['arms']:
        arm_dir = directory / arm
        arm_dir.mkdir(exist_ok=True)
        if (arm_dir / 'result.json').exists():
            result = json.loads((arm_dir / 'result.json').read_text())
            if result['parent_sha256'] != parent_sha or result['source_hashes'] != hashes:
                raise RuntimeError('Finished personal result identity changed')
            continue
        if time.time() >= deadline:
            return False
        if identity() != hashes:
            raise RuntimeError('Personal source changed during live jobs')
        started = time.time()
        user, optimizer, init_logits = None, None, None
        history, step = [], 0
        info = dict(baseline_audit)
        user_seed = CONFIG['seed'] + case['order'] * 101
        if arm.startswith('icl'):
            demos, demonstration_info = server.demonstrations(st, support, task, arm == 'icl_shuffled')
            info.update(demonstration_info)
            after = server.predict(qt, query, task, demonstrations=demos)
            support_after = None
            parameter_count = 0
            info['demonstration_tokens'] = len(demos)
            del demos
        else:
            user = make_user(arm, server, user_seed)
            parameter_count = sum(p.numel() for p in user.parameters())
            optimizer = torch.optim.AdamW(user.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
            assert_optimizer(optimizer, user, server)
            state_path = arm_dir / 'last.pt'
            if state_path.exists():
                saved = torch.load(state_path, map_location='cpu', weights_only=False)
                if saved['parent_sha256'] != parent_sha or saved['source_hashes'] != hashes or saved['case'] != case:
                    raise RuntimeError('Personal resume identity mismatch')
                user.load_state_dict(saved['user'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                step, history, init_logits = saved['step'], saved['history'], saved['initial_logits']
                del saved
            else:
                init_logits = server.predict(qt, query, task, user)

            def save():
                server.assert_frozen()
                atomic_save(dict(user={n: p.detach().cpu() for n, p in user.state_dict().items()},
                    optimizer=optimizer.state_dict(), step=step, history=history, initial_logits=init_logits,
                    parent_sha256=parent_sha, source_hashes=hashes, case=case), state_path)

            for step in range(step + 1, CONFIG['steps'] + 1):
                if time.time() >= deadline:
                    step -= 1
                    save()
                    server.activate(None)
                    return False
                rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], case['order'], step]))
                selected = rng.integers(len(support), size=CONFIG['batch_size']).tolist()
                batch_rows = [support[i] for i in selected]
                check_rows(batch_rows, case, gradient=True)
                optimizer.zero_grad(set_to_none=True)
                totals = [0., 0., 0.]
                for offset in range(0, len(selected), CONFIG['microbatch']):
                    ix = selected[offset:offset + CONFIG['microbatch']]
                    logits = server.forward(st[ix], [support[i] for i in ix], task, user, step=step, training=True)
                    kd = arm.endswith('_kd')
                    loss, ce, kl = support_loss(logits, target[ix], teacher[ix] if kd else None,
                        teacher_correct[ix] if kd else None, CONFIG['kd_weight'], CONFIG['kd_temperature'])
                    weight = len(ix) / CONFIG['batch_size']
                    (loss * weight).backward()
                    for j, value in enumerate((loss, ce, kl)):
                        totals[j] += float(value.detach()) * weight
                norm = torch.nn.utils.clip_grad_norm_(user.parameters(), CONFIG['clip_grad_norm'], error_if_nonfinite=True)
                if step == 1 and float(norm) <= 0:
                    raise RuntimeError(f'No personal gradients for {arm}')
                optimizer.step()
                history.append(dict(step=step, loss=totals[0], ce=totals[1], kl=totals[2],
                                    grad_norm=float(norm), lr=CONFIG['learning_rate']))
                if step % 10 == 0 or step == CONFIG['steps']:
                    save()
                    write_json(OUT / f'progress_gpu{worker}.json', dict(state='adapting', case=case['key'],
                        arm=arm, steps=CONFIG['steps'], time=time.time(), **history[-1]))
            after = server.predict(qt, query, task, user)
            support_after = server.predict(st, support, task, user)
            atomic_save({n: v.detach().cpu() for n, v in user.state_dict().items()}, arm_dir / 'user.pt')
            info['teacher_approved_support_fraction'] = float(teacher_correct.float().mean()) if arm.endswith('_kd') else None
        server.activate(None)
        server.assert_frozen()
        adapted_metric = metrics(after, query, case['classes'])
        np.savez_compressed(arm_dir / 'predictions.npz',
            query_ids=np.asarray(case['query_ids']), labels=np.asarray([r['label'] for r in query]),
            baseline_logits=baseline['query_logits'].numpy(), logits=after.numpy(),
            initial_logits=np.empty((0, case['classes'])) if init_logits is None else init_logits.numpy())
        write_json(arm_dir / 'result.json', dict(status='complete', arm=arm, case=case,
            parent_sha256=parent_sha, source_hashes=hashes, selected_parent_step=server.parent_step,
            steps=step, trainable_parameters=parameter_count, support_n=len(support), query_n=len(query),
            baseline=base_metric, adapted=adapted_metric,
            initial=None if init_logits is None else metrics(init_logits, query, case['classes']),
            support=None if support_after is None else metrics(support_after, support, case['classes']),
            delta_pp=100 * (adapted_metric['balanced_accuracy'] - base_metric['balanced_accuracy']),
            shared_weights_frozen=True, query_used_for_gradients=False, query_used_for_selection=False,
            independent_test_estimate=False, role=CONFIG['role'], seconds=time.time() - started,
            history=history, audit=info))
        print(json.dumps(dict(case=case['key'], arm=arm, bacc=adapted_metric['balanced_accuracy'], seconds=time.time()-started)), flush=True)
        del user, optimizer, after, support_after
        torch.cuda.empty_cache()
    write_json(directory / 'complete.json', dict(case=case['key'], arms=CONFIG['arms'], parent_sha256=parent_sha))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_device(0)
    hashes = identity()
    plan = json.loads((OUT / 'plan.json').read_text())
    queue = json.loads((OUT / 'queue.json').read_text())
    if plan['source_hashes'] != hashes or queue['source_hashes'] != hashes:
        raise RuntimeError('Personal plan or queue source mismatch')
    write_json(OUT / f'progress_gpu{args.worker}.json', dict(state='loading_frozen_parent',
        time=time.time(), pid=os.getpid(), parent_sha256=queue['parent_sha256']))
    server = PersonalServer(queue['parent_sha256'])
    store = TrialStore()
    if json.loads(json.dumps(store.stats)) != plan['source_array_stats']:
        raise RuntimeError('Prepared signal array identities changed after planning')
    write_json(OUT / f'progress_gpu{args.worker}.json', dict(state='smoke_or_waiting_for_smoke',
        time=time.time(), pid=os.getpid(), selected_parent_step=server.parent_step))
    smoke_path = OUT / 'smoke.json'
    if args.worker == 0 and not smoke_path.exists():
        smoke(server, store, plan['cases'][0], OUT)
    while not smoke_path.exists():
        if time.time() >= queue['deadline']:
            return
        time.sleep(5)
    if not json.loads(smoke_path.read_text())['passed']:
        raise RuntimeError('Personal smoke failed')
    for case in plan['cases']:
        if time.time() >= queue['deadline']:
            break
        directory = OUT / 'cases' / case['key']
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'complete.json').exists():
            continue
        with (directory / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (directory / 'complete.json').exists():
                continue
            if not run_case(server, store, case, queue['deadline'], hashes, queue['parent_sha256'], args.worker):
                break
    write_json(OUT / f'progress_gpu{args.worker}.json', dict(state='worker_done', time=time.time(), pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
