"""Support-only personal residual adaptation across all 23 original held-out MI users."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .benchmark_data import classification_metrics
from .checkpoint import restore_model, seed_rng, tensor_fingerprint
from .distributed import atomic_save
from .multitask_data import TrialStore, write_json
from .personalization import (CONFIG, DATA, ROOT, ResidualAdapter,
    check_support_rows, copy_trainable, frozen_hash, ids_hash, load_encoder,
    load_parent, make_manifest, source_identity, trainable_state, fixed_server_versions)
from .runtime import digest
from .task_options import make_variant


def report(path, **values):
    values['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    write_json(path, values)
    print(json.dumps(values, allow_nan=False), flush=True)


def guard_factory(root, extra=()):
    hashes = source_identity()
    files = [Path(CONFIG['parent'])/n for n in ('best.pt', 'complete.json', 'config.json')]
    files += [Path(CONFIG['reference_run'])/'prepared.json', *map(Path, extra)]
    identity = json.loads((Path(CONFIG['parent'])/'config.json').read_text())
    for name, expected in identity['jev_shard_stats'].items():
        path = Path('laya/JEV-27B-VL')/name
        if [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError('Native JEV checkpoint differs from parent')
        files.append(path)
    stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files}
    def guard():
        if (root/'cancelled.json').exists():
            raise RuntimeError('User cancelled this run')
        if source_identity() != hashes or any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns,
                                              Path(p).stat().st_ctime_ns) != s for p,s in stats.items()):
            raise RuntimeError('Live code/config/checkpoint/input identity changed')
    return hashes, guard


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'prepared.json').exists():
        proof = json.loads((root/'prepared.json').read_text())
        if proof['code_sha256'] != source_identity():
            raise ValueError('Prepared source identity changed')
        for path, sha in proof['artifacts_sha256'].items():
            if digest(path) != sha:
                raise ValueError(f'Prepared artifact changed: {path}')
        print('PREPARATION_ALREADY_COMPLETE', flush=True)
        return
    hashes, guard = guard_factory(root)
    source = Path(CONFIG['reference_run'])
    done = json.loads((source/'complete.json').read_text())
    if done['status'] != 'complete' or not done['paired_verified']['same_support_query_batches_and_baseline']:
        raise ValueError('Reference personal experiment was not completed and paired')
    previous = json.loads((source/'prepared.json').read_text())
    for path, sha in previous['artifacts_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'Reference inputs changed: {path}')
    store, saved = TrialStore(), load_parent()
    if (previous['parent_checkpoint_sha256'] != digest(Path(CONFIG['parent'])/'best.pt')
            or saved['identity']['cache_indexes'] !=
            {store.tasks[t]['task_id']:m['index_sha256'] for t,m in store.meta.items()}):
        raise ValueError('Reference parent or data cache mismatch')
    artifacts = dict(previous['artifacts_sha256'])
    for smoke in (False, True):
        name = 'smoke_manifest.json' if smoke else 'manifest.json'
        original = json.loads((source/name).read_text())
        manifest = make_manifest(store, smoke)
        if manifest['cohorts'] != original['cohorts'] or manifest['cache_indexes'] != original['cache_indexes']:
            raise ValueError('Support/query cohorts differ from previous experiment')
        lookup = {c['key']:c for c in original['cases']}
        for c in manifest['cases']:
            reference_key = f'{c["dataset"]}_s{c["subject"]}_{c["signal_arm"]}_k{c["shots_per_class"]}_seed{c["seed"]}'
            prior = lookup[reference_key]
            for field in ('support_ids','query_ids','support_sha256','query_sha256','original_split'):
                if c[field] != prior[field]:
                    raise ValueError(f'Unmatched reference trial list: {reference_key}/{field}')
            c['reference_case'] = reference_key
        write_json(root/name, manifest)
        artifacts[str(root/name)] = digest(root/name)
    guard()
    proof = dict(code_sha256=hashes, parent_step=saved['step'], encoder_sha256=previous['encoder_sha256'],
                 parent_checkpoint_sha256=previous['parent_checkpoint_sha256'],
                 input_root=str(source), source_train_only=True, target_used_for_projection_fit=False,
                 direct_text_readout_used=False, inputs_identical_to_reference=True,
                 artifacts_sha256=artifacts,
                 personal_cases=len(make_manifest(store)['cases']), subjects=len(make_manifest(store)['cohorts']))
    write_json(root/'prepared.json', proof)
    report(root/'prepare_progress.json', status='complete', personal_cases=proof['personal_cases'],
           subjects=proof['subjects'], input_root=str(source))


def load_server(saved, encoder):
    from .jev import JEVBridge, load_jev
    text, weight, bias, tokenizer = load_jev('cuda')
    model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    encoder.requires_grad_(True)
    model.add_module('eeg_encoder', encoder)
    restore_model(model, saved)
    model.requires_grad_(False).eval()
    model.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if getattr(model.text.config, 'attention_dropout', 0) != 0:
        raise ValueError('Matched deterministic control requires zero LM attention dropout')
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.text.modules()):
        raise ValueError('Unexpected nonzero language dropout')
    return model, tokenizer


def suffix_for(task, tokenizer):
    variant = make_variant(task, 0, 'label_independent', 0, DATA['seed'], 0)
    return torch.tensor(tokenizer.encode(variant['suffix'], add_special_tokens=False), device='cuda')


@torch.no_grad()
def predict(tokens, rows, server, tokenizer, task):
    server.eval()
    jev_scores = []
    suffix = suffix_for(task, tokenizer)
    for start in range(0, len(rows), CONFIG['jev_batch']):
        z = tokens[start:start+min(CONFIG['jev_batch'],len(rows)-start)].cuda()
        valid = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
        logits = server(z, valid, suffix_ids=[suffix]*len(z), num_choices=len(task['classes']))['logits']
        jev_scores.append(logits.cpu())
    return dict(jev=torch.cat(jev_scores).numpy())


def metrics(rows, scores, classes):
    labels = [int(r['label']) for r in rows]
    return {kind: classification_metrics(labels, value, classes) for kind,value in scores.items()}


def worker(root, worker_id, smoke=False):
    prepared = json.loads((root/'prepared.json').read_text())
    if prepared['code_sha256'] != source_identity():
        raise ValueError('Prepared inputs belong to another implementation')
    for path,sha in prepared['artifacts_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'Prepared data changed: {path}')
    input_root = Path(prepared['input_root'])
    manifest_path = root/('smoke_manifest.json' if smoke else 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    hashes, guard = guard_factory(root, [root/'prepared.json', *prepared['artifacts_sha256']])
    cases = [c for c in manifest['cases'] if c['worker'] == worker_id]
    if not cases or len({c['arm'] for c in cases}) != 1:
        raise ValueError('Expected exactly one arm per worker')
    out = root/('smoke' if smoke else 'episodes')
    out.mkdir(exist_ok=True)
    progress = root/f'{"smoke" if smoke else "worker"}_{worker_id}.json'
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    saved = load_parent()
    encoder = load_encoder(saved)
    if frozen_hash(encoder) != prepared['encoder_sha256']:
        raise ValueError('Initial Tower differs from reference')
    report(progress, status='loading_jev', worker=worker_id, cases=len(cases), arm=cases[0]['arm'])
    server, tokenizer = load_server(saved, encoder)
    del saved
    seed_rng(CONFIG['seed'])
    personal = ResidualAdapter(bottleneck=CONFIG['adapter_width']).cuda()
    if any(p.requires_grad for p in server.parameters()):
        raise ValueError('The complete parent, including language LoRA, must be frozen')
    server_version = fixed_server_versions(server)
    server_buffer_hash = tensor_fingerprint(dict(server.named_buffers()))
    parameters = list(personal.parameters())
    initial_state = trainable_state(personal)
    fixed_encoder_hash = frozen_hash(encoder)
    n_params = sum(p.numel() for p in parameters)
    if n_params != 33088:
        raise ValueError('Unexpected personal parameter count')
    parameter_groups = [dict(params=parameters, lr=CONFIG['learning_rate'])]
    tasks = TrialStore().tasks
    cached_subject, cache = None, None
    started, completed = time.monotonic(), 0
    report(progress, status='ready', worker=worker_id, cases=len(cases), signal_parameters=n_params,
           lm_parameters=0, lm_total_parameters=sum(p.numel() for p in server.text.parameters()))
    for case in cases:
        guard()
        if stop:
            report(progress, status='stopped_between_cases', worker=worker_id, completed=completed)
            return
        directory = out/case['key']
        if (directory/'complete.json').exists():
            prior = json.loads((directory/'complete.json').read_text())
            if prior['case'] != case or prior['code_sha256'] != hashes:
                raise ValueError('Existing episode belongs to another protocol')
            completed += 1
            continue
        directory.mkdir(exist_ok=True)
        key = (case['dataset'], case['subject'])
        if cached_subject != key:
            cache = torch.load(input_root/'subjects'/f'{key[0]}_s{key[1]}.pt', weights_only=False, map_location='cpu')
            cached_subject = key
        ix = {r['id']:i for i,r in enumerate(cache['rows'])}
        support_rows = [cache['rows'][ix[k]] for k in case['support_ids']]
        query_rows = [cache['rows'][ix[k]] for k in case['query_ids']]
        check_support_rows(support_rows, case)
        if smoke:
            query_rows = query_rows[:8]
        si, qi = [ix[r['id']] for r in support_rows], [ix[r['id']] for r in query_rows]
        base_support = cache['tokens'][si].cuda()
        base_query = cache['tokens'][qi]
        task = tasks[case['task']]
        copy_trainable(personal, initial_state)
        server.eval()
        with torch.no_grad():
            identity_error = float((personal(base_support)-base_support).abs().max())
        if identity_error != 0:
            raise RuntimeError(f'Non-identity initialization: {identity_error}')
        baseline_path = out/f'baseline_{key[0]}_s{key[1]}_worker{worker_id}.pt'
        if baseline_path.exists():
            baseline = torch.load(baseline_path, weights_only=False, map_location='cpu')
            if baseline['ids'] != [r['id'] for r in query_rows]:
                raise ValueError('Baseline query IDs changed')
        else:
            baseline = dict(ids=[r['id'] for r in query_rows],
                            scores=predict(base_query, query_rows, server, tokenizer, task))
            atomic_save(baseline, baseline_path)
        baseline_metrics = metrics(query_rows, baseline['scores'],len(task['classes']))
        old_path = input_root/('smoke' if smoke else 'episodes')/case['reference_case']/'complete.json'
        old_result = json.loads(old_path.read_text())
        if baseline_metrics['jev'] != old_result['baseline']['jev']:
            raise ValueError('Common baseline differs from prior InfoNCE experiment')
        seed_rng(case['seed'])
        generator = np.random.default_rng(case['seed'])
        labels = torch.tensor([int(r['label']) for r in support_rows], device='cuda')
        label_array = labels.cpu().numpy()
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=CONFIG['weight_decay'])
        total = 3 if smoke else CONFIG['steps']
        history, batches = [], []
        suffix = suffix_for(task,tokenizer)
        torch.cuda.reset_peak_memory_stats()
        training_started = time.monotonic()
        server.text.train()
        encoder.eval()
        for step in range(1,total+1):
            indices = [int(generator.choice(np.flatnonzero(label_array==j%len(task['classes']))))
                       for j in range(CONFIG['batch_size'])]
            batch_rows = [support_rows[i] for i in indices]
            check_support_rows(batch_rows,case)
            batches.extend(r['id'] for r in batch_rows)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for start in range(0,len(indices),CONFIG['microbatch']):
                sub = indices[start:start+CONFIG['microbatch']]
                z = personal(base_support[sub])
                valid = torch.ones(z.shape[:2],device='cuda',dtype=torch.bool)
                loss = server(z,valid,labels=labels[sub],suffix_ids=[suffix]*len(sub),
                              num_choices=len(task['classes']))['loss']
                (loss*(len(sub)/len(indices))).backward()
                loss_sum += float(loss.detach())*len(sub)/len(indices)
            norm = torch.nn.utils.clip_grad_norm_(parameters,CONFIG['clip_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%10==0 or step==total:
                history.append(dict(step=step,support_batch_ce=loss_sum,signal_grad_norm=float(norm),
                                    lm_grad_norm=0.0))
                report(progress,status='adapting',worker=worker_id,case=case['key'],step=step,total=total,
                       completed=completed,cases=len(cases),elapsed_seconds=time.monotonic()-started,loss=loss_sum)
                guard()
        torch.cuda.synchronize()
        train_seconds = time.monotonic()-training_started
        del optimizer
        server.eval()
        updated_state = trainable_state(personal)
        if tensor_fingerprint(updated_state) == tensor_fingerprint(initial_state):
            raise RuntimeError('Personal adapter did not update')
        if frozen_hash(encoder) != fixed_encoder_hash:
            raise RuntimeError('Frozen Tower changed')
        if (fixed_server_versions(server) != server_version
                or tensor_fingerprint(dict(server.named_buffers())) != server_buffer_hash
                or any(p.grad is not None for p in server.parameters())):
            raise RuntimeError('Frozen JEV/Tower/connector/head or buffers changed')
        batch_hash = ids_hash([dict(id=k) for k in batches])
        if batch_hash != old_result['training_batch_ids_sha256']:
            raise ValueError('Calibration batch stream differs from prior experiment')
        with torch.no_grad():
            adapted_tokens = torch.cat([personal(base_query[i:i+32].cuda()).cpu()
                                        for i in range(0, len(base_query), 32)])
            support_tokens = personal(base_support).cpu()
        prediction = predict(adapted_tokens,query_rows,server,tokenizer,task)
        support_prediction = predict(support_tokens,support_rows,server,tokenizer,task)
        # Identical batch shape before/after reload, independent of bf16 batching roundoff.
        with torch.no_grad():
            probe_tokens = personal(base_query[:4].cuda()).cpu()
        probe = predict(probe_tokens,query_rows[:4],server,tokenizer,task)
        checkpoint = dict(case=case,personal=updated_state,code_sha256=hashes,
                          parent_checkpoint_sha256=prepared['parent_checkpoint_sha256'],step=total)
        atomic_save(checkpoint,directory/'adapter.pt')
        copy_trainable(personal,initial_state)
        del checkpoint
        restored = torch.load(directory/'adapter.pt',map_location='cpu',weights_only=False,mmap=True)
        copy_trainable(personal,restored['personal'])
        with torch.no_grad():
            restored_tokens = personal(base_query[:4].cuda()).cpu()
        check = predict(restored_tokens,query_rows[:4],server,tokenizer,task)
        if not torch.equal(restored_tokens,probe_tokens) or any(not np.array_equal(check[k],probe[k]) for k in check):
            raise RuntimeError('Saved personal/LM adapter reload changed prediction')
        del restored
        atomic_save(dict(ids=[r['id'] for r in query_rows],labels=[int(r['label']) for r in query_rows],
                         baseline=baseline['scores'],adapted=prediction),directory/'predictions.pt')
        result = dict(case=case,smoke=smoke,steps=total,objective='jev_ce',code_sha256=hashes,
            baseline=baseline_metrics,adapted=metrics(query_rows,prediction,len(task['classes'])),
            support=metrics(support_rows,support_prediction,len(task['classes'])),
            final_support_ce=float(F.cross_entropy(torch.from_numpy(support_prediction['jev']),labels.cpu())),
            trainable_parameters=n_params,signal_trainable_parameters=n_params,lm_trainable_parameters=0,
            lora_locations=[],train_seconds=train_seconds,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            edge_local_training=False,peak_includes_jev=True,training_batch_ids_sha256=batch_hash,
            parent_encoder_frozen_verified=True,
            jev_base_connector_head_frozen_verified=True,lm_lora_updated=False,
            initialization_identity_max_error=identity_error,checkpoint_reload_verified=True,
            query_used_for_selection=False,original_benchmark_test_modified=False,
            reference_batch_ids_verified=True,reference_baseline_verified=True)
        write_json(directory/'support_history.json',history)
        write_json(directory/'complete.json',result)
        completed += 1
        report(progress,status='episode_complete',worker=worker_id,case=case['key'],completed=completed,cases=len(cases),
               metrics=result['adapted'],train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started)
    report(progress,status='complete',worker=worker_id,completed=completed,cases=len(cases),
           elapsed_seconds=time.monotonic()-started,signal_parameters=n_params,lm_parameters=0)


def aggregate_results(results):
    """Paired subject means; dataset macro is distinct from a 23-user mean."""
    rows, per_subject = [], []
    for result in results:
        c = result['case']
        before = result['baseline']['jev']['balanced_accuracy']
        after = result['adapted']['jev']['balanced_accuracy']
        per_subject.append(dict(dataset=c['dataset'], subject=c['subject'],
            support_n=result['support']['jev']['n'], query_n=result['adapted']['jev']['n'],
            baseline_bacc=before, adapted_bacc=after, delta_pp=100*(after-before),
            support_accuracy=result['support']['jev']['accuracy'],
            train_seconds=result['train_seconds']))
    keys = [(r['dataset'], r['subject']) for r in per_subject]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate user in aggregate')
    for dataset in CONFIG['datasets']:
        rs = [r for r in per_subject if r['dataset'] == dataset]
        if not rs:
            raise ValueError('Missing dataset')
        deltas = np.array([r['delta_pp'] for r in rs])
        rows.append(dict(dataset=dataset, subjects=len(rs),
            support_n=sum(r['support_n'] for r in rs), query_n=sum(r['query_n'] for r in rs),
            baseline_bacc=float(np.mean([r['baseline_bacc'] for r in rs])),
            adapted_bacc=float(np.mean([r['adapted_bacc'] for r in rs])),
            delta_pp=float(deltas.mean()), improved=int((deltas>1e-10).sum()),
            worse=int((deltas < -1e-10).sum()), tied=int((np.abs(deltas)<=1e-10).sum()),
            mean_training_seconds=float(np.mean([r['train_seconds'] for r in rs]))))
    return dict(rows=rows, per_subject=per_subject,
        dataset_equal_mean={k:float(np.mean([r[k] for r in rows]))
                            for k in ('baseline_bacc','adapted_bacc','delta_pp')},
        subject_equal_mean={k:float(np.mean([r[k] for r in per_subject]))
                            for k in ('baseline_bacc','adapted_bacc','delta_pp')})


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    results = []
    for case in manifest['cases']:
        result = json.loads((root/'episodes'/case['key']/'complete.json').read_text())
        if (result['case'] != case or result['smoke'] or result['query_used_for_selection']
                or result['steps'] != CONFIG['steps'] or result['code_sha256'] != source_identity()):
            raise ValueError('Incomplete or mismatched result')
        results.append(result)
    if len(results) != 23:
        raise ValueError('Expected all 23 held-out users')
    summary = aggregate_results(results)
    summary.update(config=CONFIG, complete=True, original_benchmark_test=False,
        aggregation='Per-dataset subject-equal means; dataset-equal and 23-user-equal means separately reported',
        limitations='One seed, fixed 100 steps; 2a only two users; subject8 previously inspected. Personal supervised adaptation, not zero-shot.')
    write_json(root/'summary.json', summary)
    labels = {
        'bcic_2a': ('BCIC-IV-2a','5,184 (2,880/1,152/1,152)','9 (5/2/2)','固定跨被试；个人 T→E'),
        'motor_mv_img': ('PhysioMI','9,837 (6,210/1,734/1,893)','109 (69/19/21)','固定跨被试；个人前四→后两想象 run')}
    lines = ['# 全部留出用户：个人残差 adapter → 冻结 JEV', '',
        '每人每类5条（共20条）support，seed42，固定100步；仅33,088个adapter参数更新。',
        '逐人重置到同一父模型；以下原划分计数不等于个人support/query数量。', '',
        '| 数据集 | 处理后窗口总计（父模型原训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（父模型原训／验／测） | 本轮目标用户 | Support/query 条数 | 不适配 BAcc（%） | 适配 BAcc（%） | 增量（百分点） | 提升/下降/持平人数 |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary['rows']:
        title, windows, subjects, protocol = labels[r['dataset']]
        lines.append(f"| {title} | {windows} | 运动想象（4类） | {protocol} | {subjects} | {r['subjects']} | {r['support_n']}/{r['query_n']} | {100*r['baseline_bacc']:.2f} | {100*r['adapted_bacc']:.2f} | {r['delta_pp']:+.2f} | {r['improved']}/{r['worse']}/{r['tied']} |")
    macro = summary['dataset_equal_mean']
    lines += ['', f"两库等权：{100*macro['baseline_bacc']:.2f}% → {100*macro['adapted_bacc']:.2f}%，增量 {macro['delta_pp']:+.2f} 个百分点。", '',
        '## 逐用户配对结果', '',
        '| 数据集 | 用户 | Support/query | 不适配 BAcc（%） | 适配 BAcc（%） | 增量（百分点） | Support Acc（%） |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summary['per_subject']:
        lines.append(f"| {labels[r['dataset']][0]} | {r['subject']} | {r['support_n']}/{r['query_n']} | {100*r['baseline_bacc']:.2f} | {100*r['adapted_bacc']:.2f} | {r['delta_pp']:+.2f} | {100*r['support_accuracy']:.2f} |")
    lines += ['', summary['limitations'], '']
    (root/'summary.md').write_text('\\n'.join(lines).replace('\\n', '\n'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','worker','summary'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.mode=='worker':
        torch.cuda.set_device(0)
        worker(args.root,args.worker,args.smoke)
    elif args.mode=='prepare':
        prepare(args.root)
    else:
        summarize(args.root)


if __name__=='__main__':
    main()
