"""Prepare a source-only text readout, then run isolated personal B/C episodes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .benchmark_data import classification_metrics
from .checkpoint import restore_model, seed_rng, tensor_fingerprint
from .distributed import atomic_save
from .leaf_data import LEAFSignalAdapter
from .multitask_data import TrialStore, write_json
from .personalization import (CONFIG, DATA, ROOT, ResidualAdapter, TextReadout,
    check_support_rows, copy_trainable, frozen_hash, ids_hash, install_lora, load_encoder,
    load_parent, make_manifest, projection_initialization, projection_module, QVDelta,
    source_identity, trainable_state)
from .runtime import digest
from .task_options import make_variant


def report(path, **values):
    values['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    write_json(path, values)
    print(json.dumps(values, allow_nan=False), flush=True)


def guard_factory(root, extra=()):
    hashes = source_identity()
    files = [Path(CONFIG['parent'])/n for n in ('best.pt', 'complete.json', 'config.json')]
    files += [Path(CONFIG['projection_parent'])/'best.pt', *map(Path, extra)]
    stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files}
    def guard():
        if (root/'cancelled.json').exists():
            raise RuntimeError('User cancelled this run')
        if source_identity() != hashes or any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns,
                                              Path(p).stat().st_ctime_ns) != s for p,s in stats.items()):
            raise RuntimeError('Live code/config/checkpoint identity changed')
    return hashes, guard


def dense_inputs(adapter, rows):
    groups = adapter.batch(rows)
    result = [None] * len(rows)
    for _, indices, pieces in groups:
        if len(pieces) != 1 or pieces[0].shape[1:] != (65, 800):
            raise ValueError('This MI pilot requires one complete four-second trial')
        for i, x in zip(indices, pieces[0]):
            result[i] = torch.from_numpy(x.copy())
    return torch.stack(result)


@torch.no_grad()
def encode_dense(encoder, x, batch):
    return torch.cat([encoder(x[i:i+batch]).cpu() for i in range(0, len(x), batch)])


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
    seed_rng(CONFIG['seed'])
    store, adapter = TrialStore(), LEAFSignalAdapter()
    saved = load_parent()
    if saved['identity']['cache_indexes'] != {store.tasks[t]['task_id']:m['index_sha256'] for t,m in store.meta.items()}:
        raise ValueError('Personalization and parent caches differ')
    encoder = load_encoder(saved)
    encoder_hash = frozen_hash(encoder)
    manifest = make_manifest(store)
    smoke_manifest = make_manifest(store, smoke=True)
    for name, value in [('manifest.json', manifest), ('smoke_manifest.json', smoke_manifest)]:
        write_json(root/name, value)
    report(root/'prepare_progress.json', status='encoding_source_train_only', parent_step=saved['step'])
    source = {}
    for tid, task in enumerate(store.tasks):
        name = task['task_id']
        if name not in CONFIG['datasets']:
            continue
        rows = [r for i,r in enumerate(store.rows[tid]) if r['split'] == 'train' and i not in store.excluded[tid]]
        features = []
        for start in range(0, len(rows), CONFIG['encode_batch']):
            part = rows[start:start+CONFIG['encode_batch']]
            x = dense_inputs(adapter, part)
            with torch.no_grad():
                features.append(encoder(x).mean(1).cpu())
            if start % (CONFIG['encode_batch']*10) == 0:
                guard()
                report(root/'prepare_progress.json', status='encoding_source', dataset=name, done=start, total=len(rows))
        source[name] = dict(features=torch.cat(features).cuda(),
                            labels=torch.tensor([int(r['label']) for r in rows], device='cuda'),
                            ids_sha256=ids_hash(rows), n=len(rows), subjects=sorted({r['subject'] for r in rows}, key=int))
    projection, scale = projection_initialization()
    readout = TextReadout(projection, scale).cuda()
    optimizer = torch.optim.AdamW(readout.projection.parameters(), lr=CONFIG['projection_fit_lr'], weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(CONFIG['seed'])
    names = sorted(source)
    history = []
    for step in range(1, CONFIG['projection_fit_steps']+1):
        name = names[(step-1) % len(names)]
        data = source[name]
        # Class-balanced frozen source features; never target/support/query here.
        targets = data['labels'].cpu().numpy()
        ix = np.array([rng.choice(np.flatnonzero(targets == j % 4)) for j in range(CONFIG['projection_fit_batch'])])
        z = data['features'][ix, None, :]
        _, loss = readout(z, torch.ones(z.shape[:2], device='cuda', dtype=torch.bool), name, data['labels'][ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.projection.parameters(), CONFIG['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            guard()
            history.append(dict(step=step, dataset=name, loss=float(loss.detach())))
            report(root/'prepare_progress.json', status='fitting_source_projection', **history[-1])
    readout.requires_grad_(False).eval()
    atomic_save(dict(projection=readout.projection.cpu().state_dict(), log_scale=readout.log_scale.cpu(),
                     source={n:{k:v for k,v in d.items() if k not in ('features','labels')} for n,d in source.items()}),
                root/'text_readout.pt')
    write_json(root/'projection_history.json', history)
    del source, readout, optimizer
    (root/'subjects').mkdir(exist_ok=True)
    all_rows = {r['id']:r for rows in store.rows.values() for r in rows}
    artifacts = [root/'text_readout.pt', root/'manifest.json', root/'smoke_manifest.json']
    for cohort in smoke_manifest['cohorts'] + manifest['cohorts']:
        guard()
        rows = [all_rows[k] for k in cohort['support_pool_ids'] + cohort['query_ids']]
        x = torch.cat([dense_inputs(adapter, rows[i:i+32]) for i in range(0,len(rows),32)])
        tokens = encode_dense(encoder, x, CONFIG['encode_batch'])
        path = root/'subjects'/f'{cohort["dataset"]}_s{cohort["subject"]}.pt'
        atomic_save(dict(rows=rows, x=x, tokens=tokens), path)
        artifacts.append(path)
        report(root/'prepare_progress.json', status='preparing_subject', dataset=cohort['dataset'], subject=cohort['subject'])
    if frozen_hash(encoder) != encoder_hash:
        raise RuntimeError('Source projection preparation changed the frozen Tower')
    proof = dict(code_sha256=hashes, parent_step=saved['step'], encoder_sha256=encoder_hash,
                 parent_checkpoint_sha256=digest(Path(CONFIG['parent'])/'best.pt'),
                 projection_parent_sha256=digest(Path(CONFIG['projection_parent'])/'best.pt'),
                 source_train_only=True, target_used_for_projection_fit=False, encoder_frozen_verified=True,
                 artifacts_sha256={str(p):digest(p) for p in artifacts},
                 personal_cases=len(manifest['cases']), subjects=len(manifest['cohorts']),
                 smoke_cases=len(smoke_manifest['cases']))
    write_json(root/'prepared.json', proof)
    report(root/'prepare_progress.json', status='complete', personal_cases=proof['personal_cases'], subjects=proof['subjects'])


def load_readout(root):
    saved = torch.load(root/'text_readout.pt', weights_only=True, map_location='cpu')
    head = projection_module()
    head.load_state_dict(saved['projection'], strict=True)
    return TextReadout(head, saved['log_scale']).cuda().requires_grad_(False).eval()


def load_server(saved, encoder):
    from .jev import JEVBridge, load_jev
    text, weight, bias, tokenizer = load_jev('cuda')
    model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    encoder.requires_grad_(True)
    model.add_module('eeg_encoder', encoder)
    restore_model(model, saved)
    model.requires_grad_(False).eval()
    model.text.gradient_checkpointing_disable()
    return model, tokenizer


def server_versions(server):
    return {n:p._version for n,p in server.named_parameters() if not n.startswith('eeg_encoder.')}


@torch.no_grad()
def predict(tokens, rows, readout, server, tokenizer, task):
    text_scores, jev_scores = [], []
    classes = len(task['classes'])
    for start in range(0, len(rows), CONFIG['jev_batch']):
        part = rows[start:start+CONFIG['jev_batch']]
        z = tokens[start:start+len(part)].cuda()
        valid = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
        text_scores.append(readout(z, valid, task['task_id'])[0].cpu())
        # Dummy label: the prompt/order must not depend on the unknown query label.
        variants = [make_variant(task, 0, r['id'], 0, DATA['seed'], 0) for r in part]
        suffixes = [torch.tensor(tokenizer.encode(v['suffix'], add_special_tokens=False), device='cuda') for v in variants]
        logits = server(z, valid, suffix_ids=suffixes, num_choices=classes)['logits']
        jev_scores.append(logits.cpu())
    return dict(text=torch.cat(text_scores).numpy(), jev=torch.cat(jev_scores).numpy())


def metrics(rows, scores, classes):
    labels = [int(r['label']) for r in rows]
    return {kind: classification_metrics(labels, value, classes) for kind,value in scores.items()}


def worker(root, worker_id, smoke=False):
    prepared = json.loads((root/'prepared.json').read_text())
    if prepared['code_sha256'] != source_identity():
        raise ValueError('Prepared inputs belong to another implementation')
    for p,sha in prepared['artifacts_sha256'].items():
        if digest(p) != sha:
            raise ValueError(f'Prepared data changed: {p}')
    manifest_path = root/('smoke_manifest.json' if smoke else 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    hashes, guard = guard_factory(root, [root/'text_readout.pt', manifest_path, root/'prepared.json'])
    cases = [c for c in manifest['cases'] if c['worker'] == worker_id]
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
    report(progress, status='loading_frozen_jev', worker=worker_id, cases=len(cases))
    server, tokenizer = load_server(saved, encoder)
    del saved
    if any(p.requires_grad for p in server.parameters()):
        raise RuntimeError('JEV initialization was not completely frozen')
    server_version = server_versions(server)
    readout = load_readout(root)
    readout_hash = frozen_hash(readout)
    if not cases:
        raise ValueError('Worker has no assigned cases')
    arm = cases[0]['arm']
    if any(c['arm'] != arm for c in cases):
        raise ValueError('Worker must have one fixed trainable structure')
    seed_rng(CONFIG['seed'])
    if arm == 'residual':
        personal = ResidualAdapter(bottleneck=CONFIG['adapter_width']).cuda()
        lora_locations = []
    else:
        lora_locations = install_lora(encoder)
        personal = encoder
    parameters = [p for p in personal.parameters() if p.requires_grad]
    initial_state = trainable_state(personal)
    fixed_encoder_hash = frozen_hash(encoder)
    n_params = sum(p.numel() for p in parameters)
    tasks = TrialStore().tasks
    cached_subject, cache = None, None
    started = time.monotonic()
    completed = 0
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
            cache = torch.load(root/'subjects'/f'{key[0]}_s{key[1]}.pt', weights_only=False, map_location='cpu')
            cached_subject = key
        ix = {r['id']:i for i,r in enumerate(cache['rows'])}
        support_rows = [cache['rows'][ix[k]] for k in case['support_ids']]
        query_rows = [cache['rows'][ix[k]] for k in case['query_ids']]
        check_support_rows(support_rows, case)
        if smoke:
            query_rows = query_rows[:8]
        si = [ix[r['id']] for r in support_rows]
        qi = [ix[r['id']] for r in query_rows]
        x_support, base_support = cache['x'][si].cuda(), cache['tokens'][si].cuda()
        x_query, base_query = cache['x'][qi], cache['tokens'][qi]
        task = tasks[case['task']]
        baseline_path = out/f'baseline_{key[0]}_s{key[1]}_worker{worker_id}.pt'
        copy_trainable(personal, initial_state)
        encoder.eval()
        with torch.no_grad():
            if arm == 'residual':
                identity_tokens = personal(base_support)
                identity_error = float((identity_tokens-base_support).abs().max())
            else:
                deltas = [m for m in encoder.modules() if isinstance(m, QVDelta)]
                for module in deltas:
                    module.enabled = False
                reference_tokens = encoder(x_support[:4])
                for module in deltas:
                    module.enabled = True
                identity_tokens = encoder(x_support[:4])
                identity_error = float((identity_tokens-reference_tokens).abs().max())
        if identity_error > 1e-5:
            raise RuntimeError(f'Non-identity personal initialization: {identity_error}')
        if baseline_path.exists():
            baseline = torch.load(baseline_path, weights_only=False, map_location='cpu')
            if baseline['ids'] != [r['id'] for r in query_rows]:
                raise ValueError('Baseline uses different query trials')
        else:
            baseline = dict(ids=[r['id'] for r in query_rows],
                            scores=predict(base_query, query_rows, readout, server, tokenizer, task))
            atomic_save(baseline, baseline_path)
        seed_rng(case['seed'])
        generator = np.random.default_rng(case['seed'])
        labels = torch.tensor([int(r['label']) for r in support_rows], device='cuda')
        label_array = labels.cpu().numpy()
        optimizer = torch.optim.AdamW(parameters, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        total = 3 if smoke else CONFIG['steps']
        history, batches = [], []
        torch.cuda.reset_peak_memory_stats()
        training_started = time.monotonic()
        for step in range(1, total+1):
            indices = [int(generator.choice(np.flatnonzero(label_array == j % len(task['classes']))))
                       for j in range(CONFIG['batch_size'])]
            batch_rows = [support_rows[i] for i in indices]
            check_support_rows(batch_rows, case)
            batches.extend(r['id'] for r in batch_rows)
            z = personal(base_support[indices]) if arm == 'residual' else encoder(x_support[indices])
            valid = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
            _, loss = readout(z, valid, case['dataset'], labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == total:
                history.append(dict(step=step, support_loss=float(loss.detach()), grad_norm=float(norm)))
                report(progress, status='adapting', worker=worker_id, case=case['key'], step=step,
                       total=total, completed=completed, cases=len(cases), elapsed_seconds=time.monotonic()-started)
                guard()
        torch.cuda.synchronize()
        train_seconds = time.monotonic()-training_started
        updated_state = trainable_state(personal)
        if tensor_fingerprint(updated_state) == tensor_fingerprint(initial_state):
            raise RuntimeError('Personal parameters did not update')
        if frozen_hash(encoder) != fixed_encoder_hash or frozen_hash(readout) != readout_hash:
            raise RuntimeError('Frozen encoder/text readout was modified')
        if server_versions(server) != server_version:
            raise RuntimeError('Frozen JEV parameter changed')
        with torch.no_grad():
            adapted_tokens = torch.cat([personal(base_query[i:i+32].cuda()).cpu()
                          for i in range(0,len(base_query),32)]) if arm == 'residual' else encode_dense(encoder, x_query, 32)
            support_tokens = personal(base_support) if arm == 'residual' else encoder(x_support)
            support_scores, support_loss = readout(support_tokens,
                torch.ones(support_tokens.shape[:2], device='cuda', dtype=torch.bool), case['dataset'], labels)
        prediction = predict(adapted_tokens, query_rows, readout, server, tokenizer, task)
        with torch.no_grad():
            reload_reference = personal(base_query[:4].cuda()).cpu() if arm == 'residual' else encoder(x_query[:4]).cpu()
        checkpoint = dict(case=case, personal=updated_state, code_sha256=hashes,
                          parent_checkpoint_sha256=prepared['parent_checkpoint_sha256'],
                          readout_sha256=prepared['artifacts_sha256'][str(root/'text_readout.pt')], step=total)
        atomic_save(checkpoint, directory/'adapter.pt')
        # Round-trip serialization proves the saved small artifact reproduces tokens.
        copy_trainable(personal, initial_state)
        restored = torch.load(directory/'adapter.pt', map_location='cpu', weights_only=False)
        copy_trainable(personal, restored['personal'])
        with torch.no_grad():
            restored_tokens = personal(base_query[:4].cuda()).cpu() if arm == 'residual' else encoder(x_query[:4]).cpu()
        if not torch.equal(restored_tokens, reload_reference):
            raise RuntimeError('Saved personal adapter reload changed output')
        atomic_save(dict(ids=[r['id'] for r in query_rows], labels=[int(r['label']) for r in query_rows],
                        baseline=baseline['scores'], adapted=prediction), directory/'predictions.pt')
        baseline_metrics = metrics(query_rows, baseline['scores'], len(task['classes']))
        final_metrics = metrics(query_rows, prediction, len(task['classes']))
        result = dict(case=case, smoke=smoke, steps=total, code_sha256=hashes,
            baseline=baseline_metrics, adapted=final_metrics,
            support_text=classification_metrics(label_array, support_scores.detach().cpu().numpy(), len(task['classes'])),
            final_support_loss=float(support_loss), trainable_parameters=n_params, lora_locations=lora_locations,
            train_seconds=train_seconds, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_includes_frozen_jev=True, edge_hardware_benchmarked=False,
            training_batch_ids_sha256=ids_hash([dict(id=k) for k in batches]),
            parent_encoder_frozen_verified=True, text_readout_frozen_verified=True, jev_frozen_verified=True,
            initialization_identity_max_error=identity_error, checkpoint_reload_verified=True,
            query_used_for_selection=False, original_benchmark_test_modified=False)
        write_json(directory/'support_history.json', history)
        write_json(directory/'complete.json', result)
        completed += 1
        report(progress, status='episode_complete', worker=worker_id, case=case['key'], completed=completed,
               cases=len(cases), metrics=final_metrics, train_seconds=train_seconds, elapsed_seconds=time.monotonic()-started)
    report(progress, status='complete', worker=worker_id, completed=completed, cases=len(cases),
           elapsed_seconds=time.monotonic()-started, trainable_parameters=n_params)


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    groups = defaultdict(list)
    for case in manifest['cases']:
        result = json.loads((root/'episodes'/case['key']/'complete.json').read_text())
        if result['case'] != case or result['smoke'] or result['query_used_for_selection']:
            raise ValueError('Incomplete or mismatched personal result')
        groups[(case['dataset'], case['arm'], case['shots_per_class'])].append(result)
    rows = []
    for (dataset, arm, shots), results in sorted(groups.items()):
        for readout in ('text','jev'):
            subjects = defaultdict(list)
            for r in results:
                subjects[r['case']['subject']].append(r)
            baseline, adapted, baseline_acc, adapted_acc = [], [], [], []
            for rs in subjects.values():
                baseline.append(np.mean([r['baseline'][readout]['balanced_accuracy'] for r in rs]))
                adapted.append(np.mean([r['adapted'][readout]['balanced_accuracy'] for r in rs]))
                baseline_acc.append(np.mean([r['baseline'][readout]['accuracy'] for r in rs]))
                adapted_acc.append(np.mean([r['adapted'][readout]['accuracy'] for r in rs]))
            delta = np.array(adapted)-baseline
            rows.append(dict(dataset=dataset, arm=arm, shots_per_class=shots, total_support_trials=shots*4,
                readout=readout, subjects=len(subjects), episodes=len(results),
                baseline_subject_mean_bacc=float(np.mean(baseline)), adapted_subject_mean_bacc=float(np.mean(adapted)),
                baseline_subject_mean_acc=float(np.mean(baseline_acc)), adapted_subject_mean_acc=float(np.mean(adapted_acc)),
                delta_pp=float(np.mean(delta)*100), worse_subject_fraction=float(np.mean(delta<0)),
                subject_bacc_std=float(np.std(adapted,ddof=1)) if len(adapted)>1 else None,
                mean_training_seconds=float(np.mean([r['train_seconds'] for r in results]))))
    write_json(root/'summary.json', dict(rows=rows, config=CONFIG, aggregation='mean seeds within subject, then equal mean subjects',
         original_benchmark_test=False, protocol='held-out user support/query cross-session or cross-run', complete=True))
    counts = {
        'bcic_2a': ('BCIC-IV-2a','5,184（2,880/1,152/1,152）','9（5/2/2）','原跨被试；个人 T→E'),
        'motor_mv_img': ('PhysioMI','9,837（6,210/1,734/1,893）','109（69/19/21）','原跨被试；个人前四→后两想象 run')}
    lines = ['# 个人少样本校准：B / C', '',
             '原全局窗口和被试数仅描述父模型划分；本表不是原 EEG-FM-Bench 完整测试。',
             '每个 event 一个四秒窗口。B=输出残差 adapter，C=Tower 内 Q/V LoRA-r8；JEV/连接器/文本读出全部冻结。',
             '固定 100 步、每个设置 3 个种子；先平均同一被试的种子，再对被试等权平均。0-shot 是不做个人适配，不表示未见任务。', '']
    lookup = {(r['dataset'],r['arm'],r['shots_per_class'],r['readout']):r for r in rows}
    for readout in ('text','jev'):
        lines += [f'## {"直接文本匹配" if readout == "text" else "冻结 JEV"}', '',
            '| 数据集 | 处理后窗口数：总计（原全局训／验／测） | 任务种类（分类数） | 划分协议 | 被试数：总计（原全局训／验／测） | 每人校准 trial（每类） | 独立 query 总数 | 0-shot Acc / BAcc（%） | B Acc / BAcc（%） | C Acc / BAcc（%） |',
            '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name,shots_list in CONFIG['datasets'].items():
            title,windows,subjects,protocol = counts[name]
            query_n = sum(len(c['query_ids']) for c in manifest['cohorts'] if c['dataset']==name)
            for shots in shots_list:
                b,c = (lookup[(name,arm,shots,readout)] for arm in ('residual','lora'))
                values = [f'{b["baseline_subject_mean_acc"]*100:.2f} / {b["baseline_subject_mean_bacc"]*100:.2f}']
                values += [f'{r["adapted_subject_mean_acc"]*100:.2f} / {r["adapted_subject_mean_bacc"]*100:.2f}' for r in (b,c)]
                lines.append(f'| {title} | {windows} | 运动想象（4 类） | {protocol} | {subjects} | {shots*4}（{shots}） | {query_n} | '+ ' | '.join(values)+' |')
        lines.append('')
    lines += ['2a 仅 2 个独立目标被试，种子不能充当额外被试；PhysioMI 为同日跨 run，不宣称跨日。',
              '所有校准标签均来自显式 support，未用 query 挑选 checkpoint 或超参数；负迁移比例、被试间标准差和耗时见 summary.json。',
              '显存包含本机共置的冻结 JEV，仅为流程测试；尚未测量实际边缘设备时延，token 传输不自动保证隐私。', '']
    (root/'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','worker','summary'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    # Identical attention path with and without LoRA; avoid eval-only fused dispatch.
    torch.backends.mha.set_fastpath_enabled(False)
    if args.mode != 'summary':
        torch.cuda.set_device(0)
    if args.mode == 'prepare':
        prepare(args.root)
    elif args.mode == 'worker':
        worker(args.root, args.worker, args.smoke)
    else:
        summarize(args.root)


if __name__ == '__main__':
    main()
