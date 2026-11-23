"""Support-only JEV CE: matched signal adapters with frozen or adapted language LoRA."""
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
from .personalization import (CONFIG, DATA, ROOT, ResidualAdapter, TextReadout, QVDelta,
    check_support_rows, copy_trainable, frozen_hash, ids_hash, install_lora, load_encoder,
    load_parent, make_manifest, projection_module, source_identity, trainable_state,
    configure_lm, fixed_server_versions, copy_named)
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
                 reused_projection_without_updates=True, inputs_identical_to_reference=True,
                 artifacts_sha256=artifacts,
                 personal_cases=len(make_manifest(store)['cases']), subjects=len(make_manifest(store)['cohorts']))
    write_json(root/'prepared.json', proof)
    report(root/'prepare_progress.json', status='complete', personal_cases=proof['personal_cases'],
           subjects=proof['subjects'], input_root=str(source))


def load_readout(input_root):
    saved = torch.load(input_root/'text_readout.pt', weights_only=True, map_location='cpu')
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
def predict(tokens, rows, readout, server, tokenizer, task):
    server.eval()
    text_scores, jev_scores = [], []
    suffix = suffix_for(task, tokenizer)
    for start in range(0, len(rows), CONFIG['jev_batch']):
        z = tokens[start:start+min(CONFIG['jev_batch'],len(rows)-start)].cuda()
        valid = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
        text_scores.append(readout(z, valid, task['task_id'])[0].cpu())
        logits = server(z, valid, suffix_ids=[suffix]*len(z), num_choices=len(task['classes']))['logits']
        jev_scores.append(logits.cpu())
    return dict(text=torch.cat(text_scores).numpy(), jev=torch.cat(jev_scores).numpy())


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
    readout = load_readout(input_root)
    readout_hash = frozen_hash(readout)
    seed_rng(CONFIG['seed'])
    signal_arm = cases[0]['signal_arm']
    if signal_arm == 'residual':
        personal = ResidualAdapter(bottleneck=CONFIG['adapter_width']).cuda()
        lora_locations = []
    else:
        lora_locations = install_lora(encoder)
        personal = encoder
    lm_parameters = configure_lm(server, cases[0]['train_lm_lora'])
    lm_count = sum(p.numel() for p in lm_parameters.values())
    if lm_count != (CONFIG['lm_lora_expected_parameters'] if cases[0]['train_lm_lora'] else 0):
        raise ValueError(f'Unexpected LM LoRA count: {lm_count}')
    initial_lm = {n:p.detach().cpu().clone() for n,p in lm_parameters.items()}
    lm_initial_hash = tensor_fingerprint(initial_lm)
    server_version = fixed_server_versions(server)
    parameters = [p for p in personal.parameters() if p.requires_grad]
    initial_state = trainable_state(personal)
    fixed_encoder_hash = frozen_hash(encoder)
    n_params = sum(p.numel() for p in parameters)
    parameter_groups = [dict(params=parameters,lr=CONFIG['learning_rate'])]
    if lm_parameters:
        parameter_groups.append(dict(params=list(lm_parameters.values()),lr=CONFIG['lm_learning_rate']))
    tasks = TrialStore().tasks
    cached_subject, cache = None, None
    started, completed = time.monotonic(), 0
    report(progress, status='ready', worker=worker_id, cases=len(cases), signal_parameters=n_params,
           lm_parameters=lm_count, lm_total_parameters=sum(p.numel() for p in server.text.parameters()))
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
        x_support, base_support = cache['x'][si].cuda(), cache['tokens'][si].cuda()
        x_query, base_query = cache['x'][qi], cache['tokens'][qi]
        task = tasks[case['task']]
        copy_trainable(personal, initial_state)
        copy_named(lm_parameters, initial_lm)
        server.eval()
        with torch.no_grad():
            if signal_arm == 'residual':
                identity_error = float((personal(base_support)-base_support).abs().max())
            else:
                deltas = [m for m in encoder.modules() if isinstance(m,QVDelta)]
                for m in deltas: m.enabled = False
                reference = encoder(x_support[:4])
                for m in deltas: m.enabled = True
                identity_error = float((encoder(x_support[:4])-reference).abs().max())
        if identity_error != 0:
            raise RuntimeError(f'Non-identity initialization: {identity_error}')
        baseline_path = out/f'baseline_{key[0]}_s{key[1]}_worker{worker_id}.pt'
        if baseline_path.exists():
            baseline = torch.load(baseline_path, weights_only=False, map_location='cpu')
            if baseline['ids'] != [r['id'] for r in query_rows]:
                raise ValueError('Baseline query IDs changed')
        else:
            baseline = dict(ids=[r['id'] for r in query_rows],
                            scores=predict(base_query, query_rows, readout, server, tokenizer, task))
            atomic_save(baseline, baseline_path)
        baseline_metrics = metrics(query_rows, baseline['scores'],len(task['classes']))
        old_path = input_root/('smoke' if smoke else 'episodes')/case['reference_case']/'complete.json'
        old_result = json.loads(old_path.read_text())
        if baseline_metrics != old_result['baseline']:
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
                z = personal(base_support[sub]) if signal_arm=='residual' else encoder(x_support[sub])
                valid = torch.ones(z.shape[:2],device='cuda',dtype=torch.bool)
                loss = server(z,valid,labels=labels[sub],suffix_ids=[suffix]*len(sub),
                              num_choices=len(task['classes']))['loss']
                (loss*(len(sub)/len(indices))).backward()
                loss_sum += float(loss.detach())*len(sub)/len(indices)
            norm = torch.nn.utils.clip_grad_norm_(parameters,CONFIG['clip_grad_norm'],error_if_nonfinite=True)
            lm_norm = torch.nn.utils.clip_grad_norm_(list(lm_parameters.values()),CONFIG['clip_grad_norm'],
                                                     error_if_nonfinite=True) if lm_parameters else torch.tensor(0.)
            optimizer.step()
            if step==1 or step%10==0 or step==total:
                history.append(dict(step=step,support_batch_ce=loss_sum,signal_grad_norm=float(norm),
                                    lm_grad_norm=float(lm_norm)))
                report(progress,status='adapting',worker=worker_id,case=case['key'],step=step,total=total,
                       completed=completed,cases=len(cases),elapsed_seconds=time.monotonic()-started,loss=loss_sum)
                guard()
        torch.cuda.synchronize()
        train_seconds = time.monotonic()-training_started
        del optimizer
        server.eval()
        updated_state = trainable_state(personal)
        updated_lm = {n:p.detach().cpu().clone() for n,p in lm_parameters.items()}
        lm_updated = tensor_fingerprint(updated_lm)!=lm_initial_hash
        if tensor_fingerprint(updated_state)==tensor_fingerprint(initial_state) or lm_updated!=bool(lm_parameters):
            raise RuntimeError('Unexpected signal/LM update audit')
        if frozen_hash(encoder)!=fixed_encoder_hash or frozen_hash(readout)!=readout_hash:
            raise RuntimeError('Frozen Tower/text readout changed')
        if fixed_server_versions(server)!=server_version:
            raise RuntimeError('Frozen JEV base/connector/head changed')
        batch_hash = ids_hash([dict(id=k) for k in batches])
        if batch_hash != old_result['training_batch_ids_sha256']:
            raise ValueError('Calibration batch stream differs from prior experiment')
        with torch.no_grad():
            adapted_tokens = (torch.cat([personal(base_query[i:i+32].cuda()).cpu() for i in range(0,len(base_query),32)])
                              if signal_arm=='residual' else encode_dense(encoder,x_query,32))
            support_tokens = personal(base_support).cpu() if signal_arm=='residual' else encode_dense(encoder,x_support,32)
        prediction = predict(adapted_tokens,query_rows,readout,server,tokenizer,task)
        support_prediction = predict(support_tokens,support_rows,readout,server,tokenizer,task)
        # Identical batch shape before/after reload, independent of bf16 batching roundoff.
        with torch.no_grad():
            probe_tokens = personal(base_query[:4].cuda()).cpu() if signal_arm=='residual' else encoder(x_query[:4]).cpu()
        probe = predict(probe_tokens,query_rows[:4],readout,server,tokenizer,task)
        checkpoint = dict(case=case,personal=updated_state,lm_lora=updated_lm,code_sha256=hashes,
                          parent_checkpoint_sha256=prepared['parent_checkpoint_sha256'],step=total)
        atomic_save(checkpoint,directory/'adapter.pt')
        copy_trainable(personal,initial_state)
        copy_named(lm_parameters,initial_lm)
        del updated_lm, checkpoint
        restored = torch.load(directory/'adapter.pt',map_location='cpu',weights_only=False,mmap=True)
        copy_trainable(personal,restored['personal'])
        copy_named(lm_parameters,restored['lm_lora'])
        with torch.no_grad():
            restored_tokens = personal(base_query[:4].cuda()).cpu() if signal_arm=='residual' else encoder(x_query[:4]).cpu()
        check = predict(restored_tokens,query_rows[:4],readout,server,tokenizer,task)
        if not torch.equal(restored_tokens,probe_tokens) or any(not np.array_equal(check[k],probe[k]) for k in check):
            raise RuntimeError('Saved personal/LM adapter reload changed prediction')
        del restored
        atomic_save(dict(ids=[r['id'] for r in query_rows],labels=[int(r['label']) for r in query_rows],
                         baseline=baseline['scores'],adapted=prediction),directory/'predictions.pt')
        result = dict(case=case,smoke=smoke,steps=total,objective='jev_ce',code_sha256=hashes,
            baseline=baseline_metrics,adapted=metrics(query_rows,prediction,len(task['classes'])),
            support=metrics(support_rows,support_prediction,len(task['classes'])),
            final_support_ce=float(F.cross_entropy(torch.from_numpy(support_prediction['jev']),labels.cpu())),
            trainable_parameters=n_params+lm_count,signal_trainable_parameters=n_params,lm_trainable_parameters=lm_count,
            lora_locations=lora_locations,train_seconds=train_seconds,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            edge_local_training=False,peak_includes_jev=True,training_batch_ids_sha256=batch_hash,
            parent_encoder_frozen_verified=True,text_readout_frozen_verified=True,
            jev_base_connector_head_frozen_verified=True,lm_lora_updated=lm_updated,
            initialization_identity_max_error=identity_error,checkpoint_reload_verified=True,
            query_used_for_selection=False,original_benchmark_test_modified=False,
            reference_batch_ids_verified=True,reference_baseline_verified=True)
        write_json(directory/'support_history.json',history)
        write_json(directory/'complete.json',result)
        completed += 1
        report(progress,status='episode_complete',worker=worker_id,case=case['key'],completed=completed,cases=len(cases),
               metrics=result['adapted'],train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started)
    report(progress,status='complete',worker=worker_id,completed=completed,cases=len(cases),
           elapsed_seconds=time.monotonic()-started,signal_parameters=n_params,lm_parameters=lm_count)


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    groups = defaultdict(list)
    for case in manifest['cases']:
        result = json.loads((root/'episodes'/case['key']/'complete.json').read_text())
        if result['case']!=case or result['smoke'] or result['query_used_for_selection']:
            raise ValueError('Incomplete or mismatched result')
        groups[(case['dataset'],case['arm'],case['shots_per_class'])].append(result)
    rows = []
    for (dataset,arm,shots),results in sorted(groups.items()):
        for readout in ('text','jev'):
            subjects = defaultdict(list)
            for r in results: subjects[r['case']['subject']].append(r)
            def average(stage,metric):
                return [float(np.mean([r[stage][readout][metric] for r in rs])) for rs in subjects.values()]
            base = average('baseline','balanced_accuracy')
            adapted = average('adapted','balanced_accuracy')
            delta = np.array(adapted)-base
            rows.append(dict(dataset=dataset,arm=arm,shots_per_class=shots,total_support_trials=shots*4,
                readout=readout,subjects=len(subjects),episodes=len(results),seeds=CONFIG['seeds'],
                baseline_subject_mean_bacc=float(np.mean(base)),adapted_subject_mean_bacc=float(np.mean(adapted)),
                baseline_subject_mean_acc=float(np.mean(average('baseline','accuracy'))),
                adapted_subject_mean_acc=float(np.mean(average('adapted','accuracy'))),
                support_subject_mean_bacc=float(np.mean(average('support','balanced_accuracy'))),
                delta_pp=float(np.mean(delta)*100),worse_subject_fraction=float(np.mean(delta<0)),
                subject_bacc_std=float(np.std(adapted,ddof=1)),mean_training_seconds=float(np.mean([r['train_seconds'] for r in results]))))
    write_json(root/'summary.json',dict(rows=rows,config=CONFIG,complete=True,
        aggregation='mean seeds within subject, then equal mean subjects',original_benchmark_test=False,
        limitations='Seed42 exploratory pilot; no smaller LM evaluated; base versus trainable parameter counts differ.'))
    lookup = {(r['dataset'],r['arm'],r['shots_per_class'],r['readout']):r for r in rows}
    labels = {'bcic_2a':('BCIC-IV-2a','5,184（2,880/1,152/1,152）','9（5/2/2）','跨被试；个人 T→E'),
              'motor_mv_img':('PhysioMI','9,837（6,210/1,734/1,893）','109（69/19/21）','跨被试；个人前四→后两想象 run')}
    lines = ['# JEV 分类损失：冻结 LM 与更新 LM LoRA', '',
        'Seed42；所有组固定 100 步、batch16、相同 support/query。B=残差 adapter，C=Tower LoRA-r8。',
        '更新 LM 指既有语言层 LoRA-r32，不是 27B 全参微调。连接器/head 冻结。不是模型规模对照，也不再是纯边缘适配。','']
    for readout in ('jev','text'):
        lines += [f'## {"JEV 决策" if readout=="jev" else "直接文本诊断"}', '',
            '| 数据集 | 处理后窗口总计（原训／验／测） | 任务种类（分类数） | 划分协议 | 被试总计（原训／验／测） | 每类／总校准 trial | 0-shot BAcc（%） | B＋冻结 LM | B＋LM LoRA | C＋冻结 LM | C＋LM LoRA |',
            '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for dataset,shots_list in CONFIG['datasets'].items():
            title,windows,subjects,protocol = labels[dataset]
            for shots in shots_list:
                rs = [lookup[(dataset,arm,shots,readout)] for arm in CONFIG['arms']]
                values = [rs[0]['baseline_subject_mean_bacc']]+[r['adapted_subject_mean_bacc'] for r in rs]
                lines.append(f'| {title} | {windows} | 运动想象（4 类） | {protocol} | {subjects} | {shots}／{shots*4} | '+
                             ' | '.join(f'{v*100:.2f}' for v in values)+' |')
        lines.append('')
    lines += ['均为被试等权 BAcc（%）；Acc、support BAcc、负迁移与耗时见 summary.json。',
              '2a 只有 2 名目标被试；PhysioMI 21 名。未根据 query 选最佳步或调整参数；旧 InfoNCE 对照须筛选同一 seed42 再比较。','']
    (root/'summary.md').write_text('\n'.join(lines))


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
