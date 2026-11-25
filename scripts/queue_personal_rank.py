"""Matched rank controls using the live personal worker unchanged.

Only this entry point overrides the in-process recipe before worker imports.
It never edits the rank4 config, source, plans, or outputs. Run after rank4 exits.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from personal import common

ROOT = common.ROOT
SPEC_PATH = ROOT / 'configs/personal_rank_followup.json'
SPEC = json.loads(SPEC_PATH.read_text())
BASE_CONFIG = copy.deepcopy(common.CONFIG)
BASE_IDENTITY = common.identity
REFERENCE = ROOT / SPEC['reference']
OUT = ROOT / SPEC['output_root']


def identity():
    return {**BASE_IDENTITY(), **{str(p): common.digest(p) for p in (
        SPEC_PATH, Path(__file__).resolve(), REFERENCE / 'plan.json')}}


def recipe(stage):
    setting = SPEC['stages'][stage]
    config = copy.deepcopy(BASE_CONFIG)
    config.update(run_name=f'personal_rank_{stage}',
        output_root=str((OUT / stage).relative_to(ROOT)),
        authorization=SPEC['authorization'], arms=[setting['arm']], dose_arms=[],
        support_sizes=[100], connector_lora_rank=setting['connector_lora_rank'],
        tower_lora_rank=setting['tower_lora_rank'])
    return config


def configure_worker(stage):
    # Must precede imports that bind CONFIG/OUT/identity from personal.common.
    if any(n in sys.modules for n in ('personal.worker', 'personal.model', 'personal.initialization')):
        raise RuntimeError('Configure the rank recipe before importing the worker')
    common.CONFIG.clear()
    common.CONFIG.update(recipe(stage))
    common.OUT = OUT / stage
    common.identity = identity


def build_plan(stage):
    from jev_train.checkpoint import tensor_fingerprint
    from personal.modules import UserParameters
    baseline = json.loads((REFERENCE / 'plan.json').read_text())
    if baseline['source_hashes'] != BASE_IDENTITY() or baseline['config'] != BASE_CONFIG:
        raise RuntimeError('Live rank4 reference source/config changed')
    config = recipe(stage)
    arm = SPEC['stages'][stage]['arm']
    user = UserParameters(config, arm=arm)
    plan = copy.deepcopy(baseline)
    plan.update(schema='personal-lora-matched-rank-v1', config=config,
        source_hashes=identity(), rank_stage=stage,
        reference=str(REFERENCE.relative_to(ROOT)),
        rank_change=SPEC['stages'][stage], initializations={arm: dict(arm=arm,
            final_sha=tensor_fingerprint(user.state_dict()), seed=config['initialization_seed'],
            source_episodes=0)}, episodes=[], jobs=[])
    plan['support_sets'] = {c['key']: {'100': c['support_ids'][:]} for c in plan['cases']}
    plan['schedules'] = {f'{c["key"]}_n100': baseline['schedules'][f'{c["key"]}_n100']
                         for c in plan['cases']}
    for worker, gpu in enumerate(config['gpus']):
        cases = plan['cases'][worker::len(config['gpus'])]
        entries = []
        for case in cases:
            key = case['key']
            episode = dict(key=f'{key}__{arm}__n100', case_key=key, arm=arm,
                support_total=100, schedule=f'{key}_n100', worker=worker, phase=0)
            plan['episodes'].append(episode)
            entries.append(episode['key'])
        plan['jobs'].append(dict(key=f'worker{worker}', gpu=gpu,
            cases=[c['key'] for c in cases], episodes=entries))
    if len(plan['episodes']) != 21 or len({e['case_key'] for e in plan['episodes']}) != 21:
        raise RuntimeError('Rank comparison must contain exactly the same21 users')
    return plan


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for stage in SPEC['stages']:
        plan = build_plan(stage)
        path = OUT / stage / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise RuntimeError(f'Existing rank plan differs: {stage}')
        common.write_json(path, plan)
    summarize()


def paired_group(new, old, cases, step):
    keys = [c['key'] for c in cases]
    if not keys or any((k, step) not in new or (k, step) not in old for k in keys):
        return None
    pairs = [(new[k, step], old[k, step]) for k in keys]
    for a, b in pairs:
        if (a['case']['query_ids'] != b['case']['query_ids'] or
                a['case']['support_ids'] != b['case']['support_ids'] or
                abs(a['baseline']['balanced_accuracy']-b['baseline']['balanced_accuracy']) > 1e-12):
            raise RuntimeError('Rank comparison changed samples or baseline predictions')
    diffs = [a['query']['balanced_accuracy']-b['query']['balanced_accuracy'] for a, b in pairs]
    return dict(users=len(keys), case_ids=keys,
        rank4_bacc=sum(b['query']['balanced_accuracy'] for _, b in pairs)/len(keys),
        rank16_bacc=sum(a['query']['balanced_accuracy'] for a, _ in pairs)/len(keys),
        delta_pp=100*sum(diffs)/len(keys), negative_users=sum(d < -1e-10 for d in diffs),
        rank4_support_accuracy=sum(b['support']['accuracy'] for _, b in pairs)/len(keys),
        rank16_support_accuracy=sum(a['support']['accuracy'] for a, _ in pairs)/len(keys),
        rank4_query_ce=sum(b['query']['ce'] for _, b in pairs)/len(keys),
        rank16_query_ce=sum(a['query']['ce'] for a, _ in pairs)/len(keys))


def read_points(root, arm):
    points = {}
    for path in root.glob(f'jobs/*/cases/*/{arm}_n100/curves/step*.json'):
        point = json.loads(path.read_text())
        if point['arm'] != arm or point['support_n'] != 100:
            raise RuntimeError('Unexpected rank curve')
        key = point['case']['key'], point['step']
        if key in points:
            raise RuntimeError('Duplicate personal rank curve')
        points[key] = point
    return points


def summarize():
    baseline = json.loads((REFERENCE / 'plan.json').read_text())
    report = dict(expected_results=42, completed_results=0, stages={},
        query_used_for_selection=False, independent_test_estimate=False)
    lines = ['# Personal LoRA rank controls', '',
        'Same21 users,100 support samples,100 updates. Exploratory queries; not a blind test.', '',
        'Rank4 controls are reused from the location study. Only complete paired groups are shown.', '']
    for stage, setting in SPEC['stages'].items():
        arm = setting['arm']
        new, old = read_points(OUT / stage, arm), read_points(REFERENCE, arm)
        complete = len(list((OUT / stage).glob('jobs/*/cases/*/*/result.json')))
        matched, strata = {}, {}
        for name in BASE_CONFIG['datasets']:
            cases = [c for c in baseline['cases'] if c['dataset'] == name]
            matched[name] = {str(s): v for s in BASE_CONFIG['eval_steps']
                if (v := paired_group(new, old, cases, s)) is not None}
            strata[name] = {exposure: {str(s): v for s in BASE_CONFIG['eval_steps']
                if (v := paired_group(new, old, [c for c in cases if c['exposure_stratum']==exposure], s)) is not None}
                for exposure in sorted({c['exposure_stratum'] for c in cases})}
        equal = {}
        for s in BASE_CONFIG['eval_steps']:
            if all(str(s) in g for g in matched.values()):
                rows = [g[str(s)] for g in matched.values()]
                equal[str(s)] = {k:sum(r[k] for r in rows)/len(rows)
                    for k in ('rank4_bacc','rank16_bacc','delta_pp')}
        report['stages'][stage] = dict(completed_results=complete, expected_results=21,
            curve_points=len(new), matched=matched, exposure=strata, equal_task=equal,
            ranks=setting, main_step=100)
        report['completed_results'] += complete
        lines += [f'## {stage}', '', f'Completed: {complete}/21 personal fits. {setting["comparison"]}.', '',
            '| Dataset | Processed windows total (train/valid/test) | Task (classes) | Personal split | Subjects total (train/valid/test) | Rank4 BAcc | Rank16 BAcc | Delta pp | Worse users |',
            '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
        task_types = {'bcic_2a':'MI (4)','seed':'Emotion (3)','hmc':'Sleep (5)','Nakanishi2015':'SSVEP (12)'}
        for name in BASE_CONFIG['datasets']:
            inv = baseline['inventory'][name]
            def count(field):
                return '/'.join(str(inv[field].get(s,0)) for s in ('train','valid','test'))
            row = matched[name].get('100')
            values = f'{row["rank4_bacc"]*100:.2f} | {row["rank16_bacc"]*100:.2f} | {row["delta_pp"]:+.2f} | {row["negative_users"]}/{row["users"]}' if row else '— | — | — | —'
            lines.append(f'| {name} | {inv["windows_total"]} ({count("window_counts")}) | {task_types[name]} | {BASE_CONFIG["cohorts"][name]["policy"]} | {inv["subjects_total"]} ({count("subject_counts")}) | {values} |')
        if '100' in equal:
            v = equal['100']
            lines += ['', f'Equal-task mean: {v["rank4_bacc"]*100:.2f}% → {v["rank16_bacc"]*100:.2f}% ({v["delta_pp"]:+.2f} pp).']
        lines += ['']
    lines += ['SEED/HMC counts are windows/epochs. Shared-training exposure strata are in summary.json. A01–A05 and SSVEP queries were in shared training; do not call the aggregate unseen-user generalization.', '']
    common.write_json(OUT / 'summary.json', report)
    path = OUT / 'report.md'
    tmp = path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(path)
    return report


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def predecessor_ready(state):
    if state['state'] in ('failed', 'stopped_by_user'):
        raise RuntimeError('Predecessor failed/stopped; rank controls will not start automatically')
    if state['state'] not in ('complete', 'budget_exhausted'):
        return False
    # Full100-sample controls must exist even if the24/48 phase hit its time limit.
    baseline = json.loads((REFERENCE / 'plan.json').read_text())
    for case in baseline['cases']:
        for arm in ('both', 'tower_both'):
            matches = list(REFERENCE.glob(f'jobs/*/cases/{case["key"]}/{arm}_n100/result.json'))
            if len(matches) != 1 or json.loads(matches[0].read_text())['status'] != 'complete':
                raise RuntimeError('Missing complete100-sample rank4 matched control')
    return True


def gpu_ready():
    value = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(a):int(b) for a,b in (r.split(',') for r in value.strip().splitlines())}
    return all(free.get(g,0) >= BASE_CONFIG['minimum_free_gpu_mib'] for g in BASE_CONFIG['gpus'])


def controller():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUT / 'queue.json'
    hashes = identity()
    old = json.loads(state_path.read_text()) if state_path.exists() else None
    if old and (old['source_hashes'] != hashes or old['state'] in ('complete','budget_exhausted','stopped_by_user')):
        raise RuntimeError('Changed identity or finished rank budget; explicit new authorization needed')
    predecessor = json.loads((REFERENCE / 'queue.json').read_text())
    state = old or dict(queued_at=time.time(), wait_deadline=time.time()+SPEC['maximum_wait_hours']*3600,
        source_hashes=hashes, state='preparing', training_started=None, deadline=None,
        predecessor_processes={str(pid):process_token(pid) for pid in
            [predecessor['controller_pid']]+predecessor.get('worker_pids',[])})
    state['controller_pid'] = os.getpid()
    children = []
    env = {**os.environ, 'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'3','MKL_NUM_THREADS':'3',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}

    def record(**values):
        state.update(values, time=time.time())
        common.write_json(state_path, state)
        print(json.dumps({k:v for k,v in state.items() if k!='source_hashes'}), flush=True)

    def guard():
        if identity() != hashes:
            raise RuntimeError('Rank followup source changed')

    try:
        prepare()
        record(state='waiting_for_rank4')
        while True:
            guard()
            predecessor = json.loads((REFERENCE / 'queue.json').read_text())
            exited = all(token is None or process_token(int(pid)) != token
                         for pid,token in state['predecessor_processes'].items())
            if predecessor_ready(predecessor) and exited and gpu_ready():
                break
            if exited and predecessor['state'] not in ('complete', 'budget_exhausted'):
                raise RuntimeError('Predecessor processes disappeared without a terminal state')
            if time.time() >= state['wait_deadline']:
                record(state='budget_exhausted', reason='Predecessor/GPU wait limit')
                return
            time.sleep(SPEC['poll_seconds'])
        if shutil.disk_usage(ROOT).free < BASE_CONFIG['minimum_free_disk_gib']*1024**3:
            raise RuntimeError('Insufficient disk for rank checkpoints')
        if common.digest(ROOT / BASE_CONFIG['parent'] / 'best.pt') != BASE_CONFIG['parent_sha256']:
            raise RuntimeError('Shared checkpoint changed')
        if state['training_started'] is None:
            state.update(training_started=time.time(), deadline=time.time()+SPEC['additional_wall_hours']*3600)
        for stage in SPEC['stages']:
            directory = OUT / stage
            plan = json.loads((directory / 'plan.json').read_text())
            if all((directory/'jobs'/j['key']/'complete.json').exists() for j in plan['jobs']):
                continue
            if time.time() >= state['deadline']:
                record(state='budget_exhausted', stage=stage)
                return
            guard()
            while not gpu_ready():
                guard()
                if time.time() >= state['deadline']:
                    record(state='budget_exhausted', stage=stage, reason='GPU availability')
                    return
                time.sleep(SPEC['poll_seconds'])
            common.write_json(directory/'queue.json',dict(source_hashes=hashes,
                parent_sha256=BASE_CONFIG['parent_sha256'], deadline=state['deadline'],
                state='personal_workers', stage=stage))
            children = []
            for worker,gpu in enumerate(BASE_CONFIG['gpus']):
                with (directory/f'gpu{worker}.log').open('a') as log:
                    child = subprocess.Popen([sys.executable,'-u','-m','scripts.queue_personal_rank',
                        '--stage',stage,'--worker',str(worker)], cwd=ROOT,
                        env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)}, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
            record(state='personal_workers', stage=stage, worker_pids=[p.pid for p in children])
            while any(p.poll() is None for p in children):
                guard()
                failed = [(p.pid,p.returncode) for p in children if p.poll() not in (None,0)]
                if failed:
                    raise RuntimeError(f'Rank worker failed: {failed}')
                if time.time() > state['deadline']+300:
                    raise RuntimeError('Rank worker did not honor its deadline')
                summarize()
                time.sleep(SPEC['poll_seconds'])
            if any(p.returncode for p in children):
                raise RuntimeError('A rank worker failed')
            complete = all((directory/'jobs'/j['key']/'complete.json').exists() for j in plan['jobs'])
            common.write_json(directory/'complete.json',dict(complete=complete,time=time.time(),source_hashes=hashes))
            if not complete:
                record(state='budget_exhausted', stage=stage)
                summarize()
                return
        summarize()
        record(state='complete', completed_results=42)
    except BaseException as exc:
        # Only children of this controller; never terminate the original rank4 job.
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        record(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--stage', choices=list(SPEC['stages']))
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker is not None:
        if args.stage is None:
            parser.error('--worker requires --stage')
        configure_worker(args.stage)
        from personal.worker import main as worker_main
        sys.argv = [sys.argv[0], '--worker', str(args.worker)]
        worker_main()
    elif args.prepare:
        prepare()
        print(json.dumps(dict(prepared=True, stages=list(SPEC['stages']), episodes=42)))
    elif args.summary:
        print(json.dumps(summarize()))
    elif args.start:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (OUT/'controller.log').open('a') as log:
            child = subprocess.Popen([sys.executable,'-u','-m','scripts.queue_personal_rank'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, output=str(OUT))))
    else:
        controller()


if __name__ == '__main__':
    main()
