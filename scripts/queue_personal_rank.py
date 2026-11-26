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
EXECUTION = OUT / 'execution.json'


def identity():
    paths = [SPEC_PATH, Path(__file__).resolve(), REFERENCE / 'plan.json']
    if EXECUTION.exists():
        paths.append(EXECUTION)
    return {**BASE_IDENTITY(), **{str(p): common.digest(p) for p in paths}}


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


def gpu_ready(gpus=None):
    value = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(a):int(b) for a,b in (r.split(',') for r in value.strip().splitlines())}
    return all(free.get(g,0) >= BASE_CONFIG['minimum_free_gpu_mib'] for g in (gpus or BASE_CONFIG['gpus']))


def check_scheduler_only_change(previous, current):
    """Only this launcher and its placement manifest may differ on migration."""
    permitted = {str(Path(__file__).resolve()), str(EXECUTION)}
    changed = {k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k)}
    if changed - permitted:
        raise RuntimeError(f'Training/data identity changed: {sorted(changed-permitted)}')
    return sorted(changed)


def episode_directory(stage, episode):
    return OUT/stage/'jobs'/f'worker{episode["worker"]}'/'cases'/episode['case_key']/f'{episode["arm"]}_n100'


def assign_lanes(entries, gpus):
    lanes = [dict(lane=i,gpu=g,entries=[],estimated_seconds=0.) for i,g in enumerate(gpus)]
    for entry in sorted(entries, key=lambda e:(-e['estimated_seconds'], e['episode'])):
        lane = min(lanes, key=lambda x:(x['estimated_seconds'],x['lane']))
        lane['entries'].append(entry)
        lane['estimated_seconds'] += entry['estimated_seconds']
    for lane in lanes:
        # Resume interrupted fits first on each lane; never redo their completed steps.
        lane['entries'].sort(key=lambda e:(e['resume_step']==0, -e['resume_step'], e['episode']))
    return lanes


def prepare_resume(gpus):
    import torch
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('GPU indices must be distinct and nonnegative')
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if EXECUTION.exists():
        raise RuntimeError('Execution placement already exists; use --start to resume it')
    previous = json.loads((OUT/'queue.json').read_text())
    if previous['state'] in ('complete','stopped_by_user'):
        raise RuntimeError('No automatic restart of completed or explicitly stopped work')
    if not predecessor_ready(json.loads((REFERENCE/'queue.json').read_text())):
        raise RuntimeError('Rank4 controls must finish before migration')
    changed = check_scheduler_only_change(previous['source_hashes'],identity())
    if not gpu_ready(gpus):
        raise RuntimeError('Requested GPUs do not have the required free memory')
    # The shared lock proves the old controller is gone; this also rejects orphan
    # local workers. No broad process termination and no checkpoint rewrites.
    own = str(Path(__file__).resolve().relative_to(ROOT)).replace('/', '.')[:-3]
    for pid in previous.get('worker_pids',[]):
        path = Path(f'/proc/{pid}/cmdline')
        if path.exists() and own.encode() in path.read_bytes():
            raise RuntimeError('An old local rank worker is still alive')
    stages = {}; completed = {}; legacy = previous['source_hashes']
    for stage in SPEC['stages']:
        plan = json.loads((OUT/stage/'plan.json').read_text())
        if plan['source_hashes'] != legacy or plan['config'] != recipe(stage):
            raise RuntimeError('Original rank plan/config differs from the admitted checkpoint identity')
        cases = {c['key']:c for c in plan['cases']}; entries = []
        for episode in plan['episodes']:
            directory = episode_directory(stage,episode)
            result = directory/'result.json'
            if result.exists():
                value = json.loads(result.read_text())
                if value['status']!='complete' or value['steps']!=100 or value['episode']!=episode:
                    raise RuntimeError('Invalid completed personal result')
                # Hash every completed artifact, not only the score summary.
                completed.update({str(f.relative_to(OUT)):common.digest(f)
                    for f in directory.rglob('*') if f.is_file()})
                continue
            last = directory/'last.pt'; step = 0; digest = None
            if last.exists():
                saved = torch.load(last,map_location='cpu',weights_only=False)
                expected = dict(source_hashes=legacy,parent_sha256=BASE_CONFIG['parent_sha256'])
                case = cases[episode['case_key']]
                if (saved['stamp']!=expected or saved['episode']!=episode or
                        saved['case']['support_ids']!=case['support_ids'] or
                        saved['case']['query_ids']!=case['query_ids'] or
                        len(saved['history'])!=saved['step']):
                    raise RuntimeError('Interrupted checkpoint identity/history mismatch')
                step = saved['step']; digest = common.digest(last)
            refs = list(REFERENCE.glob(f'jobs/*/cases/{episode["case_key"]}/{episode["arm"]}_n100/result.json'))
            if len(refs)!=1:
                raise RuntimeError('Missing paired rank4 timing reference')
            ref = json.loads(refs[0].read_text())
            train = sum(h['seconds'] for h in ref['history'])
            evaluation = sum(json.loads(f.read_text())['seconds'] for f in (refs[0].parent/'curves').glob('step*.json'))
            entries.append(dict(episode=episode['key'],resume_step=step,
                checkpoint_sha256=digest,estimated_seconds=train*(100-step)/100+evaluation+30))
        stages[stage] = dict(remaining=len(entries),lanes=assign_lanes(entries,gpus),
            original_plan_sha256=common.digest(OUT/stage/'plan.json'))
    archive = OUT/'executions'/'prior_4gpu'
    archive.mkdir(parents=True,exist_ok=False)
    common.write_json(archive/'queue.json',previous)
    for stage in SPEC['stages']:
        (archive/stage).mkdir(parents=True,exist_ok=True)
        for filename in ('queue.json','plan.json'):
            path=OUT/stage/filename
            if path.exists(): shutil.copy2(path,archive/stage/filename)
    placement = dict(schema='personal-rank-placement-v1',created=time.time(),gpus=gpus,
        authorization='2026-11-26 user moved to8GPU node and requested resume of unfinished rank controls only.',
        legacy_checkpoint_source_hashes=legacy,allowed_scheduler_changes=changed,
        stages=stages,completed_artifact_sha256=completed,
        training_code_config_data_unchanged=True,checkpoint_bytes_rewritten=False,
        original_deadline=previous['deadline'])
    common.write_json(EXECUTION,placement)
    common.write_json(OUT/'queue.json',dict(previous,state='ready_to_resume',source_hashes=identity(),
        execution_manifest=str(EXECUTION.relative_to(ROOT)),controller_pid=None,worker_pids=[],
        resumed_on_eight_gpus=True,gpus=gpus,time=time.time()))
    print(json.dumps(dict(prepared=True,gpus=gpus,stages={s:v['remaining'] for s,v in stages.items()},
        resume_steps={s:[e['resume_step'] for l in v['lanes'] for e in l['entries'] if e['resume_step']]
                      for s,v in stages.items()})),flush=True)


def checked_checkpoint_identity(expected_execution, legacy):
    actual = identity()
    if actual != expected_execution:
        raise RuntimeError('Live execution source/placement changed')
    check_scheduler_only_change(legacy,actual)
    return legacy


def run_lane(stage,lane_index):
    configure_worker(stage)
    from personal import worker as w
    w.torch.set_num_threads(3); w.torch.backends.mha.set_fastpath_enabled(False); w.torch.cuda.set_device(0)
    placement = json.loads(EXECUTION.read_text()); queue = json.loads((OUT/'queue.json').read_text())
    plan_path = OUT/stage/'plan.json'; plan = json.loads(plan_path.read_text())
    legacy = placement['legacy_checkpoint_source_hashes']; live = queue['source_hashes']
    if (common.digest(plan_path)!=placement['stages'][stage]['original_plan_sha256'] or
            plan['source_hashes']!=legacy or plan['config']!=recipe(stage)):
        raise RuntimeError('Original scientific plan changed during migration')
    checked_checkpoint_identity(live,legacy)
    # Old checkpoint stamps retain their original provenance. Each live check first
    # verifies the *new* launcher + placement; unchanged training/data are mandatory.
    w.identity = lambda: checked_checkpoint_identity(live,legacy)
    stamp = dict(source_hashes=legacy,parent_sha256=BASE_CONFIG['parent_sha256'])
    lane = placement['stages'][stage]['lanes'][lane_index]
    audit = OUT/'executions'/'eight_gpu'/stage/f'lane{lane_index}'
    audit.mkdir(parents=True,exist_ok=True)
    def progress(state,**values):
        record = dict(state=state,stage=stage,lane=lane_index,gpu=lane['gpu'],time=time.time(),pid=os.getpid(),**values)
        common.write_json(OUT/stage/f'progress_gpu{lane_index}.json',record)
        print(json.dumps(record),flush=True)
    progress('loading_frozen_parent')
    server = w.PersonalServer(BASE_CONFIG['parent_sha256']); store = w.TrialStore()
    if json.loads(json.dumps(store.stats))!=plan['source_array_stats']:
        raise RuntimeError('EEG arrays changed')
    common.write_json(audit/'tower_lora_manifest.json',server.tower_lora_manifest)
    original_smoke = w.smoke
    def isolated_smoke(server,inputs,support,case,task,arm,directory):
        return original_smoke(server,inputs,support,case,task,arm,audit)
    w.smoke = isolated_smoke
    cases = {c['key']:c for c in plan['cases']}; episodes = {e['key']:e for e in plan['episodes']}
    smoked = set()
    for entry in lane['entries']:
        episode = episodes[entry['episode']]; case = cases[episode['case_key']]
        directory = OUT/stage/'jobs'/f'worker{episode["worker"]}'
        folder = episode_directory(stage,episode)
        if (folder/'result.json').exists():
            continue
        if time.time()>=queue['deadline']:
            progress('budget_exhausted'); return
        folder.mkdir(parents=True,exist_ok=True)
        episode_lock = (folder/'execution.lock').open('a')
        fcntl.flock(episode_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        last=folder/'last.pt'; resume_step=0
        if last.exists():
            resume_step=w.torch.load(last,map_location='cpu',weights_only=False)['step']
        common.write_json(folder/'execution_provenance.json',dict(
            execution_source_hashes=live,checkpoint_source_hashes=legacy,
            placement_sha256=common.digest(EXECUTION),stage=stage,lane=lane_index,gpu=lane['gpu'],
            resume_step=resume_step,checkpoint_sha256=common.digest(last) if last.exists() else None,
            training_code_config_data_unchanged=True,time=time.time()))
        progress('resuming_episode',episode=episode['key'],resume_step=resume_step)
        cache=w.prepare_person(server,store,case,directory,stamp,progress)
        complete=w.run_episode(server,store,plan,case,episode,cache,directory,queue['deadline'],stamp,progress,smoked)
        episode_lock.close()
        if not complete:
            progress('budget_exhausted'); return
    common.write_json(audit/'complete.json',dict(stage=stage,lane=lane_index,source_hashes=live,time=time.time()))
    progress('worker_done')


def resume_controller():
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    state=json.loads((OUT/'queue.json').read_text()); placement=json.loads(EXECUTION.read_text())
    if state['state'] in ('complete','budget_exhausted','stopped_by_user'):
        raise RuntimeError('Execution finished or stopped; explicit authorization needed')
    expected=state['source_hashes']; children=[]
    if identity()!=expected:
        raise RuntimeError('Resume execution identity changed')
    def record(**values):
        state.update(values,controller_pid=os.getpid(),time=time.time())
        common.write_json(OUT/'queue.json',state)
        print(json.dumps({k:v for k,v in state.items() if k!='source_hashes'}),flush=True)
    def guard():
        checked_checkpoint_identity(expected,placement['legacy_checkpoint_source_hashes'])
    env={**os.environ,'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'3','MKL_NUM_THREADS':'3',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    try:
        guard()
        for relative,digest in placement['completed_artifact_sha256'].items():
            if common.digest(OUT/relative)!=digest:
                raise RuntimeError('A previously completed artifact changed')
        for stage,settings in placement['stages'].items():
            plan=json.loads((OUT/stage/'plan.json').read_text())
            if all((episode_directory(stage,e)/'result.json').exists() for e in plan['episodes']):
                continue
            while not gpu_ready(placement['gpus']):
                guard()
                if time.time()>=state['deadline']:
                    record(state='budget_exhausted',stage=stage);return
                time.sleep(SPEC['poll_seconds'])
            if time.time()>=state['deadline']:
                record(state='budget_exhausted',stage=stage);return
            children=[]
            for lane in settings['lanes']:
                if not lane['entries']:continue
                with (OUT/stage/f'gpu{lane["lane"]}_resume.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','scripts.queue_personal_rank',
                        '--stage',stage,'--lane',str(lane['lane'])],cwd=ROOT,
                        env={**env,'CUDA_VISIBLE_DEVICES':str(lane['gpu'])},stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
            record(state='personal_workers',stage=stage,worker_pids=[c.pid for c in children])
            while any(c.poll() is None for c in children):
                guard()
                if any(c.poll() not in (None,0) for c in children):
                    raise RuntimeError('Migrated rank worker failed; see resume logs')
                if time.time()>state['deadline']+300:
                    raise RuntimeError('Worker exceeded resume deadline')
                summarize();time.sleep(SPEC['poll_seconds'])
            if any(c.returncode for c in children):raise RuntimeError('Migrated worker failed')
            complete=all((episode_directory(stage,e)/'result.json').exists() for e in plan['episodes'])
            common.write_json(OUT/stage/'complete.json',dict(complete=complete,time=time.time(),execution_source_hashes=expected))
            if not complete:
                record(state='budget_exhausted',stage=stage);summarize();return
        for relative,digest in placement['completed_artifact_sha256'].items():
            if common.digest(OUT/relative)!=digest:raise RuntimeError('Completed artifact changed')
        summarize();record(state='complete',completed_results=42)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        record(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise


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
    parser.add_argument('--lane', type=int)
    parser.add_argument('--resume-gpus',type=int,nargs='+')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.resume_gpus is not None:
        prepare_resume(args.resume_gpus)
    elif args.lane is not None:
        if args.stage is None:parser.error('--lane requires --stage')
        run_lane(args.stage,args.lane)
    elif args.worker is not None:
        if EXECUTION.exists():raise RuntimeError('Use migrated --lane placement, not the old --worker assignment')
        if args.stage is None:
            parser.error('--worker requires --stage')
        configure_worker(args.stage)
        from personal.worker import main as worker_main
        sys.argv = [sys.argv[0], '--worker', str(args.worker)]
        worker_main()
    elif args.prepare:
        if EXECUTION.exists():raise RuntimeError('Preserve original plans during migrated execution')
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
        resume_controller() if EXECUTION.exists() else controller()


if __name__ == '__main__':
    main()
