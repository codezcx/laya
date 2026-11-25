"""21 historical users; paired LoRA locations and nested support sizes."""
import hashlib
import json
from collections import Counter
import numpy as np
from .common import CONFIG, OUT, ROOT, identity, write_json, check_rows
from .data import TrialStore, source_group, trial_key
from .initialization import audit_initializations

def row_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def select_support(pool, classes, total, seed):
    quotas = np.full(classes, total // classes, dtype=int)
    extra = np.random.default_rng(np.random.SeedSequence([seed,911])).permutation(classes)[:total % classes]
    quotas[extra] += 1
    selected = []
    for label, count in enumerate(quotas):
        rows = sorted((r for r in pool if r['label']==label),key=trial_key)
        if len(rows) < count:
            raise ValueError(f'Class {label} has {len(rows)} rows, needs {count}')
        rng = np.random.default_rng(np.random.SeedSequence([seed,label,711]))
        selected.extend(rows[int(i)] for i in rng.permutation(len(rows))[:count])
    return selected


def split_person(rows, name, classes):
    details = {}
    if name == 'bcic_2a':
        pool = sorted([r for r in rows if r['record'].endswith('T')],key=trial_key)
        query = sorted([r for r in rows if r['record'].endswith('E')],key=trial_key)
        for part in (pool,query):
            if len(part)!=288 or len({source_group(r) for r in part})!=288:
                raise ValueError('BCIC requires288 unique trials per session')
            if [sum(r['label']==c for r in part) for c in range(classes)]!=[72]*4:
                raise ValueError('Unexpected BCIC class counts')
        details['scope']='full_E_session'
    elif name == 'hmc':
        ordered = sorted(rows,key=lambda r:(r['record'],int(r['start'])))
        if len({r['record'] for r in ordered})!=1:
            raise ValueError('Expected one sleep recording per subject')
        cut = int(len(ordered)*CONFIG['hmc_support_fraction'])
        gap = CONFIG['hmc_gap_epochs']
        pool,query = ordered[:cut],ordered[cut+gap:]
        details.update(scope='full_chronological_tail_after_gap',cut_index=cut,
                       purge_ids=[r['id'] for r in ordered[cut:cut+gap]])
        if min(int(r['start']) for r in query)-max(int(r['start'])+r['samples'] for r in pool)<gap*6000:
            raise ValueError('Sleep support/query time gap is too short')
    elif name == 'seed':
        pool = sorted([r for r in rows if r['split']=='train'],key=trial_key)
        query = sorted([r for r in rows if r['split']=='test'],key=trial_key)
        details['scope']='full_original_test_windows'
    elif name == 'Nakanishi2015':
        if len(rows)!=180 or len({source_group(r) for r in rows})!=180:
            raise ValueError('Expected180 unique SSVEP trials')
        pool,query=sorted(rows,key=trial_key),None
        details['scope']='all_remaining_trials_in_original_train_pool'
    else:
        raise ValueError('Dataset lacks an audited personal split')
    support = select_support(pool,classes,CONFIG['support_total'],CONFIG['seed'])
    if query is None:
        ids = {r['id'] for r in support}
        query = [r for r in pool if r['id'] not in ids]
    if {source_group(r) for r in support} & {source_group(r) for r in query}:
        raise ValueError('Support/query share an original trial/epoch')
    if set(r['label'] for r in query)!=set(range(classes)):
        raise ValueError('Query does not cover all task classes')
    return support,query,details



def personal_batches(case):
    batches = []
    for step in range(1, CONFIG['steps'] + 1):
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], case['order'], step]))
        ix = rng.integers(len(case['support_ids']), size=CONFIG['batch_size']).tolist()
        batches.append(dict(step=step, indices=ix, ids=[case['support_ids'][i] for i in ix]))
    return batches


def build(store):
    previous = json.loads((ROOT / CONFIG['paired_reference'] / 'plan.json').read_text())
    old = {(c['dataset'], c['subject']): c for c in previous['cases']}
    targets = []
    for tid, task in enumerate(store.tasks):
        name, classes = task['task_id'], len(task['classes'])
        for subject in CONFIG['cohorts'][name]['subjects']:
            rows = [r for r in store.rows[tid] if r['subject'] == subject]
            support, query, _ = split_person(rows, name, classes)
            case = old[name, subject]
            if tid != case['task'] or classes != case['classes']:
                raise RuntimeError('Task interpretation changed')
            for field, selected in [('support_ids', support), ('query_ids', query)]:
                if [r['id'] for r in selected] != case[field]:
                    raise RuntimeError('Historical target pairing changed')
            check_rows(support, case, gradient=True)
            check_rows(query, case)
            targets.append(case)
    for path, stat in store.stats.items():
        if list(stat) != previous['source_array_stats'][path]:
            raise RuntimeError('Historical target EEG array changed')
    byid={r['id']:r for rows in store.rows.values() for r in rows}
    support_sets={c['key']:{str(n):nested_support(c,byid,n) for n in CONFIG['support_sizes']} for c in targets}
    schedules={f'{c["key"]}_n{n}':personal_batches(dict(c,support_ids=support_sets[c['key']][str(n)]))
               for c in targets for n in CONFIG['support_sizes']}
    worker_cases=[[] for _ in CONFIG['gpus']]
    for dataset in CONFIG['datasets']:
        for i,case in enumerate(c for c in targets if c['dataset']==dataset):
            worker_cases[i%len(worker_cases)].append(case['key'])
    jobs=[]; episodes=[]
    for worker,keys in enumerate(worker_cases):
        entries=[]
        for phase,n in enumerate(CONFIG['support_sizes']):
            arms=CONFIG['arms'] if n==100 else CONFIG['dose_arms']
            # Reset after every condition; reproducibly balance processing order.
            for key in keys:
                case=next(c for c in targets if c['key']==key)
                rng=np.random.default_rng(np.random.SeedSequence([42,case['order'],phase,197]))
                for arm in rng.permutation(arms).tolist():
                    ep=dict(key=f'{key}__{arm}__n{n}',case_key=key,arm=arm,support_total=n,
                            schedule=f'{key}_n{n}',worker=worker,phase=phase)
                    entries.append(ep['key']); episodes.append(ep)
        jobs.append(dict(key=f'worker{worker}',gpu=CONFIG['gpus'][worker],cases=keys,episodes=entries))
    # 21 people do not divide evenly over four GPUs. Spread the eight conditions
    # of the surplus 2a person evenly, avoiding a two-hour tail on one device.
    donor=max(range(len(jobs)),key=lambda i:len(jobs[i]['cases']))
    donor_keys=jobs[donor]['cases']
    spill=next(c['key'] for c in reversed(targets) if c['key'] in donor_keys and c['dataset']=='bcic_2a')
    spill_episodes=[e for e in episodes if e['case_key']==spill]
    for i,ep in enumerate(spill_episodes): ep['worker']=i%len(jobs)
    for worker,job in enumerate(jobs):
        orders={c['key']:c['order'] for c in targets}
        assigned=sorted((e for e in episodes if e['worker']==worker),key=lambda e:(e['phase'],orders[e['case_key']]))
        job['episodes']=[e['key'] for e in assigned]
        job['cases']=list(dict.fromkeys(e['case_key'] for e in assigned))
    if CONFIG['eval_steps'] != [0, 5, 10, 20, 50, 100]:
        raise RuntimeError('Unapproved evaluation schedule')
    return dict(schema='personal-lora-location-dose-v1', config=CONFIG, source_hashes=identity(),
        cases=targets, support_sets=support_sets, schedules=schedules,episodes=episodes,jobs=jobs,
        initializations=audit_initializations(),
        counts={n:sum(c['dataset']==n for c in targets) for n in CONFIG['datasets']},
        inventory=store.inventory, source_indexes=store.meta, source_array_stats=store.stats,
        original_splits_modified=False, offline_retraining=False,
        original_valid_touched=True, original_valid_policy='Only historically authorized personal users/memberships',
        query_used_for_selection=False, independent_test_estimate=False)


def nested_support(case,byid,total):
    if total==100:
        return case['support_ids'][:]
    classes=case['classes']; quotas=np.full(classes,total//classes,dtype=int)
    extra=np.random.default_rng(np.random.SeedSequence([CONFIG['seed'],911])).permutation(classes)[:total%classes]
    quotas[extra]+=1
    selected=[]
    for label,count in enumerate(quotas):
        choices=[key for key in case['support_ids'] if byid[key]['label']==label]
        if len(choices)<count:
            raise RuntimeError('Historical100 support cannot supply requested nested subset')
        selected.extend(choices[:count])
    if len(selected)!=total or len(set(selected))!=total:
        raise RuntimeError('Invalid nested support set')
    return selected


def main():
    plan = build(TrialStore())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise RuntimeError('Existing plan differs; refusing overwrite')
    write_json(path, plan)
    print(json.dumps(dict(targets=plan['counts'], episodes=len(plan['episodes']),eval_steps=CONFIG['eval_steps'],
        cases=[dict(key=c['key'], support=len(c['support_ids']), query=len(c['query_ids'])) for c in plan['cases']])), flush=True)


if __name__ == '__main__':
    main()
