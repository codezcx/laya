"""Two matched methods, fixed per-user support/query, explicit shared-data exposure."""
import hashlib
import json
from collections import Counter
import numpy as np
from .common import CONFIG, OUT, ROOT, identity, write_json
from .data import TrialStore, source_group, trial_key


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
    elif name == 'seed':
        pool = sorted([r for r in rows if r['split']=='train'],key=trial_key)
        query = sorted([r for r in rows if r['split']=='test'],key=trial_key)
        details['scope']='full_original_test_windows'
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
    elif name == 'Nakanishi2015':
        if len(rows)!=180 or len({source_group(r) for r in rows})!=180:
            raise ValueError('Expected180 unique SSVEP trials')
        pool,query = sorted(rows,key=trial_key),None
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


def build(store):
    cases = []
    prior = json.loads((ROOT / CONFIG['paired_reference'] / 'plan.json').read_text())
    old_cases = {c['subject']:c for c in prior['cases']}
    for tid,task in enumerate(store.tasks):
        name = task['task_id']
        for subject in CONFIG['cohorts'][name]['subjects']:
            rows = [r for r in store.rows[tid] if r['subject']==subject]
            support,query,details = split_person(rows,name,len(task['classes']))
            sids,qids = [r['id'] for r in support],[r['id'] for r in query]
            if len(sids)!=100 or len(set(sids))!=100 or set(sids)&set(qids):
                raise ValueError('Support/query identity violation')
            original = sorted({r['split'] for r in rows})
            seen = subject in store.train_subjects[tid]
            query_seen = sum(r['split']=='train' for r in query)
            strata = ('shared_subject_unseen' if not seen else
                      'shared_subject_seen_query_in_train_pool' if query_seen else
                      'shared_subject_seen_query_not_in_train_pool')
            key = f'bcic_2a_A{int(subject):02d}' if name=='bcic_2a' else f'{name}_s{subject}'
            reference = ROOT/CONFIG['historical_reference']/'episodes'/f'bcic_2a_s{subject}_residual_frozen_k5_seed42'
            case = dict(key=key,dataset=name,task=tid,subject=subject,original_split=original[0] if len(original)==1 else 'train_valid_test',
                original_splits=original,order=len(cases),classes=len(task['classes']),
                support_ids=sids,query_ids=qids,support_sha256=row_hash(sids),query_sha256=row_hash(qids),
                row_splits={r['id']:r['split'] for r in support+query},
                support_records=sorted({r['record'] for r in support}),query_records=sorted({r['record'] for r in query}),
                support_class_counts=dict(Counter(str(r['label']) for r in support)),
                query_class_counts=dict(Counter(str(r['label']) for r in query)),
                support_original_groups=len({source_group(r) for r in support}),
                query_original_groups=len({source_group(r) for r in query}),
                shared_train_seen_subject=seen,query_in_shared_train_pool=query_seen,exposure_stratum=strata,
                unit=CONFIG['cohorts'][name]['unit'],split_policy=CONFIG['cohorts'][name]['policy'],**details,
                historical_prediction=str(reference/'predictions.pt') if name=='bcic_2a' and (reference/'predictions.pt').exists() else None)
            if name=='bcic_2a' and subject in old_cases:
                old = old_cases[subject]
                for field in ('support_ids','query_ids','order'):
                    if case[field]!=old[field]:
                        raise RuntimeError(f'Prior four-user pairing changed: {field}')
            cases.append(case)
    return dict(schema='personal-multitask-exposure-v1',config=CONFIG,source_hashes=identity(),cases=cases,
        counts={n:sum(c['dataset']==n for c in cases) for n in CONFIG['datasets']},
        inventory=store.inventory,source_indexes=store.meta,source_array_stats=store.stats,
        original_splits_modified=False,query_used_for_selection=False,independent_test_estimate=False)


def main():
    plan=build(TrialStore())
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(plan)):
        raise RuntimeError('Existing plan differs; refusing overwrite')
    write_json(path,plan)
    print(json.dumps(dict(counts=plan['counts'],cases=[dict(key=c['key'],support=len(c['support_ids']),
        query=len(c['query_ids']),exposure=c['exposure_stratum']) for c in plan['cases']])),flush=True)


if __name__=='__main__':
    main()
