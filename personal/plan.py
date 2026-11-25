"""Audited source-train episodes and historically matched target-user partitions."""
import hashlib
import json
from collections import Counter
import numpy as np
from .common import CONFIG, OUT, ROOT, identity, write_json, check_rows, check_source_rows
from .data import TrialStore, source_group, trial_key
from .meta import episode_schedule

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
    previous = json.loads((ROOT / CONFIG['paired_reference'] / 'plan.json').read_text())
    old = {(c['dataset'], c['subject']): c for c in previous['cases']}
    targets, sources, schedules = [], [], {}
    for tid, task in enumerate(store.tasks):
        name, classes = task['task_id'], len(task['classes'])
        cohort = CONFIG['cohorts'][name]
        if set(cohort['subjects']) & set(cohort['source_subjects']):
            raise RuntimeError('Source and target subjects overlap')
        for subject in cohort['subjects']:
            rows = [r for r in store.rows[tid] if r['subject'] == subject]
            support, query, _ = split_person(rows, name, classes)
            prior = old[name, subject]
            for field, selected in [('support_ids', support), ('query_ids', query)]:
                if [r['id'] for r in selected] != prior[field]:
                    raise RuntimeError('Historical target pairing changed')
            case = dict(prior, task=tid)
            if case['original_split'] != 'test' or case['shared_train_seen_subject']:
                raise RuntimeError('Pilot targets must be original-test, shared-train-unseen users')
            check_rows(support, case, gradient=True)
            check_rows(query, case)
            targets.append(case)
        for subject in cohort['source_subjects']:
            rows = [r for r in store.rows[tid] if r['subject'] == subject]
            if not rows or {r['split'] for r in rows} != {'train'}:
                raise RuntimeError('Source subject must belong exclusively to original train')
            support, query_pool, details = split_person(rows, name, classes)
            query = select_support(query_pool, classes, CONFIG['offline']['query_total'], CONFIG['seed'] + 1)
            case = dict(key=f'{name}_source_s{subject}', dataset=name, task=tid, subject=subject,
                classes=classes, support_ids=[r['id'] for r in support], query_ids=[r['id'] for r in query],
                original_split='train', support_records=sorted({r['record'] for r in support}),
                query_records=sorted({r['record'] for r in query}),
                support_class_counts=dict(Counter(str(r['label']) for r in support)),
                query_class_counts=dict(Counter(str(r['label']) for r in query)), **details)
            for part, values in [('support', support), ('query', query)]:
                check_source_rows(values, case, part)
                case[part + '_sha256'] = row_hash(case[part + '_ids'])
            if {source_group(r) for r in support} & {source_group(r) for r in query}:
                raise RuntimeError('Source meta-support/query original group overlap')
            sources.append(case)
        schedules[name] = episode_schedule([s for s in sources if s['dataset'] == name], CONFIG)
    return dict(schema='personal-fomaml-init-v1', config=CONFIG, source_hashes=identity(),
        cases=targets, sources=sources, schedules=schedules,
        counts={n:sum(c['dataset']==n for c in targets) for n in CONFIG['datasets']},
        source_counts={n:sum(c['dataset']==n for c in sources) for n in CONFIG['datasets']},
        inventory=store.inventory, source_indexes=store.meta, source_array_stats=store.stats,
        original_splits_modified=False, offline_gradients_original_train_only=True,
        original_valid_touched=False, query_used_for_selection=False, independent_test_estimate=False)


def main():
    plan = build(TrialStore())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise RuntimeError('Existing plan differs; refusing overwrite')
    write_json(path, plan)
    print(json.dumps(dict(targets=plan['counts'], sources=plan['source_counts'],
        cases=[dict(key=c['key'], support=len(c['support_ids']), query=len(c['query_ids'])) for c in plan['cases']])), flush=True)


if __name__ == '__main__':
    main()
