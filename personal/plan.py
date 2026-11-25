"""Exact historical users and batches, plus pinned common-seed initializations."""
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
    previous = json.loads((ROOT / CONFIG['initialization_reference'] / 'plan.json').read_text())
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
            if case['original_split'] != 'test' or case['shared_train_seen_subject']:
                raise RuntimeError('Targets must remain the same original-test users')
            check_rows(support, case, gradient=True)
            check_rows(query, case)
            targets.append(case)
    if {j['key'] for j in CONFIG['jobs']} != {c['key'] for c in targets}:
        raise RuntimeError('GPU jobs do not match the planned users')
    for path, stat in store.stats.items():
        if list(stat) != previous['source_array_stats'][path]:
            raise RuntimeError('Historical target EEG array changed')
    schedules = {c['key']: personal_batches(c) for c in targets}
    if CONFIG['eval_steps'] != [0, 5, 10, 20, 50, 100]:
        raise RuntimeError('Unapproved evaluation schedule')
    return dict(schema='personal-matched-curves-v1', config=CONFIG, source_hashes=identity(),
        cases=targets, schedules=schedules, initializations=audit_initializations(),
        counts={n:sum(c['dataset']==n for c in targets) for n in CONFIG['datasets']},
        inventory=store.inventory, source_indexes=store.meta, source_array_stats=store.stats,
        original_splits_modified=False, offline_retraining=False,
        original_valid_touched=False, query_used_for_selection=False, independent_test_estimate=False)


def main():
    plan = build(TrialStore())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise RuntimeError('Existing plan differs; refusing overwrite')
    write_json(path, plan)
    print(json.dumps(dict(targets=plan['counts'], eval_steps=CONFIG['eval_steps'],
        cases=[dict(key=c['key'], support=len(c['support_ids']), query=len(c['query_ids'])) for c in plan['cases']])), flush=True)


if __name__ == '__main__':
    main()
