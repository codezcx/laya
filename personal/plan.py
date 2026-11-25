"""100 T-session trials per held-out user, all 288 E-session trials query-only."""
import hashlib
import json
import numpy as np
from .common import CONFIG, OUT, ROOT, identity, write_json
from .data import TrialStore, trial_key


def row_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def partitions(rows):
    support, query = [], []
    for row in rows:
        if not row['record'].endswith(('T', 'E')):
            raise ValueError('Unknown session')
        (support if row['record'].endswith('T') else query).append(row)
    for part in (support, query):
        if len(part) != 288 or len({trial_key(r) for r in part}) != 288:
            raise ValueError('Need 288 unique trials per session')
        if [sum(r['label'] == c for r in part) for c in range(4)] != [72] * 4:
            raise ValueError('Session must contain 72 trials per class')
    return sorted(support, key=trial_key), sorted(query, key=trial_key)


def select_support(pool, classes, shots, seed):
    selected = []
    for label in range(classes):
        rows = sorted((r for r in pool if r['label'] == label), key=trial_key)
        if len(rows) < shots:
            raise ValueError(f'Class {label} needs {shots} trials')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label, 711]))
        selected.extend(rows[int(i)] for i in rng.permutation(len(rows))[:shots])
    return selected


def build(store):
    cases = []
    for order, (subject, split) in enumerate(CONFIG['subjects'].items()):
        pool, query = partitions([r for r in store.rows[0] if r['subject'] == subject])
        support = select_support(pool, 4, CONFIG['shots_per_class'], CONFIG['seed'])
        if len(support) != CONFIG['support_total']:
            raise RuntimeError('Support count does not match authorization')
        sids, qids = [r['id'] for r in support], [r['id'] for r in query]
        if set(sids) & set(qids):
            raise RuntimeError('Support/query overlap')
        reference = ROOT / CONFIG['historical_reference'] / 'episodes' / f'bcic_2a_s{subject}_residual_frozen_k5_seed42'
        if (reference / 'complete.json').exists():
            old = json.loads((reference / 'complete.json').read_text())
            if old['case']['query_ids'] != qids or not set(old['case']['support_ids']).issubset(sids):
                raise RuntimeError('Historical query/nested support changed')
        cases.append(dict(key=f'bcic_2a_A{int(subject):02d}', dataset='bcic_2a', task=0,
            subject=subject, original_split=split, order=order, classes=4,
            support_ids=sids, query_ids=qids, support_sha256=row_hash(sids), query_sha256=row_hash(qids),
            support_records=[f'A{int(subject):02d}T'], query_records=[f'A{int(subject):02d}E'],
            support_class_counts={str(c):CONFIG['shots_per_class'] for c in range(4)}, shared_train_seen_subject=False,
            historical_prediction=str(reference / 'predictions.pt') if (reference/'predictions.pt').exists() else None))
    return dict(schema='personal-heldout-session-v1', config=CONFIG, source_hashes=identity(), cases=cases,
        skipped=[], counts={'bcic_2a':len(cases)}, source_indexes=store.meta, source_array_stats=store.stats,
        query_used_for_selection=False, original_splits_modified=False, independent_test_estimate=False)


def main():
    result = build(TrialStore())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(result)):
        raise RuntimeError('Existing personal plan differs; refusing overwrite')
    write_json(path, result)
    print(json.dumps(dict(counts=result['counts'], cases=[dict(subject=c['subject'], split=c['original_split'],
        support=len(c['support_ids']), query=len(c['query_ids'])) for c in result['cases']])), flush=True)


if __name__ == '__main__':
    main()
