"""Fixed, shared validation IDs; train monitoring; test prohibited in pilot."""
import numpy as np


def require_split(split):
    if split not in {'train', 'valid'}:
        raise RuntimeError('Pilot does not permit test-set model evaluation')


def fixed_training_rows(store, tid, cap, seed=42):
    rows = [r for i, r in enumerate(store.rows[tid]) if r['split'] == 'train' and i not in store.excluded[tid]]
    rng = np.random.default_rng(np.random.SeedSequence([seed, tid, 733]))
    return [rows[i] for i in sorted(rng.choice(len(rows), min(cap, len(rows)), replace=False))]


def validation_rows(store, tid, cap=512):
    """Retain legacy label-independent 128; expand with balanced class/subject queues.

    The remainder is sampled, not score-selected. Accuracy is descriptive of
    this subset only; full validation remains the final comparator.
    """
    all_rows = store.evaluation(tid, 'valid')
    if cap is None or len(all_rows) <= cap:
        return all_rows
    if cap < 128:
        raise ValueError('Selection subset must include legacy 128')
    selected = {r['id'] for r in store.evaluation(tid, 'valid', 128)}
    rng = np.random.default_rng(np.random.SeedSequence([42, tid, 734]))
    queues = {}
    for row in all_rows:
        if row['id'] not in selected:
            queues.setdefault((row['label'], row['subject']), []).append(row['id'])
    for queue in queues.values():
        rng.shuffle(queue)
    labels = sorted({key[0] for key in queues})
    cursor = {label: 0 for label in labels}
    while len(selected) < cap:
        for label in labels:
            keys = sorted(k for k, q in queues.items() if k[0] == label and q)
            if keys:
                key = keys[cursor[label] % len(keys)]
                selected.add(queues[key].pop())
                cursor[label] += 1
            if len(selected) == cap:
                break
    return [r for r in all_rows if r['id'] in selected]
