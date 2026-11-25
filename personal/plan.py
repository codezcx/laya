"""Reconstruct recording provenance, then split only the authorized valid pool."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from jev_train.data import TrialStore
from jev_train.runtime import ROOT
from .common import CONFIG, OUT, identity, write_json


MATCHED_CONFIG_KEYS = ('parent', 'original_split', 'seed', 'steps', 'batch_size', 'microbatch',
    'eval_batch', 'icl_batch', 'encode_batch', 'learning_rate', 'weight_decay', 'clip_grad_norm',
    'adapter_width', 'prompt_tokens', 'prompt_init_text', 'connector_lora_rank',
    'connector_lora_scale', 'prefix_layers', 'prefix_tokens', 'kd_weight',
    'kd_temperature', 'teacher_variants', 'arms')


def paired_reference():
    if not CONFIG.get('paired_reference'):
        return {}
    root = ROOT / CONFIG['paired_reference']
    queue = json.loads((root / 'queue.json').read_text())
    if queue['state'] not in ('stopped_by_user', 'complete', 'budget_exhausted'):
        raise RuntimeError('Paired reference must be stopped and immutable')
    plan = json.loads((root / 'plan.json').read_text())
    for key in MATCHED_CONFIG_KEYS:
        if plan['config'][key] != CONFIG[key]:
            raise RuntimeError(f'Paired experiment silently changed {key}')
    if plan['config'].get('shots_per_class') != 5:
        raise RuntimeError('Expected previous 5-per-class reference')
    return {case['key']: case for case in plan['cases']}


def exact_total_support(pool, query, reference, total):
    if len(pool) != total:
        raise ValueError(f'Complete support pool has {len(pool)} trials, requires exactly {total}')
    ids = [r['id'] for r in pool]
    if len(set(ids)) != total or not set(reference['support_ids']).issubset(ids):
        raise RuntimeError('Expanded support must contain every previous support trial exactly once')
    if [r['id'] for r in query] != reference['query_ids']:
        raise RuntimeError('Paired query IDs/order changed')
    if set(ids) & set(reference['query_ids']):
        raise RuntimeError('Expanded support borrowed a query trial')
    return sorted(pool, key=lambda r: (r['label'], r['id']))


def row_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def provenance(dataset, subject, shard):
    """Return one record/trial/label triple per HDF5 row; never reprocess EEG."""
    with h5py.File(shard, 'r', locking=False) as f:
        labels = np.asarray(f['all/Y'][:], dtype=int)
    raw = ROOT / 'datasets/leaf13/raw'
    result = []
    if dataset == 'MI_BCIC_IV2a':
        import scipy.io
        # Author's exporter concatenates E before T, NOT T before E.
        for session in ('E', 'T'):
            value = scipy.io.loadmat(raw / f'MI_BCIC_IV2a_mat/{subject}{session}.mat')['data'][0]
            for run in range(3, len(value)):
                yy = value[run][0, 0][2][:, 0].astype(int) - 1
                result.extend(dict(record=session, trial=f'{session}:{run}:{i}', label=int(y))
                              for i, y in enumerate(yy))
    elif dataset == 'MI_PhysioNet':
        import mne
        for run in range(3, 15):
            path = raw / f'MI_PhysioNet/{subject}/{subject}R{run:02d}.edf'
            recording = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
            end = int(round(recording.n_times * 200 / recording.info['sfreq']))
            for event, annotation in enumerate(recording.annotations):
                desc = str(annotation['description'])
                if desc not in ('T1', 'T2'):
                    continue
                start = int(round((float(annotation['onset']) - recording.first_time) * 200))
                if start < 0 or start + 800 >= end:
                    continue
                result.append(dict(record=f'R{run:02d}', trial=f'R{run:02d}:{event}',
                                   label=int(desc[-1]) - 1))
            recording.close()
    elif dataset == 'MI_OpenBMI':
        # Pinned author's proc_one concatenates s1 train/test, then s2 train/test,
        # each block contains exactly 100 trials in these prepared 400-row shards.
        if len(labels) != 400:
            raise ValueError('Unexpected OpenBMI export length; cannot infer sessions')
        result = [dict(record=f'session{i // 200 + 1}', trial=str(i), label=int(y))
                  for i, y in enumerate(labels)]
    elif dataset == 'SSVEP_OpenBMI':
        sources = json.loads(Path(shard).with_suffix('.sources.json').read_text())
        mapping = {'12.0': 0, '12': 0, '8.57': 1, '8.6': 1, '6.67': 2, '6.6': 2, '5.45': 3, '5.4': 3}
        result = [dict(record='session' + str(s['session']),
                       trial=f"{s['session']}:{s['run']}:{s['trial']}", label=mapping[str(s['label'])])
                  for s in sources]
    else:
        raise ValueError('Dataset lacks an audited independent-trial protocol')
    if [r['label'] for r in result] != labels.tolist():
        raise ValueError(f'{dataset}/{subject}: reconstructed raw labels/order differ from HDF5')
    if len({r['trial'] for r in result}) != len(result):
        raise ValueError('Multiple windows from one trial require group-level splitting')
    return result


def partitions(rows, dataset, provenance_rows):
    support, query = [], []
    for row in rows:
        p = provenance_rows[row['index']]
        record = p['record']
        if dataset == 'MI_BCIC_IV2a':
            is_support = record == 'T'
        elif dataset == 'MI_PhysioNet':
            is_support = int(record[1:]) <= 10
        elif dataset == 'MI_OpenBMI':
            is_support = record == 'session1'
        else:
            is_support = record == sorted({p['record'] for p in provenance_rows})[0]
        target = support if is_support else query
        target.append(dict(row, record=record, trial=p['trial']))
    if {r['record'] for r in support} & {r['record'] for r in query}:
        raise ValueError('Support and query recordings overlap')
    return support, query


def select_support(pool, classes, shots, seed):
    selected = []
    for label in range(classes):
        rows = sorted((r for r in pool if r['label'] == label), key=lambda r: r['id'])
        if len(rows) < shots:
            raise ValueError(f'Class {label} has {len(rows)} support trials, needs {shots}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label, 711]))
        selected.extend(rows[int(i)] for i in rng.permutation(len(rows))[:shots])
    return selected


def build(store):
    groups, skipped = defaultdict(list), []
    references = paired_reference()
    for tid, task in enumerate(store.tasks):
        name = task['task_id']
        if name not in CONFIG['datasets']:
            continue
        by_subject = defaultdict(list)
        for row in store.rows[tid]:
            if row['split'] == CONFIG['original_split']:
                by_subject[row['subject']].append(row)
        for subject, rows in sorted(by_subject.items()):
            key = f'{name}_{subject}'
            reference = references.get(key)
            if CONFIG.get('paired_reference') and reference is None:
                raise RuntimeError(f'Missing paired reference for {key}')
            try:
                origin = provenance(name, subject, rows[0]['path'])
                pool, query = partitions(rows, name, origin)
                classes = len(task['classes'])
                if {r['label'] for r in query} != set(range(classes)):
                    raise ValueError('Query partition lacks at least one class')
                if CONFIG.get('support_total'):
                    support = exact_total_support(pool, query, reference, CONFIG['support_total'])
                else:
                    support = select_support(pool, classes, CONFIG['shots_per_class'], CONFIG['seed'])
            except (ValueError, FileNotFoundError) as exc:
                skipped.append(dict(dataset=name, subject=subject, reason=str(exc)))
                continue
            sids, qids = [r['id'] for r in support], [r['id'] for r in query]
            train_subjects = {r['subject'] for r in store.rows[tid] if r['split'] == 'train'}
            case = dict(key=key, dataset=name, task=tid, subject=subject,
                original_split='valid', support_ids=sids, query_ids=qids,
                support_sha256=row_hash(sids), query_sha256=row_hash(qids),
                support_records=sorted({r['record'] for r in support}),
                query_records=sorted({r['record'] for r in query}),
                shared_train_seen_subject=subject in train_subjects, classes=classes,
                support_provenance=[{k: r[k] for k in ('id', 'record', 'trial')} for r in support],
                query_provenance=[{k: r[k] for k in ('id', 'record', 'trial')} for r in query],
                support_class_counts={str(k): sum(r['label'] == k for r in support) for k in range(classes)})
            if reference is not None:
                # Filtering to BCIC must NOT renumber user-specific RNG seeds.
                case['order'] = reference['order']
                case['reference'] = dict(root=CONFIG['paired_reference'],
                    support_ids=reference['support_ids'], query_sha256=reference['query_sha256'],
                    old_support_n=len(reference['support_ids']))
                for arm in CONFIG['arms']:
                    result = ROOT / CONFIG['paired_reference'] / 'cases' / key / arm / 'result.json'
                    if not result.exists() or json.loads(result.read_text())['case'] != reference:
                        raise RuntimeError(f'Missing/mismatched completed reference: {key}/{arm}')
            groups[name].append(case)
    # Each GPU receives all tasks rather than one permanently confounded dataset.
    cases = []
    for i in range(max(map(len, groups.values()), default=0)):
        for name in CONFIG['datasets']:
            if i < len(groups[name]):
                cases.append(groups[name][i])
    for index, case in enumerate(cases):
        case.setdefault('order', index)
    if not cases:
        raise RuntimeError('No eligible personal cases')
    return dict(schema='personal-valid-v1', config=CONFIG, source_hashes=identity(),
        cases=cases, skipped=skipped, query_used_for_selection=False,
        original_splits_modified=False, independent_test_estimate=False,
        counts={n: len(groups[n]) for n in CONFIG['datasets']},
        source_indexes={store.tasks[t]['task_id']: m['index_sha256'] for t, m in store.meta.items()})


def main():
    result = build(TrialStore())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError('Existing personal plan differs; refusing silent overwrite')
    write_json(path, result)
    print(json.dumps(dict(counts=result['counts'], skipped=result['skipped'], cases=len(result['cases']))), flush=True)


if __name__ == '__main__':
    main()
