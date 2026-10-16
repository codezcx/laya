"""Fixed source-only partitions and matched frozen EEG representations."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.feature_selection import SelectKBest, mutual_info_classif

ROOT = Path('artifacts/multitask_bridge_v30/source_cache')
DATASETS = ('Cho2017', 'Lee2019_MI', 'Dreyer2023')
BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    """Atomic replacement of this experiment's own status/results files."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def read_rows():
    rows = [json.loads(line) for line in (ROOT / 'rows.jsonl').read_text().splitlines()]
    audit = json.loads((ROOT / 'verified.json').read_text())
    if digest(ROOT / 'rows.jsonl') != audit['manifest_sha256']:
        raise ValueError('Source manifest changed')
    return rows


def enrich_rows(rows, dataset):
    selected = []
    for index, row in enumerate(rows):
        if row['dataset'] != dataset:
            continue
        r = dict(row, source_index=index)
        match = re.search(r'_run-(.*?)_trial-(\d+)_event-(\d+)_', r['path'])
        if match is None:
            raise ValueError(f'Unrecognized trial metadata: {r["path"]}')
        r.update(run=match[1], trial_number=int(match[2]), event_number=int(match[3]))
        if r['class_id'] != r['label'] + 3 or r['label'] not in (0, 1):
            raise ValueError('Not binary left/right MI')
        selected.append(r)
    if not selected:
        raise ValueError(f'No rows for {dataset}')
    for key in ('path', 'signal_sha256'):
        if len({r[key] for r in selected}) != len(selected):
            raise ValueError(f'Duplicate {key} in source rows')
    return selected


def check_protocol(rows, protocol, csp_fit):
    parts = [protocol[k] for k in ('train', 'validation', 'test')]
    for part in parts:
        if not part or len(part) != len(set(part)):
            raise ValueError('Empty or duplicated partition')
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            for key in ('path', 'signal_sha256'):
                if {rows[j][key] for j in a} & {rows[j][key] for j in b}:
                    raise ValueError(f'Trial leakage: {key}')
    subjects = [{rows[i]['subject'] for i in part} for part in parts]
    if protocol['target'] == 'task':
        if any(subjects[i] & subjects[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Task probe must be subject-disjoint')
        if any({rows[i]['label'] for i in part} != {0, 1} for part in parts):
            raise ValueError('Missing task class')
    else:
        if not subjects[0] == subjects[1] == subjects[2]:
            raise ValueError('Identity probe must be closed-set')
        if len(subjects[0]) < 2:
            raise ValueError('Identity probe needs multiple people')
        if subjects[0] & {rows[i]['subject'] for i in csp_fit}:
            raise ValueError('CSP saw identity-probe participants')
        if protocol['domain'] == 'cross_session':
            keys = lambda part: {(rows[i]['subject'], rows[i]['session']) for i in part}
            if (keys(parts[0]) | keys(parts[1])) & keys(parts[2]):
                raise ValueError('Session leakage')
        if protocol['domain'] == 'cross_run':
            keys = lambda part: {(rows[i]['subject'], rows[i]['session'], rows[i]['run']) for i in part}
            if any(keys(parts[i]) & keys(parts[j]) for i in range(3) for j in range(i + 1, 3)):
                raise ValueError('Run leakage')
    if any(set(csp_fit) & set(p) for p in parts[1:]):
        raise ValueError('CSP fit overlaps validation/test')


def build_protocols(rows, seed=42):
    """Holdout subjects unchanged; use source-train people for inner selection."""
    dataset = rows[0]['dataset']
    train_subjects = sorted({r['subject'] for r in rows if r['split'] == 'train'}, key=int)
    heldout = sorted({r['subject'] for r in rows if r['split'] == 'validation'}, key=int)
    if set(train_subjects) & set(heldout) or len(train_subjects) < 4:
        raise ValueError('Invalid source split')
    shuffled = np.random.default_rng(seed).permutation(train_subjects).tolist()
    val_subjects = set(shuffled[:max(2, int(np.ceil(.2 * len(shuffled))))])
    fit_subjects = set(train_subjects) - val_subjects
    task = dict(name='task_subject_holdout', target='task', domain='cross_subject',
                train=[i for i, r in enumerate(rows) if r['subject'] in fit_subjects],
                validation=[i for i, r in enumerate(rows) if r['subject'] in val_subjects],
                test=[i for i, r in enumerate(rows) if r['subject'] in heldout])
    identity = dict(name='identity', target='subject', train=[], validation=[], test=[])
    for subject in heldout:
        ids = [i for i, r in enumerate(rows) if r['subject'] == subject]
        if dataset == 'Lee2019_MI':
            sessions = sorted({rows[i]['session'] for i in ids}, key=int)
            if len(sessions) != 2:
                raise ValueError('Lee requires exactly two sessions')
            earlier = sorted([i for i in ids if rows[i]['session'] == sessions[0]],
                             key=lambda i: rows[i]['event_number'])
            cut = int(.7 * len(earlier))
            identity['train'] += earlier[:cut - 3]
            identity['validation'] += earlier[cut + 3:]
            identity['test'] += [i for i in ids if rows[i]['session'] == sessions[1]]
            identity['domain'] = 'cross_session'
        elif dataset == 'Dreyer2023':
            runs = sorted({rows[i]['run'] for i in ids})
            if len(runs) != 6:
                raise ValueError('Dreyer identity panel requires six runs')
            for key, keep in [('train', runs[:3]), ('validation', runs[3:4]), ('test', runs[4:])]:
                identity[key] += [i for i in ids if rows[i]['run'] in keep]
            identity['domain'] = 'cross_run'
        elif dataset == 'Cho2017':
            # Export indices are grouped by label, NOT authenticated timestamps.
            # This is explicitly a weak within-record diagnostic, never temporal DG.
            for label in (0, 1):
                ordered = sorted([i for i in ids if rows[i]['label'] == label],
                                 key=lambda i: rows[i]['trial_number'])
                first, second = int(.6 * len(ordered)), int(.8 * len(ordered))
                identity['train'] += ordered[:first - 2]
                identity['validation'] += ordered[first + 2:second - 2]
                identity['test'] += ordered[second + 2:]
            identity['domain'] = 'within_record_index_blocks_only'
        else:
            raise ValueError(dataset)
    identity['name'] = 'identity_' + identity['domain']
    protocols = [task, identity]
    for label, name in [(0, 'left_to_right'), (1, 'right_to_left')]:
        cross = {k: v for k, v in identity.items() if k not in ('train', 'validation', 'test')}
        cross['name'] = 'identity_' + name
        cross['train'] = [i for i in identity['train'] if rows[i]['label'] == label]
        cross['validation'] = [i for i in identity['validation'] if rows[i]['label'] == label]
        cross['test'] = [i for i in identity['test'] if rows[i]['label'] != label]
        protocols.append(cross)
    for protocol in protocols:
        check_protocol(rows, protocol, task['train'])
    return protocols


def verify_source_cache():
    manifest_hash = digest(ROOT / 'rows.jsonl')
    verified = []
    for shard in range(4):
        folder = ROOT / str(shard)
        audit = json.loads((folder / 'complete.json').read_text())
        if audit['manifest_sha256'] != manifest_hash:
            raise ValueError('Shard manifest mismatch')
        for name, expected in audit['hashes'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Cache checksum failed: {shard}/{name}')
        verified.append(dict(shard=shard, audit_sha256=digest(folder / 'complete.json')))
        print(f'Verified source shard {shard}', flush=True)
    return dict(manifest_sha256=manifest_hash, shards=verified)


def load_reve(rows):
    wanted = {r['source_index']: j for j, r in enumerate(rows)}
    x = np.empty((len(rows), 12 * 512), dtype=np.float32)
    seen = set()
    for shard in range(4):
        folder = ROOT / str(shard)
        ids = np.load(folder / 'indices.npy')
        lengths = np.load(folder / 'lengths.npy')
        tokens = np.load(folder / 'tokens.npy', mmap_mode='r')
        for j, source_id in enumerate(ids):
            source_id = int(source_id)
            if source_id in wanted:
                if source_id in seen or lengths[j] != 12:
                    raise ValueError('Duplicate or invalid MI token length')
                value = tokens[j, :12].astype(np.float32).reshape(-1)
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite REVE features')
                x[wanted[source_id]] = value
                seen.add(source_id)
    if seen != set(wanted):
        raise ValueError('Incomplete feature coverage')
    return x


def load_signals(rows):
    # Reuse exactly the implementation used to generate frozen tokens, including
    # its trial-level signal hash check, filtering, crop and common amplitude scale.
    from eeg_align.cache import signals_for_file
    cfg = json.loads(Path('configs/data.json').read_text())
    def one(row):
        signals, channels = signals_for_file([row], cfg)
        if channels != ['C3', 'Cz', 'C4'] or signals.shape != (1, 3, 600):
            raise ValueError('Unexpected matched input')
        return signals[0]
    x = np.empty((len(rows), 3, 600), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, value in enumerate(pool.map(one, rows)):
            x[i] = value
            if (i + 1) % 1000 == 0:
                print(f'Matched raw signals {i + 1}/{len(rows)}', flush=True)
    return x


def fit_fbcsp(signals, labels, fit_ids):
    """One shared transform fitted only on the declared representation-fit set."""
    import mne
    from mne.decoding import CSP
    mne.set_log_level('ERROR')
    csps, values = [], []
    fit_ids = np.asarray(fit_ids, dtype=int)
    for band in BANDS:
        filtered = sosfiltfilt(butter(4, band, btype='bandpass', fs=200, output='sos'),
                              signals.astype(np.float64), axis=-1)
        csp = CSP(n_components=2, reg='oas', log=True, norm_trace=False, cov_est='epoch')
        csp.fit(filtered[fit_ids], labels[fit_ids])
        values.append(csp.transform(filtered))
        csps.append(csp)
        print(f'Fitted shared CSP {band}', flush=True)
    full = np.concatenate(values, axis=1).astype(np.float32)
    selector = SelectKBest(lambda x, y: mutual_info_classif(x, y, random_state=42), k=6)
    selector.fit(full[fit_ids], labels[fit_ids])
    # Avoid serializing a lambda: retain selected indices and MI scores explicitly.
    selected = selector.get_support(indices=True)
    fitted = dict(csps=csps, bands=BANDS, fit_indices=fit_ids,
                  selected_indices=selected, selection_scores=selector.scores_)
    return {'fbcsp14': full, 'fbcsp6': full[:, selected]}, fitted


def prepare_dataset(dataset, out):
    out = Path(out)
    rows = enrich_rows(read_rows(), dataset)
    protocols = build_protocols(rows)
    save_json(out / 'rows.json', rows)
    save_json(out / 'protocols.json', protocols)
    save_json(out / 'status.json', dict(phase='loading_matched_signals'))
    signals = load_signals(rows)
    save_json(out / 'status.json', dict(phase='fitting_shared_fbcsp'))
    features, fitted = fit_fbcsp(signals, np.array([r['label'] for r in rows]), protocols[0]['train'])
    joblib.dump(fitted, out / 'fbcsp.joblib')
    del signals
    save_json(out / 'status.json', dict(phase='loading_frozen_reve'))
    features['reve6144'] = load_reve(rows)
    for name, x in features.items():
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite features')
        np.save(out / f'{name}.npy', x)
    save_json(out / 'features.json', dict(
        shapes={k: list(v.shape) for k, v in features.items()},
        hashes={f'{k}.npy': digest(out / f'{k}.npy') for k in features},
        fbcsp_sha256=digest(out / 'fbcsp.joblib'),
        manifest_sha256=digest(ROOT / 'rows.jsonl'),
        representation_fit_subjects=sorted({rows[i]['subject'] for i in protocols[0]['train']}, key=int),
        frozen_encoder='REVE pretrained base; no SFT weights; pretraining overlap not excluded',
        preprocessing='Existing 200 Hz, 1-30 Hz, first 600 samples of task epoch, C3/Cz/C4, common_scale',
        limitation='No absolute-amplitude information removed by common_scale can be recovered.'))
    return rows, protocols, features
