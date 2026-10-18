"""Prepare held-out 2b pairs using the source protocol; never fit or select a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from eeg_align.cache import digest
from eeg_tasks.attribute_questions import paired_normalize


def cue_and_bounds(trial_matlab, fs=250):
    if fs != 250:
        raise ValueError('Expected native 250 Hz 2b signal')
    # Official feedback protocol: cue is 3 seconds after trial start.
    cue = int(trial_matlab) - 1 + 3 * fs
    center = int(round(cue * 200 / fs))
    return cue, [(center - 500, center - 100), (center + 100, center + 500)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/MNE-bnci-data/~bci/database/004-2014'))
    p.add_argument('--manifest', type=Path, default=Path('artifacts/multitask_prototypes_v1/external_2b'))
    p.add_argument('--output', type=Path, default=Path('artifacts/jev_paired_qa/external_2b'))
    args = p.parse_args()
    paths = [args.manifest / f'rows_{i}.jsonl' for i in range(4)]
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(rows) == 2840 and len({r['path'] for r in rows}) == 2840
    assert all(r['split'] == 'external_2b' and r['dataset'] == 'BNCI2014_004' for r in rows)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[int(r['subject'])].append((i, r))
    assert set(groups) == set(range(1, 10))
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = np.lib.format.open_memmap(args.output/'pairs.npy', mode='w+', dtype='float32', shape=(2840, 2, 3, 400))
    labels = np.empty(2840, dtype=np.int64)
    sources = [None] * 2840
    raw_hashes = {}
    seen = set()
    for subject, members in sorted(groups.items()):
        raw_path = args.raw / f'B{subject:02d}E.mat'
        raw_hashes[str(raw_path)] = digest(raw_path)
        sessions = np.atleast_1d(loadmat(raw_path, squeeze_me=True, struct_as_record=False)['data'])
        assert len(sessions) == 2
        signals = []
        for raw in sessions:
            assert int(raw.fs) == 250 and raw.X.shape[1] == 6
            eeg = np.asarray(raw.X[:, :3], dtype=np.float64).T  # EOG excluded.
            assert np.isfinite(eeg).all()
            filtered = sosfiltfilt(butter(4, [1, 30], btype='bandpass', fs=250, output='sos'), eeg, axis=-1)
            signals.append(resample_poly(filtered, 4, 5, axis=-1))
        for index, row in members:
            with open(row['path'], 'rb') as f:
                meta = pickle.load(f)
            session = {'3test': 0, '4test': 1}[row['session']]
            ordinal = int(meta['event_index'])
            key = (subject, session, ordinal)
            assert key not in seen
            seen.add(key)
            raw = sessions[session]
            cue, bounds = cue_and_bounds(raw.trial[ordinal], int(raw.fs))
            label = int(raw.y[ordinal]) - 1
            assert label in (0, 1) and label == row['label']
            assert meta['label'] == ('left_hand' if label == 0 else 'right_hand')
            assert int(meta['subject']) == subject and meta['session'] == row['session']
            assert abs(float(meta['annotation_onset']) - cue/250) < 1e-9
            assert bounds[0][0] >= 0 and bounds[1][1] <= signals[session].shape[-1]
            pair, scale = paired_normalize(*(signals[session][:, a:b] for a, b in bounds))
            assert np.isfinite(pair).all() and np.max(np.abs(pair.mean(-1))) < 1e-5
            assert abs(float(np.mean(pair.astype(np.float64)**2)) - 1) < 1e-6
            pairs[index] = pair
            labels[index] = label
            sources[index] = dict(index=index, subject=subject, session=row['session'], event_index=ordinal,
                label=label, original_path=row['path'], raw_file=str(raw_path), cue_sample_250hz=cue,
                bounds_200hz=bounds, shared_scale=scale,
                cue_rounding_seconds=(bounds[0][0]+500)/200-cue/250)
        assert len(members) == sum(len(np.atleast_1d(s.trial)) for s in sessions)
        print('Prepared external subject', subject, len(members), flush=True)
    pairs.flush()
    del pairs
    np.save(args.output/'labels.npy', labels)
    (args.output/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources))
    max_rounding = max(abs(r['cue_rounding_seconds']) for r in sources)
    assert max_rounding <= .0025 + 1e-9
    report = dict(n=2840, channels=['C3', 'Cz', 'C4'], sampling_rate=200, shape=[2840, 2, 3, 400],
        status='prepared; not encoded or evaluated', labels=np.bincount(labels, minlength=2).tolist(),
        preprocessing='Native bipolar channels; continuous 1-30 Hz fourth-order zero-phase Butterworth at 250 Hz; resample 4/5; per-segment/channel demean; shared pair RMS; no clipping',
        windows_seconds=[[-2.5, -.5], [.5, 2.5]], max_cue_rounding_seconds=max_rounding,
        limitation='Reference contains warning beep at cue -1s; preparatory reference, not certified rest. Bipolar channels differ from source montage. Target not used for training or selection.',
        source_protocol='https://www.bbci.de/competition/iv/desc_2b.pdf',
        manifests={str(f):digest(f) for f in paths}, raw_sha256=raw_hashes,
        files={n:digest(args.output/n) for n in ['pairs.npy','labels.npy','sources.jsonl']},
        code_sha256=digest(__file__), normalization_sha256=digest('eeg_tasks/attribute_questions.py'))
    (args.output/'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['n','labels','max_cue_rounding_seconds','status']}), flush=True)


if __name__ == '__main__':
    main()
