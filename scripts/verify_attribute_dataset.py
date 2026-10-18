"""Audit coverage, split isolation, copied signals, and independently recompute QA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.build_attribute_dataset import digest, write_json


def independent_power(x, side, channels):
    """Direct FFT Welch computation, independent of the generator's scipy.welch."""
    sl = {'full': slice(None), 'early': slice(0, 300), 'late': slice(300, 600)}[side['window']]
    x = np.asarray(x[[channels.index(c) for c in side['channels']], sl], dtype=np.float64)
    spectra = []
    for start in range(0, x.shape[-1] - 199, 100):
        segment = x[:, start:start+200]
        segment = (segment-segment.mean(-1, keepdims=True)) * np.hanning(201)[:-1]
        p = np.abs(np.fft.rfft(segment, axis=-1)) ** 2
        p[:, 1:-1] *= 2
        spectra.append(p)
    p = np.mean(spectra, axis=0)
    band = {'alpha': slice(8, 13), 'beta': slice(13, 31)}[side['band']]
    return float(np.mean(p[:, band].sum(-1) / p[:, 1:31].sum(-1)))


def verify(root):
    complete = json.loads((root/'complete.json').read_text())
    assert not (root/'BUILDING.json').exists()
    for name, expected in complete['files'].items():
        assert digest(root/name) == expected, name
    manifest = Path(complete['source_manifest'])
    assert digest(manifest) == complete['source_manifest_sha256']
    original = [json.loads(s) for s in manifest.read_text().splitlines()]
    raw = np.load(complete['source_signals'], mmap_mode='r')
    arrays, sources = {}, {}
    for task in ['mi', 'emotion']:
        arrays[task] = np.load(root/f'signals/{task}.npy', mmap_mode='r')
        sources[task] = [json.loads(s) for s in (root/f'{task}_sources.jsonl').read_text().splitlines()]
        for i, source in enumerate(sources[task]):
            r = source['source_metadata']
            idx = source['source_row_index']
            assert r == original[idx] and r['split'] == 'train'
            assert source['signal_index'] == i
            signal = arrays[task][i]
            assert hashlib.sha256(signal.tobytes()).hexdigest() == source['signal_sha256']
            np.testing.assert_array_equal(signal, raw[idx, :signal.shape[0]])
    actual = [s['source_row_index'] for group in sources.values() for s in group]
    expected = [i for i, r in enumerate(original) if r['split'] == 'train']
    assert len(actual) == len(set(actual)) and sorted(actual) == expected
    stats = json.loads((root/'statistics.json').read_text())
    report = {}
    for spec in json.loads((root/'question_catalog.json').read_text()):
        task = spec['task']
        n = len(sources[task])
        selected = set(np.random.default_rng(42).choice(n, size=min(24, n), replace=False).tolist())
        counts, seen, by_dataset = Counter(), set(), {}
        with (root/'questions'/f'{spec["id"]}.jsonl').open() as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                src = sources[task][i]
                assert item['sample_id'] == src['sample_id'] and item['split'] == 'train'
                assert item['id'] not in seen
                seen.add(item['id'])
                inp = item['input']
                assert inp['signal_index'] == i and inp['signal_file'] == src['signal_file']
                assert inp['question'] == spec['question'] and set(inp['options']) == {'A', 'B', 'C'}
                ev = item['evidence']
                db = 10*np.log10(max(ev['first_relative_power'], 1e-12)/max(ev['second_relative_power'], 1e-12))
                np.testing.assert_allclose(db, ev['difference_db'], atol=1e-12)
                answer = 'A' if db > 1 else 'C' if db < -1 else 'B'
                assert item['target']['answer'] == answer
                counts[answer] += 1
                ds = src['source_metadata']['dataset']
                by_dataset.setdefault(ds, Counter())[answer] += 1
                if i in selected:
                    for side, key in [('first', 'first_relative_power'), ('second', 'second_relative_power')]:
                        value = independent_power(arrays[task][i], spec[side], inp['channels'])
                        np.testing.assert_allclose(value, ev[key], rtol=1e-9, atol=1e-12)
        assert len(seen) == n and dict(counts) == stats[spec['id']]['answers']
        assert by_dataset == stats[spec['id']]['datasets']
        report[spec['id']] = dict(n=n, independently_recomputed=len(selected), answers=dict(counts),
                                 majority_baseline=max(counts.values())/n)
        print(spec['id'], dict(counts), flush=True)
    assert sum(r['n'] for r in report.values()) == complete['question_records']
    write_json(root/'verification.json', dict(status='passed', source_windows=len(actual),
        question_records=complete['question_records'], questions=report,
        verifier_sha256=digest(__file__),
        checks=['all artifact hashes', 'all signal copies and per-window hashes',
                'exact training-only coverage', 'all evidence-to-target comparisons',
                '24 independent FFT recomputations per question', 'answer counts by dataset']))
    print('Verification passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('datasets/eeg_attribute_qa'))
    verify(parser.parse_args().root)
