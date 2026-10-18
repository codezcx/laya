"""Build separable EEG attribute QA for one source split (CPU only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from eeg_tasks.attribute_questions import OPTIONS, specifications, relative_power, measure
from eeg_tasks.channels import CHANNELS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def build(args):
    out = args.output
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    audit = json.loads((args.signals.parent / 'complete.json').read_text())
    manifest_hash = digest(args.manifest)
    if audit['manifest_sha256'] != manifest_hash:
        raise ValueError('Signal cache/manifest mismatch')
    print('Verifying complete signal-cache SHA256...', flush=True)
    signal_hash = digest(args.signals)
    if audit['signals_sha256'] != signal_hash:
        raise ValueError('Signal cache checksum mismatch')
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    x = np.load(args.signals, mmap_mode='r')
    if x.shape != (len(rows), 30, 600) or len(rows) != audit['n']:
        raise ValueError('Unexpected signal dimensions')
    if {r['split'] for r in rows} != {'train', 'validation'}:
        raise ValueError('Unexpected input split')
    for key in ['subject', 'path']:
        train = {(r['dataset'], r[key]) for r in rows if r['split'] == 'train'}
        val = {(r['dataset'], r[key]) for r in rows if r['split'] == 'validation'}
        if train & val:
            raise ValueError(f'Train/validation overlap: {key}')
    allowed = {'Cho2017', 'Dreyer2023', 'Lee2019_MI', 'SEED', 'FACED-NEMAR'}
    if {r['dataset'] for r in rows} != allowed:
        raise ValueError('Unexpected source datasets')
    out.mkdir(parents=True)
    (out / 'questions').mkdir()
    (out / 'signals').mkdir()
    specs = specifications()
    write_json(out / 'question_catalog.json', specs)
    write_json(out / 'BUILDING.json', {'status': 'incomplete; do not train until complete.json exists'})
    counts, example, total = {}, {}, 0
    for task, channels in [('mi', ['C3', 'Cz', 'C4']), ('emotion', CHANNELS)]:
        ids = [i for i, r in enumerate(rows) if r['split'] == args.split
               and (r['dataset'] in {'SEED', 'FACED-NEMAR'}) == (task == 'emotion')]
        signal_file = f'signals/{task}.npy'
        dest = np.lib.format.open_memmap(out / signal_file, mode='w+', dtype=np.float32,
                                       shape=(len(ids), len(channels), 600))
        selected = [s for s in specs if s['task'] == task]
        handles = {s['id']: (out / 'questions' / (s['id'] + '.jsonl')).open('w') for s in selected}
        stats = {s['id']: {'answers': Counter(), 'datasets': {}} for s in selected}
        try:
            with (out / f'{task}_sources.jsonl').open('w') as source:
                for start in range(0, len(ids), 128):
                    batch_ids = ids[start:start + 128]
                    signals = np.asarray(x[batch_ids, :len(channels)])
                    if not np.isfinite(signals).all():
                        raise ValueError('Nonfinite signal')
                    dest[start:start + len(batch_ids)] = signals
                    powers = {name: relative_power(signals[..., sl]) for name, sl in
                              [('full', slice(None)), ('early', slice(0, 300)), ('late', slice(300, 600))]}
                    for j, original in enumerate(batch_ids):
                        # Provenance/class labels are never part of the model input.
                        source.write(json.dumps(dict(sample_id=f'source-{original:06d}',
                            source_row_index=original, signal_file=signal_file, signal_index=start+j,
                            signal_sha256=hashlib.sha256(signals[j].tobytes()).hexdigest(),
                            source_metadata=rows[original]), ensure_ascii=False) + '\n')
                    for spec in selected:
                        a, b, db, answers = measure(powers, channels, spec)
                        for j, original in enumerate(batch_ids):
                            r = rows[original]
                            sid = f'source-{original:06d}'
                            item = dict(id=f'{sid}:{spec["id"]}', sample_id=sid, split=args.split,
                                question_type=spec['id'],
                                input=dict(signal_file=signal_file, signal_index=start+j, sample_rate=200,
                                           channels=channels, question=spec['question'], options=OPTIONS),
                                target=dict(answer=str(answers[j])),
                                evidence=dict(first_relative_power=float(a[j]), second_relative_power=float(b[j]),
                                              difference_db=float(db[j])),
                                provenance=dict(dataset=r['dataset'], subject=r['subject'], source_row_index=original))
                            handles[spec['id']].write(json.dumps(item, ensure_ascii=False) + '\n')
                            stat = stats[spec['id']]
                            stat['answers'][str(answers[j])] += 1
                            stat['datasets'].setdefault(r['dataset'], Counter())[str(answers[j])] += 1
                            example.setdefault(spec['id'], item)
                    if start % 4096 == 0:
                        print(f'{task}: {start + len(batch_ids)}/{len(ids)} windows', flush=True)
        finally:
            for h in handles.values():
                h.close()
        dest.flush()
        del dest
        counts.update(stats)
        total += len(ids)
    write_json(out / 'examples.json', example)
    write_json(out / 'statistics.json', counts)
    shutil.copyfile('docs/ATTRIBUTE_DATASET.md', out / 'README.md')
    if args.split == 'validation':
        note = '# This directory contains SOURCE VALIDATION ONLY\n\n'
        note += f'{total} held-out windows; no training examples. See complete.json for exact contents.\n\n'
        (out/'README.md').write_text(note + (out/'README.md').read_text())
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name != 'BUILDING.json'}
    write_json(out / 'complete.json', dict(schema_version=1, source_windows=total,
        question_records=sum(sum(s['answers'].values()) for s in counts.values()),
        source_manifest=str(args.manifest.resolve()), source_manifest_sha256=manifest_hash,
        source_signals=str(args.signals.resolve()), source_signals_sha256=signal_hash,
        code_sha256={p: digest(p) for p in ['scripts/build_attribute_dataset.py', 'eeg_tasks/attribute_questions.py']},
        files=files, threshold_db=1.0, threshold_fitting='none; fixed descriptive tolerance',
        source_splits=[args.split], sfreq=200, samples=600,
        spectral_method='Welch Hann 200 samples, overlap100, nfft200; band-bin sum / 1–30 Hz bin sum',
        band_bins_hz={'theta': '[4,8)', 'alpha': '[8,13)', 'beta': '[13,30]'},
        input_contract='Only input is fed to model; target/evidence/provenance/source_metadata must not be included',
        temporal_reference='Window start, NOT cue onset or resting baseline'))
    (out / 'BUILDING.json').unlink()
    print(f'Completed {total} windows, {sum(sum(s["answers"].values()) for s in counts.values())} QA records: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/multitask_bridge_v30/source_cache/rows.jsonl'))
    parser.add_argument('--signals', type=Path, default=Path('artifacts/jev_eeg/source_signals/signals.npy'))
    parser.add_argument('--output', type=Path, default=Path('datasets/eeg_attribute_qa'))
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    build(parser.parse_args())
