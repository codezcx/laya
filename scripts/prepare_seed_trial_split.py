"""Reindex existing SEED windows into the pinned paper's ordered 9/3/3 trials.

Never rewrite signals or alter another dataset. QC is inherited only after
verifying the old frozen summary, indexes and array identities. No model runs.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import digest
from eeg_align.multitask_data import write_json
from eeg_align.signal_quality import load_quality
from eeg_align.split_policy import SEED_PROTOCOL, seed_split, validate_split_frame
from scripts.prepare_multitask_splits import extract_methods, summary, COMMIT


def prepare(config, split_root, cache_root, quality_root):
    old_split, old_cache = Path(config['split_root']), Path(config['cache_root'])
    if digest(old_split / 'summary.json') != config['split_summary_sha256']:
        raise ValueError('Source splits changed')
    old_summary = json.loads((old_split / 'summary.json').read_text())
    if old_summary['upstream_commit'] != COMMIT:
        raise ValueError('Wrong benchmark commit')
    reports = load_quality(config)
    # Exercise the actual pinned official function, not an approximate ratio.
    cls = extract_methods('data/dataset/seeds/seed.py', 'SeedBuilder', {'_divide_by_uniform_label'})
    official = cls._divide_by_uniform_label(pd.DataFrame([{'label': json.dumps(list(range(15)))}]))
    upstream_map = {i: r['split'] for r in official.to_dict('records') for i in json.loads(r['label'])}
    if list(seed_split(np.arange(15))) != [upstream_map[i] for i in range(15)]:
        raise ValueError('Local policy disagrees with pinned upstream')
    for path in [split_root, cache_root, quality_root]:
        if path.exists():
            raise FileExistsError(path)
    for path in [split_root, cache_root, quality_root]:
        path.mkdir(parents=True)
    new_summary = copy.deepcopy(old_summary)
    qc = json.loads((Path(config['quality_root']) / 'summary.json').read_text())
    qc.update(cache_root=str(cache_root.resolve()),
              derivation='Inherited fixed signal QC; only SEED trial allocation changed; source array identities verified',
              source_quality_sha256=config['quality_summary_sha256'],
              finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    for name, manifest in old_summary['datasets'].items():
        source = old_split / name
        if (digest(source / 'records.csv') != manifest['records_sha256']
                or json.loads((source / 'manifest.json').read_text()) != manifest):
            raise ValueError(f'Source allocation changed: {name}')
        shutil.copytree(source, split_root / name)
        if name != 'seed':
            (cache_root / name).symlink_to((old_cache / name).resolve(), target_is_directory=True)
            shutil.copy2(Path(config['quality_root']) / f'{name}.json', quality_root / f'{name}.json')
            continue
        frame = pd.read_csv(old_cache / name / 'index.csv', dtype={'subject': str}, keep_default_na=False)
        original = frame.copy(deep=True)
        frame['split'] = seed_split(frame['event'].to_numpy())
        record_frame = pd.read_csv(source / 'records.csv', dtype={'subject': str}, keep_default_na=False)
        expanded = []
        for row in record_frame.to_dict('records'):
            for event in range(15):
                expanded.append(dict(row, event=event, split=upstream_map[event]))
        records = pd.DataFrame(expanded)
        records['allocation_row'] = np.arange(len(records))
        new_manifest = copy.deepcopy(manifest)
        new_manifest.update(split_protocol=SEED_PROTOCOL, record_unit='source_recording_trial',
                            counts=summary(records), n_original_trials=len(records),
                            source_records_sha256=manifest['records_sha256'],
                            split_method='SeedBuilder._divide_by_uniform_label; ordered event IDs 0:9/9:12/12:15')
        validate_split_frame(name, new_manifest, records)
        validate_split_frame(name, new_manifest, frame, windows=True)
        if (len(records) != 675 or frame['subject'].nunique() != 15
                or len(frame) != 37890 or not original.drop(columns='split').equals(frame.drop(columns='split'))):
            raise ValueError('Unexpected SEED coverage or modified window identity')
        record_keys = set(zip(records['record_id'], records['event'], records['subject']))
        if set(zip(frame['record_id'], frame['event'], frame['subject'])) != record_keys:
            raise ValueError('Window trial identities disagree with source recordings')
        records.to_csv(split_root / name / 'records.csv', index=False)
        new_manifest['records_sha256'] = digest(split_root / name / 'records.csv')
        write_json(split_root / name / 'manifest.json', new_manifest)
        new_summary['datasets'][name] = new_manifest
        folder = cache_root / name
        folder.mkdir()
        for filename in sorted(set(frame['array'])):
            (folder / filename).symlink_to((old_cache / name / filename).resolve())
        frame.to_csv(folder / 'index.csv', index=False)
        counts = {s: dict(epochs=len(g), subjects=g['subject'].nunique(),
                         labels={str(k): int(v) for k,v in g['label'].value_counts().items()})
                  for s,g in frame.groupby('split')}
        complete = json.loads((old_cache / name / 'complete.json').read_text())
        complete.update(counts=counts, records_sha256=new_manifest['records_sha256'],
                        index_sha256=digest(folder / 'index.csv'), split_protocol=SEED_PROTOCOL,
                        resplit_code_sha256=digest(__file__), source_index_sha256=digest(old_cache / name / 'index.csv'),
                        preprocessing='Signals unchanged; source preprocessing code hash retained; online normalization is encoder-specific')
        write_json(folder / 'complete.json', complete)
        report = copy.deepcopy(reports[name])
        if report['exclusions'] or report['invalid_heldout']:
            raise ValueError('SEED QC exclusions require explicit re-audit before reallocation')
        report.update(index_sha256=complete['index_sha256'],
                      before_counts={s:c['epochs'] for s,c in counts.items()},
                      effective_counts={s:c['epochs'] for s,c in counts.items()})
        write_json(quality_root / f'{name}.json', report)
        qc['datasets'][name].update(report_sha256=digest(quality_root / f'{name}.json'),
                                   before_counts=report['before_counts'], effective_counts=report['effective_counts'])
    new_summary.update(refresh=dict(base_summary_sha256=config['split_summary_sha256'],
        changed_datasets=['seed'], warm_start_from_old_trained_checkpoint=False),
        preparation_code_sha256=digest(__file__))
    write_json(split_root / 'summary.json', new_summary)
    qc['split_summary_sha256'] = digest(split_root / 'summary.json')
    write_json(quality_root / 'summary.json', qc)
    updated = dict(config, split_root=str(split_root), cache_root=str(cache_root), quality_root=str(quality_root),
                   split_summary_sha256=digest(split_root / 'summary.json'),
                   quality_summary_sha256=digest(quality_root / 'summary.json'))
    load_quality(updated)
    print(json.dumps({k:updated[k] for k in ['split_root','cache_root','quality_root',
                     'split_summary_sha256','quality_summary_sha256']}, indent=2))
    print('SEED_COUNTS', json.dumps(counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--quality-root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.source_config.read_text())
    prepare(config.get('config', config), args.split_root, args.cache_root, args.quality_root)
