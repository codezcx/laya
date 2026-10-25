"""Freeze a current joint checkpoint's LEAF queries and fit bounded linear probes.

Uses the existing subject splits and online preprocessing. No test rows, encoder
updates, validation tuning or Qwen allocation. This diagnoses readable information;
it is not an optimizer-/budget-matched replacement for joint JEV training.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from eeg_align.benchmark_data import catalog, classification_metrics
from eeg_align.checkpoint import capture_model, restore_model, seed_rng, tensor_fingerprint
from eeg_align.leaf_runtime import CONFIG, digest
from eeg_align.online_leaf import OnlineLEAF
from eeg_align.signal_data import ROOT as SIGNAL_ROOT, SignalStore, load_rows


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_subject_cv(cache, out):
    """Select regularization using source subjects only, then read validation once."""
    assert (cache / 'complete.json').exists()
    source_plan = json.loads((cache / 'plan.json').read_text())
    assert source_plan['train_manifest_sha256'] == digest(
        SIGNAL_ROOT / f"fold{source_plan['fold']}" / 'train.jsonl')
    assert source_plan['validation_manifest_sha256'] == digest(
        SIGNAL_ROOT / f"fold{source_plan['fold']}" / 'validation.jsonl')
    source_rows = {r['id']: r for r in load_rows('train', source_plan['fold'])
                   if r['dataset'] in source_plan['datasets']}
    arrays = np.load(cache / 'features.npz')
    candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(cache=str(cache.resolve()), feature_sha256=digest(cache / 'features.npz'),
                source_plan_sha256=digest(cache / 'plan.json'), script_sha256=digest(__file__),
                C_candidates=candidates, selection='mean source-subject held-out BA; ties choose smaller C',
                folds='leave one training subject out; scaler and classifier refit in every fold',
                estimator=dict(class_weight='balanced', solver='lbfgs', max_iter=2000, tol=1e-4),
                limitation='Probe-only selection: the fixed encoder previously saw all source subjects. '
                           'Inner scores are not end-to-end unseen-subject estimates. Outer validation '
                           'was already used in earlier diagnostics; this is not independent test evidence.')
    write_json(out / 'plan.json', plan)
    results, models = {}, {}
    started = time.monotonic()

    def fit(x, y, c):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight='balanced', solver='lbfgs', max_iter=2000, tol=1e-4, random_state=42))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y)
        return model, [str(w.message) for w in caught]

    with threadpool_limits(limits=3):
        for ds in source_plan['datasets']:
            x = arrays[f'{ds}_train']
            y = arrays[f'{ds}_train_labels']
            rr = [source_rows[i] for i in arrays[f'{ds}_train_ids']]
            assert np.array_equal(y, [r['label'] for r in rr])
            assert all(r['dataset'] == ds and r['split'] == 'train' for r in rr)
            groups = np.array([r['subject'] for r in rr])
            subjects = sorted(set(groups))
            assert len(subjects) >= 3
            classes = len(np.unique(y))
            scores = []
            for c in candidates:
                folds = []
                for held in subjects:
                    train, held_out = groups != held, groups == held
                    assert not set(groups[train]) & set(groups[held_out])
                    model, messages = fit(x[train], y[train], c)
                    assert np.array_equal(model.classes_, np.arange(classes))
                    assert model[0].n_samples_seen_ == train.sum()
                    score = classification_metrics(y[held_out], model.predict_proba(x[held_out]), classes)
                    folds.append(dict(subject=str(held), train_n=int(train.sum()),
                                      metric=score, convergence_warnings=messages))
                value = dict(C=c, subject_macro_BA=float(np.mean([
                    f['metric']['balanced_accuracy'] for f in folds])), folds=folds)
                scores.append(value)
                print(json.dumps(dict(dataset=ds, C=c, source_cv_BA=value['subject_macro_BA'])), flush=True)
                write_json(out / 'progress.json', dict(dataset=ds, completed_candidates=scores,
                                                       elapsed_seconds=time.monotonic() - started))
            chosen = max(scores, key=lambda v: (v['subject_macro_BA'], -v['C']))
            models[ds], messages = fit(x, y, chosen['C'])
            results[ds] = dict(selected_C=chosen['C'], source_cv_BA=chosen['subject_macro_BA'],
                               candidates=scores, final_fit_warnings=messages,
                               final_iterations=models[ds][-1].n_iter_.tolist())
        # Freeze every choice on disk before accessing any outer validation features/labels.
        write_json(out / 'selection.json', results)
        with (out / 'predictions.jsonl').open('x') as predictions:
            for ds, model in models.items():
                for split in ['train', 'validation']:
                    x, y = arrays[f'{ds}_{split}'], arrays[f'{ds}_{split}_labels']
                    probabilities = model.predict_proba(x)
                    results[ds][split] = classification_metrics(y, probabilities, len(model.classes_))
                    for sample, label, p in zip(arrays[f'{ds}_{split}_ids'], y, probabilities):
                        predictions.write(json.dumps(dict(id=str(sample), dataset=ds, split=split,
                                                          label=int(label), probabilities=p.tolist())) + '\n')
                print(json.dumps({k: v for k, v in dict(dataset=ds, **results[ds]).items()
                                  if k != 'candidates'}), flush=True)
        joblib.dump(models, out / 'probes.joblib')
        write_json(out / 'metrics.json', results)
        write_json(out / 'complete.json', dict(elapsed_seconds=time.monotonic() - started,
                                               datasets=source_plan['datasets'], no_test_access=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--source-cv-cache', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=[
        'BNCI2014001', 'BNCI2014009', 'Nakanishi2015'])
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.source_cv_cache:
        if args.checkpoint or args.expected_step is not None:
            parser.error('--source-cv-cache cannot be combined with a checkpoint')
        source_subject_cv(args.source_cv_cache, args.out)
        return
    if args.checkpoint is None or args.expected_step is None:
        parser.error('Extraction requires --checkpoint and --expected-step')
    assert args.batch_size > 0 and len(set(args.datasets)) == len(args.datasets)
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    seed_rng(42)
    tasks = {t['task_id']: t for t in catalog()}
    assert set(args.datasets) <= tasks.keys()
    before = args.checkpoint.stat()
    source = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert source['format_version'] == 2 and source['step'] == args.expected_step
    assert source['config']['bn_policy'] == 'frozen', 'This diagnosis uses the frozen-BN branch'
    for key in ['checkpoint', 'upstream_commit', 'fold']:
        assert source['config'][key] == CONFIG[key], key
    for key, path in [
        ('data_manifest_sha256', SIGNAL_ROOT / f"fold{CONFIG['fold']}" / 'complete.json'),
        ('task_sha256', Path('configs/benchmark_tasks.json')),
        ('instructions_sha256', Path('artifacts/benchmark/instructions.npy')),
    ]:
        assert source['config'][key] == digest(path), key
    for path, expected in source['config']['code_sha256'].items():
        assert digest(path) == expected, f'Source changed: {path}'

    encoder = OnlineLEAF(bn_policy='frozen')
    prefix = 'eeg_encoder.'
    state = dict(format_version=2, **{
        kind: {k[len(prefix):]: v for k, v in source[kind].items() if k.startswith(prefix)}
        for kind in ['trainable', 'buffers']
    })
    restore_model(encoder, state)  # Strict tensor names, dtype and shapes, including BN buffers.
    restored = capture_model(encoder)
    fingerprints = {kind: tensor_fingerprint(state[kind]) for kind in ['trainable', 'buffers']}
    assert all(tensor_fingerprint(restored[kind]) == fingerprints[kind] for kind in fingerprints)
    after = args.checkpoint.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns), 'Source checkpoint changed while loading'
    del source, state

    rows = {split: [r for r in load_rows(split, CONFIG['fold']) if r['dataset'] in args.datasets]
            for split in ['train', 'validation']}
    assert not {r['id'] for r in rows['train']} & {r['id'] for r in rows['validation']}
    assert not {(r['dataset'], r['subject']) for r in rows['train']} & {
        (r['dataset'], r['subject']) for r in rows['validation']}
    assert all(r['tokens'] == 8 for rr in rows.values() for r in rr)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(restored, args.out / 'encoder.pt')
    del restored
    plan = dict(
        checkpoint=str(args.checkpoint.resolve()), step=args.expected_step, bn_policy='frozen',
        encoder_sha256=fingerprints, snapshot_sha256=digest(args.out / 'encoder.pt'),
        script_sha256=digest(__file__), git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sklearn_version=sklearn.__version__, torch_version=torch.__version__,
        fold=CONFIG['fold'], datasets=args.datasets, batch_size=args.batch_size,
        representation='8 final normalized LEAF query tokens flattened to 2048; exact online preprocessing',
        encoder_mode='eval; all parameters frozen; CUDA bfloat16 autocast as in joint validation',
        estimator=dict(scaler='StandardScaler fit on training rows only', classifier='LogisticRegression',
                       C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, tol=1e-4,
                       random_state=42),
        train_manifest_sha256=digest(SIGNAL_ROOT / f"fold{CONFIG['fold']}" / 'train.jsonl'),
        validation_manifest_sha256=digest(SIGNAL_ROOT / f"fold{CONFIG['fold']}" / 'validation.jsonl'),
        counts={ds: {split: sum(r['dataset'] == ds for r in rr) for split, rr in rows.items()}
                for ds in args.datasets},
        limitation='Diagnostic only: classifier objective/optimizer/data exposure differ from JEV. '
                   'No hyperparameter search or test evaluation; checkpoint already selected on validation.')
    write_json(args.out / 'plan.json', plan)
    encoder.cuda().eval().requires_grad_(False)
    before_parameters = tensor_fingerprint(dict(encoder.named_parameters()))
    before_buffers = tensor_fingerprint(dict(encoder.named_buffers()))
    store = SignalStore()
    features, models, results = {}, {}, {}
    started = time.monotonic()
    with (args.out / 'predictions.jsonl').open('x') as predictions, threadpool_limits(limits=3):
        for ds in args.datasets:
            selected = {split: [r for r in rr if r['dataset'] == ds] for split, rr in rows.items()}
            for split, rr in selected.items():
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(rr), args.batch_size):
                        batch = rr[start:start + args.batch_size]
                        tokens, mask = encoder(store.batch(batch), len(batch), unfrozen=False)
                        assert mask[:, :8].all() and not mask[:, 8:].any()
                        x = tokens[:, :8].reshape(len(batch), 2048).cpu().numpy()
                        assert np.isfinite(x).all()
                        chunks.append(x)
                        if start % (args.batch_size * 8) == 0:
                            progress = dict(dataset=ds, split=split, encoded=start + len(batch),
                                            total=len(rr), elapsed_seconds=time.monotonic() - started)
                            write_json(args.out / 'progress.json', progress)
                            print(json.dumps(progress), flush=True)
                features[f'{ds}_{split}'] = np.concatenate(chunks)
                features[f'{ds}_{split}_ids'] = np.array([r['id'] for r in rr])
                features[f'{ds}_{split}_labels'] = np.array([r['label'] for r in rr])
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, tol=1e-4, random_state=42))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(features[f'{ds}_train'], features[f'{ds}_train_labels'])
            results[ds] = dict(convergence_warnings=[str(w.message) for w in caught],
                               iterations=model[-1].n_iter_.tolist())
            assert np.array_equal(model.classes_, np.arange(len(tasks[ds]['classes'])))
            for split, rr in selected.items():
                probs = model.predict_proba(features[f'{ds}_{split}'])
                results[ds][split] = classification_metrics(
                    features[f'{ds}_{split}_labels'], probs, len(tasks[ds]['classes']))
                for row, probability in zip(rr, probs):
                    predictions.write(json.dumps(dict(row, probabilities=probability.tolist())) + '\n')
            models[ds] = model
            write_json(args.out / 'metrics.json', results)
            print(json.dumps(dict(dataset=ds, **results[ds])), flush=True)
        assert tensor_fingerprint(dict(encoder.named_parameters())) == before_parameters
        assert tensor_fingerprint(dict(encoder.named_buffers())) == before_buffers
        np.savez(args.out / 'features.npz', **features)
        joblib.dump(models, args.out / 'probes.joblib')
        write_json(args.out / 'complete.json', dict(
            elapsed_seconds=time.monotonic() - started, encoder_parameters_unchanged=True,
            encoder_buffers_unchanged=True, metrics=results,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))


if __name__ == '__main__':
    main()
