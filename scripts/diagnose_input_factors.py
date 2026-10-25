"""Source-only 2x2 input diagnosis for the existing frozen LEAF control.

Keep the cached four-second signals/filtering, encoder and fixed probe recipe.
Cross common-average reference with trial/whole-subject robust normalization.
No outer validation/test samples, encoder updates, or hyperparameter selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_align.benchmark_data import catalog, classification_metrics, resample
from eeg_align.checkpoint import capture_model, restore_model, seed_rng, tensor_fingerprint
from eeg_align.leaf_runtime import CONFIG, UP, digest
from eeg_align.online_leaf import OnlineLEAF
from eeg_align.signal_data import ROOT as SIGNAL_ROOT, SignalStore, load_rows
from leaf_datasets.electrode_unifier import ElectrodeUnifier

ARMS = {'original_trial': (False, 'trial'), 'original_subject': (False, 'subject'),
        'car_trial': (True, 'trial'), 'car_subject': (True, 'subject')}
SOURCE = Path('artifacts/benchmark/cross_subject_fold0_bn_control/representation_probe_0200/initial_encoder_control')


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def transform(x, spatial, car, scope):
    x = x.copy()
    if car:
        x -= x.mean(axis=1, keepdims=True)
    axis = (1, 2) if scope == 'trial' else None
    limits = np.percentile(x, [.1, 25, 50, 75, 99.9], axis=axis)
    scale = limits[3] - limits[1]
    assert np.all(scale > 0) and np.isfinite(x).all()
    if scope == 'trial':
        x = (np.clip(x, limits[0, :, None, None], limits[4, :, None, None])
             - limits[2, :, None, None]) / scale[:, None, None]
    else:
        x = (np.clip(x, limits[0], limits[4]) - limits[2]) / scale
    return np.einsum('ec,bct->bet', spatial, x).astype(np.float32), limits.tolist()


def fit(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000,
        tol=1e-4, random_state=42))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y)
    assert not caught, [str(w.message) for w in caught]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    assert args.batch_size > 0 and torch.cuda.is_available()
    torch.set_num_threads(3)
    seed_rng(42)
    started = time.monotonic()
    source_plan = json.loads((SOURCE/'plan.json').read_text())
    assert source_plan['step'] == 100 and source_plan['bn_policy'] == 'frozen'
    assert digest(SOURCE/'encoder.pt') == source_plan['snapshot_sha256']
    assert digest(SIGNAL_ROOT/'fold0/train.jsonl') == source_plan['train_manifest_sha256']
    rows = [r for r in load_rows('train', 0) if r['dataset'] == 'BNCI2014001']
    assert len(rows) == 2016 and {r['subject'] for r in rows} == set(map(str, range(3, 10)))
    tid = rows[0]['task']
    assert all(r['task'] == tid and r['split'] == 'train' for r in rows)
    folder = SIGNAL_ROOT/'BNCI2014001'
    meta = json.loads((folder/'complete.json').read_text())
    assert len(meta['channels']) == 22 and meta['sampling_rate'] == 250
    assert not any('-' in name for name in meta['channels'])
    unifier = ElectrodeUnifier('LEAF-ch65')
    aliases = {k.upper(): k for k in unifier.reverse_lookup}
    names = [aliases[n.upper()] for n in meta['channels']]
    spatial = unifier.map_to_template(np.eye(22, dtype=np.float32)[None], names)[0]
    signals = np.load(folder/'signals.npy', mmap_mode='r')
    # Populate only the selected dataset: the normal constructor reads an example
    # from every dataset, including rows outside this diagnostic population.
    baseline = SignalStore.__new__(SignalStore)
    baseline.meta, baseline.arrays = {tid: meta}, {tid: signals}
    baseline.keep, baseline.spatial = {tid: list(range(22))}, {tid: spatial}
    encoder = OnlineLEAF(bn_policy='frozen')
    state = torch.load(SOURCE/'encoder.pt', map_location='cpu', weights_only=False)
    restore_model(encoder, state)
    restored = capture_model(encoder)
    assert all(tensor_fingerprint(restored[k]) == source_plan['encoder_sha256'][k]
               for k in ['trainable', 'buffers'])
    del state, restored
    code_files = [Path(__file__), *Path('eeg_align').glob('*.py'),
                  UP/'leaf_datasets/electrode_unifier.py', Path('configs/benchmark_train.json')]
    guards = {str(p): digest(p) for p in code_files}
    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(source_plan_sha256=digest(SOURCE/'plan.json'), source_snapshot_sha256=digest(SOURCE/'encoder.pt'),
                code_sha256=guards, arms=ARMS, batch_size=args.batch_size, n=2016,
                subjects=list(map(str, range(3, 10))), query_ids=[r['id'] for r in rows],
                outer_validation_and_test_rows_used=False, encoder_updates=False,
                preprocessing='Same cached 4s 0.1-75Hz epochs and polyphase resampling in all arms; '
                              'CAR after resampling, before clipping/scaling; same 65-channel mapping.',
                normalization='Subject arms pool all 288 unlabeled trials per subject, including '
                              'held inner-fold inputs; transductive diagnostic, not online performance.',
                probe='Fixed C=1; source-only leave-one-subject-out; scaler/probe refit in each fold. '
                      'Also evaluate all arms through the fold baseline probe without refitting.',
                limitations=['Exploratory source data only, not an independent generalization estimate.',
                             'Pretraining exposure to these subjects is not excluded.',
                             'No claim that CAR removes only noise or preserves all task information.',
                             'Does not isolate frequency band or six-second FIR context.',
                             'Not a new algorithm or novelty claim.'])
    write(args.out/'plan.json', plan)
    encoder.cuda().eval().requires_grad_(False)
    before = {kind: tensor_fingerprint(dict(iterator())) for kind, iterator in
              [('parameters', encoder.named_parameters), ('buffers', encoder.named_buffers)]}
    features = {a: np.empty((len(rows), 2048), np.float32) for a in ARMS}
    checks = {}
    with torch.inference_mode():
        for subject in plan['subjects']:
            ix = np.array([i for i, r in enumerate(rows) if r['subject'] == subject])
            rr = [rows[i] for i in ix]
            raw = np.array(signals[[r['source_index'] for r in rr]], dtype=np.float32)
            x = resample(raw, 250)
            stats = dict(raw_selected_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                         common_component_energy_fraction=float(np.sum(x.mean(1, keepdims=True)**2)*22/np.sum(x*x)),
                         arms={})
            for arm, (car, scope) in ARMS.items():
                values, limits = transform(x, spatial, car, scope)
                assert values.shape == (288, 65, 800) and np.isfinite(values).all()
                if arm == 'original_trial':
                    reference = baseline.batch(rr)[0][2][0]
                    np.testing.assert_array_equal(values, reference)
                    stats['baseline_input_exact'] = True
                    # Trial-wise normalization must not depend on unrelated rows.
                    singleton, _ = transform(x[:1], spatial, car, scope)
                    np.testing.assert_array_equal(singleton, values[:1])
                stats['arms'][arm] = dict(limits=limits, normalized_rms=float(np.sqrt(np.mean(values**2))))
                for start in range(0, len(ix), args.batch_size):
                    z = values[start:start+args.batch_size]
                    q, mask = encoder([(tid, list(range(len(z))), [z])], len(z), unfrozen=False)
                    assert mask[:, :8].all() and not mask[:, 8:].any()
                    features[arm][ix[start:start+len(z)]] = q[:, :8].reshape(len(z), 2048).cpu().numpy()
                print(json.dumps(dict(subject=subject, arm=arm, elapsed=time.monotonic()-started)), flush=True)
            checks[subject] = stats
            write(args.out/'input_checks.json', checks)
    for kind, iterator in [('parameters', encoder.named_parameters), ('buffers', encoder.named_buffers)]:
        assert tensor_fingerprint(dict(iterator())) == before[kind]
    del encoder
    torch.cuda.empty_cache()
    labels = np.array([r['label'] for r in rows])
    groups = np.array([r['subject'] for r in rows])
    np.savez(args.out/'features.npz', **features, labels=labels, subjects=groups,
             ids=np.array([r['id'] for r in rows]))
    results = {a: [] for a in ARMS}
    probabilities = {}
    with threadpool_limits(limits=3):
        for held in plan['subjects']:
            train, test = groups != held, groups == held
            baseline_model = fit(features['original_trial'][train], labels[train])
            for arm in ARMS:
                model = baseline_model if arm == 'original_trial' else fit(features[arm][train], labels[train])
                p = model.predict_proba(features[arm][test])
                fixed = baseline_model.predict_proba(features[arm][test])
                original = baseline_model.predict_proba(features['original_trial'][test])
                result = dict(subject=held, matched_probe=classification_metrics(labels[test], p, 4),
                              fixed_baseline_probe=classification_metrics(labels[test], fixed, 4),
                              fixed_probe_prediction_flip=float(np.mean(fixed.argmax(1) != original.argmax(1))),
                              iterations=int(model[-1].n_iter_.max()))
                results[arm].append(result)
                probabilities[f'{arm}_{held}_matched'] = p
                probabilities[f'{arm}_{held}_fixed'] = fixed
            write(args.out/'fold_results.json', results)
            print(json.dumps(dict(completed_probe_subject=held, elapsed=time.monotonic()-started)), flush=True)
    np.savez(args.out/'probabilities.npz', **probabilities)
    means = {a: float(np.mean([r['matched_probe']['balanced_accuracy'] for r in rr])) for a, rr in results.items()}
    interaction = means['car_subject'] - means['original_subject'] - means['car_trial'] + means['original_trial']
    assert all(digest(p) == h for p, h in guards.items()), 'Imported source/config changed during diagnosis'
    summary = dict(matched_probe_mean_BA=means, interaction_difference_of_differences=interaction,
                   no_encoder_updates=True, encoder_parameters_and_buffers_unchanged=True,
                   no_outer_validation_or_test_rows=True, all_fits_converged=True,
                   subject_pooled_normalization_is_transductive=True, elapsed_seconds=time.monotonic()-started,
                   files_sha256={name: digest(args.out/name) for name in
                                 ['plan.json','features.npz','fold_results.json','probabilities.npz','input_checks.json']})
    write(args.out/'complete.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
