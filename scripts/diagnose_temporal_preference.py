"""Locked source-only longitudinal feasibility pilot; see TEMPORAL_PILOT.md.

Standalone CPU analysis; does not import or modify the training implementation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = list(range(3, 10))
SESSIONS = ['03T', '04E', '05E']
SEED = 20261025


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def covariances(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x @ x.transpose(0, 2, 1) / x.shape[-1]


def fit_predict(train_cov, train_y, query_cov, arm):
    if arm == 0:
        cc = [train_cov[train_y == k].mean(0) for k in [0, 1]]
        cc = [.9*c + .1*np.trace(c)/3*np.eye(3) for c in cc]
        eigenvalues, vectors = eigh(cc[0], cc[0] + cc[1])
        np.testing.assert_allclose(cc[0] @ vectors,
                                   (cc[0] + cc[1]) @ vectors * eigenvalues,
                                   atol=1e-9, rtol=1e-7)
        w = vectors[:, [0, -1]]
    else:
        w = np.eye(3)

    def features(cov):
        variances = np.einsum('ci,ncd,di->ni', w, cov, w)
        assert np.all(variances > 0)
        return np.log(variances)

    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto'))
    model.fit(features(train_cov), train_y)
    probability = model.predict_proba(features(query_cov))
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(1), 1)
    return probability


def ba(y, probability):
    return float(balanced_accuracy_score(y, probability.argmax(-1)))


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(scores, probabilities, labels):
    # shape: subject, session, arm; matched independently within each session.
    delta = scores[:, :, 0] - scores[:, :, 1]
    early, later = delta[:, 0], delta[:, 1:].mean(1)
    r = correlation(early, later)
    null = [correlation(early, later[list(order)])
            for order in itertools.permutations(range(len(SUBJECTS)))]
    assert r is not None and all(x is not None for x in null)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(SUBJECTS), size=(20000, len(SUBJECTS)))
    bootstrap_r = [correlation(early[ix], later[ix]) for ix in draws]
    valid = [x for x in bootstrap_r if x is not None]
    personal = scores[:, 0].argmax(-1)
    global_choice = np.array([scores[np.arange(7) != i, 0].mean(0).argmax()
                              for i in range(7)])
    decision = []
    for i, subject in enumerate(SUBJECTS):
        for s in [1, 2]:
            pp = probabilities[i, s]
            yy = labels[i, s, 80:]
            each = [ba(yy, pp[a]) for a in [0, 1]]
            decision.append(dict(subject=subject, session=SESSIONS[s],
                                 personal_arm=int(personal[i]), global_arm=int(global_choice[i]),
                                 personal=each[personal[i]], global_choice=each[global_choice[i]],
                                 fixed_A=each[0], fixed_B=each[1],
                                 ensemble=ba(yy, pp.mean(0)), retrospective_oracle=max(each)))
    gain = np.array([np.mean([v['personal']-v['global_choice'] for v in decision
                             if v['subject'] == subject]) for subject in SUBJECTS])
    return dict(
        subjects=SUBJECTS, sessions=SESSIONS, arms=['CSP_LDA', 'ChannelPower_LDA'],
        matched_refit_BA=scores.tolist(), delta_A_minus_B=delta.tolist(),
        repeatability=dict(early_vs_mean_later_r=r,
                           pairwise_r={f'{a}_{b}': correlation(delta[:, a], delta[:, b])
                                       for a, b in [(0, 1), (0, 2), (1, 2)]},
                           bootstrap_percentile_95=np.quantile(valid, [.025, .975]).tolist(),
                           bootstrap_degenerate=20000-len(valid),
                           exact_person_permutation_two_sided_p=float(np.mean(
                               np.abs(null) >= abs(r)-1e-12)), permutation_n=len(null)),
        decisions=decision,
        decision_value=dict(per_person_gain=gain.tolist(), mean_gain=float(gain.mean()),
                            bootstrap_percentile_95=np.quantile(gain[draws].mean(1), [.025, .975]).tolist(),
                            mean_BA={key: float(np.mean([v[key] for v in decision]))
                                     for key in ['personal', 'global_choice', 'fixed_A', 'fixed_B',
                                                 'ensemble', 'retrospective_oracle']}),
        limits=['Seven source users, classical pipelines only; exploratory not confirmatory.',
                'No separation of transient session effects and measurement error.',
                'No foundation-model trust, causal neural trait, or new-method claim.',
                'Feedback protocol and stable artifacts remain possible explanations.',
                'Small-sample bootstrap coverage is not guaranteed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = ROOT/'docs/conference_research_2026/TEMPORAL_PILOT.md'
    guards = {str(p): digest(p) for p in [Path(__file__), protocol]}
    manifest = json.loads((ROOT/'datasets/benchmark/signals/fold0/complete.json').read_text())
    split = manifest['datasets']['BNCI2014004']['subjects']
    assert {int(k) for k, v in split.items() if v == 'train'} == set(SUBJECTS)
    plan = dict(protocol_sha256=digest(protocol), code_sha256=guards,
                subjects=SUBJECTS, sessions=SESSIONS, seed=SEED, train_trials=80,
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                training_code_imported=False, outer_validation_and_test_accessed=False)
    write(args.out/'plan.json', plan)
    data, hashes, structure = {}, {}, []
    sos = butter(4, [8, 30], fs=250, btype='bandpass', output='sos')
    for subject in SUBJECTS:
        selected = []
        for suffix in ['T', 'E']:
            p = args.raw_root/f'B{subject:02d}{suffix}.mat'
            hashes[str(p)] = digest(p)
            raw_sessions = loadmat(p, simplify_cells=True)['data']
            assert len(raw_sessions) == (3 if suffix == 'T' else 2)
            selected.extend(raw_sessions[2:] if suffix == 'T' else raw_sessions)
        for si, raw in enumerate(selected):
            y = np.asarray(raw['y'], dtype=int)-1
            assert len(y) == 160 and raw['fs'] == 250
            assert np.array_equal(np.unique(y), [0, 1])
            starts = np.asarray(raw['trial'], dtype=int)-1
            assert np.all(np.diff(starts) > 1125)
            x = np.stack([raw['X'][t+750:t+1875, :3].T for t in starts])
            assert x.shape == (160, 3, 1125) and np.isfinite(x).all()
            x = sosfiltfilt(sos, x, axis=-1)[:, :, 125:]
            cov = covariances(x)
            assert np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(cov) > 0)
            blocks = [np.bincount(y[j:j+40], minlength=2).tolist() for j in range(0, 160, 40)]
            assert all(min(b) > 0 for b in blocks)
            data[subject, si] = cov, y
            structure.append(dict(subject=subject, session=SESSIONS[si], n=160,
                                  block_labels=blocks, artifact_flagged=int(np.count_nonzero(raw['artifacts'])),
                                  first_onset_sample=int(starts[0]),
                                  trial_spacing_min=int(np.diff(starts).min())))
    # Finish structure validation for every included session before any classifier fit.
    write(args.out/'structure.json', structure)
    write(args.out/'input_hashes.json', hashes)
    scores = np.empty((7, 3, 2))
    local_probs = np.empty((7, 3, 2, 80, 2))
    history_probs = np.empty_like(local_probs)
    labels = np.stack([np.stack([data[s, d][1] for d in range(3)]) for s in SUBJECTS])
    order = list(itertools.product(range(7), range(3)))
    np.random.default_rng(SEED).shuffle(order)
    blocks = []
    with threadpool_limits(limits=2):
        for i, si in order:
            subject = SUBJECTS[i]
            cov, y = data[subject, si]
            history_cov = np.concatenate([data[subject, d][0] for d in range(si)]+[cov[:80]])
            history_y = np.concatenate([data[subject, d][1] for d in range(si)]+[y[:80]])
            for a in [0, 1]:
                p = fit_predict(cov[:80], y[:80], cov[80:], a)
                local_probs[i, si, a] = p
                scores[i, si, a] = ba(y[80:], p)
                history_probs[i, si, a] = (p if si == 0 else
                    fit_predict(history_cov, history_y, cov[80:], a))
                blocks.append(dict(subject=subject, session=SESSIONS[si], arm=a,
                                   local_query_block_BA=[ba(y[80+j:100+j], p[j:j+20])
                                                        for j in range(0, 80, 20)],
                                   local_train_n=80, history_train_n=len(history_y)))
            print(json.dumps(dict(subject=subject, session=SESSIONS[si],
                                  seconds=round(time.monotonic()-started, 2))), flush=True)
    np.savez(args.out/'predictions.npz', local=local_probs, history=history_probs,
             labels=labels, subjects=SUBJECTS, sessions=SESSIONS, trial_indices=np.arange(160))
    result = summarize(scores, history_probs, labels)
    result['block_diagnostics'] = blocks
    result['elapsed_seconds'] = time.monotonic()-started
    assert all(digest(p) == h for p, h in {**guards, **hashes}.items())
    result['artifact_sha256'] = {p.name: digest(p) for p in args.out.iterdir() if p.is_file()}
    write(args.out/'result.json', result)
    print(json.dumps({k: result[k] for k in ['repeatability', 'decision_value']}), flush=True)


if __name__ == '__main__':
    main()
