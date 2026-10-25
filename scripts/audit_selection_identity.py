"""Exploratory identity control on the mapped Yang2025 author score matrix.

No training. Compares actual early choices with their shuffled assignment,
holding marginal model-choice counts fixed; does not estimate adapted V(m).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def identity_summary(early, later):
    n, k = early.shape
    choices = early.argmax(1)
    onehot = np.eye(k)[choices]
    frequencies = onehot.mean(0)
    selected = later[np.arange(n), choices]
    mixture = later @ frequencies
    contributions = ((onehot-frequencies)*later).mean(0)
    return choices, selected, mixture, contributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.result.read_text())
    eligible = np.array(source['analysis_subjects'])-1
    original = np.array(source['original_number_order_scores'])
    orders = np.array(source['chronological_session_indices'])
    scores = np.stack([original[i, orders[i]] for i in eligible])
    assert scores.shape == (42, 3, 5) and np.isfinite(scores).all()
    assert source['subjects_with_nonchronological_numbering'] == []
    n, repetitions = 42, 20000
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, n, (repetitions, n))
    permutations = np.stack([rng.permutation(n) for _ in range(repetitions)])
    groups = []
    for name, indices in [('all_five', list(range(5))), ('EEGNet_deepConvNet', [2, 3])]:
        early, later = scores[:, 0, indices], scores[:, 1:, indices].mean(1)
        choices, selected, mixture, contributions = identity_summary(early, later)
        actual, expected = float(selected.mean()), float(mixture.mean())
        increment = actual-expected
        assert abs(contributions.sum()-increment) < 1e-12
        # The same model is assigned once to each user, with fixed arm counts:
        # additive user difficulty and additive arm means cannot change T.
        centered = later-later.mean(1, keepdims=True)-later.mean(0, keepdims=True)+later.mean()
        _, a, b, _ = identity_summary(early, centered)
        assert abs((a-b).mean()-increment) < 1e-12
        boot_freq = np.eye(len(indices))[choices[draws]].mean(1)
        boot_mean_later = later[draws].mean(1)
        bootstrap = selected[draws].mean(1)-(boot_freq*boot_mean_later).sum(1)
        shuffled = later[np.arange(n)[None, :], choices[permutations]].mean(1)-expected
        extreme = int((np.abs(shuffled) >= abs(increment)-1e-12).sum())
        nominal_p = (extreme+1)/(repetitions+1)
        # Exact variance of a uniformly random assignment of the donor choices.
        assignment_matrix = later[:, choices]
        residual = (assignment_matrix-assignment_matrix.mean(0, keepdims=True)
                    -assignment_matrix.mean(1, keepdims=True)+assignment_matrix.mean())
        exact_permutation_variance = float((residual**2).sum()/(n*n*(n-1)))
        loo = []
        for i in range(n):
            _, a, b, _ = identity_summary(np.delete(early, i, axis=0), np.delete(later, i, axis=0))
            loo.append(float((a-b).mean()))
        global_choices = np.array([np.delete(early, i, axis=0).mean(0).argmax() for i in range(n)])
        global_scores = later[np.arange(n), global_choices]
        groups.append(dict(
            group=name, arms=[source['arms'][i] for i in indices],
            choice_counts=np.bincount(choices, minlength=len(indices)).tolist(),
            personal_mean=actual, shuffled_expected_mean=expected,
            identity_increment=increment,
            bootstrap_percentile_95=np.quantile(bootstrap, [.025, .975]).tolist(),
            leave_one_person_out_range=[min(loo), max(loo)],
            leave_one_person_out_increment=loo,
            model_contributions=contributions.tolist(),
            person_contributions=(selected-mixture).tolist(),
            no_self_donor_expected_mean=float((n*expected-actual)/(n-1)),
            no_self_donor_increment=float(n*increment/(n-1)),
            global_mean=float(global_scores.mean()),
            personal_minus_global=float((selected-global_scores).mean()),
            permutation=dict(repetitions=repetitions, extreme_count=extreme,
                             two_sided_p=nominal_p, mc_standard_error=float(np.sqrt(nominal_p*(1-nominal_p)/repetitions)),
                             central_95=np.quantile(shuffled, [.025, .975]).tolist(),
                             sampled_mean=float(shuffled.mean()),
                             sampled_variance=float(shuffled.var(ddof=1)),
                             exact_variance=exact_permutation_variance)))
    order = np.argsort([g['permutation']['two_sided_p'] for g in groups])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(groups)-rank)*groups[i]['permutation']['two_sided_p']))
        groups[i]['permutation']['holm_p_two_groups'] = running
    result = dict(kind='exploratory_posthoc_identity_control_not_adapted_policy_value',
                  analysis_subjects=(eligible+1).tolist(), n_people=n, groups=groups,
                  seed=20261025, repetitions=repetitions,
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [args.result, Path(__file__)]},
                  numpy_version=np.__version__,
                  limitations=['Author CV scores, not new predictions or equal-history adapted models.',
                               'Exploratory after viewing all scores and earlier analyses; not independent confirmation.',
                               'Permutation interpretation requires exchangeable person correspondence under the null; '
                               'not a randomized causal intervention or a test of neural mechanism.',
                               'Bootstrap recomputes both choice frequencies and future score means; '
                               'it does not include new training, trial noise or uncertainty of learned selectors.',
                               'A positive identity increment need not beat the strongest globally chosen model.',
                               'Rare model choices and ceiling effects can dominate; all leave-one-person results retained.'])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(groups=groups), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
