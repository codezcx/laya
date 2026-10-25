"""Sensitivity planning for a frozen policy's paired future user-level gain.

Not observed power, not a power calculation for overlapping cross-validation,
and not a guarantee that a longitudinal EEG experiment will follow this model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, nct, norm, t


def rejection_probability(n, standardized_mean, alpha):
    critical = t.ppf(1-alpha/2, n-1)
    noncentrality = standardized_mean*np.sqrt(n)
    probability = nct.sf(critical, n-1, noncentrality) + nct.cdf(-critical, n-1, noncentrality)
    if not np.isfinite(probability):
        # T=(Z+noncentrality)/sqrt(V/df), V~chi2(df), Z~N(0,1).
        # Some SciPy versions return NaN in an extreme noncentral tail.
        probability, error = quad(
            lambda v: chi2.pdf(v, n-1)*(norm.sf(critical*np.sqrt(v/(n-1))-noncentrality)
                              + norm.cdf(-critical*np.sqrt(v/(n-1))-noncentrality)),
            0, np.inf, epsabs=1e-10, epsrel=1e-10)
        assert error < 1e-8
    return float(probability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads(args.author_result.read_text())
    gaps = np.array(observed['score_selection']['per_person_difference'])*100
    assert len(gaps) == 42 and np.isfinite(gaps).all()
    # Variability anchor only. Never feed the observed mean into power.
    sd = float(gaps.std(ddof=1))
    rows = []
    for alpha in [.05, .05/3]:
        for target_power in [.8, .9]:
            for n in [7, 20, 30, 42, 62, 100, 200]:
                # Avoid unnecessary extreme noncentralities in older SciPy
                # implementations. Grow a finite bracket around the root.
                upper = .1
                while rejection_probability(n, upper, alpha) < target_power:
                    upper *= 2
                    assert upper <= 10
                assert np.isfinite(rejection_probability(n, upper, alpha))
                mde = brentq(lambda d: rejection_probability(n, d, alpha)-target_power, 0, upper)
                rows.append(dict(n_people=n, alpha=alpha, target_power=target_power,
                                 mde_in_population_sd=mde,
                                 illustrative_mde_pp_at_author_sd=mde*sd))
    # Independent Monte Carlo check of the reference model, not EEG simulations.
    rng = np.random.default_rng(20261025)
    checks = []
    n, n_sims = 42, 20000
    for effect in [0.0, next(r['mde_in_population_sd'] for r in rows
                          if r['n_people'] == n and r['alpha'] == .05
                          and r['target_power'] == .8)]:
        sample = rng.normal(effect, 1, (n_sims, n))
        statistic = sample.mean(1)/(sample.std(1, ddof=1)/np.sqrt(n))
        count = int((np.abs(statistic) > t.ppf(.975, n-1)).sum())
        estimate = count/n_sims
        z = norm.ppf(.975)
        center = (estimate+z*z/(2*n_sims))/(1+z*z/n_sims)
        radius = z*np.sqrt(estimate*(1-estimate)/n_sims+z*z/(4*n_sims*n_sims))/(1+z*z/n_sims)
        expected = rejection_probability(n, effect, .05)
        assert abs(estimate-expected) < 5*np.sqrt(expected*(1-expected)/n_sims)
        checks.append(dict(n_people=n, simulations=n_sims, standardized_effect=effect,
                           rejection_fraction=estimate, wilson_95=[center-radius, center+radius],
                           noncentral_t_probability=expected))
    assert abs(rejection_probability(42, 0, .05)-.05) < 1e-12
    result = dict(
        kind='planning_sensitivity_under_frozen_policy_iid_normal_user_differences',
        inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [args.author_result, Path(__file__)]},
        versions=dict(numpy=np.__version__, scipy=scipy.__version__),
        planning_assumptions=['Policies and model configurations fixed on independent development people.',
                              'One paired future-session-average gain per independent confirmation person.',
                              'Normally distributed user gains with a common unknown SD for the exact t model.',
                              'Single two-sided primary contrast alpha=.05; alpha=.05/3 is a separate '
                              'conservative scenario for three confirmatory contrasts, not a selected result.',
                              'No dropout assumed; n is usable confirmation people, not enrollment or trial count.'],
        author_sd_pp=sd,
        author_sd_usage='Illustrative scale anchor only. Retrospective author CV scores are not prospective '
                        'equal-history policy gains, and their global choices share data. '
                        'This SD is not a validated variance estimate for the future design.',
        no_observed_effect_used=True, no_minimum_practical_effect_assumed=True,
        sensitivity=rows, reference_model_simulation_checks=checks,
        exclusions=['Not power for a bootstrap interval, nested CV, learned policies refitted within folds, '
                    'heterogeneous mixed effects, or the stability variance component.',
                    'Not a test of equivalence or absence of useful personalization.',
                    'Cannot count sessions or seeds as additional independent people.',
                    'Practical effect threshold and confirmation data are still unspecified.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(author_sd_pp=sd, reference=[r for r in rows if r['alpha']==.05
                             and r['target_power']==.8], checks=checks)), flush=True)


if __name__ == '__main__':
    main()
