"""Exploratory secondary analysis of Yang2025's published score matrix.

No model fitting or deployable-personalization claim. See AUTHOR_SCORE_AUDIT.md.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ARMS = ['CSP', 'FBCSP', 'EEGNet', 'deepConvNet', 'FBCNet']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(x, y):
    x, y = x-x.mean(-1, keepdims=True), y-y.mean(-1, keepdims=True)
    denominator = np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
    return np.divide((x*y).sum(-1), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--headers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    matrix = np.full((51, 3, 5), np.nan)
    current = None
    rows = list(csv.reader(args.csv.open(encoding='utf-8-sig')))
    assert rows[1][2:7] == ARMS
    for row in rows[2:]:
        if row[0].strip():
            if not row[0].strip().isdigit():
                continue
            current = int(row[0])-1
        if row[1].strip() not in ['1', '2', '3']:
            continue
        session = int(row[1])-1
        assert 0 <= current < 51 and np.isnan(matrix[current, session]).all()
        matrix[current, session] = [float(v) for v in row[2:7]]
    assert np.isfinite(matrix).all() and np.all((matrix >= 0) & (matrix <= 1))
    headers = json.loads(args.headers.read_text())['files']
    assert len(headers) == 153 and all('date' in h for h in headers)
    dates = {}
    for h in headers:
        pp = Path(h['path']).parts
        subject, session = int(pp[0][4:])-1, int(pp[1][4:])
        assert (subject, session) not in dates
        dates[subject, session] = datetime.strptime(h['date']+' '+h['time'], '%d.%m.%y %H.%M.%S')
    orders = np.array([sorted(range(3), key=lambda d: dates[i, d]) for i in range(51)])
    eligible = np.array([i for i in range(51) if len({dates[i, d].date() for d in range(3)}) == 3])
    excluded = [i+1 for i in range(51) if i not in eligible]
    # Metadata-only eligibility amendment, recorded before any score statistic.
    assert excluded == [12, 16, 19, 21, 22, 29, 39, 41, 51]
    scores = np.stack([matrix[i, orders[i]] for i in eligible])
    n = len(eligible)
    assert n == 42
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, n, size=(20000, n))
    pairs = []
    for a, b in itertools.combinations(range(5), 2):
        delta = scores[:, :, a]-scores[:, :, b]
        early, later = delta[:, 0], delta[:, 1:].mean(1)
        boot = corr(early[draws], later[draws])
        assert np.isfinite(boot).all()
        loo = [float(corr(np.delete(early, i), np.delete(later, i))) for i in range(n)]
        pairs.append(dict(arms=[ARMS[a], ARMS[b]], pearson_r=float(corr(early, later)),
                          spearman_r=float(spearmanr(early, later).statistic),
                          bootstrap_percentile_95=np.quantile(boot, [.025, .975]).tolist(),
                          leave_one_person_out_range=[min(loo), max(loo)],
                          early_delta=early.tolist(), mean_later_delta=later.tolist(),
                          original_number_order_r=float(corr(matrix[eligible, 0, a]-matrix[eligible, 0, b],
                              (matrix[eligible, 1:, a]-matrix[eligible, 1:, b]).mean(1)))))
    # Retrospective score-selection illustration. Not history-adapted V(m).
    personal = scores[:, 0].argmax(-1)
    global_choice = np.array([scores[np.arange(n) != i, 0].mean(0).argmax() for i in range(n)])
    personal_later = np.array([scores[i, 1:, personal[i]].mean() for i in range(n)])
    global_later = np.array([scores[i, 1:, global_choice[i]].mean() for i in range(n)])
    gap = personal_later-global_later
    result = dict(
        kind='exploratory_author_score_reanalysis_not_new_model_experiment',
        input_sha256={str(p): digest(p) for p in [args.csv, args.headers, Path(__file__)]},
        n_source_people=51, n_source_sessions=153, n_analysis_people=n,
        analysis_subjects=(eligible+1).tolist(), excluded_repeated_date_subjects=excluded,
        arms=ARMS, original_number_order_scores=matrix.tolist(),
        chronological_session_indices=orders.tolist(),
        chronological_dates=[[dates[i, d].isoformat() for d in orders[i]] for i in range(51)],
        subjects_with_nonchronological_numbering=[i+1 for i in range(51) if list(orders[i]) != [0, 1, 2]],
        source_means=matrix.mean((0, 1)).tolist(),
        original_number_order_day_means=matrix[eligible].mean(0).tolist(),
        chronological_day_means=scores.mean(0).tolist(),
        fraction_at_least_95_percent=(scores >= .95).mean((0, 1)).tolist(),
        per_model_min=scores.min((0, 1)).tolist(), per_model_max=scores.max((0, 1)).tolist(),
        pairs=pairs,
        score_selection=dict(personal_choice_counts=np.bincount(personal, minlength=5).tolist(),
                             global_choice_counts=np.bincount(global_choice, minlength=5).tolist(),
                             personal_mean=float(personal_later.mean()), global_mean=float(global_later.mean()),
                             mean_difference=float(gap.mean()),
                             bootstrap_percentile_95=np.quantile(gap[draws].mean(1), [.025, .975]).tolist(),
                             per_person_difference=gap.tolist(),
                             retrospective_late_oracle=float(scores[:, 1:].max(-1).mean()),
                             caveat='Conditional retrospective score comparison: no historical-label adaptation, '
                                    'no chronological trial prefix evaluation, no new training. '
                                    'Bootstrap conditions on observed leave-one-person-out global choices.'),
        limitations=['No trialwise predictions or repeated initialization/fold logs; cannot deattenuate reliability.',
                     'Ten pair intervals are exploratory marginal intervals, not simultaneous coverage.',
                     'Recording clock validity and identity matching of author score rows remain assumptions.',
                     'No foundation model in this score matrix; not evidence about frozen encoders.'])
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    primary = next(p for p in pairs if p['arms'] == ['EEGNet', 'deepConvNet'])
    x, y = np.array(primary['early_delta'])*100, np.array(primary['mean_later_delta'])*100
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    axs[0].scatter(x, y, s=28, alpha=.75, color='#2166ac')
    axs[0].axhline(0, color='gray', linewidth=.7); axs[0].axvline(0, color='gray', linewidth=.7)
    axs[0].set(xlabel='Earliest day: EEGNet − deepConvNet (pp)',
               ylabel='Mean later days: EEGNet − deepConvNet (pp)',
               title=f'{n} people; Pearson r = {primary["pearson_r"]:.2f}')
    for i in np.flatnonzero((abs(x) > 10) | (abs(y) > 7)):
        axs[0].annotate(str(eligible[i]+1), (x[i], y[i]), fontsize=8)
    for a, label, color in [(2, 'EEGNet', '#2166ac'), (3, 'deepConvNet', '#d95f02')]:
        axs[1].plot([1, 2, 3], scores[:, :, a].mean(0)*100, marker='o', label=label, color=color)
        axs[1].plot([1, 2, 3], matrix[eligible, :, a].mean(0)*100, linestyle='--', alpha=.55, color=color)
    axs[1].set(xlabel='Day rank / original session number', ylabel='Mean author accuracy (%)',
               title='Solid: date order; dashed: session number', xticks=[1, 2, 3])
    axs[1].legend()
    fig.suptitle('Exploratory reanalysis of Yang2025 author scores; no new model fits', fontsize=10)
    fig.savefig(args.out/'diagnostic.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(nonchronological=len(result['subjects_with_nonchronological_numbering']),
                          means=result['source_means'], primary_pair=primary,
                          score_selection=result['score_selection'])), flush=True)


if __name__ == '__main__':
    main()
