"""Full-cohort paired summaries only: never average different users across arms."""
import csv
import json
from .common import CONFIG, OUT, write_json


def aggregate(points, cases, arms, steps):
    expected = {c['key']: c for c in cases}
    index = {}
    for p in points:
        key = p['case']['key']
        if key not in expected or p['case'] != expected[key] or p['arm'] not in arms or p['step'] not in steps:
            raise RuntimeError('Unexpected curve identity')
        group = (key, p['arm'], p['step'])
        if group in index:
            raise RuntimeError('Duplicate curve point')
        index[group] = p
    datasets = sorted({c['dataset'] for c in cases})
    matched, pending = {}, {}
    for dataset in datasets:
        keys = [c['key'] for c in cases if c['dataset'] == dataset]
        matched[dataset], pending[dataset] = {}, {}
        for step in steps:
            missing = [(key, arm) for key in keys for arm in arms if (key, arm, step) not in index]
            if missing:
                pending[dataset][str(step)] = missing
                continue
            entry = {}
            for key in keys:
                reference = index[key, arms[0], step]['baseline']
                if any(index[key, arm, step]['baseline'] != reference for arm in arms):
                    raise RuntimeError('No-adaptation baseline differs across methods')
            for arm in arms:
                group = [index[key, arm, step] for key in keys]
                metrics = {f'{part}_{metric}': sum(p[part][metric] for p in group)/len(group)
                           for part in ('query', 'support')
                           for metric in ('accuracy', 'balanced_accuracy', 'ce')}
                metrics['baseline_balanced_accuracy'] = sum(p['baseline']['balanced_accuracy'] for p in group)/len(group)
                metrics['delta_vs_ordinary_pp'] = 100*sum(
                    index[key, arm, step]['query']['balanced_accuracy'] -
                    index[key, 'ordinary_lora', step]['query']['balanced_accuracy'] for key in keys)/len(keys)
                metrics['delta_vs_unadapted_pp'] = 100*(metrics['query_balanced_accuracy']-metrics['baseline_balanced_accuracy'])
                metrics['negative_vs_ordinary_users'] = sum(
                    index[key, arm, step]['query']['balanced_accuracy'] <
                    index[key, 'ordinary_lora', step]['query']['balanced_accuracy'] - 1e-10 for key in keys)
                entry[arm] = dict(users=len(keys), case_ids=keys, **metrics)
            matched[dataset][str(step)] = entry
    equal = {}
    for step in steps:
        if all(str(step) in matched[d] for d in datasets):
            equal[str(step)] = {}
            for arm in arms:
                names = [n for n in matched[datasets[0]][str(step)][arm]
                         if n not in ('users', 'case_ids', 'negative_vs_ordinary_users')]
                equal[str(step)][arm] = {
                    n:sum(matched[d][str(step)][arm][n] for d in datasets)/len(datasets) for n in names}
    return dict(matched=matched, equal_task=equal, pending=pending, curve_points=len(index),
        expected_curve_points=len(cases)*len(arms)*len(steps), final_step=steps[-1],
        original_evaluation_allocation='test', query_selection=False, independent_test_estimate=False)


def plot(report, directory, arms, steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    labels = dict(ordinary_lora='Ordinary LoRA', source_erm='Source ERM init', fomaml='FOMAML init')
    groups = [('bcic_2a', '2a: same 2 users'), ('hmc', 'HMC: same 2 users'), ('equal_task', 'Equal-task mean: 4 users')]
    series = dict(report['matched'], equal_task=report['equal_task'])
    colors = ['#0072B2', '#D55E00', '#009E73']
    for filename, rows in [
        ('adaptation_curves', [('query_balanced_accuracy', 'Full query BAcc (%)', 100),
                               ('delta_vs_ordinary_pp', 'Gain vs ordinary LoRA (pp)', 1)]),
        ('support_query_curves', [('support_accuracy', 'Full support accuracy (%)', 100),
                                 ('support_ce', 'Full support CE', 1),
                                 ('query_ce', 'Full query CE', 1)])]:
        fig, axes = plt.subplots(len(rows), len(groups), figsize=(14, 3.3*len(rows)),
                                 squeeze=False, constrained_layout=True)
        for col, (group, title) in enumerate(groups):
            values = series.get(group, {})
            for row, (metric, ylabel, scale) in enumerate(rows):
                ax = axes[row, col]
                for arm, color, marker in zip(arms, colors, ['o', 's', '^']):
                    y = [scale*values[str(s)][arm][metric] if str(s) in values else np.nan for s in steps]
                    ax.plot(steps, y, label=labels[arm], color=color, marker=marker, markersize=4)
                if not values:
                    ax.text(.5, .5, 'Waiting for all paired users / arms', transform=ax.transAxes,
                            ha='center', fontsize=9)
                if metric == 'delta_vs_ordinary_pp':
                    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
                ax.set(title=title, xlabel='Personal optimizer step', ylabel=ylabel, xlim=(-2, 102))
                ax.set_xticks(steps)
                ax.grid(alpha=.25)
                ax.spines[['top', 'right']].set_visible(False)
                ax.margins(y=.12)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Matched seed 42; fixed cohorts; exploratory original-test query; no checkpoint selection')
        for extension in ('png', 'pdf'):
            fig.savefig(directory / f'{filename}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)


_last_curve_count = -1


def summary():
    global _last_curve_count
    plan = json.loads((OUT / 'plan.json').read_text())
    paths = sorted((OUT / 'jobs').glob('*/cases/*/*/curves/step*.json'))
    points = [json.loads(path.read_text()) for path in paths]
    if any(p['stamp']['source_hashes'] != plan['source_hashes'] for p in points):
        raise RuntimeError('Curve provenance changed')
    report = aggregate(points, plan['cases'], CONFIG['arms'], CONFIG['eval_steps'])
    report['completed_unique_results'] = len(list((OUT / 'jobs').glob('*/cases/*/*/result.json')))
    report['expected_unique_results'] = len(plan['cases'])*len(CONFIG['arms'])
    report['role'] = CONFIG['role']
    write_json(OUT / 'summary.json', report)
    if len(points) != _last_curve_count:
        with (OUT / 'curve_points.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['dataset', 'user', 'arm', 'step', 'query_bacc', 'query_acc', 'query_ce',
                             'support_acc', 'support_ce', 'support_n', 'query_n'])
            for p in points:
                writer.writerow([p['case']['dataset'], p['case']['subject'], p['arm'], p['step'],
                    p['query']['balanced_accuracy'], p['query']['accuracy'], p['query']['ce'],
                    p['support']['accuracy'], p['support']['ce'],
                    len(p['case']['support_ids']), len(p['case']['query_ids'])])
        plot(report, OUT / 'figures', CONFIG['arms'], CONFIG['eval_steps'])
        _last_curve_count = len(points)
    return report
