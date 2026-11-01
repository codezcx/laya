"""Plot recorded multitask losses and fixed-subset validation; no model loading.

Original metrics are read only. Missing records are gaps, not interpolated data.
Training begins at step 1; an unrecorded step-0 loss is never invented.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / 'configs/training_curves.mplstyle'
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#777777'
DISPLAY = {'bcic_2a': 'BCIC-IV-2a', 'motor_mv_img': 'PhysioMI',
           'seed': 'SEED', 'seed_vii': 'SEED-VII', 'tuab': 'TUAB',
           'tuev': 'TUEV', 'hmc': 'HMC', 'workload': 'Workload',
           'adftd': 'ADFTD', 'CHB_MIT': 'CHB-MIT'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trailing_mean(values, window):
    """Full-window trailing mean, with NaN gaps and no startup extrapolation."""
    if window < 1:
        raise ValueError('Smoothing window must be positive')
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1:] = np.convolve(values, np.ones(window) / window, mode='valid')
    return result


def padded_limits(values, fraction=.12, minimum_pad=.5, lower=None, upper=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Cannot choose an axis range without finite data')
    lo, hi = float(values.min()), float(values.max())
    pad = max(minimum_pad, (hi - lo) * fraction)
    return (max(lower, lo - pad) if lower is not None else lo - pad,
            min(upper, hi + pad) if upper is not None else hi + pad)


def read_records(folder):
    identity_path = folder / 'config.json'
    identity = json.loads(identity_path.read_text())
    config = identity['config']
    source_paths = [identity_path, folder / 'metrics.jsonl']
    rows = [json.loads(line) for line in source_paths[1].read_text().splitlines() if line.strip()]
    training = [r for r in rows if r['status'] == 'training']
    if not training:
        raise ValueError('No recorded training loss')
    steps = [r['step'] for r in training]
    if steps != sorted(set(steps)) or min(steps) < 1 or max(steps) > config['steps']:
        raise ValueError('Duplicate, out-of-order, or invalid training steps')
    if any(not np.isfinite(r['loss']) or r['loss'] < 0 for r in training):
        raise ValueError('Invalid recorded cross-entropy')
    files = sorted(folder.glob('valid_step*_subset.json'))
    validation = [json.loads(f.read_text()) for f in files]
    if not validation or validation[0]['step'] != 0:
        raise ValueError('The validation baseline at step 0 is required')
    valid_steps = [v['step'] for v in validation]
    if valid_steps != sorted(set(valid_steps)):
        raise ValueError('Duplicate or out-of-order validation steps')
    datasets = list(validation[0]['datasets'])
    for report in validation:
        if set(report['datasets']) != set(datasets):
            raise ValueError('Validation task coverage changed')
        for name, metrics in report['datasets'].items():
            if metrics['n'] != validation[0]['datasets'][name]['n']:
                raise ValueError('Validation sample count changed')
            for key in ('accuracy', 'balanced_accuracy'):
                if not np.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                    raise ValueError('Invalid validation metric')
        recomputed = np.mean([m['balanced_accuracy'] for m in report['datasets'].values()])
        if not np.isclose(recomputed, report['macro_balanced_accuracy'], rtol=0, atol=1e-12):
            raise ValueError('Recorded macro balanced accuracy does not match dataset mean')
    source_paths += files
    return config, training, validation, source_paths


def validation_series(reports, expected_steps, name=None):
    by_step = {v['step']: v for v in reports}
    acc, balanced = [], []
    for step in expected_steps:
        if step not in by_step:
            acc.append(np.nan)
            balanced.append(np.nan)
            continue
        metrics = by_step[step]['datasets']
        if name is None:
            acc.append(100 * np.mean([m['accuracy'] for m in metrics.values()]))
            balanced.append(100 * by_step[step]['macro_balanced_accuracy'])
        else:
            acc.append(100 * metrics[name]['accuracy'])
            balanced.append(100 * metrics[name]['balanced_accuracy'])
    return np.asarray(acc), np.asarray(balanced)


def decorate(ax, total, ticks, unfreeze):
    ax.set_xlim(0, total)
    ax.set_xticks(ticks)
    ax.set_xlabel('Training update')
    ax.grid(axis='y', color='#dddddd', linewidth=.6)
    ax.axvline(unfreeze, color='#555555', linestyle=':', linewidth=1)


def plot_validation(ax, steps, accuracy, balanced):
    ax.plot(steps, balanced, color=BLUE, marker='o', label='Balanced accuracy')
    ax.plot(steps, accuracy, color=ORANGE, marker='s', linestyle='--', label='Accuracy')
    ax.set_ylim(*padded_limits(np.r_[accuracy, balanced], lower=0, upper=100))
    ax.set_ylabel('Validation score (%)')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_name = json.loads((ROOT / 'configs/benchmark_train.json').read_text())['run_name']
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'artifacts/benchmark' / default_name)
    parser.add_argument('--arm', choices=['mean_pool','cls','cross_attention','qformer','leaf_frozen','bsr_unfrozen'], required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Replace only this script\'s named plot outputs')
    args = parser.parse_args()
    folder = args.run_dir.resolve() / 'train' / args.arm
    output = args.out or args.run_dir / 'figures' / args.arm
    stems = ['training_validation_curves', 'validation_by_dataset', 'training_loss_by_dataset']
    names = [f'{stem}.{ext}' for stem in stems for ext in ['png', 'pdf', 'svg']]
    names += ['training.csv', 'validation.csv', 'validation_macro.csv', 'figure_manifest.json']
    if not args.overwrite and any((output / name).exists() for name in names):
        raise FileExistsError('Plot outputs already exist; use --overwrite deliberately')
    config, train, validation, sources = read_records(folder)
    source_hashes = {str(p): sha256(p) for p in sources}
    output.mkdir(parents=True, exist_ok=True)
    total = config['steps']
    unfreeze = config['bridge_only_steps'] + 1
    validation_cap = config['validation_per_dataset']
    ticks = np.unique(np.r_[0, np.arange(config['validation_every'], total + 1, config['validation_every']), total])
    x = np.arange(1, max(r['step'] for r in train) + 1)
    raw = np.full(len(x), np.nan)
    for r in train:
        raw[r['step'] - 1] = r['loss']
    # Three complete 17-task cycles in the current run; descriptive, not an estimator of test loss.
    window = 3 * len(config['datasets'])
    smooth = trailing_mean(raw, window)
    acc, balanced = validation_series(validation, ticks)
    best = int(np.nanargmax(balanced))
    axes_ranges = {}

    def save(fig, stem):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{stem}.{ext}', dpi=300, facecolor='white', transparent=False)
        plt.close(fig)

    with plt.style.context(str(STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        fig.subplots_adjust(left=.065, right=.975, bottom=.23, top=.80, wspace=.25)
        fig.suptitle(f'{args.arm} → JEV | Multitask training curves', fontsize=16, x=.065, ha='left', y=.96)
        fig.text(.065, .89, f'{len(train):,} updates recorded · 17 training tasks · fixed validation subsets · seed {config["seed"]}', fontsize=10)
        ax = axes[0]
        ax.plot(x, raw, color=GREY, linewidth=.45, alpha=.8, label='Raw per-update cross-entropy')
        ax.plot(x, smooth, color=BLUE, linewidth=2.2, label=f'Trailing {window}-update mean')
        ax.set_title('a  Training loss', loc='left')
        ax.set_ylabel('Cross-entropy loss')
        ax.set_ylim(*padded_limits(raw, minimum_pad=.08, lower=0))
        ax.legend(loc='upper right')
        ax = axes[1]
        plot_validation(ax, ticks, acc, balanced)
        ax.set_title('b  Nine-dataset macro validation (zoomed y)', loc='left')
        ax.legend(loc='lower right')
        ax.annotate(f'Best balanced accuracy\n{balanced[best]:.2f}% @ {ticks[best]}',
                    xy=(ticks[best], balanced[best]), xytext=(-118, 22), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color=BLUE), fontsize=9, color=BLUE)
        for ax in axes:
            decorate(ax, total, ticks, unfreeze)
        fig.text(.065, .105, f'Loss: different tasks have different class counts; raw values are retained. No step-0 training loss was logged.', fontsize=9)
        fig.text(.065, .055, f'Validation: up to {validation_cap} fixed windows/dataset; no smoothing. Dotted line: JEV adapter/head enabled at {unfreeze}.', fontsize=9)
        axes_ranges['overview'] = [list(ax.get_ylim()) for ax in axes]
        save(fig, stems[0])

        datasets = list(validation[0]['datasets'])
        fig, axes = plt.subplots(3, 3, figsize=(12, 10.3))
        fig.subplots_adjust(left=.075, right=.98, bottom=.11, top=.86, hspace=.48, wspace=.31)
        fig.suptitle('Validation curves by dataset', fontsize=16, x=.075, ha='left', y=.97)
        fig.text(.075, .93, f'Fixed subset: up to {validation_cap} windows/dataset · from step 0 · independent zoomed y axes', fontsize=10)
        handles = [Line2D([], [], color=BLUE, marker='o', label='Balanced accuracy'),
                   Line2D([], [], color=ORANGE, marker='s', linestyle='--', label='Accuracy')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .915), ncol=2)
        for ax, name in zip(axes.flat, datasets):
            a, b = validation_series(validation, ticks, name)
            plot_validation(ax, ticks, a, b)
            ax.set_title(DISPLAY.get(name, name), loc='left')
            decorate(ax, total, [0, 680, 1360, 2040], unfreeze)
            axes_ranges[name] = list(ax.get_ylim())
        fig.text(.075, .045, 'Lines only connect recorded evaluations. Validation CE is saved in JSON; these panels are not full-validation or test scores.', fontsize=9)
        save(fig, stems[1])

        fig, axes = plt.subplots(6, 3, figsize=(12, 15.5))
        fig.subplots_adjust(left=.075, right=.98, bottom=.06, top=.91, hspace=.58, wspace=.3)
        fig.suptitle('Training loss by dataset', fontsize=16, x=.075, ha='left', y=.975)
        fig.text(.075, .95, 'Raw task-specific batches + trailing 9-observation mean · independent y axes · no invented step-0 loss', fontsize=10)
        for ax, name in zip(axes.flat, config['datasets']):
            rows = [r for r in train if r['dataset'] == name]
            tx = np.asarray([r['step'] for r in rows])
            y = np.asarray([r['loss'] for r in rows])
            ax.plot(tx, y, color=GREY, marker='.', markersize=2.5, linewidth=.6, label='Raw')
            ax.plot(tx, trailing_mean(y, 9), color=BLUE, linewidth=1.8, label='Trailing 9 mean')
            ax.set_title(DISPLAY.get(name, name), loc='left', fontsize=11)
            ax.set_ylabel('Cross-entropy', fontsize=9)
            if len(y): ax.set_ylim(*padded_limits(y, minimum_pad=.05, lower=0))
            decorate(ax, total, [0, 1020, 2040], unfreeze)
        axes.flat[-1].axis('off')
        axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc='upper left')
        fig.text(.075, .02, 'Budget: 120 updates/dataset. Loss scales differ with class count; compare trends within a panel, not absolute heights between tasks.', fontsize=9)
        save(fig, stems[2])

    train_csv = []
    for r in train:
        value = smooth[r['step'] - 1]
        train_csv.append(dict(step=r['step'], dataset=r['dataset'], loss=r['loss'],
                              trailing_mean=value if np.isfinite(value) else '',
                              batch_accuracy=r['accuracy']))
    valid_csv, macro_csv = [], []
    for v in validation:
        for name, m in v['datasets'].items():
            valid_csv.append(dict(step=v['step'], dataset=name, n=m['n'], accuracy=m['accuracy'],
                                  balanced_accuracy=m['balanced_accuracy']))
        macro_csv.append(dict(step=v['step'], macro_balanced_accuracy=v['macro_balanced_accuracy'],
                             macro_accuracy=float(np.mean([m['accuracy'] for m in v['datasets'].values()]))))
    write_csv(output / 'training.csv', train_csv)
    write_csv(output / 'validation.csv', valid_csv)
    write_csv(output / 'validation_macro.csv', macro_csv)
    if source_hashes != {str(p): sha256(p) for p in sources}:
        raise RuntimeError('Source logs changed while rendering; rerun from a consistent snapshot')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources_sha256=source_hashes,
        script_sha256=sha256(__file__), style_sha256=sha256(STYLE),
        versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
        train_updates=len(train), task_update_counts=dict(Counter(r['dataset'] for r in train)),
        validation_steps=[v['step'] for v in validation], validation_scope=f'fixed <= {validation_cap}-window subset per dataset, NOT full validation/test',
        missing_training_steps=sorted(set(x.tolist()) - {r['step'] for r in train}),
        missing_validation_steps=sorted(set(ticks.tolist()) - {v['step'] for v in validation}),
        smoothing=dict(overall=f'trailing arithmetic mean, {window} updates, requires full window; no gaps bridged',
                       per_dataset='trailing arithmetic mean, 9 task observations; raw observations retained'),
        missing_step0_training_loss='not recorded, not fabricated', validation_loss='available in JSON, not plotted in these panels',
        axes_limits=axes_ranges, uncertainty='not estimated; one run, no confidence intervals',
        first_last_51_mean_loss=[float(np.mean(raw[:window])), float(np.mean(raw[-window:]))],
        first_last_macro_balanced_accuracy=[validation[0]['macro_balanced_accuracy'], validation[-1]['macro_balanced_accuracy']],
        selection=dict(step=int(ticks[best]), macro_balanced_accuracy=float(balanced[best] / 100)),
        description='Recorded training CE and fixed-subset validation; no direction of change is assumed.',
        software_process_reference='https://doi.org/10.48550/arXiv.2609.00065',
        outputs_sha256={name: sha256(output / name) for name in names if name != 'figure_manifest.json'})
    (output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output.resolve()), train_updates=len(train),
                         validation_steps=manifest['validation_steps'], selection=manifest['selection'])))


if __name__ == '__main__':
    main()
