#!/usr/bin/env python3
"""High-dimensional geometry and descriptive plots; no model fitting/selection.

PCA is fit only to the displayed validation embeddings for visualization. It is
not a predictive evaluation. All geometrical scores use the original dimensions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_samples

ARMS = ['mean_pool', 'cls', 'cross_attention', 'qformer']
STAGES = ['bsr_mean', 'readout_mean', 'readout_flat', 'connector_mean', 'decision']
DISPLAY = dict(bcic_2a='2a', motor_mv_img='PhysioMI', seed='SEED', seed_vii='SEED-VII',
               tuab='TUAB', tuev='TUEV', hmc='HMC', workload='Workload', adftd='ADFTD')
COLORS = ['#0072B2', '#D55E00', '#009E73', '#AA4499', '#000000', '#997700', '#882255', '#44AA99', '#332288']
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('Cosine distance undefined for zero vectors')
    return x/norms


def pair_separation(dist, labels):
    """Class-macro within distance and class-pair-macro between distance.

    Exclude self-pairs. Missing within pairs return None, never a fabricated zero.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    within, between = [], []
    for i, c in enumerate(classes):
        ix = np.flatnonzero(labels == c)
        if len(ix) > 1:
            d = dist[np.ix_(ix, ix)]
            within.append(float(d[np.triu_indices(len(ix), 1)].mean()))
        for other in classes[i+1:]:
            jx = np.flatnonzero(labels == other)
            between.append(float(dist[np.ix_(ix, jx)].mean()))
    if not within or not between:
        return dict(within=None, between=None, gap=None)
    a, b = float(np.mean(within)), float(np.mean(between))
    return dict(within=a, between=b, gap=(b-a)/max(a, b) if max(a, b) > 1e-12 else None)


def geometry(x, labels, subjects):
    unit = normalize(x)
    dist = np.maximum(0., 1.-unit@unit.T)
    np.fill_diagonal(dist, 0.)
    sep = pair_separation(dist, labels)
    raw = np.asarray(x, dtype=np.float64)
    centered = raw-raw.mean(axis=0)
    gram = centered@centered.T
    d2 = np.maximum(0., np.diag(gram)[:, None]+np.diag(gram)[None]-2*gram)
    euclidean = pair_separation(np.sqrt(d2), labels)
    trace = float(np.trace(gram))
    rank = trace**2/float(np.square(gram).sum()) if trace > 1e-12 else 0.
    subject_gaps = []
    eligible = []
    for subject in np.unique(subjects):
        ix = np.flatnonzero(subjects == subject)
        s = pair_separation(dist[np.ix_(ix, ix)], labels[ix])
        if s['gap'] is not None:
            subject_gaps.append(s['gap'])
            eligible.append(str(subject))
    sil = silhouette_samples(dist, labels, metric='precomputed')
    return dict(n=len(x), dimensions=x.shape[1],
        within_class_cosine=sep['within'], between_class_cosine=sep['between'], class_gap=sep['gap'],
        class_macro_silhouette=float(np.mean([sil[labels == c].mean() for c in np.unique(labels)])),
        euclidean_class_gap=euclidean['gap'],
        within_condition_cosine=float(dist[np.triu_indices(len(dist), 1)].mean()),
        rms_centered=float(np.sqrt(trace/np.prod(x.shape))),
        participation_rank=rank,
        subject_controlled_gap=float(np.mean(subject_gaps)) if subject_gaps else None,
        eligible_subjects=len(eligible), eligible_subject_ids=eligible)


def pca2(x):
    model = PCA(n_components=2, svd_solver='randomized', random_state=42)
    projected = model.fit_transform(x)
    return projected, model.explained_variance_ratio_, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run/'representation_analysis'
    for arm in ARMS:
        if not (root/arm/'complete.json').exists():
            raise RuntimeError(f'Incomplete extraction: {arm}')
    manifests = {a: json.loads((root/a/'manifest.json').read_text()) for a in ARMS}
    if any(manifests[a] != manifests[ARMS[0]] for a in ARMS):
        raise ValueError('Arms have different samples/prompts')
    manifest = manifests[ARMS[0]]
    tasks = manifest['tasks']
    names = [t['task_id'] for t in tasks]
    data = {a: {n: np.load(root/a/f'{n}.npz') for n in names} for a in ARMS}
    metrics, decisions = [], []
    for arm in ARMS:
        unit_means = {n: normalize(data[arm][n]['decision']).mean(0) for n in names}
        full = json.loads((args.run/'train'/arm/'valid_step002040_full.json').read_text())
        for task in tasks:
            name = task['task_id']
            d = data[arm][name]
            labels, subjects, probs = d['labels'], d['subjects'], d['probabilities']
            if not np.array_equal(d['ids'], data[ARMS[0]][name]['ids']):
                raise ValueError('Window ordering mismatch')
            if not np.allclose(d['bsr_mean'], data[ARMS[0]][name]['bsr_mean'], atol=1e-5, rtol=1e-5):
                raise ValueError('Frozen BSR differs between arms')
            count = len(task['classes'])
            pred = probs.argmax(1)
            cm = confusion_matrix(labels, pred, labels=np.arange(count))
            bacc = float(np.mean(np.diag(cm)/cm.sum(1)))
            original = np.load(args.run/'train'/arm/f'valid_{name}_predictions.npz')
            original_index = {key: i for i, key in enumerate(original['ids'])}
            original_probs = original['probabilities'][[original_index[key] for key in d['ids']]]
            original_pred = original_probs.argmax(1)
            original_cm = confusion_matrix(labels, original_pred, labels=np.arange(count))
            null_pred = int(d['null_probabilities'].argmax(1)[0])
            summary = dict(arm=arm, dataset=name, classes=count, n=len(labels),
                subjects=len(set(subjects)), accuracy=float((labels==pred).mean()), balanced_accuracy=bacc,
                chance_adjusted_bacc=(bacc-1/count)/(1-1/count),
                full_validation_balanced_accuracy=full['datasets'][name]['balanced_accuracy'],
                original_same_subset_bacc=float(np.mean(np.diag(original_cm)/original_cm.sum(1))),
                saved_prediction_disagreements=int((pred!=original_pred).sum()),
                modal_prediction_fraction=float(np.bincount(pred, minlength=count).max()/len(pred)),
                prediction_counts=np.bincount(pred, minlength=count).tolist(), confusion=cm.tolist(),
                mean_confidence=float(probs.max(1).mean()),
                probability_rms_change=float(np.sqrt(np.mean((probs-probs.mean(0))**2))),
                null_slot_prediction=null_pred, null_slot_agreement=float((pred==null_pred).mean()),
                mean_token_cosine=float(np.nanmean(d['token_cosine'])) if count and np.isfinite(d['token_cosine']).any() else None)
            other_dist = float(np.mean([1-unit_means[name]@unit_means[other] for other in names if other!=name]))
            for stage in STAGES:
                values = geometry(d[stage], labels, subjects)
                metric = dict(arm=arm, dataset=name, stage=stage, **values)
                if stage == 'decision':
                    summary.update(values)
                    summary['between_conditions_cosine'] = other_dist
                    summary['within_over_between_condition'] = values['within_condition_cosine']/other_dist
                metrics.append(metric)
            decisions.append(summary)
            print(arm, name, f'BAcc {bacc:.3f}', 'gap', summary['class_gap'], flush=True)
    (root/'geometry.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    (root/'decisions.json').write_text(json.dumps(decisions, indent=2, allow_nan=False)+'\n')
    for filename, rows in [('geometry.csv', metrics), ('decisions.csv', decisions)]:
        with (root/filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    figure_meta = {}
    def save(fig, name, caption):
        for extension in ['png', 'svg']:
            fig.savefig(figures/f'{name}.{extension}', dpi=180, facecolor='white')
        figure_meta[name] = dict(caption=caption, inches=fig.get_size_inches().tolist(), dpi=180)
        plt.close(fig)
    def matrix(ax, values, rownames, colnames, title, fmt='.2f', **kwargs):
        values = np.asarray(values, dtype=float)
        im = ax.imshow(values, aspect='auto', **kwargs)
        ax.set_yticks(range(len(rownames)), rownames)
        ax.set_xticks(range(len(colnames)), colnames, rotation=25, ha='right')
        ax.set_title(title, fontsize=11)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                rgb = np.array(im.cmap(im.norm(value))[:3])
                linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
                luminance = float(linear@np.array([.2126, .7152, .0722]))
                color = 'black' if luminance > .179 else 'white'
                ax.text(j, i, format(value, fmt) if np.isfinite(value) else 'NA',
                        ha='center', va='center', fontsize=8, color=color)
        return im
    lookup = {(m['arm'], m['dataset'], m['stage']): m for m in metrics}
    decision_lookup = {(m['arm'], m['dataset']): m for m in decisions}
    labels = [DISPLAY[n]+f' ({len(t["classes"])}c)' for n, t in zip(names, tasks)]
    with plt.rc_context({'font.size': 9, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), layout='constrained')
        specifications = [
            ('within_condition_cosine', 'Within-condition distance\n(smaller = tighter)', '.1e', 'viridis'),
            ('within_over_between_condition', 'Within / between conditions\n(smaller = task-dominated geometry)', '.1e', 'viridis'),
            ('class_gap', 'Class separation gap\n(0 = no mean-distance separation)', '.3f', 'RdBu_r'),
            ('balanced_accuracy', 'Diagnostic subset BAcc\n(not full validation)', '.3f', 'viridis')]
        for ax, (key, title, fmt, cmap) in zip(axes, specifications):
            values = np.array([[decision_lookup[a,n][key] for a in ARMS] for n in names])
            if key == 'class_gap':
                bound = max(abs(values.min()), abs(values.max()), .01)
                opts = dict(norm=TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound))
            else:
                opts = dict(vmin=0., vmax=max(float(values.max()), .01))
            im = matrix(ax, values, labels, ARMS, title, fmt, cmap=cmap, **opts)
            fig.colorbar(im, ax=ax, shrink=.6)
        fig.suptitle('Decision-state geometry: original 5120 dimensions | identical validation windows')
        save(fig, 'geometry_overview', 'High-dimensional cosine distances and class separation versus subset BAcc. No 2D projection enters these scores.')

        fig, axes = plt.subplots(1, 4, figsize=(18, 6), layout='constrained')
        bound = max(.01, max(abs(m['class_gap']) for m in metrics if m['class_gap'] is not None))
        for ax, arm in zip(axes, ARMS):
            values = [[lookup[arm,n,s]['class_gap'] for s in STAGES] for n in names]
            im = matrix(ax, values, labels, ['BSR mean','Readout mean','All readout','Connector mean','Decision'],
                        arm, '.3f', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
        fig.colorbar(im, ax=axes, shrink=.6, label='(between-class - within-class) / max(between, within)')
        fig.suptitle('Where is class separation present? | all scores use original dimensions')
        save(fig, 'separation_by_stage', 'Class-macro within/between cosine gap at five extraction points. All readout flattens ordered slots, not just their average. BSR mean cannot summarize all patch information.')

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout='constrained')
        for ax, key, title in zip(axes[:2], ['class_macro_silhouette', 'subject_controlled_gap'],
                ['Class-macro silhouette\n(nearest competing class)', 'Class gap within the same subject\n(eligible subjects only; NA is not zero)']):
            values = np.array([[decision_lookup[a,n][key] for a in ARMS] for n in names], dtype=float)
            bound = max(float(np.nanmax(abs(values))), .01)
            rownames = labels if key == 'class_macro_silhouette' else [f'{DISPLAY[n]} (s={decision_lookup[ARMS[0],n]["eligible_subjects"]})' for n in names]
            im = matrix(ax, values, rownames, ARMS, title, '.3f',
                        cmap=plt.colormaps['RdBu_r'].with_extremes(bad='#dddddd'),
                        norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
            fig.colorbar(im, ax=ax, shrink=.6)
        values = [[decision_lookup[a,n]['mean_token_cosine'] for a in ARMS[2:]] for n in names]
        im = matrix(axes[2], values, labels, ARMS[2:], 'Within-window query-slot similarity\n(32 slots; higher = more similar directions)',
                    '.5f', cmap='viridis', vmin=.98, vmax=1.)
        fig.colorbar(im, ax=axes[2], shrink=.6, label='Cosine similarity; displayed range 0.98 to 1.00')
        fig.suptitle('Alternative explanations: local class overlap, subject effects, query redundancy')
        save(fig, 'subject_and_slot_diagnostics', 'Nearest-class silhouette, equally weighted eligible-subject class gaps, and mean off-diagonal similarity of 32 readout slots. Subject controls are undefined for subject-fixed diagnostic labels. Slot similarity alone does not prove identical information.')

        # Common PCA coordinates across all checkpoints, never four unrelated bases.
        blocks = [normalize(data[a][n]['decision']).astype(np.float32) for a in ARMS for n in names]
        xy, variance, model = pca2(np.concatenate(blocks))
        np.savez_compressed(root/'global_pca.npz', coordinates=xy, variance=variance,
                            components=model.components_, mean=model.mean_)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained', sharex=True, sharey=True)
        offset = 0
        for ax, arm in zip(axes.flat, ARMS):
            for j, n in enumerate(names):
                length = len(data[arm][n]['labels'])
                z = xy[offset:offset+length]; offset += length
                ax.scatter(z[:,0], z[:,1], s=9, alpha=.45, color=COLORS[j], marker=MARKERS[j], label=DISPLAY[n])
            ax.set_title(arm); ax.set_xlabel('Common PC1'); ax.set_ylabel('Common PC2')
        axes.flat[-1].legend(ncol=3, fontsize=8, loc='best')
        fig.suptitle(f'Decision states by task: one common PCA | variance {variance.sum():.1%}\nL2-normalized; projection is descriptive, not a separability test')
        save(fig, 'decision_pca_tasks', 'One PCA fit jointly to all four checkpoints; identical axes. Colors and markers identify datasets, not task classes.')

        per_task = {}
        for n in names:
            blocks = [normalize(data[a][n]['decision']).astype(np.float32) for a in ARMS]
            # Remove only arm/task mean; preserve the scale of within-task variation.
            blocks = [x-x.mean(0) for x in blocks]
            projected, variance, model = pca2(np.concatenate(blocks))
            per_task[n] = (projected, variance)
            np.savez_compressed(root/f'within_{n}_pca.npz', coordinates=projected, variance=variance,
                                components=model.components_, mean=model.mean_)
        for ai, arm in enumerate(ARMS):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15), layout='constrained')
            for ax, task in zip(axes.flat, tasks):
                n = task['task_id']; y = data[arm][n]['labels']
                all_xy, variance = per_task[n]
                z = all_xy[ai*len(y):(ai+1)*len(y)]
                for ci, cls in enumerate(task['classes']):
                    mask = y == ci
                    name = cls['option']
                    if n == 'tuev':
                        name = ['Spike/sharp','Generalized PED','Lateralized PED','Eye movement','Artifact','Background'][ci]
                    ax.scatter(z[mask,0], z[mask,1], s=15, alpha=.65, color=COLORS[ci], marker=MARKERS[ci], label=f'{ci}: {name}')
                spread = np.ptp(all_xy, axis=0)
                ax.set_xlim(all_xy[:,0].min()-.05*spread[0], all_xy[:,0].max()+.05*spread[0])
                ax.set_ylim(all_xy[:,1].min()-.05*spread[1], all_xy[:,1].max()+.05*spread[1])
                ax.set_title(f'{DISPLAY[n]} | 2-PC variance {variance.sum():.0%}')
                ax.set_xlabel('Task-specific PC1'); ax.set_ylabel('Task-specific PC2')
                ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
            fig.suptitle(f'{arm}: classes within each condition | arm/task mean removed\nEach task has its own PCA, shared across checkpoints; axes identical for that task across arms')
            save(fig, f'within_task_classes_{arm}', 'Class-labelled validation windows after arm/task centering. Coordinate bases differ across tasks but are shared across checkpoints for each task. Centering is for display only.')

        texts = {a: np.load(root/a/'text.npz') for a in ARMS}
        values = {a: normalize(texts[a]['condition'])@normalize(texts[a]['condition']).T for a in ARMS}
        minimum = min(v.min() for v in values.values())
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
        for ax, arm in zip(axes.flat, ARMS):
            im = matrix(ax, values[arm], [DISPLAY[n] for n in names], [DISPLAY[n] for n in names], arm,
                        '.2f', cmap='viridis', vmin=minimum, vmax=1.)
        fig.colorbar(im, ax=axes, shrink=.6, label='Cosine similarity (scale shown; not a CLIP similarity)')
        fig.suptitle('Text-only condition: final [decision] state, no EEG slots\nSame template and canonical candidate list; not a sentence-embedding benchmark')
        save(fig, 'text_condition_similarity', 'Contextual text-only task-condition similarity. Shared prompt structure and decoder anisotropy may produce high similarity; this alone does not establish text semantic failure.')

        candidate_means = []
        for task in tasks:
            indices = [i for i,c in enumerate(manifest['candidate_labels']) if c['dataset']==task['task_id']]
            row=[]
            for a in ARMS:
                u=normalize(texts[a]['candidate'][indices]); d=1-u@u.T
                row.append(float(d[np.triu_indices(len(indices),1)].mean()))
            candidate_means.append(row)
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        im = matrix(ax, candidate_means, labels, ARMS, 'Contextual candidate-text distance\nA diagnostic prompt, not a trained contrastive embedding', '.3f', cmap='viridis', vmin=0.)
        fig.colorbar(im, ax=ax, label='Mean pairwise cosine distance among candidate texts')
        save(fig, 'candidate_text_distance', 'Task/question plus one candidate per input, with fixed trailing representation marker. Actual EEG forwards never receive the true candidate as input.')

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained', sharex=True, sharey=True)
        for ax, arm in zip(axes.flat, ARMS):
            for i, n in enumerate(names):
                d=decision_lookup[arm,n]
                ax.scatter(d['class_gap'],d['chance_adjusted_bacc'],color=COLORS[i],marker=MARKERS[i],s=50)
                ax.annotate(DISPLAY[n], (d['class_gap'],d['chance_adjusted_bacc']), xytext=(4,4),textcoords='offset points',fontsize=8)
            ax.axhline(0,color='gray',linestyle='--'); ax.axvline(0,color='gray',linestyle=':')
            ax.set_title(arm); ax.set_xlabel('High-dimensional class separation gap')
            ax.set_ylabel('(BAcc - chance) / (1 - chance)')
        fig.suptitle('Exploratory association, not causal evidence | nine dependent task observations per checkpoint')
        save(fig, 'separation_vs_performance', 'Chance-adjusted diagnostic subset BAcc versus high-dimensional class separation. Different tasks, class counts, subjects and data distributions confound causal interpretation.')

        for arm in ARMS:
            fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout='constrained')
            for ax, task in zip(axes.flat,tasks):
                n=task['task_id']; d=decision_lookup[arm,n]; cm=np.array(d['confusion'])
                normalized=cm/cm.sum(1,keepdims=True)
                im=matrix(ax,normalized,[f'{i} (n={k})' for i,k in enumerate(cm.sum(1))],list(range(len(cm))),
                          f'{DISPLAY[n]} | BAcc {d["balanced_accuracy"]:.1%}', '.2f',cmap='viridis',vmin=0.,vmax=1.)
                ax.set_xlabel('Predicted class index'); ax.set_ylabel('True class index')
            fig.colorbar(im,ax=axes,shrink=.5,label='Recall fraction per true class')
            fig.suptitle(f'{arm}: diagnostic subset confusion matrices | class names in manifest.json')
            save(fig,f'confusion_{arm}', 'Row-normalized confusion matrices; labels include true-class counts, indices map to manifest class options.')
    (root/'figure_manifest.json').write_text(json.dumps(dict(figures=figure_meta,
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=42,
        numpy=np.__version__, matplotlib=matplotlib.__version__,
        pca='L2-normalized; randomized SVD; fit only for descriptive visualization',
        geometry='Original dimensions; cosine and unnormalized Euclidean separation; descriptive, no significance claims',
        candidate_text_mean_distances=candidate_means),indent=2)+'\n')


if __name__ == '__main__':
    main()
