"""Frozen support profiles: split-half repeatability and source record transfer.

Target query embeddings are never indexed. No fitted subject classifier, no
target-driven descriptor selection, and no trial-level significance claims.
"""
import csv
import itertools
import json
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .common import CONFIG, OUT, ROOT, PARENT, digest, write_json

DESCRIPTORS = ('tower_scalar', 'connector_scalar', 'tower_featurewise')


def moments(x, featurewise=False):
    x = x.double()
    axes = (1,) if featurewise else (1, 2)
    return torch.stack((x.mean(axes), x.square().mean(axes)), dim=1).numpy()


def profile(values, indices):
    pooled = values[indices].mean(axis=0)
    return np.concatenate((np.atleast_1d(pooled[0]),
                           np.atleast_1d(np.sqrt(np.maximum(pooled[1] - pooled[0]**2, 0)))))


def balanced_halves(labels, ordering, seed=None):
    a, b = [], []
    rng = np.random.default_rng(seed)
    for label in sorted(set(labels)):
        ix = np.array([i for i in ordering if labels[i] == label])
        if seed is not None:
            ix = rng.permutation(ix)
        n = len(ix) // 2
        if not n:
            raise ValueError('Each class needs at least two support samples')
        a.extend(ix[:n].tolist())
        b.extend(ix[-n:].tolist())
    return a, b


def compare(a, b, permutation=False):
    a, b = np.asarray(a), np.asarray(b)
    d = np.sqrt(((a[:, None] - b[None])**2).mean(-1))
    n = len(a)
    if n < 2 or d.shape != (n, n) or not np.isfinite(d).all():
        raise ValueError('Need at least two paired finite subject profiles')
    # Row = reference individual; column = held-out profile. Ties get fractional credit.
    ties = np.isclose(d, d.min(axis=0, keepdims=True), rtol=1e-9, atol=1e-12)
    acc = np.diag(ties) / ties.sum(axis=0)
    off = d[~np.eye(n, dtype=bool)]
    result = dict(subjects=n, chance=1/n, matching_accuracy=float(acc.mean()),
                  same_person_distance=float(np.diag(d).mean()),
                  other_person_distance=float(off.mean()),
                  same_over_other=float(np.diag(d).mean()/off.mean()) if off.mean() else None,
                  per_subject_match_credit=acc.tolist(), distances=d.tolist())
    if permutation:
        costs = [d[np.arange(n), perm].mean() for perm in itertools.permutations(range(n))]
        result['exact_pairing_permutation_p'] = float(np.mean(np.asarray(costs) <= np.diag(d).mean()+1e-12))
        result['permutation_unit'] = 'whole subject profile, not EEG trial'
    return result


def metadata(name, ids):
    wanted = set(ids)
    rows = {}
    path = ROOT / CONFIG['cache_root'] / name / 'index.csv'
    expected = json.loads((PARENT / 'config.json').read_text())['cache_indexes'][name]
    if digest(path) != expected:
        raise RuntimeError('Frozen source index changed')
    with path.open() as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            key = f'{name}:{i}'
            if key in wanted:
                rows[key] = dict(row, id=key, label=int(row['label']))
    if set(rows) != wanted:
        raise RuntimeError('Missing diagnostic IDs')
    return [rows[key] for key in ids]


def load_case(case, source, connector):
    if source:
        ids = case['support_ids'] + case['query_ids']
        path = ROOT / CONFIG['initialization_reference'] / 'jobs' / (case['dataset']+'_source_erm') / 'sources' / case['key'] / 'features.pt'
    else:
        ids = case['support_ids']  # Do not index ANY target-query tensor.
        path = ROOT / CONFIG['matched_reference'] / 'jobs' / case['key'] / 'cases' / case['key'] / 'features.pt'
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if saved['stamp']['parent_sha256'] != CONFIG['parent_sha256']:
        raise RuntimeError('Diagnostic cache parent mismatch')
    loc = {key:i for i,key in enumerate(saved['ids'])}
    if len(loc) != len(saved['ids']):
        raise RuntimeError('Duplicate cache IDs')
    tokens = saved['tokens'][[loc[key] for key in ids]]
    rows = metadata(case['dataset'], ids)
    if any(r['subject'] != case['subject'] or r['split'] != ('train' if source else 'test') for r in rows):
        raise RuntimeError('Diagnostic crossed subject/split boundary')
    values = dict(tower_scalar=moments(tokens), tower_featurewise=moments(tokens, True))
    inputs = []
    with torch.no_grad():
        for part in tokens.split(8):
            first = F.layer_norm(part.float(), (256,), connector['0.weight'], connector['0.bias'])
            second = F.gelu(F.linear(first, connector['1.weight'], connector['1.bias']))
            # Two moments per linear input; retain the moment axis for exact pooling.
            inputs.append(np.stack((moments(first), moments(second)), axis=-1))
    values['connector_scalar'] = np.concatenate(inputs)
    return dict(case=case, rows=rows, values=values,
                provenance=dict(path=str(path), selected_ids=ids, parent_sha256=CONFIG['parent_sha256'],
                                size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))


def summarize_group(items, descriptor, mode, seed=None):
    a, b, indices = [], [], []
    for item in items:
        case, rows = item['case'], item['rows']
        n = len(case['support_ids'])
        labels = [r['label'] for r in rows]
        if mode == 'record_transfer':
            ia, ib = list(range(n)), list(range(n,len(rows)))
        else:
            def chronology(i):
                r = rows[i]
                # 2a starts/events reset each run; array rows retain run order.
                return ((r['record_id'],int(r['row'])//48,int(r['event']))
                        if case['dataset']=='bcic_2a' else (r['record_id'],int(r['start'])))
            order = sorted(range(n), key=chronology)
            ia, ib = balanced_halves(labels[:n], order, seed)
        if Counter(labels[i] for i in ia) != Counter(labels[i] for i in ib):
            raise RuntimeError('Unequal class composition confounds identity matching')
        a.append(profile(item['values'][descriptor], ia))
        b.append(profile(item['values'][descriptor], ib))
        indices.append(dict(subject=case['subject'], a=[rows[i]['id'] for i in ia], b=[rows[i]['id'] for i in ib]))
    return dict(compare(a,b,permutation=mode=='record_transfer'), pairs=indices)


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues)); largest = 0.
    for rank, i in enumerate(order):
        largest = max(largest, min(1.,(len(pvalues)-rank)*pvalues[i]))
        result[i] = largest
    return result.tolist()


def main():
    torch.set_num_threads(3)
    destination = OUT / 'diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / CONFIG['initialization_reference'] / 'plan.json'
    target_path = ROOT / CONFIG['matched_reference'] / 'plan.json'
    source_plan = json.loads(source_path.read_text())
    target_plan = json.loads(target_path.read_text())
    protocol = dict(descriptors=DESCRIPTORS, seed=42, random_splits=100,
        primary='class-balanced chronological split halves; source record transfer',
        source_plan_sha256=digest(source_path), target_plan_sha256=digest(target_path),
        target_ids={c['key']:c['support_ids'] for c in target_plan['cases']},
        target_query_used=False, original_valid_used=False,
        distance='raw RMS Euclidean, no scaling or fitted classifier',
        standard_deviation='population SD pooled over selected trials and tokens',
        repeat_range='resampling sensitivity, not population confidence intervals',
        hmc_transfer='early versus late SAME night; not cross-night validation')
    write_json(destination/'protocol.json', protocol)
    parent = torch.load(PARENT/'best.pt', map_location='cpu', mmap=True, weights_only=False)
    connector = {k.removeprefix('connector.'):v for k,v in parent['trainable'].items() if k.startswith('connector.')}
    groups = {}
    for name in CONFIG['datasets']:
        for kind, cases in [('source',source_plan['sources']),('target',target_plan['cases'])]:
            items = [load_case(c,kind=='source',connector) for c in cases if c['dataset']==name]
            group = dict(subjects=[it['case']['subject'] for it in items], descriptors={},
                         provenance=[it['provenance'] for it in items])
            for descriptor in DESCRIPTORS:
                result = dict(chronological_halves=summarize_group(items,descriptor,'halves'))
                repeated = [summarize_group(items,descriptor,'halves',seed=42+i)['matching_accuracy'] for i in range(100)]
                result['random_halves'] = dict(mean=float(np.mean(repeated)), min=float(np.min(repeated)),
                                              max=float(np.max(repeated)), repeats=100)
                if kind=='source':
                    result['record_transfer'] = summarize_group(items,descriptor,'record_transfer')
                group['descriptors'][descriptor] = result
            groups[name+'_'+kind] = group
            print(json.dumps(dict(group=name+'_'+kind, results={d:{k:v.get('matching_accuracy',v.get('mean')) for k,v in r.items()} for d,r in group['descriptors'].items()})),flush=True)
    records = [v['record_transfer'] for k,g in groups.items() if k.endswith('_source') for v in g['descriptors'].values()]
    for record, adjusted in zip(records,holm_adjust([r['exact_pairing_permutation_p'] for r in records])):
        record['holm_p_across_six_transfer_comparisons'] = adjusted
    write_json(destination/'results.json', dict(protocol=protocol,groups=groups,
        limitations=['Subject recognition is not proof of beneficial personal adaptation.',
                     'Stable recording/montage effects may masquerade as neural individual traits.',
                     'Source subjects were used in shared training; target support repeatability is separate.',
                     'Only two target subjects per task; no population generalization claims.']))
    plot(groups,destination)


def plot(groups,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,name in enumerate(CONFIG['datasets']):
        group=groups[name+'_source']
        for col,descriptor in enumerate(DESCRIPTORS):
            ax=axes[row,col]; result=group['descriptors'][descriptor]['record_transfer']
            im=ax.imshow(result['distances'],cmap='viridis')
            ax.set_xticks(range(len(group['subjects'])),group['subjects'])
            ax.set_yticks(range(len(group['subjects'])),group['subjects'])
            ax.set_xlabel('Held-out record/segment subject'); ax.set_ylabel('Support subject')
            ax.set_title(f'{name}: {descriptor}\nMatch {result["matching_accuracy"]:.0%}; chance {result["chance"]:.1%}',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.7,label='RMS distance')
    fig.suptitle('Class-balanced source profiles: 2a across sessions; HMC within one night')
    fig.savefig(destination/'source_profile_transfer.png',dpi=180)
    fig.savefig(destination/'source_profile_transfer.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
