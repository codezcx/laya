"""Compact question index, randomized option semantics, and source-only metrics."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .cache import ROOT, digest
from eeg_tasks.attribute_questions import OPTIONS

PERMUTATIONS = list(itertools.permutations(range(3)))


class AttributeData:
    def __init__(self, root, split):
        self.root = Path(root)
        complete = json.loads((self.root/'complete.json').read_text())
        if complete.get('schema_version') != 1:
            raise ValueError('Paired MI schema requires new segment-aware REVE tokens and training loader; old single-window cache is incompatible')
        if complete['source_splits'] != [split]:raise ValueError('Wrong attribute split')
        if json.loads((self.root/'verification.json').read_text())['status'] != 'passed':
            raise ValueError('Attribute QA audit missing')
        if complete['source_manifest_sha256'] != digest(ROOT/'rows.jsonl'):
            raise ValueError('Token/attribute manifest mismatch')
        source = [json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
        self.specs = json.loads((self.root/'question_catalog.json').read_text())
        self.datasets = sorted({r['dataset'] for r in source})
        self.records = []
        self.groups = {}
        self.root_hash = digest(self.root/'complete.json')
        for q, spec in enumerate(self.specs):
            name = f'questions/{spec["id"]}.jsonl'
            if digest(self.root/name) != complete['files'][name]:raise ValueError('Question checksum mismatch')
            with (self.root/name).open() as f:
                for line in f:
                    r = json.loads(line)
                    i = r['provenance']['source_row_index']
                    sr = source[i]
                    if r['split'] != split or sr['split'] != split:raise ValueError('Split leakage')
                    if r['input']['question'] != spec['question'] or r['input']['options'] != OPTIONS:
                        raise ValueError('Unexpected question or option semantics')
                    if sr['dataset'] != r['provenance']['dataset'] or sr['subject'] != r['provenance']['subject']:
                        raise ValueError('Provenance mismatch')
                    a = 'ABC'.index(r['target']['answer'])
                    ds = self.datasets.index(sr['dataset'])
                    k = len(self.records)
                    self.records.append((i, q, a, ds))
                    self.groups.setdefault((q, ds, a), []).append(k)
        self.records = np.asarray(self.records, dtype=np.int64)
        if len(self.records) != complete['question_records']:raise ValueError('QA count mismatch')

    def panel(self, per_group=32):
        rng = np.random.default_rng(20261018)
        return np.array(sorted(i for values in self.groups.values()
                               for i in rng.choice(values, min(per_group, len(values)), replace=False)), dtype=np.int64)


class QuestionTokens:
    def __init__(self, tokenizer, specs, device='cuda'):
        self.ids = {}
        for q, spec in enumerate(specs):
            for k, perm in enumerate(PERMUTATIONS):
                options = '\n'.join(f'{letter}) {OPTIONS["ABC"[semantic]]}'
                                    for letter, semantic in zip('ABC', perm))
                suffix = ('\nEEG recording ends.\n[question] ' + spec['question'] +
                          '\n[options]\n' + options + '\n[decision]:')
                self.ids[q, k] = torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), device=device)

    def batch(self, records, permutations):
        suffixes = [self.ids[int(q), int(p)] for q, p in zip(records[:, 1], permutations)]
        labels = [PERMUTATIONS[int(p)].index(int(a)) for a, p in zip(records[:, 2], permutations)]
        return suffixes, torch.tensor(labels, device=suffixes[0].device)


def semantic_predictions(letter_predictions, permutations):
    return np.array([PERMUTATIONS[int(p)][int(a)] for a, p in zip(letter_predictions, permutations)])


def summarize(data, ids, predictions):
    rows = data.records[ids]
    correct = predictions == rows[:, 2]
    result = {}
    for q, spec in enumerate(data.specs):
        sel = rows[:, 1] == q
        if not sel.any():continue
        datasets = {}
        for ds in sorted(set(rows[sel, 3])):
            ii = sel & (rows[:, 3] == ds)
            confusion = np.zeros((3, 3), dtype=int)
            np.add.at(confusion, (rows[ii, 2], predictions[ii]), 1)
            counts = confusion.sum(1)
            recall = np.divide(confusion.diagonal(), counts, out=np.full(3, np.nan), where=counts>0)
            datasets[data.datasets[ds]] = dict(n=int(ii.sum()), accuracy=float(correct[ii].mean()),
                balanced_accuracy=float(np.nanmean(recall)),
                recalls=[None if np.isnan(v) else float(v) for v in recall], confusion=confusion.tolist(),
                majority_baseline=float(counts.max()/counts.sum()))
        result[spec['id']] = dict(n=int(sel.sum()), accuracy=float(correct[sel].mean()),
            balanced_accuracy=float(np.mean([v['balanced_accuracy'] for v in datasets.values()])),
            datasets=datasets)
    task = {name:float(np.mean([v['balanced_accuracy'] for key, v in result.items() if key.startswith(name+'_')]))
            for name in ['mi','emotion']}
    return dict(n=len(ids), questions=result, MI=task['mi'], emotion=task['emotion'],
                selection_score=float(np.mean(list(task.values()))), accuracy=float(correct.mean()))
