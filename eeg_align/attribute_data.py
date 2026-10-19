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
    def __init__(self, root, split, include_emotion_class=False):
        self.root = Path(root)
        complete = json.loads((self.root/'complete.json').read_text())
        if complete.get('schema_version') != 2:
            raise ValueError('Expected paired MI schema version 2')
        if complete['source_splits'] != [split]:raise ValueError('Wrong attribute split')
        audit=json.loads((self.root/'verification.json').read_text())
        if audit['status'] != 'passed' or audit['complete_sha256'] != digest(self.root/'complete.json'):
            raise ValueError('Attribute QA audit missing')
        if complete['source_manifest_sha256'] != digest(ROOT/'rows.jsonl'):
            raise ValueError('Token/attribute manifest mismatch')
        source = [json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
        self.specs = json.loads((self.root/'question_catalog.json').read_text())
        self.subjects = {i:(r['dataset'],r['subject']) for i,r in enumerate(source)}
        extra = None
        self.classification_hash = None
        if include_emotion_class:
            extra_path=self.root/'classification_complete.json'
            extra=json.loads(extra_path.read_text())
            if extra['base_sha256'] != digest(self.root/'complete.json') or extra['split'] != split:
                raise ValueError('Classification supplement provenance mismatch')
            if extra['source_manifest_sha256'] != digest(ROOT/'rows.jsonl'):
                raise ValueError('Classification source manifest mismatch')
            self.classification_hash=digest(extra_path)
            self.specs.append(extra['specification'])
        self.datasets = sorted({r['dataset'] for r in source})
        self.choice_counts = np.array([len(s.get('options', OPTIONS)) for s in self.specs])
        self.records = []
        self.groups = {}
        self.root_hash = digest(self.root/'complete.json')
        for q, spec in enumerate(self.specs):
            name = f'questions/{spec["id"]}.jsonl'
            if digest(self.root/name) != (extra['question_sha256'] if spec['id']=='emotion_class' else complete['files'][name]):raise ValueError('Question checksum mismatch')
            with (self.root/name).open() as f:
                for line in f:
                    r = json.loads(line)
                    i = r['provenance']['source_row_index']
                    sr = source[i]
                    if r['split'] != split or sr['split'] != split:raise ValueError('Split leakage')
                    if r['input']['question'] != spec['question'] or r['input']['options'] != spec.get('options', OPTIONS):
                        raise ValueError('Unexpected question or option semantics')
                    if sr['dataset'] != r['provenance']['dataset'] or sr['subject'] != r['provenance']['subject']:
                        raise ValueError('Provenance mismatch')
                    a = 'ABC'.index(r['target']['answer'])
                    if spec['id']=='emotion_class' and (sr['dataset'] not in ['SEED','FACED-NEMAR'] or a!=sr['class_id']):
                        raise ValueError('Emotion label mismatch')
                    ds = self.datasets.index(sr['dataset'])
                    k = len(self.records)
                    self.records.append((i, q, a, ds))
                    self.groups.setdefault((q, ds, a), []).append(k)
        self.records = np.asarray(self.records, dtype=np.int64)
        if len(self.records) != complete['question_records']+(extra['n'] if extra else 0):raise ValueError('QA count mismatch')

    def panel(self, per_group=32, classification_per_group=128):
        rng = np.random.default_rng(20261018)
        return np.array(sorted(i for (q,_,_),values in self.groups.items()
                               for i in (values if per_group>1 and self.specs[q]['id']=='mi_class' else
                                         rng.choice(values,min(classification_per_group if per_group>1 and self.specs[q]['id']=='emotion_class' else per_group,len(values)),replace=False))),dtype=np.int64)


class QuestionTokens:
    def __init__(self, tokenizer, specs, device='cuda'):
        self.ids = {}
        self.counts=[len(s.get('options',OPTIONS)) for s in specs]
        for q, spec in enumerate(specs):
            opts=spec.get('options',OPTIONS)
            for k, perm in enumerate(option_permutations(len(opts))):
                options = '\n'.join(f'{letter}) {opts["ABC"[semantic]]}'
                                    for letter, semantic in zip('ABC', perm))
                suffix = ('\nEEG recording ends.\n[question] ' + spec['question'] +
                          '\n[options]\n' + options + '\n[decision]:')
                self.ids[q, k] = torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), device=device)

    def batch(self, records, permutations):
        suffixes = [self.ids[int(q), int(p)] for q, p in zip(records[:, 1], permutations)]
        labels = [option_permutations(self.counts[int(q)])[int(p)].index(int(a)) for q,a,p in zip(records[:,1],records[:,2],permutations)]
        return suffixes, torch.tensor(labels, device=suffixes[0].device)


def option_permutations(n):
    return list(itertools.permutations(range(n)))


def draw_permutations(data, records, rng, fixed_mi_class=False, fixed_emotion_class=False):
    # Consume identical RNG draws so the matched attribute control is unchanged.
    orders=np.array([rng.integers(len(option_permutations(int(data.choice_counts[q])))) for q in records[:,1]])
    if fixed_mi_class:
        orders[[data.specs[q]['id']=='mi_class' for q in records[:,1]]]=0
    if fixed_emotion_class:
        orders[[data.specs[q]['id']=='emotion_class' for q in records[:,1]]]=0
    return orders


def semantic_predictions(letter_predictions, permutations, counts=None):
    if counts is None:counts=[3]*len(letter_predictions)
    return np.array([option_permutations(int(n))[int(p)][int(a)] for a,p,n in zip(letter_predictions,permutations,counts)])


def mask_choices(logits,counts):
    counts=torch.as_tensor(counts,device=logits.device)
    return logits.masked_fill(torch.arange(logits.shape[1],device=logits.device)[None]>=counts[:,None],float('-inf'))


def training_weights(data,draw_counts=None,attribute_loss_weight=1.):
    category=lambda q: data.specs[q]['id'] if data.specs[q]['id'] in ['mi_class','emotion_class'] else data.specs[q].get('task','emotion')
    joint=any(s['id']=='emotion_class' for s in data.specs)
    masses={'mi_class':.4,'mi':.1,'emotion_class':.4,'emotion':.1} if joint else {'mi_class':.25,'mi':.25,'emotion':.5}
    if attribute_loss_weight not in (0.,1.):raise ValueError("Expected attribute loss weight 0 or 1")
    for c in ["mi","emotion"]:masses[c]*=attribute_loss_weight
    if draw_counts is None:draw_counts={k:1 for k in data.groups}
    groups={c:sum(n for (q,_,_),n in draw_counts.items() if category(q)==c) for c in masses}
    batch=sum(draw_counts.values())
    return np.array([batch*masses[category(q)]/groups[category(q)] for q in range(len(data.specs))])


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
            n=len(spec.get('options',OPTIONS))
            confusion = np.zeros((n, n), dtype=int)
            np.add.at(confusion, (rows[ii, 2], predictions[ii]), 1)
            counts = confusion.sum(1)
            recall = np.divide(confusion.diagonal(), counts, out=np.full(n, np.nan), where=counts>0)
            datasets[data.datasets[ds]] = dict(n=int(ii.sum()), accuracy=float(correct[ii].mean()),
                balanced_accuracy=float(np.nanmean(recall)),
                recalls=[None if np.isnan(v) else float(v) for v in recall], confusion=confusion.tolist(),
                majority_baseline=float(counts.max()/counts.sum()))
        result[spec['id']] = dict(n=int(sel.sum()), accuracy=float(correct[sel].mean()),
            balanced_accuracy=float(np.mean([v['balanced_accuracy'] for v in datasets.values()])),
            datasets=datasets)
    task = {name:float(np.mean([v['balanced_accuracy'] for key, v in result.items() if key.startswith(name+'_') and key not in ['mi_class','emotion_class']]))
            for name in ['mi','emotion']}
    classification=result.get('mi_class',{}).get('balanced_accuracy')
    score=.25*task['mi']+.25*classification+.5*task['emotion'] if classification is not None else float(np.mean(list(task.values())))
    emotion_classification=result.get('emotion_class',{}).get('balanced_accuracy')
    if emotion_classification is not None and classification is not None:
        score=.5*classification+.5*emotion_classification
    return dict(n=len(ids), questions=result, MI=task['mi'], MI_classification=classification,
                emotion_classification=emotion_classification, emotion=task['emotion'],
                selection_score=score, accuracy=float(correct.mean()))
