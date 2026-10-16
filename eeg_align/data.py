"""Verified source token cache and task-separated metrics."""
import json
import numpy as np
import torch
from .cache import ROOT,digest

class TokenCache:
    def __init__(self):
        verified = json.loads((ROOT/'verified.json').read_text())
        assert verified['manifest_sha256'] == digest(ROOT/'rows.jsonl')
        self.rows = [json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
        self.location = {}; self.arrays = []; self.lengths = []
        for shard in range(4):
            folder = ROOT/str(shard)
            self.arrays.append(np.load(folder/'tokens.npy', mmap_mode='r'))
            self.lengths.append(np.load(folder/'lengths.npy'))
            for j,i in enumerate(np.load(folder/'indices.npy')):
                assert int(i) not in self.location
                self.location[int(i)] = (shard,j)
        assert sorted(self.location) == list(range(len(self.rows)))

    def batch(self, ids, device):
        locations = [self.location[int(i)] for i in ids]
        lengths = [int(self.lengths[s][j]) for s,j in locations]
        x = np.stack([self.arrays[s][j,:max(lengths)] for s,j in locations])
        tokens = torch.from_numpy(x.astype(np.float32)).to(device)
        mask = torch.arange(max(lengths),device=device)[None] < torch.tensor(lengths,device=device)[:,None]
        labels = torch.tensor([self.rows[int(i)]['class_id'] for i in ids], device=device)
        return tokens,mask,labels

def metrics(rows, logits):
    y = np.array([r['class_id'] for r in rows]); pred = logits.argmax(1)
    task = (y>=3); routing = (pred>=3)
    oracle = logits.copy()
    oracle[task,:3] = -np.inf; oracle[~task,3:] = -np.inf
    within = oracle.argmax(1); result = {}
    for ds in sorted({r['dataset'] for r in rows}):
        people = []
        for subject in sorted({r['subject'] for r in rows if r['dataset']==ds}):
            ii = np.array([i for i,r in enumerate(rows) if r['dataset']==ds and r['subject']==subject])
            classes = np.unique(y[ii])
            people.append(dict(joint=float(np.mean([np.mean(pred[ii][y[ii]==c]==c) for c in classes])),
                               oracle=float(np.mean([np.mean(within[ii][y[ii]==c]==c) for c in classes])),
                               routing=float(np.mean(routing[ii]==task[ii]))))
        result[ds] = {k:float(np.mean([v[k] for v in people])) for k in ['joint','oracle','routing']}
    tasks = [['SEED','FACED-NEMAR'], ['Cho2017','Dreyer2023','Lee2019_MI']]
    score = float(np.mean([np.mean([result[d]['joint'] for d in names]) for names in tasks]))
    return dict(datasets=result, selection_score=score, n=len(y),
        MI=float(np.mean([result[d]['joint'] for d in tasks[1]])),
        emotion=float(np.mean([result[d]['joint'] for d in tasks[0]])))
