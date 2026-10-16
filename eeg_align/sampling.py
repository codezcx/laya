"""Balanced grouped sampling with explicit coverage rounds and deterministic rank slices."""
import math
import numpy as np

class BalancedCoverageSampler:
    def __init__(self,groups,per_group=8,seed=42):
        if not groups or per_group<1:raise ValueError('Nonempty groups and positive per_group required')
        self.groups={k:np.asarray(groups[k],dtype=np.int64) for k in sorted(groups)}
        if any(len(v)==0 for v in self.groups.values()):raise ValueError('Empty group')
        flat=np.concatenate(list(self.groups.values()))
        if len(np.unique(flat))!=len(flat):raise ValueError('Indices must be unique across groups')
        self.per_group=per_group;self.rng=np.random.default_rng(seed)
        self.steps_per_round=max(math.ceil(len(v)/per_group) for v in self.groups.values())
        self.step=0;self.round=0;self.queues={};self.positions={}
    def _reset(self):
        self.round+=1
        self.queues={k:self.rng.permutation(v) for k,v in self.groups.items()}
        self.positions={k:0 for k in self.groups}
    def next_batch(self):
        if self.step%self.steps_per_round==0:self._reset()
        batch=[]
        for key,values in self.groups.items():
            drawn=[]
            while len(drawn)<self.per_group:
                if self.positions[key]==len(values):
                    self.queues[key]=self.rng.permutation(values);self.positions[key]=0
                n=min(self.per_group-len(drawn),len(values)-self.positions[key]);start=self.positions[key]
                drawn.extend(self.queues[key][start:start+n].tolist());self.positions[key]+=n
            batch.append(drawn)
        self.step+=1
        return np.asarray(batch,dtype=np.int64)
    @staticmethod
    def rank_batch(grouped,rank,world_size):
        if grouped.ndim!=2 or world_size<1 or not 0<=rank<world_size or grouped.shape[1]%world_size:
            raise ValueError('Group count per batch must divide evenly across ranks')
        per_rank=grouped.shape[1]//world_size
        return grouped[:,rank*per_rank:(rank+1)*per_rank].reshape(-1)
