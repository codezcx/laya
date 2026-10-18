"""Balanced grouped sampling with explicit coverage rounds and deterministic rank slices."""
import math
import copy
import hashlib
import json
import numpy as np

class BalancedCoverageSampler:
    def __init__(self,groups,per_group=8,seed=42):
        if not groups:raise ValueError('Nonempty groups required')
        self.counts={k:per_group[k] if isinstance(per_group,dict) else per_group for k in groups}
        if any(not isinstance(n,int) or n<1 for n in self.counts.values()):raise ValueError('Positive integer draws required')
        if isinstance(per_group,dict) and set(per_group)!=set(groups):raise ValueError('Draw counts must match groups')
        self.groups={k:np.array(groups[k],dtype=np.int64,copy=True) for k in sorted(groups)}
        if any(len(v)==0 for v in self.groups.values()):raise ValueError('Empty group')
        flat=np.concatenate(list(self.groups.values()))
        if len(np.unique(flat))!=len(flat):raise ValueError('Indices must be unique across groups')
        self.per_group=per_group;self.rng=np.random.default_rng(seed)
        self.steps_per_round=max(math.ceil(len(v)/self.counts[k]) for k,v in self.groups.items())
        self.step=0;self.round=0;self.queues={};self.positions={}
    def _signature(self):
        payload=[(repr(k),v.tolist()) for k,v in self.groups.items()]
        return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
    def state_dict(self):
        return dict(version=1,groups_sha256=self._signature(),per_group=self.per_group,
                    steps_per_round=self.steps_per_round,step=self.step,round=self.round,
                    rng=copy.deepcopy(self.rng.bit_generator.state),
                    queues=[self.queues[k].tolist() for k in self.groups] if self.queues else [],
                    positions=[self.positions[k] for k in self.groups] if self.positions else [])
    def load_state_dict(self,state):
        if (state.get('version')!=1 or state['groups_sha256']!=self._signature()
                or state['per_group']!=self.per_group or state['steps_per_round']!=self.steps_per_round):
            raise ValueError('Sampler state belongs to a different data partition or batch configuration')
        step,round_=state['step'],state['round']
        if not isinstance(step,int) or step<0 or round_!=math.ceil(step/self.steps_per_round):
            raise ValueError('Invalid sampler progress')
        expected=len(self.groups) if step else 0
        if len(state['queues'])!=expected or len(state['positions'])!=expected:
            raise ValueError('Incomplete sampler queues')
        queues={};positions={}
        for (key,values),queue,pos in zip(self.groups.items(),state['queues'],state['positions']):
            queue=np.asarray(queue,dtype=np.int64)
            if not np.array_equal(np.sort(queue),np.sort(values)) or not isinstance(pos,int) or not 0<=pos<=len(values):
                raise ValueError('Corrupt sampler queue or cursor')
            queues[key]=queue.copy();positions[key]=pos
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.step=step;self.round=round_;self.queues=queues;self.positions=positions;self.rng=rng
    def _reset(self):
        self.round+=1
        self.queues={k:self.rng.permutation(v) for k,v in self.groups.items()}
        self.positions={k:0 for k in self.groups}
    def next_batch(self):
        if self.step%self.steps_per_round==0:self._reset()
        batch=[]
        for key,values in self.groups.items():
            drawn=[]
            count=self.counts[key]
            while len(drawn)<count:
                if self.positions[key]==len(values):
                    self.queues[key]=self.rng.permutation(values);self.positions[key]=0
                n=min(count-len(drawn),len(values)-self.positions[key]);start=self.positions[key]
                drawn.extend(self.queues[key][start:start+n].tolist());self.positions[key]+=n
            batch.append(drawn)
        self.step+=1
        return np.asarray(batch,dtype=np.int64) if len(set(self.counts.values()))==1 else np.concatenate(batch).astype(np.int64)
    @staticmethod
    def rank_batch(grouped,rank,world_size):
        if grouped.ndim!=2 or world_size<1 or not 0<=rank<world_size or grouped.shape[1]%world_size:
            raise ValueError('Group count per batch must divide evenly across ranks')
        per_rank=grouped.shape[1]//world_size
        return grouped[:,rank*per_rank:(rank+1)*per_rank].reshape(-1)
