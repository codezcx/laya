"""Frozen task+class text prototypes and symmetric multi-positive EEG/text loss."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather
from transformers import AutoModel, AutoTokenizer

TEXTS = ['Task: motor imagery recognition. Imagined movement: left hand.',
         'Task: motor imagery recognition. Imagined movement: right hand.']
TEXT_MODEL = 'artifacts/models/Qwen3-Embedding-0.6B'


@torch.no_grad()
def text_prototypes(device='cuda'):
    tok=AutoTokenizer.from_pretrained(TEXT_MODEL,local_files_only=True,padding_side='left')
    model,info=AutoModel.from_pretrained(TEXT_MODEL,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to(device).eval().requires_grad_(False)
    inputs=tok(TEXTS,padding=True,return_tensors='pt').to(device)
    # Left-padded sequences end at a real token; follow the local model card.
    vectors=F.normalize(model(**inputs).last_hidden_state[:,-1].float(),dim=-1)
    assert vectors.shape[0]==2 and torch.isfinite(vectors).all()
    del model;torch.cuda.empty_cache()
    return vectors


def multipositive_loss(z,y,text,temperature=.07,distributed=True):
    """Average log probability over ALL same-label positives, in both directions.

    Duplicate class texts remain positive, never false negatives. Every rank
    evaluates the same global loss. Differentiable gather sums backward gradients;
    the caller averages parameter gradients once across ranks.
    """
    z=F.normalize(z.float(),dim=-1)
    if distributed and dist.get_world_size()>1:
        count=torch.tensor([len(z)],device=z.device);counts=[torch.empty_like(count) for _ in range(dist.get_world_size())]
        dist.all_gather(counts,count);sizes=[int(n.item()) for n in counts];width=max(sizes)
        zz=all_gather(F.pad(z,(0,0,0,width-len(z))))
        yy=[torch.empty(width,dtype=y.dtype,device=y.device) for _ in sizes]
        dist.all_gather(yy,F.pad(y,(0,width-len(y))))
        z=torch.cat([a[:n] for a,n in zip(zz,sizes)]);y=torch.cat([a[:n] for a,n in zip(yy,sizes)])
    logits=z@text[y].T/temperature
    positive=y[:,None].eq(y[None,:]).float();target=positive/positive.sum(-1,keepdim=True)
    return -.5*((target*logits.log_softmax(-1)).sum(-1).mean()+(target*logits.T.log_softmax(-1)).sum(-1).mean())
