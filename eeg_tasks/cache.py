"""Frozen REVE feature shards, plus frozen Qwen text prototypes."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from .data import config,write_json

@torch.no_grad()
def features(shard):
    torch.set_num_threads(4);cfg=config();root=Path(cfg['output']);base=json.loads(Path(cfg['base_config']).read_text())
    assert (root/'data.done.json').exists()
    out=root/'features';out.mkdir(exist_ok=True)
    if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
    x=np.load(root/'signals.npy',mmap_mode='r');ids=np.arange(shard,len(x),4)
    model,info=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.eval().requires_grad_(False).cuda()
    bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
    pos=bank(cfg['channels']).detach().cuda()
    results=[];start=time.monotonic()
    for begin in range(0,len(ids),64):
        batch=torch.from_numpy(np.array(x[ids[begin:begin+64]])).cuda()
        stride=model.config.patch_size-model.config.patch_overlap;tail=(-(batch.shape[-1]-model.config.patch_size))%stride
        batch=torch.nn.functional.pad(batch,(0,tail),mode='replicate')
        z=model(batch,pos[None].expand(len(batch),-1,-1))
        assert z.shape[1]==3 and z.shape[-1]==512
        results.append(z.mean(2).flatten(1).float().cpu().numpy())
        if begin%1024==0:print(shard,begin,len(ids),flush=True)
    np.savez(out/f'{shard}.npz',indices=ids,features=np.concatenate(results))
    write_json(out/f'{shard}.done.json',{'n':len(ids),'seconds':time.monotonic()-start})

@torch.no_grad()
def text():
    from eeg_align.model import load_text_only
    torch.set_num_threads(4);cfg=config();root=Path(cfg['output']);base=json.loads(Path(cfg['base_config']).read_text())
    tok=AutoTokenizer.from_pretrained(base['qwen_checkpoint'],local_files_only=True)
    model,_,audit=load_text_only(base['qwen_checkpoint'],[tok.encode('A',add_special_tokens=False)[0],tok.encode('B',add_special_tokens=False)[0]],'cuda:0',torch.bfloat16)
    model.eval().requires_grad_(False)
    texts=[s for t in cfg['tasks'] for s in t['texts']];vectors=[]
    for s in texts:
        ids=torch.tensor(tok.encode(s,add_special_tokens=False),device='cuda')[None]
        z=model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state.float().mean(1)
        vectors.append(torch.nn.functional.normalize(z,dim=-1).cpu())
    vectors=torch.cat(vectors);np.save(root/'text_prototypes.npy',vectors.numpy())
    write_json(root/'text_prototypes.json',{'texts':texts,'method':'frozen Qwen3VL language backbone, plain text, mean of final token hidden states, L2 normalize; not a dedicated embedding model','checkpoint':base['qwen_checkpoint'],'cosine_matrix':(vectors@vectors.T).tolist(),'loading_audit':audit})
    print('text prototypes',vectors.shape,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--text',action='store_true');a=p.parse_args()
    text() if a.text else features(a.shard)
