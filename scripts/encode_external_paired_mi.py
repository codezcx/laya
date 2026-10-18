"""Encode audited external pairs with frozen REVE and verify against a source cache anchor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from eeg_align.cache import digest
from eeg_align.paired_cache import PairedTokenCache


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('artifacts/jev_paired_qa/external_2b'))
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.15,torch.device(args.device))
    manifest=args.input/'complete.json';prepared=json.loads(manifest.read_text())
    for name,sha in prepared['files'].items():assert digest(args.input/name)==sha
    assert prepared['shape']==[2840,2,3,400]
    out=args.input/'encoded';out.mkdir(exist_ok=False)
    cfg=json.loads(Path('configs/encoder.json').read_text())
    model,info=AutoModel.from_pretrained(cfg['reve_checkpoint'],trust_remote_code=True,
        local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to(args.device).eval().requires_grad_(False)
    bank=AutoModel.from_pretrained(cfg['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
    positions=bank(['C3','Cz','C4']).detach().to(args.device)
    assert model.config.embed_dim==512 and model.config.patch_size==200 and model.config.patch_overlap==20

    def encode(pair):
        x=torch.as_tensor(np.array(pair),device=args.device).reshape(-1,3,400)
        x=torch.nn.functional.pad(x,(0,160),mode='replicate')
        z=model(x,positions[None].expand(len(x),-1,-1))
        assert z.shape==(len(x),3,3,512) and torch.isfinite(z).all()
        return z.reshape(len(pair),18,512).cpu().numpy()

    # Known source pair checks segment/token ordering and compatibility without target labels.
    source=Path('datasets/eeg_attribute_qa')
    anchor=json.loads((source/'mi_sources.jsonl').read_text().splitlines()[0])
    pair=np.stack([np.load(source/f'signals/mi_{s}.npy',mmap_mode='r')[anchor['signal_index']]
                   for s in ['baseline','task']])[None]
    current=encode(pair)
    cached,mask,cuts=PairedTokenCache().batch([anchor['source_row_index']],'cpu')
    assert cuts==[9] or list(cuts)==[9]
    assert int(mask.sum())==18
    reference=cached[:,:18].numpy().astype(np.float32)
    rel=float(np.linalg.norm(current-reference)/np.linalg.norm(reference))
    assert rel<.001,('Source cache compatibility failed',rel)
    pairs=np.load(args.input/'pairs.npy',mmap_mode='r')
    tokens=np.lib.format.open_memmap(out/'tokens.npy',mode='w+',dtype='float16',shape=(len(pairs),18,512))
    for start in range(0,len(pairs),32):
        tokens[start:start+32]=encode(pairs[start:start+32]).astype(np.float16)
        if start%640==0:print('External REVE encoding',start,len(pairs),flush=True)
    tokens.flush()
    assert np.isfinite(tokens).all()
    del tokens
    checkpoint_files={str(f):digest(f) for folder in cfg.values() for f in Path(folder).iterdir()
                      if f.suffix in ['.json','.py','.safetensors']}
    report=dict(n=2840,shape=[2840,18,512],layout='baseline9 then task9; channels C3,Cz,C4; three patches/channel',
        encoder_config=cfg,encoder_files_sha256=checkpoint_files,input_complete_sha256=digest(manifest),
        source_anchor_index=anchor['source_row_index'],source_anchor_relative_l2_error=rel,
        source_cache_audit_sha256=digest('artifacts/jev_paired_qa/tokens/verified.json'),
        tokens_sha256=digest(out/'tokens.npy'),code_sha256=digest(__file__),
        status='frozen tokens ready; no target model evaluation or training performed')
    (out/'complete.json').write_text(json.dumps(report,indent=2))
    print('Complete; source anchor relative L2 error:',rel,flush=True)


if __name__=='__main__':main()
