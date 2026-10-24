"""Encode new task instructions using LEAF's exact MPNet mean-pooling recipe."""
import sys,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_align.benchmark_data import catalog

def main():
    torch.set_num_threads(4)
    folder=ROOT/'artifacts/benchmark/mpnet'
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModel.from_pretrained(folder,local_files_only=True).eval().requires_grad_(False)
    cached_text=(ROOT.parent/'LEAF/text_embeddings/mpnet-base.txt').read_text().splitlines()
    cached=np.load(ROOT.parent/'LEAF/text_embeddings/mpnet-base.npy')
    texts=list(dict.fromkeys(t['instruction'] for t in catalog()))
    with torch.inference_mode():
        x=tokenizer(texts,padding=True,return_tensors='pt');h=model(**x).last_hidden_state
        mask=x['attention_mask'][...,None];emb=F.normalize((h*mask).sum(1)/mask.sum(1),dim=-1).numpy()
    check={s:float(emb[i]@cached[cached_text.index(s)]) for i,s in enumerate(texts) if s in cached_text}
    assert min(check.values())>.9999,check
    out=ROOT/'artifacts/benchmark';np.save(out/'instructions.npy',emb)
    (out/'instructions.json').write_text(json.dumps(dict(texts=texts,existing_cache_cosine=check,model='sentence-transformers/all-mpnet-base-v2',pooling='masked mean then L2 normalization'),indent=2))
    print(check)
if __name__=='__main__':main()
