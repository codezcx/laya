import requests,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
root=Path('artifacts/models/Qwen3-Embedding-0.6B');root.mkdir(parents=True,exist_ok=True)
record=dict(repo='Qwen/Qwen3-Embedding-0.6B',revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
(root/'download_provenance.json').write_text(json.dumps(record,indent=2))
info=HfApi().model_info(record['repo'],revision=record['revision'],files_metadata=True)
names={'config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','README.md','modules.json','sentence_bert_config.json','config_sentence_transformers.json'}
for item in info.siblings:
 if item.rfilename not in names:continue
 p=root/item.rfilename
 if p.exists() and p.stat().st_size==item.size:continue
 r=requests.get(f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{item.rfilename}",timeout=180);r.raise_for_status()
 assert len(r.content)==item.size
 p.write_bytes(r.content)
f=next(f for f in info.siblings if f.rfilename=='model.safetensors');size=f.size
url=f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/model.safetensors"
chunk=16*1024*1024
folder=root/'ranges';folder.mkdir(exist_ok=True)
def get(i):
 start=i*chunk;b=min(size,(i+1)*chunk)-1;p=folder/str(i)
 a=start
 if p.exists() and p.stat().st_size==b-a+1:return p
 for retry in range(12):
  try:
   tmp=p.with_suffix('.part');a=start+(tmp.stat().st_size if tmp.exists() else 0)
   with requests.get(url+f'?range_part={i}&offset={a}',headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(30,120)) as r:
    r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {a}-{b}/'),dict(r.headers)
    tmp=p.with_suffix('.part')
    with tmp.open('ab') as out:
     for block in r.iter_content(1024*1024):out.write(block)
    assert tmp.stat().st_size==b-start+1
    tmp.replace(p);print('part',i,'done',flush=True);return p
  except Exception as e:print('retry',i,retry,str(e)[:150],flush=True);time.sleep(2)
 raise RuntimeError(i)
p=root/'model.safetensors'
valid=p.exists() and p.stat().st_size==size and hashlib.sha256(p.read_bytes()).hexdigest()==f.lfs.sha256
if not valid:
 with ThreadPoolExecutor(max_workers=16) as pool:parts=list(pool.map(get,range((size+chunk-1)//chunk)))
 temp=root/'model.safetensors.assembled';h=hashlib.sha256()
 with temp.open('wb') as out:
  for part in parts:
   data=part.read_bytes();out.write(data);h.update(data)
 assert h.hexdigest()==f.lfs.sha256
 temp.replace(p)
records=[]
for item in info.siblings:
 p=root/item.rfilename
 if not p.is_file() or p.name.startswith('download_'):continue
 sha=hashlib.sha256(p.read_bytes()).hexdigest();assert p.stat().st_size==item.size
 if item.lfs:assert sha==item.lfs.sha256
 records.append(dict(file=item.rfilename,bytes=p.stat().st_size,sha256=sha))
(root/'download_complete.json').write_text(json.dumps(dict(**record,files=records),indent=2))
print('COMPLETE',flush=True)
