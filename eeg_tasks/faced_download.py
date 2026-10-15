"""Download a pinned public NEMAR FACED recording and verify annex SHA256."""
import argparse,hashlib,json,re,time,urllib.request
from pathlib import Path
ROOT=Path('artifacts/faced_source_v14/raw')

def fetch(url):
 with urllib.request.urlopen(url,timeout=60) as r:return r.read()

def download(subject):
 ROOT.mkdir(parents=True,exist_ok=True);pin=ROOT/'source_revision.json'
 if not pin.exists():
  revision=json.loads(fetch('https://api.github.com/repos/nemarDatasets/nm000112/commits/main'))['sha'];pin.write_text(json.dumps({'repository':'https://github.com/nemarDatasets/nm000112','revision':revision},indent=2))
 revision=json.loads(pin.read_text())['revision'];base=f'https://raw.githubusercontent.com/nemarDatasets/nm000112/{revision}/';sub=f'sub-{subject:03d}';prefix=f'{sub}_task-watchingVideoClips';folder=ROOT/sub;folder.mkdir(exist_ok=True)
 if (folder/'download.done.json').exists():raise FileExistsError(folder)
 for filename in ['README.md','dataset_description.json']:
  path=ROOT/filename
  if not path.exists():path.write_bytes(fetch(base+filename))
 for suffix in ['channels.tsv','eeg.json','events.tsv','events.json']:
  filename=f'{prefix}_{suffix}';(folder/filename).write_bytes(fetch(base+f'{sub}/eeg/{filename}'))
 filename=f'{prefix}_eeg.bdf';pointer=fetch(base+f'{sub}/eeg/{filename}').decode().strip();key=pointer.split('/')[-1];match=re.fullmatch(r'SHA256E-s(\d+)--([0-9a-f]{64})\.bdf',key)
 if not match:raise ValueError('Unexpected annex pointer')
 size,digest=int(match[1]),match[2];url='https://nemar.s3.amazonaws.com/nm000112/objects/'+key;dest=folder/filename;part=dest.with_suffix('.part');start=time.time()
 if dest.exists():raise FileExistsError(dest)
 h=hashlib.sha256();count=0
 with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as f:
  assert response.status==200
  while True:
   chunk=response.read(4*1024*1024)
   if not chunk:break
   f.write(chunk);h.update(chunk);count+=len(chunk)
   if count%(64*1024*1024)==0:print(subject,count,size,flush=True)
 assert count==size and h.hexdigest()==digest,'Incomplete or corrupt download'
 part.rename(dest);result={'subject':subject,'revision':revision,'url':url,'size':size,'sha256':digest,'seconds':time.time()-start,'license':json.loads((ROOT/'dataset_description.json').read_text())['License']};(folder/'download.done.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=0);a=p.parse_args();assert 0<=a.subject<123;download(a.subject)
