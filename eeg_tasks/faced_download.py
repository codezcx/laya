"""Download a pinned public NEMAR FACED recording and verify annex SHA256."""
import argparse,hashlib,json,re,time,urllib.request
from pathlib import Path
ROOT=Path('artifacts/faced_source_v14/raw')

def fetch(url):
 with urllib.request.urlopen(url,timeout=60) as r:return r.read()

def read_range(url,start,end,total):
 request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
 with urllib.request.urlopen(request,timeout=90) as response:
  if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise ValueError('Server did not honor exact byte range')
  chunk=response.read(end-start+2)
 if len(chunk)!=end-start+1:raise IOError('Incomplete or oversized range')
 return chunk

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
 # Each completed range is independently length-checked before appending.
 count=part.stat().st_size if part.exists() else 0
 if count>size:raise ValueError('Partial file is larger than expected object')
 (folder/'download.request.json').write_text(json.dumps({'url':url,'expected_size':size,'expected_sha256':digest,'resume_offset':count,'revision':revision},indent=2))
 with part.open('ab') as f:
  while count<size:
   end=min(size-1,count+4*1024*1024-1);expected=end-count+1
   for attempt in range(3):
    try:
     chunk=read_range(url,count,end,size)
     break
    except Exception:
     if attempt==2:raise
     time.sleep(2**attempt)
   f.write(chunk);f.flush();count+=len(chunk)
   print(subject,count,size,flush=True)
 h=hashlib.sha256()
 with part.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert part.stat().st_size==size and h.hexdigest()==digest,'Incomplete or corrupt download'
 part.rename(dest);result={'subject':subject,'revision':revision,'url':url,'size':size,'sha256':digest,'seconds':time.time()-start,'license':json.loads((ROOT/'dataset_description.json').read_text())['License']};(folder/'download.done.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=0);a=p.parse_args();assert 0<=a.subject<123;download(a.subject)
