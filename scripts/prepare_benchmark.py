"""Export original EEG arrays for mmap loading; no frozen feature computation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_align.benchmark_data import catalog,read_dataset
from eeg_align.leaf_runtime import digest
from eeg_align.signal_data import ROOT as OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for tid,task in enumerate(catalog()):
        ds=task['task_id']
        if a.datasets and ds not in a.datasets:continue
        dest=OUT/ds;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            assert json.loads((dest/'complete.json').read_text())['catalog_sha256']==digest('configs/benchmark_tasks.json');continue
        start=time.monotonic();x,y,subjects,names,fs=read_dataset(task)
        target=np.lib.format.open_memmap(dest/'signals.npy',mode='w+',dtype=np.float32,shape=x.shape)
        accepted=[];excluded=[]
        with (dest/'rows.jsonl').open('w') as rows:
            for offset in range(0,len(x),128):
                end=min(len(x),offset+128);block=np.asarray(x[offset:end],dtype=np.float32);target[offset:end]=block
                good=np.isfinite(block).all(axis=(1,2)) & np.any(block!=0,axis=(1,2))
                for j in range(len(block)):
                    i=offset+j
                    if not good[j]:excluded.append(dict(source_index=i,reason='all_zero_or_nonfinite'));continue
                    accepted.append(i);subject=str(subjects[i]);group='01' if ds=='CHB_MIT' and subject=='21' else subject
                    row=dict(id=f'{ds}-{i}',parent_id=f'{ds}-{i}',task=tid,dataset=ds,subject=group,original_subject=subject,label=int(y[i]),source_index=i,tokens=8*int(np.ceil(x.shape[-1]/fs/10)))
                    rows.write(json.dumps(row)+'\n')
                (dest/'progress.json').write_text(json.dumps(dict(exported=end,total=len(x),seconds=time.monotonic()-start)))
        target.flush();del target
        report=dict(dataset=ds,n=len(x),valid_n=len(accepted),first_valid_index=accepted[0],channels=names,sampling_rate=fs,shape=list(x.shape),dtype='float32',excluded=excluded,source=task['source'],catalog_sha256=digest('configs/benchmark_tasks.json'),files_sha256={n:digest(dest/n) for n in ['signals.npy','rows.jsonl']},seconds=time.monotonic()-start)
        (dest/'complete.json').write_text(json.dumps(report,indent=2));print('EXPORTED',ds,report['valid_n'],report['seconds'],flush=True);del x
if __name__=='__main__':main()
