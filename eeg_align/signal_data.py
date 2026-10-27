"""Subject-disjoint manifests and online EEG loading for joint LEAF/JEV SFT."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .benchmark_data import catalog,to_electrodes,resample,segment,bipolar_matrix

ROOT=Path('datasets/benchmark/signals')


class SignalStore:
    def __init__(self,datasets=None,official_source=None):
        from .leaf_runtime import UP
        from leaf_datasets.electrode_unifier import ElectrodeUnifier
        self.arrays={};self.meta={};self.spatial={};self.keep={}
        unifier=ElectrodeUnifier('LEAF-ch65');aliases={k.upper():k for k in unifier.reverse_lookup}
        for tid,task in enumerate(catalog()):
            ds=task['task_id'];folder=ROOT/ds
            if datasets is not None and ds not in datasets:continue
            if official_source is None:
                self.meta[tid]=json.loads((folder/'complete.json').read_text())
                self.arrays[tid]=np.load(folder/'signals.npy',mmap_mode='r')
            else:
                self.meta[tid]=dict(dataset=ds,channels=list(official_source.channel_names),
                                    sampling_rate=float(official_source.sampling_rate))
                self.arrays[tid]=official_source.eeg_data
            meta=self.meta[tid]
            # Montage geometry comes from metadata, never from a held-out waveform.
            names=bipolar_matrix(meta['channels'])[1] if any('-' in c for c in meta['channels']) else meta['channels']
            keep=[i for i,c in enumerate(names) if c.upper() not in {'A1','A2'}];self.keep[tid]=keep
            canonical=[aliases[names[i].upper()] for i in keep]
            self.spatial[tid]=unifier.map_to_template(np.eye(len(keep),dtype=np.float32)[None],canonical)[0]

    def batch(self,rows):
        grouped=defaultdict(list)
        for i,r in enumerate(rows):grouped[r['task']].append(i)
        result=[]
        for tid,indices in grouped.items():
            meta=self.meta[tid];source=[rows[i]['source_index'] for i in indices]
            raw=np.array(self.arrays[tid][source],dtype=np.float32)
            xx,_,_=to_electrodes(raw,meta['channels']);xx=resample(xx[:,self.keep[tid]],meta['sampling_rate'])
            limits=np.percentile(xx,[.1,25,50,75,99.9],axis=(1,2));scale=limits[3]-limits[1]
            fallback=scale<=0;scale[fallback]=xx[fallback].std(axis=(1,2))
            if (scale<=0).any() or not np.isfinite(xx).all():raise ValueError('Invalid online signal batch')
            xx=(np.clip(xx,limits[0,:,None,None],limits[4,:,None,None])-limits[2,:,None,None])/scale[:,None,None]
            xx=np.einsum('ec,bct->bet',self.spatial[tid],xx).astype(np.float32)
            if meta['dataset']=='SEED':xx=np.clip(xx,-10,10)
            pieces=[np.pad(v,((0,0),(0,0),(0,(-v.shape[-1])%100))) for v in segment(xx)]
            result.append((tid,indices,pieces))
        return result
