"""Verify current paired inputs, four-GPU smoke, then train the current approach."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from eeg_align.cache import digest
from eeg_align.paired_cache import verify, DATA
from eeg_align.attribute_data import AttributeData

OUT=Path('artifacts/jev_paired_qa')


def main():
    assert not (OUT/'run').exists() and not (OUT/'smoke').exists()
    def status(phase,**kw):
        (OUT/'status.json').write_text(json.dumps(dict(phase=phase,timestamp=time.time(),**kw),indent=2))
    status('verifying_inputs')
    init=json.loads((OUT/'initialization.json').read_text())
    assert digest(OUT/'initialization.pt')==init['sha256']
    verify()
    for root in DATA:
        complete=json.loads((root/'complete.json').read_text())
        audit=json.loads((root/'verification.json').read_text())
        assert complete['schema_version']==2 and audit['status']=='passed'
        assert audit['complete_sha256']==digest(root/'complete.json')
        for name,sha in complete['files'].items():assert digest(root/name)==sha
        data=AttributeData(root,complete['source_splits'][0],include_emotion_class=True)
        assert {s['id'] for s in data.specs} >= {'mi_class','emotion_class'}
        assert len(data.groups)==62
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH='.',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    for name,extra in [('smoke',['--smoke']),('run',[])]:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            '-m','eeg_align.train_jev','--out',str(OUT/name),*extra]
        with (OUT/f'{name}.log').open('w') as log:
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(name,pid=process.pid,command=cmd)
            code=process.wait()
        if code:
            status('failed',stage=name,returncode=code)
            raise RuntimeError(f'{name} failed: {code}')
        assert (OUT/name/'complete.json').exists()
    status('complete')


if __name__=='__main__':main()
