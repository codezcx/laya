"""Two matched coverage-controlled arms, two GPUs each; external evaluation after freeze."""
import os,json,subprocess,sys,time,signal
from pathlib import Path
from eeg_align.cache import ROOT,digest

OUT=Path('artifacts/jev_eeg_coverage/runs')

def main():
    OUT.mkdir(parents=True,exist_ok=False);status=OUT/'status.json'
    def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,**kw),indent=2))
    update('verifying_source_inputs')
    for shard in range(4):
        root=ROOT/str(shard);audit=json.loads((root/'complete.json').read_text())
        assert audit['manifest_sha256']==digest(ROOT/'rows.jsonl')
        for name,sha in audit['hashes'].items():assert digest(root/name)==sha
    provenance=json.loads(Path('artifacts/jev_eeg/checkpoint_audit.json').read_text())
    for name,sha in provenance['files'].items():assert digest(name)==sha
    smoke=json.loads(Path('artifacts/jev_eeg_coverage/smoke/resume_verified.json').read_text())
    assert smoke['exact_match']
    for name,sha in smoke['code_sha256'].items():assert digest(name)==sha
    jobs=[]
    def launch(gpus,command,name):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,PYTHONPATH='.',OMP_NUM_THREADS='4')
        with (OUT/f'{name}.log').open('a') as log:
            process=subprocess.Popen([sys.executable,*command],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return name,process
    def wait_all(jobs):
        while True:
            codes={n:p.poll() for n,p in jobs}
            failures={n:c for n,c in codes.items() if c not in [None,0]}
            if failures:
                for n,p in jobs:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                update('failed',failures=failures);raise RuntimeError(failures)
            if all(c==0 for c in codes.values()):return
            time.sleep(10)
    for arm,gpus in [('projector','0,1'),('lora','2,3')]:
        name=f'{arm}_seed42'
        jobs.append(launch(gpus,['-m','torch.distributed.run','--standalone','--nproc-per-node=2',
            '-m','eeg_align.train_jev','--arm',arm,'--rounds','2','--per-group','4','--out',str(OUT/name)],name))
    update('source_training',pids={n:p.pid for n,p in jobs});wait_all(jobs)
    freeze=OUT/'source_freeze.json'
    for n,p in jobs:
        done=json.loads((OUT/n/'complete.json').read_text());assert not done['smoke']
        assert done['best_sha256']==digest(OUT/n/'best.pt')
    freeze.write_text(json.dumps(dict(runs={n:digest(OUT/n/'best.pt') for n,p in jobs},
        selection='complete source validation at coverage-round endpoints; targets unused',timestamp=time.time()),indent=2))
    update('full_source_and_external_evaluation')
    jobs=[launch(str(gpu),['-m','eeg_align.evaluate_jev','--run',str(OUT/n),'--freeze',str(freeze)],n) for gpu,(n,p) in enumerate(jobs)]
    wait_all(jobs);update('complete')

if __name__=='__main__':main()
