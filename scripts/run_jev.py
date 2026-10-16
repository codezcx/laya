"""Four GPU runs: projector control, two LoRA seeds, LoRA plus unfrozen REVE."""
import os,json,subprocess,sys,time
from pathlib import Path
from eeg_align.cache import ROOT,digest
from eeg_align.signals import RAW

OUT=Path('artifacts/jev_eeg/runs')

def main():
    OUT.mkdir(parents=True,exist_ok=False);status=OUT/'status.json'
    def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,**kw),indent=2))
    update('verifying_source_inputs')
    for shard in range(4):
        root=ROOT/str(shard);audit=json.loads((root/'complete.json').read_text())
        assert audit['manifest_sha256']==digest(ROOT/'rows.jsonl')
        for name,sha in audit['hashes'].items():assert digest(root/name)==sha
    raw=json.loads((RAW/'complete.json').read_text())
    assert raw['signals_sha256']==digest(RAW/'signals.npy') and raw['manifest_sha256']==digest(ROOT/'rows.jsonl')
    assert Path('artifacts/jev_eeg/checkpoint_audit.json').exists()
    arms=[('projector',42),('lora',42),('lora',43),('lora_encoder',42)];jobs=[]
    def launch(gpu,command,name):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4')
        log=(OUT/f'{name}.log').open('a')
        process=subprocess.Popen([sys.executable,*command],env=env,stdout=log,stderr=subprocess.STDOUT)
        return name,process,log
    for gpu,(arm,seed) in enumerate(arms):
        name=f'{arm}_seed{seed}'
        jobs.append(launch(gpu,['-m','eeg_align.train_jev','--arm',arm,'--seed',str(seed),'--out',str(OUT/name)],name))
    update('source_training',pids={n:p.pid for n,p,l in jobs})
    def wait_all(jobs):
        while True:
            codes={n:p.poll() for n,p,l in jobs}
            failures={n:c for n,c in codes.items() if c not in [None,0]}
            if failures:
                for n,p,l in jobs:
                    if p.poll() is None:p.terminate()
                update('failed',failures=failures);raise RuntimeError(failures)
            if all(c==0 for c in codes.values()):break
            time.sleep(10)
    wait_all(jobs)
    freeze=OUT/'source_freeze.json'
    freeze.write_text(json.dumps(dict(runs={n:digest(OUT/n/'best.pt') for n,p,l in jobs},
        selection='fixed source validation panel; targets unused',timestamp=time.time()),indent=2))
    update('full_source_and_external_evaluation')
    jobs=[launch(gpu,['-m','eeg_align.evaluate_jev','--run',str(OUT/n),'--freeze',str(freeze)],n) for gpu,(n,p,l) in enumerate(jobs)]
    wait_all(jobs);update('complete')

if __name__=='__main__':main()
