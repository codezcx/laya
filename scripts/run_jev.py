"""Audit source inputs, run real four-GPU QA smoke, then bounded attribute training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from eeg_align.cache import ROOT, digest

OUT = Path('artifacts/jev_attribute_qa')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw):
        (OUT/'status.json').write_text(json.dumps(dict(phase=phase, timestamp=time.time(), **kw), indent=2))
    status('verifying_inputs')
    for shard in range(4):
        folder = ROOT/str(shard)
        audit = json.loads((folder/'complete.json').read_text())
        assert audit['manifest_sha256'] == digest(ROOT/'rows.jsonl')
        for name, sha in audit['hashes'].items():assert digest(folder/name) == sha
    previous = Path('artifacts/jev_eeg_coverage/runs/lora_seed42')
    assert digest(previous/'best.pt') == json.loads((previous/'complete.json').read_text())['best_sha256']
    for root in ['datasets/eeg_attribute_qa','datasets/eeg_attribute_qa_validation']:
        assert json.loads((Path(root)/'verification.json').read_text())['status'] == 'passed'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONPATH='.',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    for name, extra in [('smoke',['--smoke']), ('run',[])]:
        cmd = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
               '-m','eeg_align.train_jev','--out',str(OUT/name),*extra]
        with (OUT/f'{name}.log').open('w') as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            status(name, pid=process.pid, command=cmd)
            code = process.wait()
        if code:
            status('failed', stage=name, returncode=code)
            raise RuntimeError(f'{name} failed: {code}')
        assert (OUT/name/'complete.json').exists()
    status('complete')


if __name__ == '__main__':main()
