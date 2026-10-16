"""Four CPU extraction workers, followed by all-arm frozen evaluation."""
import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path('artifacts/emotion_power_v26_external_controller')
    root.mkdir(exist_ok=False)
    def status(**value):
        (root / 'status.json').write_text(json.dumps(value, indent=2))
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    command = ['laya/.venv-eeg/bin/python', '-u', '-m', 'eeg_tasks.emotion_power_external_v26']
    try:
        jobs = []
        for shard in range(4):
            with (root / f'{shard}.log').open('x') as log:
                jobs.append(subprocess.Popen(command + ['--shard', str(shard)], stdout=log, stderr=subprocess.STDOUT, env=env))
        status(state='extracting', pids=[p.pid for p in jobs])
        codes = [p.wait() for p in jobs]
        if any(codes):
            raise RuntimeError(codes)
        status(state='evaluating')
        with (root / 'evaluation.log').open('x') as log:
            subprocess.run(command + ['--evaluate'], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        status(state='complete')
    except Exception as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
