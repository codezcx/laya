"""Wait for the four known feature workers, then run audited source-only fitting."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('artifacts/mi_covariance_v23_controller')


def main():
    status_path = ROOT / 'fit_status.json'
    with status_path.open('x') as f:
        json.dump({'state': 'waiting_for_features'}, f)
    def status(**value):
        tmp = ROOT / 'fit_status.tmp'
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(status_path)
    try:
        jobs = json.loads((ROOT / 'jobs.json').read_text())
        deadline = time.monotonic() + 3600
        while True:
            pending = [j for j in jobs if not Path(f"artifacts/mi_covariance_v23/{j['shard']}.done.json").exists()]
            if not pending:
                break
            for job in pending:
                proc = Path(f"/proc/{job['pid']}/stat")
                if not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                    raise RuntimeError(f'Worker ended without completion: {job}')
            if time.monotonic() > deadline:
                raise TimeoutError('Feature wait exceeded one hour; no automatic restart')
            time.sleep(5)
        status(state='auditing_and_fitting')
        env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
        with (ROOT / 'fit.log').open('x') as log:
            p = subprocess.Popen(['laya/.venv-eeg/bin/python', '-u', '-m', 'eeg_tasks.mi_covariance_fit_v23'],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            status(state='auditing_and_fitting', pid=p.pid)
            if p.wait() != 0:
                raise RuntimeError('Source fitting failed; inspect fit.log')
        status(state='source_complete', target_evaluation='not_run')
    except Exception as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
