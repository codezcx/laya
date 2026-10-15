"""One-time reconciliation of the inspected sub-100 failure; never retry new errors."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SOURCE = Path('artifacts/faced_full_v22')
ROOT = Path('artifacts/faced_v22_reconciliation')
CONTROLLER = Path('artifacts/faced_scale_v22_controller')
PYTHON = 'laya/.venv-eeg/bin/python'


def validate_repair(status, repair, audit_bytes):
    expected = [{'subject': 100, 'error': "ValueError('Flat EEG channel')"}]
    if status.get('state') != 'failed' or status.get('failed') != expected:
        raise ValueError('Only the previously inspected sub-100 failure is authorized')
    completed = [row['subject'] for row in status['completed']]
    if len(completed) != 97 or len(set(completed)) != 97 or 100 in completed:
        raise ValueError('All other 97 subjects must have completed')
    if hashlib.sha256(audit_bytes).hexdigest() != repair['audit_sha256']:
        raise ValueError('Repair audit changed after inspection')
    audit = json.loads(audit_bytes)
    if (repair['subject'], audit['subject'], audit['windows'],
            len(audit['quarantined_windows'])) != (100, 100, 267, 13):
        raise ValueError('Unexpected repair accounting')


def process_identity(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return None
    # comm may contain spaces; subsequent fields start at process state.
    fields = path.read_text().rsplit(')', 1)[1].split()
    return None if fields[0] == 'Z' else fields[19]


def main():
    ROOT.mkdir(exist_ok=False)
    def status(**value):
        tmp = ROOT / 'status.tmp'
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(ROOT / 'status.json')

    try:
        pids = {name: int(Path(f'artifacts/{name}.pid').read_text())
                for name in ['faced_full_v22_fetch', 'faced_scale_v22_controller']}
        identities = {name: process_identity(pid) for name, pid in pids.items()}
        status(state='waiting_for_original_processes', pids=pids)
        deadline = time.monotonic() + 12 * 3600
        while any(identity is not None and process_identity(pids[name]) == identity
                  for name, identity in identities.items()):
            if time.monotonic() > deadline:
                raise TimeoutError('Original processes still active; no restart')
            time.sleep(5)
        original = json.loads((SOURCE / 'cohort_status.json').read_text())
        repair = json.loads((SOURCE / 'sub100_repair.json').read_text())
        validate_repair(original, repair, (SOURCE / 'prepared/sub-100/audit.json').read_bytes())
        old_controller = json.loads((CONTROLLER / 'status.json').read_text())
        if old_controller['state'] != 'failed':
            raise ValueError('Expected original controller to stop on acquisition failure')
        shutil.copy2(SOURCE / 'cohort_status.json', ROOT / 'original_cohort_status.json')
        shutil.copy2(SOURCE / 'sub100_repair.json', ROOT / 'verified_repair.json')
        CONTROLLER.rename(ROOT / 'original_controller')
        status(state='resuming_inspected_cohort')
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path.cwd()) + os.pathsep + env.get('PYTHONPATH', '')
        with (ROOT / 'resume.log').open('w') as log:
            fetch = subprocess.Popen([PYTHON, '-u', 'scripts/fetch_faced_full_v22.py', '--resume'],
                                     stdout=log, stderr=subprocess.STDOUT, env=env)
        Path('artifacts/faced_full_v22_fetch.pid').write_text(str(fetch.pid))
        if fetch.wait() != 0:
            raise RuntimeError('Resume or full audit failed; no further retry')
        if json.loads((SOURCE / 'cohort_status.json').read_text())['state'] != 'complete':
            raise RuntimeError('Full audit did not complete')
        with (ROOT / 'scale_controller.log').open('w') as log:
            controller = subprocess.Popen([PYTHON, '-u', 'scripts/run_faced_scale_v22.py'],
                                          stdout=log, stderr=subprocess.STDOUT, env=env)
        Path('artifacts/faced_scale_v22_controller.pid').write_text(str(controller.pid))
        status(state='running_source_controls', controller_pid=controller.pid)
        if controller.wait() != 0:
            raise RuntimeError('Source controls failed; inspect logs')
        status(state='complete', target_evaluation='not_run')
    except Exception as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
