import hashlib
import json

import pytest

from scripts.reconcile_faced_v22 import validate_repair


def records():
    audit = json.dumps({'subject': 100, 'windows': 267,
                        'quarantined_windows': [{}] * 13}).encode()
    repair = {'subject': 100, 'audit_sha256': hashlib.sha256(audit).hexdigest()}
    status = {'state': 'failed', 'failed': [
        {'subject': 100, 'error': "ValueError('Flat EEG channel')"}],
        'completed': [{'subject': i} for i in range(97)]}
    return status, repair, audit


def test_accept_only_inspected_repair():
    validate_repair(*records())


@pytest.mark.parametrize('change', ['new_failure', 'running', 'incomplete', 'changed_audit'])
def test_refuse_uninspected_or_incomplete_state(change):
    status, repair, audit = records()
    if change == 'new_failure':
        status['failed'].append({'subject': 110, 'error': 'new error'})
    elif change == 'running':
        status['state'] = 'running'
    elif change == 'incomplete':
        status['completed'].pop()
    else:
        audit += b' '
    with pytest.raises(ValueError):
        validate_repair(status, repair, audit)
