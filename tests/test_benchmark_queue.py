import copy
import json
from unittest.mock import patch
import pytest
from eeg_align.runtime import CONFIG
from scripts.run_benchmark import training_command, verify_initialization, validate_resume, require_gpu_capacity


def test_one_model_uses_four_processes():
    command = training_command('python', [0,1,2,3], 'leaf_tower_mlp', 'out', smoke=True)
    assert '--nproc-per-node=4' in command
    assert command.count('eeg_align.train_joint') == 1
    assert '--smoke' in command and '--resume' not in command


def test_fresh_formal_initialization_must_match_smoke():
    proof = dict(common_connector_sha256='a', jev_adapter_head_sha256='b',
                 initial_encoder_sha256='c', buffer_sha256='d', world_size=4,
                 rank_initialization_verified=True)
    verify_initialization(proof, proof)
    for key in ('common_connector_sha256', 'jev_adapter_head_sha256', 'initial_encoder_sha256'):
        bad = dict(proof); bad[key] = 'changed'
        with pytest.raises(RuntimeError):
            verify_initialization(proof, bad)


def test_memory_gate_never_stops_peer_jobs():
    with patch('subprocess.check_output', return_value='0, 80000\n1, 60000\n'):
        with pytest.raises(RuntimeError):
            require_gpu_capacity([0,1],72000)
        assert require_gpu_capacity([0],72000) == {0:80000}


def test_resume_identity_and_completion_gate(tmp_path):
    arm = CONFIG['arms'][0]
    directory = tmp_path / 'train' / arm
    directory.mkdir(parents=True)
    identity = dict(config=CONFIG, arm=arm, smoke=False, code_sha256={})
    (directory/'config.json').write_text(json.dumps(identity))
    for filename in ('last.pt','best.pt'):
        (directory/filename).write_bytes(b'fixture')
    validate_resume(tmp_path, arm)
    (directory/'complete.json').write_text('{}')
    with pytest.raises(ValueError, match='Already complete'):
        validate_resume(tmp_path, arm)
