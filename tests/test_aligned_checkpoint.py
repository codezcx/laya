import copy
import json
from pathlib import Path
import pytest
import torch

from eeg_align.aligned_checkpoint import DATA_KEYS, load_aligned_encoder, restore_encoder_only
from eeg_align.checkpoint import capture_model, tensor_fingerprint
from eeg_align.runtime import CONFIG, digest
from scripts.run_benchmark import require_gpu_capacity, verify_initializations


def tiny():
    encoder = torch.nn.Module()
    encoder.renderer = torch.nn.Module()
    encoder.renderer.eeg2pix = torch.nn.Linear(2, 3)
    encoder.renderer.pix2eeg = torch.nn.Linear(3, 2).requires_grad_(False)
    encoder.encoder = torch.nn.Linear(3, 2)
    encoder.register_buffer('image_mean', torch.tensor([1., 2., 3.]))
    model = torch.nn.Module()
    model.encoder = encoder
    model.readout = torch.nn.Linear(2, 2)
    return model


def test_encoder_only_transfer_restores_frozen_destination_and_excludes_readout():
    source, destination = tiny(), tiny()
    original_inverse = copy.deepcopy(destination.encoder.renderer.pix2eeg.state_dict())
    original_head = copy.deepcopy(destination.readout.state_dict())
    destination.encoder.requires_grad_(False)
    assert restore_encoder_only(destination.encoder, capture_model(source)) == 5
    for name, value in source.encoder.state_dict().items():
        if not name.startswith('renderer.pix2eeg.'):
            torch.testing.assert_close(destination.encoder.state_dict()[name], value, rtol=0, atol=0)
    for name, value in original_inverse.items():
        torch.testing.assert_close(destination.encoder.renderer.pix2eeg.state_dict()[name], value)
    for name, value in original_head.items():
        torch.testing.assert_close(destination.readout.state_dict()[name], value)
    assert not any(p.requires_grad for p in destination.encoder.parameters())


def test_transfer_restores_nonpersistent_renderer_buffers():
    source, destination = tiny(), tiny()
    source.encoder.renderer.register_buffer('stft_window', torch.ones(3), persistent=False)
    destination.encoder.renderer.register_buffer('stft_window', torch.zeros(3), persistent=False)
    assert 'renderer.stft_window' not in source.encoder.state_dict()
    restore_encoder_only(destination.encoder, capture_model(source))
    torch.testing.assert_close(destination.encoder.renderer.stft_window, torch.ones(3))


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'dtype', 'nan', 'buffer', 'version'])
def test_bad_source_is_rejected_before_destination_mutation(bad):
    source, destination = tiny(), tiny()
    state = capture_model(source)
    key = 'encoder.encoder.weight'
    if bad == 'missing': del state['trainable'][key]
    if bad == 'extra': state['trainable']['encoder.unknown'] = torch.zeros(1)
    if bad == 'shape': state['trainable'][key] = torch.zeros(7)
    if bad == 'dtype': state['trainable'][key] = state['trainable'][key].double()
    if bad == 'nan': state['trainable'][key].fill_(float('nan'))
    if bad == 'buffer': del state['buffers']['encoder.image_mean']
    if bad == 'version': state['format_version'] = 1
    before = tensor_fingerprint(destination.state_dict())
    with pytest.raises(ValueError): restore_encoder_only(destination.encoder, state)
    assert tensor_fingerprint(destination.state_dict()) == before


@pytest.fixture
def saved_source(tmp_path):
    source = tiny()
    state = capture_model(source)
    state.update(step=1700, best_step=1700, identity=dict(
        arm='infonce', smoke=False, data_config={k: CONFIG[k] for k in DATA_KEYS},
        cache_indexes={'task': 'hash'}, code_sha256={
            'configs/benchmark_tasks.json': digest('configs/benchmark_tasks.json')}))
    path = tmp_path / 'best.pt'
    torch.save(state, path)
    fingerprint = tensor_fingerprint(source.encoder.state_dict())
    complete = dict(selected_step=1700, test_evaluations=0, jev_loaded=False,
                    encoder_updated=True, checkpoint_reload_verified=True,
                    selected_encoder_sha256=fingerprint)
    (tmp_path / 'complete.json').write_text(json.dumps(complete))
    config = copy.deepcopy(CONFIG)
    config['aligned_sources']['infonce'] = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                                               selected_step=1700, encoder_sha256=fingerprint)
    return source, config


def test_source_provenance_and_strict_loaded_fingerprint(saved_source):
    source, config = saved_source
    # Release-only inverse weights must match the source release exactly.
    destination = copy.deepcopy(source.encoder).requires_grad_(False)
    with torch.no_grad(): destination.encoder.weight.zero_()
    audit = load_aligned_encoder(destination, {'aligned_source':'infonce'}, config, {'task':'hash'})
    assert audit['selected_step'] == 1700
    assert audit['encoder_sha256'] == tensor_fingerprint(source.encoder.state_dict())
    with pytest.raises(ValueError, match='cache indexes'):
        load_aligned_encoder(destination, {'aligned_source':'infonce'}, config, {'task':'different'})
    config['split_summary_sha256'] = 'different'
    with pytest.raises(ValueError, match='data protocol'):
        load_aligned_encoder(destination, {'aligned_source':'infonce'}, config, {'task':'hash'})


def test_source_hash_and_completion_gate(saved_source):
    source, config = saved_source
    spec = config['aligned_sources']['infonce']
    path = Path(spec['checkpoint'])
    done = json.loads((path.parent/'complete.json').read_text())
    done['test_evaluations'] = 1
    (path.parent/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError, match='completion gates'):
        load_aligned_encoder(source.encoder, {'aligned_source':'infonce'}, config, {'task':'hash'})
    spec['checkpoint_sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        load_aligned_encoder(source.encoder, {'aligned_source':'infonce'}, config, {'task':'hash'})


def test_five_arm_initializations_match_by_source_and_readout():
    config = copy.deepcopy(CONFIG)
    config.pop('matched_control', None)
    config['arms'] = ['a', 'b']
    config['arm_settings'] = {a:dict(readout='qformer', aligned_source='infonce') for a in config['arms']}
    proofs = {a: dict(common_connector_sha256='bridge', jev_adapter_head_sha256='jev',
                     initial_encoder_sha256=config['aligned_sources']['infonce']['encoder_sha256'],
                     readout_sha256='qformer') for a in config['arms']}
    verify_initializations(proofs, config)
    proofs['b']['readout_sha256'] = 'different'
    with pytest.raises(RuntimeError, match='Unmatched qformer'):
        verify_initializations(proofs, config)


def test_gpu_headroom_allows_existing_small_jobs_but_rejects_overcommit(monkeypatch):
    monkeypatch.setattr('scripts.run_benchmark.subprocess.check_output',
                        lambda *a, **k: '0, 78570\n1, 65000\n')
    assert require_gpu_capacity([0], 72000) == {0:78570}
    with pytest.raises(RuntimeError, match='Insufficient GPU headroom'):
        require_gpu_capacity([0, 1], 72000)
