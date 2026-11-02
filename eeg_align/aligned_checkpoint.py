"""Strict encoder-only warm start from a completed, split-matched alignment run."""
import json
from pathlib import Path
import torch

from .checkpoint import tensor_fingerprint
from .runtime import digest


DATA_KEYS = ('split_summary_sha256', 'quality_summary_sha256', 'cache_root',
             'datasets', 'input_units', 'preprocessing_hz', 'checkpoint_sha256')


def source_spec(setting, config):
    name = setting.get('aligned_source')
    return None if name is None else config['aligned_sources'][name]


@torch.no_grad()
def restore_encoder_only(encoder, saved):
    """Restore all forward parameters and buffers, never the pooling/projection head.

    The upstream inverse renderer was frozen and absent from the small checkpoint;
    only that reconstruction-only module may retain its verified release weights.
    Validate the entire payload before touching the destination.
    """
    if saved.get('format_version') != 2:
        raise ValueError('Aligned checkpoint must include complete buffers')
    parameters = dict(encoder.named_parameters())
    buffers = dict(encoder.named_buffers())
    transferred = {}
    for kind, expected in (('trainable', parameters), ('buffers', buffers)):
        source = {n.removeprefix('encoder.'): v for n, v in saved[kind].items()
                  if n.startswith('encoder.')}
        names = {n for n in expected if not n.startswith('renderer.pix2eeg.')}
        if set(source) != names:
            raise ValueError(f'Aligned encoder {kind} names mismatch: '
                             f'missing={sorted(names-set(source))}, extra={sorted(set(source)-names)}')
        for name, value in source.items():
            target = expected[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f'Aligned encoder shape/dtype mismatch: {name}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite aligned encoder tensor: {name}')
        transferred.update(source)
    # named_buffers includes persistent=False STFT/montage tensors that are
    # deliberately absent from state_dict. The small checkpoint saves them all.
    targets = {**parameters, **buffers}
    for name, value in transferred.items():
        targets[name].copy_(value)
    return len(transferred)


def load_aligned_encoder(encoder, setting, config, cache_indexes):
    spec = source_spec(setting, config)
    if spec is None:
        return None
    path = Path(spec['checkpoint'])
    if path.name != 'best.pt' or digest(path) != spec['checkpoint_sha256']:
        raise ValueError('Aligned best checkpoint checksum mismatch')
    complete = json.loads((path.parent / 'complete.json').read_text())
    # Trusted local experiment payload includes RNG state; never load remote pickle.
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    identity = saved['identity']
    if identity['smoke'] or identity['arm'] != setting['aligned_source']:
        raise ValueError('Wrong aligned source arm or smoke checkpoint')
    if (complete['test_evaluations'] != 0 or complete['jev_loaded'] is not False
            or not complete['encoder_updated'] or not complete['checkpoint_reload_verified']):
        raise ValueError('Aligned source did not pass its completion gates')
    if not (saved['step'] == saved['best_step'] == complete['selected_step'] == spec['selected_step']):
        raise ValueError('Aligned checkpoint is not the validation-selected source')
    for key in DATA_KEYS:
        if identity['data_config'][key] != config[key]:
            raise ValueError(f'Aligned source data protocol mismatch: {key}')
    if identity['cache_indexes'] != cache_indexes:
        raise ValueError('Aligned source cache indexes differ from current training')
    if identity['code_sha256']['configs/benchmark_tasks.json'] != digest('configs/benchmark_tasks.json'):
        raise ValueError('Aligned source label semantics differ from current tasks')
    n = restore_encoder_only(encoder, saved)
    fingerprint = tensor_fingerprint(encoder.state_dict())
    if fingerprint != spec['encoder_sha256'] or fingerprint != complete['selected_encoder_sha256']:
        raise ValueError('Transferred encoder differs from selected alignment encoder')
    return dict(source=setting['aligned_source'], checkpoint=str(path),
                checkpoint_sha256=spec['checkpoint_sha256'], selected_step=saved['step'],
                encoder_sha256=fingerprint, transferred_tensors=n,
                readout='new initialization; source CLS and projection intentionally excluded',
                optimizer='fresh; no source optimizer/RNG/JEV weights restored')
