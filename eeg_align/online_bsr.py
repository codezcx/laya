"""Published BSR R1/SharedScale/VideoMAE -> unpooled tokens on official MI rows."""
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

from .benchmark_data import resample
from .runtime import digest


def upstream_files(config):
    root = Path(config['bsr_root']) / 'code'
    return [root / 'bsri/__init__.py', root / 'bsri/model.py', root / 'ops/render_v3.py']


class BSRSignalStore:
    """Keep original row/channel identity and entire epoch; never fit on test data."""
    def __init__(self, dataset, source):
        if dataset not in {'BNCI2014001', 'BNCI2014004'}:
            raise ValueError('Only the audited 2a/2b MI source adapters are enabled')
        self.data = source.eeg_data
        self.hz = float(source.sampling_rate)
        self.channels = [str(c).upper() for c in source.channel_names]
        if len(set(self.channels)) != len(self.channels):
            raise ValueError('Duplicate channels')
        expected = 22 if dataset == 'BNCI2014001' else 3
        if len(self.channels) != expected or any('-' in c for c in self.channels):
            raise ValueError('Unexpected source montage; no inferred channel renaming')

    def batch(self, rows):
        raw = np.array(self.data[[r['source_index'] for r in rows]], dtype=np.float32)
        if raw.ndim != 3 or raw.shape[1] != len(self.channels) or not np.isfinite(raw).all():
            raise ValueError('Invalid/nonfinite MI signal')
        if np.any(raw.std(axis=(1, 2)) <= 0):
            raise ValueError('Constant/empty MI epoch')
        x = resample(raw, self.hz) if self.hz != 200 else raw
        return torch.from_numpy(x), self.channels


def compatible_encoder_state(state, expected):
    """Lossless old q_bias/v_bias -> Transformers 5 Linear biases conversion."""
    state = dict(state)
    for name in list(state):
        if name.endswith('.q_bias'):
            prefix = name[:-len('q_bias')]
            if prefix + 'query.bias' in expected:
                q, v = state.pop(name), state.pop(prefix + 'v_bias')
                state[prefix + 'query.bias'] = q
                state[prefix + 'value.bias'] = v
                state[prefix + 'key.bias'] = torch.zeros_like(q)
    return state


class OnlineBSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        root = Path(config['bsr_root']) / 'code'
        sys.path.insert(0, str(root))
        from bsri.model import BSRenderer, IMAGENET_MEAN, IMAGENET_STD
        from ops.render_v3 import SharedScaleAdapter
        if digest(config['checkpoint']) != config['checkpoint_sha256']:
            raise ValueError('BSR release checkpoint hash mismatch')
        checkpoint = torch.load(config['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
        if (checkpoint['v3']['renderer_kind'], checkpoint['v3']['adapter_kind']) != ('r1', 'sharedscale'):
            raise ValueError('Wrong renderer/adapter recipe')
        cfg = VideoMAEConfig.from_dict(checkpoint['encoder_config'])
        cfg._attn_implementation = 'sdpa'
        self.renderer = BSRenderer(n_fft=200, system='1010')
        self.adapter = SharedScaleAdapter(output_size=cfg.image_size)
        self.encoder = VideoMAEModel(cfg)
        self.renderer.load_state_dict(checkpoint['renderer_state'], strict=True)
        self.adapter.load_state_dict(checkpoint['adapter_state'], strict=True)
        state = compatible_encoder_state(checkpoint['encoder_state'], self.encoder.state_dict())
        self.encoder.load_state_dict(state, strict=True)
        self.renderer.pix2eeg.requires_grad_(False)
        # Old VideoMAE has no trainable key bias; preserve that parameterization.
        for name, param in self.encoder.named_parameters():
            if name.endswith('.key.bias'):
                param.requires_grad_(False)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.register_buffer('image_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer('image_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1))
        self.expected_tokens = (cfg.num_frames // cfg.tubelet_size) * (cfg.image_size // cfg.patch_size)**2

    def forward(self, batch, unfrozen=False):
        from bsri.model import stft_hop_for_frames, SYSTEM_GRIDS
        x, channels = batch
        if not set(channels) <= {c for row in SYSTEM_GRIDS['1010'] for c in row if c}:
            raise ValueError('Unknown channels would be silently dropped by renderer')
        x = x.to(device=self.image_mean.device, dtype=torch.float32)
        self.renderer.hop_length = stft_hop_for_frames(x.shape[-1], self.encoder.config.num_frames)
        with (nullcontext() if unfrozen else torch.no_grad()):
            # STFT + adapter calibration stay fp32, as in the published path.
            rendered = self.renderer(x, channels)
            if rendered.shape[2] != self.encoder.config.num_frames:
                raise ValueError('STFT frame count mismatch')
            adapted = self.adapter(rendered)
            video = ((adapted - self.image_mean) / self.image_std).permute(0, 2, 1, 3, 4)
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                tokens = self.encoder(video).last_hidden_state
        if tokens.shape[1:] != (self.expected_tokens, self.encoder.config.hidden_size):
            raise ValueError('Unpooled VideoMAE token shape mismatch')
        return tokens, torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
