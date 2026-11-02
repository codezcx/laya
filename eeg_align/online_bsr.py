"""Released R1 renderer + SharedScale + unpooled VideoMAE; no text inputs."""
from pathlib import Path
import sys
import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel
from .runtime import digest
from .encoder_lora import inject_videomae_lora, is_lora


def upstream_files(config):
    root = Path(config['bsr_root']) / 'code'
    return [root / p for p in ('bsri/__init__.py', 'bsri/model.py', 'ops/render_v3.py')]


def compatible_encoder_state(state, expected):
    """Exact old query/value bias conversion; old keys had no additive bias."""
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
    def __init__(self, config, trainable=False):
        super().__init__()
        sys.path.insert(0, str(Path(config['bsr_root']) / 'code'))
        from bsri.model import BSRenderer, IMAGENET_MEAN, IMAGENET_STD, CHANNELS_1010
        from ops.render_v3 import SharedScaleAdapter
        if digest(config['checkpoint']) != config['checkpoint_sha256']:
            raise ValueError('BSR release checksum mismatch')
        ckpt = torch.load(config['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
        if (ckpt['v3']['renderer_kind'], ckpt['v3']['adapter_kind']) != ('r1', 'sharedscale'):
            raise ValueError('Wrong BSR release recipe')
        cfg = VideoMAEConfig.from_dict(ckpt['encoder_config'])
        cfg._attn_implementation = 'sdpa'
        self.renderer = BSRenderer(n_fft=200, system='1010')
        self.adapter = SharedScaleAdapter(output_size=cfg.image_size)
        self.encoder = VideoMAEModel(cfg)
        self.renderer.load_state_dict(ckpt['renderer_state'], strict=True)
        self.adapter.load_state_dict(ckpt['adapter_state'], strict=True)
        self.encoder.load_state_dict(compatible_encoder_state(ckpt['encoder_state'], self.encoder.state_dict()), strict=True)
        self.register_buffer('image_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer('image_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1))
        self.channels = frozenset(CHANNELS_1010)
        self.finetune = bool(trainable)
        self.lora_audit = None
        self.adaptation_enabled = bool(trainable)
        self.requires_grad_(trainable)
        # Released inverse renderer is a reconstruction-only head, not this path.
        self.renderer.pix2eeg.requires_grad_(False)
        if trainable:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.eval()

    def train(self, mode=True):
        return super().train(mode and self.finetune and self.adaptation_enabled)

    def enable_videomae_lora(self, spec):
        if not self.finetune or self.lora_audit is not None:
            raise ValueError('VideoMAE LoRA requires a trainable, unwrapped BSR encoder')
        # Renderer and SharedScale retain the original full-tuning policy.
        self.lora_audit = inject_videomae_lora(self.encoder, spec)
        self.set_adaptation_enabled(self.adaptation_enabled)
        return self.lora_audit

    def set_adaptation_enabled(self, enabled):
        self.adaptation_enabled = self.finetune and bool(enabled)
        for module in (self.renderer.eeg2pix, self.adapter):
            module.requires_grad_(self.adaptation_enabled)
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(self.adaptation_enabled and
                                     (getattr(self, 'lora_audit', None) is None or is_lora(name)))
        self.train(self.training)

    def forward(self, x, channels):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.adaptation_enabled):
            return self._encode(x, channels)

    def _encode(self, x, channels):
        from bsri.model import stft_hop_for_frames
        if not set(channels) <= self.channels:
            raise ValueError('Unknown renderer channels')
        x = x.to(device=self.image_mean.device, dtype=torch.float32)
        self.renderer.hop_length = stft_hop_for_frames(x.shape[-1], 16)
        rendered = self.renderer(x, channels)
        if rendered.shape[2] != 16:
            raise ValueError('STFT must produce exactly 16 frames')
        adapted = self.adapter(rendered)
        video = ((adapted - self.image_mean) / self.image_std).permute(0, 2, 1, 3, 4)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            tokens = self.encoder(video).last_hidden_state
        if tokens.shape[1:] != (1568, 768) or not torch.isfinite(tokens).all():
            raise ValueError('Invalid VideoMAE tokens')
        return tokens
