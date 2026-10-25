"""Explicit optimizer-update learning rates, including delayed encoder warmup."""
import math


class GroupLRSchedule:
    """Set the LR used by a numbered update; never step after optimizer.step().

    Each group warms up from its first active update, then reaches min_ratio at
    total_steps. A frozen encoder has LR zero as well as no gradient. This is
    independent of microbatch count, accumulation and distributed world size.
    """

    def __init__(self, optimizer, total_steps, warmup_steps, min_ratio,
                 encoder_freeze_steps=0, kind='warmup_cosine'):
        if kind not in {'constant', 'warmup_cosine'}:
            raise ValueError(f'Unknown learning-rate schedule: {kind}')
        if not 0 <= encoder_freeze_steps < total_steps:
            raise ValueError('Encoder must have at least one active update')
        if not 0 <= warmup_steps <= total_steps or not 0 <= min_ratio <= 1:
            raise ValueError('Invalid warmup or minimum LR ratio')
        self.optimizer = optimizer
        self.settings = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                             min_ratio=min_ratio, encoder_freeze_steps=encoder_freeze_steps,
                             kind=kind)
        self.names = [g['name'] for g in optimizer.param_groups]
        if len(set(self.names)) != len(self.names):
            raise ValueError('Optimizer group names must be unique')
        self.base_lrs = [float(g['lr']) for g in optimizer.param_groups]
        self.last_step = 0

    def set_step(self, step):
        cfg = self.settings
        if not 1 <= step <= cfg['total_steps']:
            raise ValueError('LR update step outside training budget')
        for name, peak, group in zip(self.names, self.base_lrs, self.optimizer.param_groups):
            frozen = cfg['encoder_freeze_steps'] if name == 'eeg_encoder' else 0
            active_step = step - frozen
            active_total = cfg['total_steps'] - frozen
            warmup = min(cfg['warmup_steps'], active_total)
            if active_step <= 0:
                multiplier = 0.
            elif cfg['kind'] == 'constant':
                multiplier = 1.
            elif active_step <= warmup:
                multiplier = active_step / warmup
            else:
                phase = (active_step - warmup) / (active_total - warmup)
                multiplier = cfg['min_ratio'] + (1. - cfg['min_ratio']) * (1. + math.cos(math.pi * phase)) / 2.
            group['lr'] = peak * multiplier
        self.last_step = step
        return {name: float(g['lr']) for name, g in zip(self.names, self.optimizer.param_groups)}

    def state_dict(self):
        return dict(**self.settings, group_names=self.names, base_lrs=self.base_lrs,
                    last_step=self.last_step)

    def load_state_dict(self, state):
        expected = dict(**self.settings, group_names=self.names, base_lrs=self.base_lrs)
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError('Saved learning-rate schedule does not match configuration')
        if state['last_step']:
            self.set_step(state['last_step'])
