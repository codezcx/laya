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
        if not 0 <= encoder_freeze_steps <= total_steps:
            raise ValueError('Encoder freeze duration must lie within the budget')
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


class ContinuationLRSchedule:
    """A declared low-peak warm restart; preserves the parent endpoint LR.

    Update numbers stay global. Adam moments are loaded separately, not reset.
    The first continuation update uses the exact parent endpoint; after warmup
    rates decay from peak_ratio * original_peak back to the parent endpoint.
    """
    def __init__(self, optimizer, parent, additional_steps, warmup_steps, peak_ratio):
        start = parent['last_step']
        if start != parent['total_steps'] or parent['kind'] != 'warmup_cosine':
            raise ValueError('Continuation requires a completed cosine schedule')
        if additional_steps < 3 or not 2 <= warmup_steps < additional_steps:
            raise ValueError('Invalid continuation or warmup budget')
        if not parent['min_ratio'] <= peak_ratio <= 1:
            raise ValueError('Continuation peak must lie between parent floor and peak')
        if [g['name'] for g in optimizer.param_groups] != parent['group_names']:
            raise ValueError('Optimizer group order changed')
        self.optimizer = optimizer
        self.settings = dict(kind='continuation_warmup_cosine', start_step=start,
            total_steps=start + additional_steps, warmup_steps=warmup_steps,
            peak_ratio=peak_ratio, floor_ratio=parent['min_ratio'],
            base_lrs=list(parent['base_lrs']), group_names=list(parent['group_names']))
        self.last_step = start

    def set_step(self, step):
        cfg = self.settings
        if not cfg['start_step'] <= step <= cfg['total_steps']:
            raise ValueError('Continuation LR step outside budget')
        local = step - cfg['start_step']
        if local <= 1:
            multiplier = cfg['floor_ratio']
        elif local <= cfg['warmup_steps']:
            progress = (local - 1) / (cfg['warmup_steps'] - 1)
            multiplier = cfg['floor_ratio'] + progress * (cfg['peak_ratio'] - cfg['floor_ratio'])
        else:
            phase = (local - cfg['warmup_steps']) / (cfg['total_steps'] - cfg['start_step'] - cfg['warmup_steps'])
            multiplier = cfg['floor_ratio'] + (cfg['peak_ratio'] - cfg['floor_ratio']) * (1 + math.cos(math.pi * phase)) / 2
        for group, base in zip(self.optimizer.param_groups, cfg['base_lrs']):
            group['lr'] = base * multiplier
        self.last_step = step
        return {g['name']: float(g['lr']) for g in self.optimizer.param_groups}

    def state_dict(self):
        return dict(**self.settings, last_step=self.last_step)

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self.settings.items()):
            raise ValueError('Continuation LR identity changed')
        self.set_step(state['last_step'])
