"""Deterministic bridge warmup followed by low-LR native JEV adapter tuning."""
import math


def rates_at(step, total, bridge_only, peaks, warmup=62, minimum=.02):
    if not 1 <= step <= total or not 0 <= bridge_only < total:
        raise ValueError('Invalid step or phase boundary')
    rates = {}
    for name, peak in peaks.items():
        start = bridge_only if name in {'jev_lora', 'decision_head'} else 0
        local, budget = step - start, total - start
        if local <= 0:
            rates[name] = 0.
            continue
        warm = min(warmup, max(1, budget // 4))
        ratio = local / warm if local <= warm else minimum + (1 - minimum) * .5 * (
            1 + math.cos(math.pi * (local - warm) / max(1, budget - warm)))
        rates[name] = peak * ratio
    return rates
