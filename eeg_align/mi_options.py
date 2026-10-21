"""Reproducible option permutations and semantic readout for binary hand MI."""
import hashlib
from .jev import SUFFIX

REVERSED_SUFFIX = SUFFIX.replace('A) Left hand motor imagery\nB) Right hand motor imagery',
                                'A) Right hand motor imagery\nB) Left hand motor imagery')


def swapped(trial_id, epoch, seed):
    return bool(hashlib.sha256(f'{seed}:{epoch}:{trial_id}'.encode()).digest()[0] & 1)


def target_slot(label, reverse):
    return 1-label if reverse else label


def right_probability(p_b, reverse):
    return 1-p_b if reverse else p_b
