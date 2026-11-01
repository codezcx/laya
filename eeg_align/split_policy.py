"""Explicit split contracts: subject holdout, or SEED's ordered trial holdout."""
import numpy as np

SEED_PROTOCOL = 'seed_within_subject_trial_9_3_3'


def seed_split(events):
    events = np.asarray(events)
    if not np.isfinite(events).all() or not np.equal(events, events.astype(int)).all():
        raise ValueError('SEED event IDs must be integers')
    if ((events < 0) | (events >= 15)).any():
        raise ValueError('SEED requires original event IDs 0 through 14')
    return np.where(events < 9, 'train', np.where(events < 12, 'valid', 'test'))


def validate_split_frame(name, manifest, frame, *, windows=False):
    if manifest.get('split_protocol') != SEED_PROTOCOL:
        if frame.groupby('subject')['split'].nunique().max() != 1:
            raise ValueError('Subject split overlap')
        return
    if name != 'seed':
        raise ValueError('Within-subject exception is restricted to SEED')
    if not np.array_equal(frame['split'].to_numpy(), seed_split(frame['event'].to_numpy())):
        raise ValueError('SEED split does not match ordered trial 9/3/3 protocol')
    if frame.groupby(['record_id', 'event'])['split'].nunique().max() != 1:
        raise ValueError('SEED trial crosses splits')
    if not frame.groupby('record_id')['event'].apply(lambda s: set(s) == set(range(15))).all():
        raise ValueError('SEED recording lacks one of its 15 original trials')
    keys = ['array', 'row'] if windows else ['record_id', 'event']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate SEED window/trial identity')
