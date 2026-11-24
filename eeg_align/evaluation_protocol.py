"""Current development roles, separate from the hash-pinned historical sampler.

The original test allocation is now used for development by user decision.
No row or manifest is relabeled; original valid is never forwarded here.
"""
import copy
import json
from pathlib import Path

from .evaluation import validation_rows

POLICY_PATH = Path('configs/evaluation_policy.json')
POLICY = json.loads(POLICY_PATH.read_text())
SELECTION_SPLIT = POLICY['selection_split']


def effective_config(source):
    config = copy.deepcopy(source)
    config.update(evaluation_policy=copy.deepcopy(POLICY), selection_split=SELECTION_SPLIT,
                  selection=POLICY['selection_criterion'], test_policy=POLICY['final_evaluation'])
    return config


def require_split(split, training=False):
    if split not in {POLICY['gradient_split'], SELECTION_SPLIT}:
        raise RuntimeError(f'Deferred/unknown split cannot be forwarded: {split}')
    if training and split != POLICY['gradient_split']:
        raise RuntimeError('Only train rows can backpropagate')


def selection_rows(store, tid, cap):
    class SelectionPool:
        # Reuse the exact historical sampling algorithm without changing its
        # hash-pinned implementation. Its legacy argument is an API alias only;
        # the real store is queried exclusively with the original test label.
        def evaluation(self, task, legacy_split, limit=None):
            if legacy_split != 'valid':
                raise ValueError('Unexpected legacy sampler request')
            return store.evaluation(task, SELECTION_SPLIT, limit)

    rows = validation_rows(SelectionPool(), tid, cap)
    if not rows or any(r['split'] != SELECTION_SPLIT for r in rows):
        raise RuntimeError('Selection pool contains an incorrect split or is empty')
    if len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate selection IDs')
    return rows


class EvaluationLedger:
    """Completed dataset passes and globally gathered sample counts, not batches.

    Persisted in checkpoints for resume. Counts include repeated evaluations;
    zero valid passes is additionally enforced at the forward boundary.
    """
    def __init__(self, state=None):
        self.counts = {f'{split}_{kind}': 0 for split in ('train', 'valid', 'test')
                       for kind in ('evaluations', 'evaluation_samples')}
        if state is not None:
            if set(state) != set(self.counts) or any(type(v) is not int or v < 0 for v in state.values()):
                raise ValueError('Invalid evaluation ledger')
            self.counts.update(state)
        if self.counts['valid_evaluations'] or self.counts['valid_evaluation_samples']:
            raise ValueError('Deferred valid pool was evaluated')

    def record(self, split, n):
        require_split(split)
        if type(n) is not int or n <= 0:
            raise ValueError('Evaluation requires a positive sample count')
        self.counts[f'{split}_evaluations'] += 1
        self.counts[f'{split}_evaluation_samples'] += n

    def state_dict(self):
        return dict(self.counts)


def verify_evaluation_contract(complete):
    ledger = EvaluationLedger(complete['evaluation_counts']).state_dict()
    if complete['evaluation_policy'] != POLICY:
        raise RuntimeError('Evaluation policy mismatch')
    for split in ('test', 'valid'):
        if complete[f'{split}_evaluations'] != ledger[f'{split}_evaluations']:
            raise RuntimeError('Evaluation counts disagree')
    if complete['smoke']:
        if ledger['test_evaluations'] or complete['development_evaluation'] is not None:
            raise RuntimeError('Smoke must be train-only')
    else:
        full = complete['development_evaluation']
        if (ledger['test_evaluations'] <= 0 or full['split'] != SELECTION_SPLIT
                or full['role'] != POLICY['selection_role'] or full['scope'] != 'full'
                or full['independent_test_estimate']):
            raise RuntimeError('Missing or mislabeled full development evaluation')
