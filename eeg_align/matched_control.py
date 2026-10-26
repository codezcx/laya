"""Matched task exposure and objective helpers, independent of model loading."""
import hashlib
import json


def select_rows(rows, datasets):
    selected = [r for r in rows if r['dataset'] in datasets]
    if set(r['dataset'] for r in selected) != set(datasets):
        raise ValueError('Requested dataset is missing from the split')
    return selected


def task_sum_loss(losses, rows, samples_per_task, weights, world=1):
    """Accumulate sum_t w_t * mean_t CE, never divide by number of tasks.

    The denominator is the complete per-task global batch, not the microbatch.
    average_gradients subsequently averages over world ranks.
    """
    scale = losses.new_tensor([weights[r['dataset']] for r in rows])
    return (losses * scale).sum() * world / samples_per_task


def ids_digest(rows):
    return hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest()


def sampling_proof(rows, sampler, seed, steps, samples_per_task, anchor):
    tasks = {r['task'] for r in rows}
    proof = []
    for step in range(1, steps + 1):
        ix = sampler(rows, step, seed, samples_per_task * len(tasks))
        selected = [rows[i] for i in ix if rows[i]['dataset'] == anchor]
        assert len(selected) == samples_per_task
        proof.append(ids_digest(selected))
    return dict(anchor=anchor, steps=steps, samples_per_update=samples_per_task,
                total_anchor_exposures=steps*samples_per_task, step_sha256=proof)
