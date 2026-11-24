"""Task-specific candidates with reproducible, label-independent permutations."""
import hashlib
import json
from pathlib import Path
import random


def load_tasks(path='configs/jev13_tasks.json'):
    tasks=json.loads(Path(path).read_text())['tasks']
    return {t['task_id']:t for t in tasks}


def make_question(task, class_id, sample_id, epoch, seed=42):
    classes=task['classes']
    if not 2 <= len(classes) <= 16:raise ValueError('JEV supports 2–16 choices')
    ids=[c['class_id'] for c in classes]
    if len(set(ids))!=len(ids) or class_id not in ids:raise ValueError('Invalid class catalog/target')
    key=json.dumps([seed,epoch,sample_id],separators=(',',':')).encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(key).digest(),'big'))
    order=list(range(len(classes)));rng.shuffle(order)
    options='\n'.join(f"{chr(65+i)}) {classes[j]['option']}" for i,j in enumerate(order))
    suffix=f"\nEEG recording ends.\n[task] {task.get('instruction', '')}\n[question] {task['question']}\n[options]\n{options}\n[decision]:"
    return dict(suffix=suffix,label=order.index(ids.index(class_id)),num_choices=len(classes),
                option_class_ids=[ids[i] for i in order],task_id=task['task_id'])


def make_variant(task, label, sample, step, seed, variant=None):
    if variant is None:
        return make_question(task, task['classes'][label]['class_id'], sample, step, seed)
    order = list(range(len(task['classes'])))
    if variant:
        order.reverse()
    options = '\n'.join(f"{chr(65+i)}) {task['classes'][j]['option']}" for i, j in enumerate(order))
    return dict(suffix=f"\nEEG recording ends.\n[task] {task.get('instruction', '')}\n[question] {task['question']}\n[options]\n{options}\n[decision]:",
                label=order.index(label), num_choices=len(order),
                option_class_ids=[task['classes'][j]['class_id'] for j in order])
