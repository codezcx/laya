"""Offline MPNet bank and CPU audit of every planned input, before GPU launch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from eeg_align.multitask_data import TrialStore, write_json
from eeg_align.bsr_data import BSRSignalAdapter
from eeg_align.runtime import digest
from .protocol import CONFIG, DATA, code_identity, build_text_manifest

STORE = None
ADAPTER = None


def initialize():
    global ADAPTER
    ADAPTER = BSRSignalAdapter()


def check_step(step):
    tid, rows = STORE.sample(step, CONFIG['global_batch'], CONFIG['seed'])
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Held-out row selected for training')
    for start in range(0, len(rows), CONFIG['microbatch']):
        for _, _, pieces in ADAPTER.batch(rows[start:start+CONFIG['microbatch']]):
            if not all(np.isfinite(p).all() for p in pieces):
                raise ValueError(f'Nonfinite planned training step {step}')
    return tid, len(rows)


def prepare_text(output):
    source = Path(CONFIG['text_model'])
    if digest(source / 'model.safetensors') != CONFIG['text_model_sha256']:
        raise ValueError('MPNet release identity mismatch')
    manifest = build_text_manifest()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(source, local_files_only=True, trust_remote_code=False).eval().requires_grad_(False)
    # Check the local checkpoint/mean-pooling recipe against the existing LEAF vectors.
    reference = json.loads((Path(DATA['text_root']) / 'instructions.json').read_text())
    old = np.load(Path(DATA['text_root']) / 'instructions.npy', allow_pickle=False)
    if digest(Path(DATA['text_root']) / 'instructions.npy') != reference['vectors_sha256']:
        raise ValueError('Reference text vectors changed')
    texts = manifest['texts'] + reference['texts']
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            inputs = tokenizer(texts[start:start+32], padding=True, return_tensors='pt', truncation=False)
            if inputs['input_ids'].shape[1] > 384:
                raise ValueError('Unexpected long caption; do not silently truncate')
            states = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'][..., None]
            vectors.append(F.normalize((states*mask).sum(1)/mask.sum(1), dim=-1).numpy())
    vectors = np.concatenate(vectors)
    cosines = np.sum(vectors[-len(old):] * old, 1)
    if not np.isfinite(vectors).all() or cosines.min() < .9999:
        raise ValueError(f'Text reference mismatch: {cosines.tolist()}')
    values = vectors[:len(manifest['texts'])]
    if values.shape[1] != CONFIG['projection_width']:
        raise ValueError('Projection width differs from text encoder')
    np.save(output / 'text_vectors.npy', values)
    manifest.update(vectors_sha256=digest(output/'text_vectors.npy'),
        model_sha256=CONFIG['text_model_sha256'], reference_cosines=cosines.tolist(),
        pooling='float32 masked mean, then L2 normalization; frozen CPU encoder',
        model_files_sha256={str(p):digest(p) for p in sorted(source.iterdir()) if p.is_file()})
    write_json(output / 'text_bank.json', manifest)
    print('TEXT_BANK_READY', len(values), 'texts', len(manifest['concepts']), 'concepts', flush=True)


def main():
    global STORE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    if (args.out/'preflight.json').exists() or (args.out/'text_bank.json').exists():
        raise FileExistsError('Preparation is immutable; inspect existing artifacts')
    hashes = code_identity()
    started = time.monotonic()
    # Fork CPU workers before constructing a Transformers model/thread pool.
    STORE = TrialStore()
    counts = {t['task_id']:0 for t in STORE.tasks}
    with ProcessPoolExecutor(max_workers=CONFIG['cpu_workers'], mp_context=mp.get_context('fork'), initializer=initialize) as pool:
        for step, (tid, n) in enumerate(pool.map(check_step, range(1, CONFIG['steps']+1), chunksize=4), 1):
            counts[STORE.tasks[tid]['task_id']] += n
            if step % 100 == 0:
                print('CHECKED_TRAIN_STEP', step, round(time.monotonic()-started, 1), flush=True)
    prepare_text(args.out)
    if hashes != code_identity():
        raise RuntimeError('Code changed during preparation')
    write_json(args.out/'preflight.json', dict(passed=True, steps=CONFIG['steps'],
        code_sha256=hashes, presentations=counts, elapsed_seconds=time.monotonic()-started,
        quality_summary_sha256=DATA['quality_summary_sha256'], test_evaluations=0))


if __name__ == '__main__':
    main()
