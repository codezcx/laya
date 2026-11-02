"""CPU preflight of the exact planned training rows and rank/microbatch boundaries."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.multitask_data import TrialStore, write_json
from eeg_align.bsr_data import BSRSignalAdapter

STORE = None
ADAPTER = None


def initialize():
    global ADAPTER
    ADAPTER = BSRSignalAdapter()


def check_step(step):
    tid, rows = STORE.sample(step, CONFIG['global_batch'], CONFIG['seed'])
    try:
        for start in range(0, len(rows), CONFIG['microbatch']):
            for _, _, pieces in ADAPTER.batch(rows[start:start + CONFIG['microbatch']]):
                if not all(np.isfinite(x).all() for x in pieces):
                    raise ValueError('Nonfinite BSR input after adaptation')
    except Exception as exc:
        raise ValueError(f'Step {step}, {STORE.tasks[tid]["task_id"]}, IDs={[r["id"] for r in rows]}: {exc}') from exc
    return tid, len(rows)


def main():
    global STORE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-step', type=int, default=1)
    parser.add_argument('--stop-step', type=int, default=CONFIG['steps'])
    args = parser.parse_args()
    if not 1 <= args.start_step <= args.stop_step:
        raise ValueError('Invalid planned update interval')
    if args.out.exists():
        raise FileExistsError(args.out)
    STORE = TrialStore()
    start = time.monotonic()
    watched = [Path('configs/benchmark_train.json'), Path('scripts/check_multitask_signals.py')] + sorted(Path('eeg_align').glob('*.py'))
    hashes = {str(p): digest(p) for p in watched}
    counts = {t['task_id']: 0 for t in STORE.tasks}
    # Linux fork shares the read-only row index; workers construct their own adapter.
    with ProcessPoolExecutor(max_workers=CONFIG['cpu_workers'], mp_context=mp.get_context('fork'), initializer=initialize) as executor:
        for step, (tid, n) in enumerate(executor.map(check_step, range(args.start_step, args.stop_step + 1), chunksize=4), args.start_step):
            counts[STORE.tasks[tid]['task_id']] += n
            if step % 100 == 0 or step == args.stop_step:
                print('CHECKED_TRAIN_STEP', step, 'elapsed_seconds', round(time.monotonic() - start, 1), flush=True)
    if hashes != {str(p): digest(p) for p in watched}:
        raise RuntimeError('Signal preflight code/config changed')
    write_json(args.out, dict(passed=True, planned_updates=args.stop_step - args.start_step + 1,
        start_step=args.start_step, stop_step=args.stop_step, sample_presentations=counts,
        heldout_model_evaluations=0, quality_summary_sha256=CONFIG['quality_summary_sha256'],
        code_sha256=hashes, elapsed_seconds=time.monotonic() - start))


if __name__ == '__main__':
    main()
