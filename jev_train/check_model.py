"""Train-only, real-weight integration check before distributed training."""
import json
import time

import torch
from torch.nn import functional as F
from .runtime import CONFIG, ROOT, write_json, code_identity
from .data import TrialStore, PreparedAdapter
from .online_leaf import OnlineLEAFTower, encode_batch
from .checkpoint import seed_rng, tensor_fingerprint
from .medjev import load_medjev
from .task_options import make_variant


def main():
    torch.set_num_threads(3)
    seed_rng(CONFIG['seed'])
    started = time.time()
    print('LOAD_MEDJEV_REAL_WEIGHTS', flush=True)
    model, _ = load_medjev()
    print('LOAD_LEAF_TOWER', flush=True)
    model.add_module('eeg_encoder', OnlineLEAFTower(CONFIG).cuda())
    groups = model.parameter_groups() + [dict(name='eeg_encoder', params=[
        p for p in model.eeg_encoder.parameters() if p.requires_grad])]
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    buffers = tensor_fingerprint(dict(model.eeg_encoder.named_buffers()))
    store = TrialStore()
    adapter = PreparedAdapter(store)
    cases = [0, max(store.rows, key=lambda t: store.rows[t][0]['samples'])]
    checks = []
    for tid in dict.fromkeys(cases):
        task = store.tasks[tid]
        rows = [r for r in store.rows[tid] if r['split'] == 'train'][:CONFIG['microbatch']]
        questions = [make_variant(task, r['label'], r['id'], 1, CONFIG['seed']) for r in rows]
        model.train()
        model.zero_grad(set_to_none=True)
        before = time.time()
        tokens, valid = encode_batch(model.eeg_encoder, adapter.batch(rows), len(rows))
        logits = model(tokens, valid, questions)
        targets = torch.tensor([q['label'] for q in questions], device=logits.device)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite integration loss')
        loss.backward()
        norms = {}
        for group in groups:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in group['params']):
                raise RuntimeError('Missing/nonfinite gradient: ' + group['name'])
            norms[group['name']] = float(torch.stack([p.grad.float().square().sum() for p in group['params']]).sum().sqrt())
            if norms[group['name']] <= 0:
                raise RuntimeError('Zero group gradient: ' + group['name'])
        if any(p.grad is not None or p._version != frozen[n] for n, p in model.named_parameters() if n in frozen):
            raise RuntimeError('Frozen base changed')
        if tensor_fingerprint(dict(model.eeg_encoder.named_buffers())) != buffers:
            raise RuntimeError('LEAF BatchNorm buffers changed')
        check = dict(dataset=task['task_id'], split='train', ids=[r['id'] for r in rows],
            signal_samples=rows[0]['samples'], eeg_tokens=tokens.shape[1], loss=float(loss.detach()),
            group_gradient_norms=norms, seconds=time.time()-before,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        checks.append(check)
        print(json.dumps(check), flush=True)
    result = dict(passed=True, source_hashes=code_identity(), architecture=model.architecture_audit,
        tower_adaptation=model.eeg_encoder.release_audit,
        checks=checks, parameters_total=sum(p.numel() for p in model.parameters()),
        parameters_by_group={g['name']:sum(p.numel() for p in g['params']) for g in groups},
        seconds=time.time()-started)
    write_json(ROOT / CONFIG['output_root'] / 'model_preflight.json', result)
    print('MODEL_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
