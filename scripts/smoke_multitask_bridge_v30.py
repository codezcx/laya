"""Real-checkpoint gradient check; synthetic input, no accuracy claim."""
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from eeg_align.model import load_text_only
from eeg_align.multitask_bridge import MultitaskBridge, PREFIX, SUFFIX, CLASSES


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    out = Path('artifacts/multitask_bridge_v30/smoke')
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    path = cfg['qwen_checkpoint']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in 'ABCDE']
    assert all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == 5
    text, head, audit = load_text_only(path, [x[0] for x in ids], 'cuda:0', torch.bfloat16)
    enc = lambda s: torch.tensor(tokenizer.encode(s, add_special_tokens=False), device='cuda')
    model = MultitaskBridge(text, head, enc(PREFIX), enc(SUFFIX)).cuda().train()
    tokens = torch.randn(2, 12, 512, device='cuda')
    valid = torch.arange(12, device='cuda')[None] < torch.tensor([12, 8], device='cuda')[:, None]
    tokens[~valid] = float('nan')
    result = model(tokens, valid, torch.tensor([0, 4], device='cuda'))
    result['loss'].backward()
    grads = [p.grad for p in model.connector.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    norm = sum(float(g.norm()) for g in grads)
    assert norm > 0 and all(p.grad is None for p in model.text.parameters())
    model.eval()
    with torch.no_grad():
        original = model(tokens, valid)['logits']
        clean = tokens.clone(); clean[~valid] = 12345
        padding_delta = float((original-model(clean, valid)['logits']).abs().max())
        assert padding_delta == 0
    payload = dict(synthetic_only=True, checkpoint=path, classes=CLASSES,
                   checkpoint_index_sha256=hashlib.sha256((Path(path)/'model.safetensors.index.json').read_bytes()).hexdigest(),
                   loading_audit=audit, loss=float(result['loss'].detach()),
                   connector_gradient_norm_sum=norm, frozen_decoder_no_grad=True,
                   padding_logit_delta=padding_delta, logits_shape=list(original.shape),
                   peak_cuda_gb=torch.cuda.max_memory_allocated()/1e9)
    (out/'complete.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
