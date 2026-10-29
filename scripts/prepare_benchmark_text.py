"""Frozen local MPNet masked-mean task vectors, with LEAF reference check."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.benchmark_data import catalog
from eeg_align.multitask_data import write_json


def main():
    torch.set_num_threads(4)
    source = ROOT / 'artifacts/benchmark/mpnet'
    texts = list(dict.fromkeys(t['instruction'] for t in catalog()))
    output = Path(CONFIG['text_root'])
    if (output / 'instructions.json').exists():
        saved = json.loads((output / 'instructions.json').read_text())
        if saved['texts'] != texts or saved['vectors_sha256'] != digest(output / 'instructions.npy'):
            raise ValueError('Instruction cache mismatch')
        return
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModel.from_pretrained(source, local_files_only=True).eval().requires_grad_(False)
    with torch.inference_mode():
        inputs = tokenizer(texts, padding=True, return_tensors='pt')
        states = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'][..., None]
        vectors = F.normalize((states * mask).sum(1) / mask.sum(1), dim=-1).numpy()
    original_texts = (Path(CONFIG['leaf_root']) / 'text_embeddings/mpnet-base.txt').read_text().splitlines()
    original_vectors = np.load(Path(CONFIG['leaf_root']) / 'text_embeddings/mpnet-base.npy')
    checks = {s: float(vectors[i] @ original_vectors[original_texts.index(s)]) for i, s in enumerate(texts) if s in original_texts}
    if not checks or min(checks.values()) < .9999:
        raise ValueError(f'MPNet reference mismatch: {checks}')
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'instructions.npy', vectors)
    write_json(output / 'instructions.json', dict(texts=texts, reference_cosines=checks, model_sha256=digest(source / 'model.safetensors'), vectors_sha256=digest(output / 'instructions.npy'), pooling='masked mean then L2 normalization; no labels'))
    print('TEXT_READY', checks, flush=True)


if __name__ == '__main__':
    main()
