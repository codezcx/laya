"""Bounded real-source overfit check, not a generalization experiment."""
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel, AutoTokenizer
from eeg_tasks.data import preprocess, normalized_window
from eeg_tasks.emotion_transfer_probe import CHANNELS, picks, readrows
from eeg_tasks.mi_reve_scaling_v24 import scale_window
from eeg_align.model import load_text_only
from eeg_align.multitask_bridge import MultitaskBridge, PREFIX, SUFFIX

ROOT = Path('artifacts/multitask_bridge_v30/real_smoke')


def select_rows():
    mi_path = Path('artifacts/mi_full_sources_v12/manifest.jsonl')
    em_path = Path('artifacts/faced_scale_v22/rows.jsonl')
    mi = readrows(mi_path)
    keep = np.load(mi_path.parent/'keep_indices.npy')
    rows = [mi[int(i)] for i in keep] + readrows(em_path)
    selected = {}
    for r in rows:
        if r['split'] != 'train':
            continue
        key = (r['dataset'], r['class_id'])
        selected.setdefault(key, r)
    expected = {(d, c) for d in ['Cho2017', 'Dreyer2023', 'Lee2019_MI'] for c in [3, 4]}
    expected |= {(d, c) for d in ['SEED', 'FACED-NEMAR'] for c in [0, 1, 2]}
    assert set(selected) == expected
    return [selected[k] for k in sorted(selected)], {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [mi_path, em_path, mi_path.parent/'keep_indices.npy']}


def read_signal(row, cfg):
    path = row['path']
    if row['dataset'] in ['Cho2017', 'Dreyer2023', 'Lee2019_MI']:
        with open(path, 'rb') as f:
            item = pickle.load(f)
        assert item['freq'] == 200 and cfg['labels'].index(item['label']) == row['class_id']-3
        x = preprocess(np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']]), 200, {'bandpass': [1, 30]})[:, :600]
        assert hashlib.sha256(normalized_window(x).tobytes()).hexdigest() == row['signal_sha256']
        return scale_window(x, 'common_scale'), ['C3', 'Cz', 'C4'], 'common_scale'
    if row['dataset'] == 'SEED':
        meta = json.loads(Path('configs/multitask_prototypes_v1.json').read_text())
        names = [s.split()[-1] for s in Path(meta['seed_channel_file']).read_text().splitlines() if s.strip()]
        data = loadmat(path)
        keys = {str(int(re.search(r'eeg(\d+)$', k).group(1))): k for k in data if re.search(r'eeg(\d+)$', k)}
        x = data[keys[row['trial']]][picks(names)]
        labels = loadmat(Path(path).parent/'label.mat')['label'].reshape(-1)
        assert int(labels[int(row['trial'])-1])+1 == row['class_id']
        fs = 200
    else:
        import mne
        from eeg_tasks.faced_prepare import read_table, video_events
        events = {str(e['video_index']): e for e in video_events(read_table(path.replace('_eeg.bdf', '_events.tsv')))}
        event = events[row['trial']]
        assert float(event['onset']) == row['event_onset_seconds']
        assert {'negative': 0, 'neutral': 1, 'positive': 2}[event['binary_label']] == row['class_id']
        raw = mne.io.read_raw_bdf(path, preload=False, verbose=False)
        fs = raw.info['sfreq']
        x = raw.get_data(picks=picks(raw.ch_names), start=round(float(event['onset'])*fs),
                         stop=round((float(event['onset'])+float(event['duration']))*fs))
        raw.close()
    x = preprocess(x, fs, {'bandpass': [1, 30]})
    start = row['start_sample']
    x = normalized_window(x[:, start:start+600])
    assert x.shape == (30, 600)
    return x, CHANNELS, 'per_channel'


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42); torch.set_num_threads(4)
    cfg = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    rows, hashes = select_rows()
    reve = AutoModel.from_pretrained(cfg['reve_checkpoint'], trust_remote_code=True, local_files_only=True).cuda().eval().requires_grad_(False)
    bank = AutoModel.from_pretrained(cfg['positions_checkpoint'], trust_remote_code=True, local_files_only=True)
    cache, audit = [], []
    for row in rows:
        x, channels, norm = read_signal(row, cfg)
        pos = bank(channels).detach().cuda()[None]
        batch = torch.from_numpy(x).cuda()[None]
        tail = (-(600-reve.config.patch_size)) % (reve.config.patch_size-reve.config.patch_overlap)
        batch = torch.nn.functional.pad(batch, (0, tail), mode='replicate')
        with torch.no_grad():
            z = reve(batch, pos)
        assert z.shape == (1, len(channels), 4, 512) and torch.isfinite(z).all()
        cache.append(z.flatten(1, 2)[0].cpu())
        audit.append(dict(row=row, channels=channels, normalization=norm,
                          signal_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                          token_shape=list(cache[-1].shape)))
        print('encoded', row['dataset'], row['class_id'], flush=True)
    torch.save(cache, ROOT/'tokens.pt')
    (ROOT/'data.json').write_text(json.dumps(dict(source_manifests=hashes, samples=audit,
        token_cache_sha256=hashlib.sha256((ROOT/'tokens.pt').read_bytes()).hexdigest(),
        synthetic=False, source_train_only=True, token_order='channel then temporal patch'), indent=2))
    del reve, bank
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(cfg['qwen_checkpoint'], local_files_only=True)
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in 'ABCDE']
    assert all(len(x)==1 for x in ids)
    text, head, loading = load_text_only(cfg['qwen_checkpoint'], [x[0] for x in ids], 'cuda:0', torch.bfloat16)
    enc = lambda s: torch.tensor(tokenizer.encode(s, add_special_tokens=False), device='cuda')
    model = MultitaskBridge(text, head, enc(PREFIX), enc(SUFFIX)).cuda()
    opt = torch.optim.AdamW(model.connector.parameters(), lr=1e-4, weight_decay=.01)
    cache = [x.cuda()[None] for x in cache]
    # Equal task, then dataset, then class mass; one microbatch per source/class.
    weights = [1/(2*3*2) if r['class_id'] >= 3 else 1/(2*2*3) for r in rows]
    start = time.monotonic()
    records = []
    for step in range(11):
        model.train(step > 0); opt.zero_grad(set_to_none=True)
        loss_sum = 0.; pred = []
        for x, row, weight in zip(cache, rows, weights):
            with torch.set_grad_enabled(step > 0):
                result = model(x, torch.ones(x.shape[:2], dtype=torch.bool, device='cuda'),
                               torch.tensor([row['class_id']], device='cuda'))
                assert torch.isfinite(result['loss'])
                if step > 0:
                    (weight*result['loss']).backward()
            loss_sum += weight*float(result['loss'].detach())
            pred.append(int(result['logits'].argmax(-1)))
        if step > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.connector.parameters(), 1.)
            assert torch.isfinite(norm) and norm > 0
            assert all(p.grad is None for p in model.text.parameters())
            opt.step()
        record = dict(step=step, pre_update_train_loss=loss_sum,
                      train_accuracy=float(np.mean([p==r['class_id'] for p,r in zip(pred, rows)])),
                      seconds=time.monotonic()-start)
        records.append(record)
        with (ROOT/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(record, flush=True)
    torch.save(model.connector.state_dict(), ROOT/'connector.pt')
    (ROOT/'complete.json').write_text(json.dumps(dict(real_source_smoke_only=True, n=len(rows),
        optimizer_steps=10, loading=loading, records=records, external_evaluation=False,
        connector_sha256=hashlib.sha256((ROOT/'connector.pt').read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
