"""Four source-only shards of final, unpooled REVE tokens for Qwen."""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from eeg_tasks.data import preprocess, normalized_window
from eeg_tasks.emotion_transfer_probe import CHANNELS, picks, readrows
from eeg_tasks.mi_reve_scaling_v24 import scale_window

ROOT = Path('artifacts/multitask_bridge_v30/source_cache')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    mi_path = Path('artifacts/mi_full_sources_v12/manifest.jsonl')
    em_path = Path('artifacts/faced_scale_v22/rows.jsonl')
    keep_path = mi_path.parent/'keep_indices.npy'
    mi = readrows(mi_path)
    rows = [mi[int(i)] for i in np.load(keep_path)] + readrows(em_path)
    assert len(rows) == 37432+32827
    assert all(r['split'] in ['train', 'validation'] for r in rows)
    for key in ['path', 'subject']:
        sets = [{(r['dataset'], r[key]) for r in rows if r['split']==s} for s in ['train', 'validation']]
        assert not sets[0] & sets[1], (key, 'split overlap')
    assert {r['class_id'] for r in rows} == set(range(5))
    assert {r['dataset'] for r in rows} == {'Cho2017','Dreyer2023','Lee2019_MI','SEED','FACED-NEMAR'}
    (ROOT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    groups = {}
    for i, r in enumerate(rows): groups.setdefault(r['path'], []).append(i)
    # Greedy balance by token count; retain each raw recording on one worker.
    work = [[] for _ in range(4)]; costs = [0]*4
    for path, ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shard = int(np.argmin(costs))
        work[shard].append(path)
        costs[shard] += len(ids)*(12 if rows[ids[0]]['class_id'] >= 3 else 120)
    (ROOT/'plan.json').write_text(json.dumps(dict(paths=work, token_costs=costs,
        n=len(rows), source_hashes={str(p):digest(p) for p in [mi_path, em_path, keep_path]},
        manifest_sha256=digest(ROOT/'rows.jsonl'), source_only=True), indent=2))


def signals_for_file(rr, cfg):
    path = rr[0]['path']; ds = rr[0]['dataset']
    if ds == 'SEED':
        names = [s.split()[-1] for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()]
        data = loadmat(path)
        keys = {str(int(re.search(r'eeg(\d+)$', k).group(1))):k for k in data if re.search(r'eeg(\d+)$', k)}
        filtered = {}; signals = []
        labels = loadmat(Path(path).parent/'label.mat')['label'].reshape(-1)
        for r in rr:
            trial = r['trial']
            assert r['class_id'] == int(labels[int(trial)-1])+1
            if trial not in filtered:
                filtered[trial] = preprocess(data[keys[trial]][picks(names)], 200, cfg)
            start = r['start_sample']
            signals.append(normalized_window(filtered[trial][:,start:start+600]))
        return np.stack(signals), CHANNELS
    if ds == 'FACED-NEMAR':
        folder = Path('artifacts/faced_full_v22/prepared')/f"sub-{int(rr[0]['subject']):03d}"
        audit = json.loads((folder/'audit.json').read_text())
        source = readrows(folder/'rows.jsonl')
        lookup = {r['signal_sha256']:i for i,r in enumerate(source)}
        assert len(lookup) == len(source)
        signals = np.load(folder/'signals.npy', mmap_mode='r')
        output = []
        for r in rr:
            i = lookup[r['signal_sha256']]
            assert all(source[i][k] == r[k] for k in ['dataset','subject','split','class_id','trial','start_sample'])
            x = np.asarray(signals[i])
            assert hashlib.sha256(x.tobytes()).hexdigest() == r['signal_sha256']
            output.append(x[picks(audit['channels'])])
        return np.stack(output), CHANNELS
    assert len(rr) == 1
    with open(path, 'rb') as f: item = pickle.load(f)
    assert item['freq'] == 200 and ['left_hand','right_hand'].index(item['label'])+3 == rr[0]['class_id']
    raw = np.stack([item['eeg'][c] for c in ['C3','CZ','C4']])
    x = preprocess(raw, 200, cfg)[:,:600]
    assert hashlib.sha256(normalized_window(x).tobytes()).hexdigest() == rr[0]['signal_sha256']
    return scale_window(x, 'common_scale')[None], ['C3','Cz','C4']


@torch.no_grad()
def extract(shard):
    torch.set_num_threads(4)
    plan = json.loads((ROOT/'plan.json').read_text())
    assert digest(ROOT/'rows.jsonl') == plan['manifest_sha256']
    rows = readrows(ROOT/'rows.jsonl'); groups = {}
    for i,r in enumerate(rows): groups.setdefault(r['path'], []).append(i)
    indices = [i for path in plan['paths'][shard] for i in groups[path]]
    out = ROOT/str(shard); out.mkdir(exist_ok=False)
    cfg = json.loads(Path('configs/multitask_prototypes_v1.json').read_text())
    base = json.loads(Path(cfg['base_config']).read_text())
    model, info = AutoModel.from_pretrained(base['reve_checkpoint'], local_files_only=True,
        trust_remote_code=True, output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model = model.cuda().eval().requires_grad_(False)
    bank = AutoModel.from_pretrained(base['positions_checkpoint'], local_files_only=True, trust_remote_code=True)
    positions = {n:bank(ch).detach().cuda() for n,ch in [(3,['C3','Cz','C4']),(30,CHANNELS)]}
    features = np.lib.format.open_memmap(out/'tokens.npy', mode='w+', dtype=np.float16, shape=(len(indices),120,512))
    lengths = []; cursor = 0
    for path in plan['paths'][shard]:
        rr = [rows[i] for i in groups[path]]
        signals, channels = signals_for_file(rr, cfg)
        assert signals.shape == (len(rr),len(channels),600) and np.isfinite(signals).all()
        for begin in range(0,len(rr),32):
            x = torch.from_numpy(signals[begin:begin+32]).cuda()
            tail = (-(600-model.config.patch_size)) % (model.config.patch_size-model.config.patch_overlap)
            x = torch.nn.functional.pad(x, (0,tail), mode='replicate')
            z = model(x, positions[len(channels)][None].expand(len(x),-1,-1))
            assert z.shape == (len(x),len(channels),4,512) and torch.isfinite(z).all()
            z = z.flatten(1,2).cpu().numpy().astype(np.float16)
            features[cursor:cursor+len(z)] = 0
            features[cursor:cursor+len(z),:z.shape[1]] = z
            cursor += len(z); lengths.extend([z.shape[1]]*len(z))
        if cursor % 100 < len(rr): print(shard,cursor,len(indices),flush=True)
    assert cursor == len(indices)
    features.flush(); del features
    np.save(out/'indices.npy', np.asarray(indices)); np.save(out/'lengths.npy', np.asarray(lengths))
    (out/'complete.json').write_text(json.dumps(dict(n=cursor, manifest_sha256=plan['manifest_sha256'],
        hashes={name:digest(out/name) for name in ['tokens.npy','indices.npy','lengths.npy']},
        checkpoint=base['reve_checkpoint'], channels={'mi':['C3','Cz','C4'],'emotion':CHANNELS},
        normalization={'mi':'common_scale','emotion':'per_channel'},
        encoder_dtype='float32', cache_dtype='float16', temporal_pooling=False), indent=2))
    print('complete',shard,cursor,flush=True)


def verify():
    plan = json.loads((ROOT/'plan.json').read_text())
    assert digest(ROOT/'rows.jsonl') == plan['manifest_sha256']
    for path, h in plan['source_hashes'].items(): assert digest(path) == h
    rows = readrows(ROOT/'rows.jsonl'); seen = []; counts = {}
    for shard in range(4):
        out = ROOT/str(shard); done = json.loads((out/'complete.json').read_text())
        assert done['manifest_sha256'] == plan['manifest_sha256']
        for name,h in done['hashes'].items(): assert digest(out/name) == h
        ids = np.load(out/'indices.npy'); lengths = np.load(out/'lengths.npy')
        x = np.load(out/'tokens.npy', mmap_mode='r')
        assert x.shape == (len(ids),120,512) and len(ids)==len(lengths)==done['n']
        assert np.array_equal(lengths, [12 if rows[int(i)]['class_id']>=3 else 120 for i in ids])
        for begin in range(0,len(ids),128):
            a = x[begin:begin+128]; ll = lengths[begin:begin+128]
            assert np.isfinite(a).all()
            mask = np.arange(120)[None] >= ll[:,None]
            assert np.all(a[mask] == 0)
        seen.extend(ids.tolist()); counts[str(shard)] = len(ids)
    assert sorted(seen) == list(range(len(rows)))
    result = dict(n=len(rows), shards=counts, exhaustive_coverage=True,
                  finite=True, padding_zero=True, manifest_sha256=plan['manifest_sha256'])
    (ROOT/'verified.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepare', action='store_true'); p.add_argument('--verify',action='store_true'); p.add_argument('--shard',type=int,choices=range(4))
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.verify: verify()
    elif a.shard is not None: extract(a.shard)
    else: p.error('Choose --prepare or --shard')
