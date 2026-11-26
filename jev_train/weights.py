"""Read-only completeness/integrity check of the user's independently downloaded weights."""
import json
from pathlib import Path

from safetensors import safe_open
from .runtime import CONFIG, ROOT, backbone_files, digest, write_json


def verify_weights():
    paths = backbone_files()
    root = Path(CONFIG['backbone_path'])
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    actual, shards = set(), {}
    for path in paths:
        if path.suffix != '.safetensors':
            continue
        expected = {name for name, shard in index['weight_map'].items() if shard == path.name}
        stat = path.stat()
        with safe_open(path, framework='pt', device='cpu') as f:
            keys = set(f.keys())
            if keys != expected:
                raise ValueError(f'Shard index/key mismatch: {path.name}')
            for name in keys:
                if any(d <= 0 for d in f.get_slice(name).get_shape()):
                    raise ValueError(f'Invalid tensor shape: {name}')
            actual.update(keys)
        sha = digest(path)
        if sha != CONFIG['backbone_shard_sha256'].get(path.name):
            raise ValueError(f'Official HuggingFace LFS SHA256 mismatch: {path.name}')
        if (stat.st_size, stat.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise RuntimeError('Weights changed during verification')
        shards[path.name] = dict(bytes=stat.st_size, sha256=sha, tensors=len(keys))
    if actual != set(index['weight_map']):
        raise ValueError('Missing indexed tensors')
    result = dict(passed=True, backbone=CONFIG['backbone_path'], shards=shards,
                  tensor_count=len(actual), remote_checksum_verified=True,
                  note='SHA256 checked against the pinned official HuggingFace LFS metadata')
    write_json(ROOT / CONFIG['output_root'] / 'weight_audit.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify_weights()), flush=True)
