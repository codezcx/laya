"""Current BSR/JEV recipe and content hashing shared by training/queue."""
import hashlib,json
from pathlib import Path
CONFIG=json.loads(Path('configs/benchmark_train.json').read_text())

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
