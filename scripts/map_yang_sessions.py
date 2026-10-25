"""Map processed Yang2025 sessions to native records using ordered labels.

Reads bounded ZIP byte ranges, validates deflate suffixes and inner MAT streams.
Does not execute downloaded code or infer chronological order from filenames.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import struct
import threading
import time
import zlib

import mne
import numpy as np
import requests
from scipy.io import loadmat


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Archive:
    def __init__(self, inventory, cache):
        self.inventory = inventory
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()

    def read(self, start, count):
        assert 0 < count <= 262144
        p = self.cache/f'{start}_{count}.bin'
        if p.exists():
            data = p.read_bytes()
            assert len(data) == count
            return data
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        end = start+count-1
        for attempt in range(3):
            try:
                with self.local.session.get(self.inventory['url'],
                        headers={'Range': f'bytes={start}-{end}'},
                        stream=True, timeout=(10, 30)) as response:
                    assert response.status_code == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{self.inventory["bytes"]}'
                    data = response.raw.read(count+1)
                    assert len(data) == count
                p.write_bytes(data)
                return data
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1)

    def header(self, entry):
        raw = self.read(entry['offset'], 512)
        fields = struct.unpack('<4s5H3I2H', raw[:30])
        assert fields[0] == b'PK\x03\x04'
        assert fields[3] == entry['compression'] == 8
        return entry['offset']+30+fields[-2]+fields[-1], fields[6]

    def event_file(self, entry, out):
        start, crc = self.header(entry)
        compressed = self.read(start, entry['compressed_bytes'])
        raw = zlib.decompress(compressed, -15)
        assert len(raw) == entry['bytes'] and zlib.crc32(raw) == crc
        out.write_bytes(raw)
        annotations = mne.read_annotations(str(out))
        indices = [i for i, d in enumerate(annotations.description) if d in ['1', '2']]
        labels = np.array([int(annotations.description[i]) for i in indices], dtype=int)
        return dict(labels=labels.tolist(), onsets=annotations.onset[indices].tolist(),
                    entry_crc_verified=True, extracted_sha256=sha(raw),
                    header_date=raw[168:176].decode(), header_time=raw[176:184].decode())

    def mat_labels(self, entry, mat_header):
        start, _ = self.header(entry)
        size = min(131072, entry['compressed_bytes'])
        tail = self.read(start+entry['compressed_bytes']-size, size)
        candidates = {}
        for i in range(len(tail)-5):
            # Stored blocks are byte-aligned after their padding. Consecutive
            # stored blocks end on a byte boundary; accept complete suffixes only.
            if tail[i] & 6:
                continue
            n, complement = struct.unpack_from('<HH', tail, i+1)
            if n ^ complement != 65535 or i+5+n > len(tail):
                continue
            try:
                decoder = zlib.decompressobj(-15)
                suffix = decoder.decompress(tail[i:], 262144)
            except zlib.error:
                continue
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                continue
            for match in re.finditer(b'\x0f\x00\x00\x00', suffix):
                pos = match.start()
                if pos+8 > len(suffix):
                    continue
                length = struct.unpack_from('<I', suffix, pos+4)[0]
                if not 0 < length <= 4096 or pos+8+length != len(suffix):
                    continue
                block = suffix[pos:pos+8+length]
                try:
                    # This verifies the inner zlib Adler32 before MAT parsing.
                    zlib.decompress(block[8:])
                    variables = loadmat(io.BytesIO(mat_header+block))
                    labels = np.asarray(variables['labels']).reshape(-1)
                except (ValueError, KeyError, zlib.error):
                    continue
                if not 150 <= len(labels) <= 250 or not set(labels) <= {1, 2}:
                    continue
                labels = labels.astype(int)
                candidates[tuple(labels)] = dict(
                    labels=labels.tolist(), label_variable_sha256=sha(block),
                    inner_adler_verified=True, outer_suffix_eof_verified=True,
                    whole_mat_crc_verified=False, tail_sha256=sha(tail),
                    source_tail_start=start+entry['compressed_bytes']-size,
                    suffix_output_bytes=len(suffix))
        assert len(candidates) == 1, f'Unique validated labels not found: {entry["path"]}'
        return next(iter(candidates.values()))


def subsequence(short, long):
    k = 0
    for value in long:
        if k < len(short) and short[k] == value:
            k += 1
    return k == len(short)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--mat-header-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 52)))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4 and set(args.subjects) <= set(range(1, 52))
    args.out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(args.inventory.read_text())
    archive = Archive(inv, args.out/'ranges')
    mat_header = args.mat_header_source.read_bytes()[:128]
    assert mat_header[126:128] == b'IM'
    jobs = []
    for entry in inv['entries']:
        path = entry['path']
        pattern = re.search(r'/sub-(\d+)/ses-(\d+)/', path)
        if not pattern:
            continue
        subject, session = map(int, pattern.groups())
        if subject not in args.subjects:
            continue
        if '/derivatives/2C dataset_processeddata/' in path and path.endswith('.mat'):
            jobs.append((subject, session, 'processed', entry))
        if '/sourcedata/2C dataset/' in path and path.endswith('/evt.bdf'):
            jobs.append((subject, session, 'native', entry))
    assert len(jobs) == 6*len(args.subjects)

    def fetch(job):
        subject, session, kind, entry = job
        try:
            if kind == 'processed':
                result = archive.mat_labels(entry, mat_header)
            else:
                result = archive.event_file(entry, args.out/f'native_{subject:03d}_{session:02d}_evt.bdf')
            result.update(subject=subject, session=session, kind=kind, archive_path=entry['path'])
            return result
        except Exception as error:
            return dict(subject=subject, session=session, kind=kind,
                        archive_path=entry['path'], error_type=type(error).__name__)

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            records.append(future.result())
            if len(records) % 30 == 0:
                print(json.dumps(dict(completed=len(records), total=len(jobs),
                                      errors=sum('error_type' in r for r in records))), flush=True)
    records.sort(key=lambda r: (r['subject'], r['kind'], r['session']))
    mappings = []
    for subject in args.subjects:
        selected = [r for r in records if r['subject'] == subject]
        if any('error_type' in r for r in selected):
            mappings.append(dict(subject=subject, status='fetch_or_parse_incomplete'))
            continue
        processed = sorted([r for r in selected if r['kind'] == 'processed'], key=lambda r: r['session'])
        native = sorted([r for r in selected if r['kind'] == 'native'], key=lambda r: r['session'])
        matches = [[subsequence(p['labels'], n['labels']) for n in native] for p in processed]
        perms = [order for order in itertools.permutations(range(3))
                 if all(matches[i][order[i]] for i in range(3))]
        mappings.append(dict(subject=subject, status='unique' if len(perms) == 1 else 'ambiguous_or_unmatched',
                             candidate_permutations=[[v+1 for v in order] for order in perms],
                             processed_n=[len(r['labels']) for r in processed],
                             native_n=[len(r['labels']) for r in native],
                             equality_matrix=[[p['labels'] == n['labels'] for n in native] for p in processed],
                             subsequence_matrix=matches))
    result = dict(inventory_sha256=sha(args.inventory.read_bytes()), script_sha256=sha(Path(__file__).read_bytes()),
                  mat_header_sha256=sha(mat_header), subjects=args.subjects, records=records, mappings=mappings,
                  limitation='Ordered label identity/subsequence supports record matching; not full waveform equivalence.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status_counts={s: sum(m['status'] == s for m in mappings)
                                        for s in sorted({m['status'] for m in mappings})},
                          mappings=mappings if len(mappings) <= 3 else None)), flush=True)


if __name__ == '__main__':
    main()
