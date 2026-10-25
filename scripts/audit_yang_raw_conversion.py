"""Bounded subject-1 native/NEMAR BDF signal comparison; no model fitting.

Downloads only ZIP compressed prefixes and corresponding converted prefixes.
Partial native entries cannot pass whole-entry CRC validation; report this limit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zlib
from urllib.parse import quote

import numpy as np
import requests
import mne

from map_yang_sessions import Archive


def header(raw):
    n = int(raw[252:256])
    size = int(raw[184:192])
    assert size == 256 * (n + 1) and len(raw) >= size
    fields = {}
    pos = 256
    for name, width in [('labels', 16), ('transducer', 80), ('units', 8),
                        ('pmin', 8), ('pmax', 8), ('dmin', 8), ('dmax', 8),
                        ('prefilter', 80), ('samples', 8), ('reserved', 32)]:
        fields[name] = [raw[pos+i*width:pos+(i+1)*width].decode().strip()
                        for i in range(n)]
        pos += n * width
    for name in ['pmin', 'pmax', 'dmin', 'dmax']:
        fields[name] = np.array(fields[name], dtype=float)
    fields['samples'] = np.array(fields['samples'], dtype=int)
    fields.update(size=size, duration=float(raw[244:252]), records=int(raw[236:244]),
                  record_bytes=int(3*sum(fields['samples'])))
    return fields


def decode(raw, h, records):
    b = np.frombuffer(raw[h['size']:h['size']+records*h['record_bytes']],
                      dtype=np.uint8).reshape(records, -1, 3).astype(np.int32)
    digital = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    digital = np.where(digital >= 2**23, digital - 2**24, digital)
    output = {}
    pos = 0
    for i, name in enumerate(h['labels']):
        samples = h['samples'][i]
        values = digital[:, pos:pos+samples].reshape(-1)
        pos += samples
        if 'Annotations' in name:
            continue
        assert name not in output
        scale = (h['pmax'][i]-h['pmin'][i])/(h['dmax'][i]-h['dmin'][i])
        output[name] = (values-h['dmin'][i])*scale+h['pmin'][i]
    return output


def range_read(url, count, path, start=0):
    if path.exists():
        raw = path.read_bytes()
        assert len(raw) == count
        return raw
    end = start+count-1
    with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                      timeout=(10, 40), stream=True) as response:
        assert response.status_code == 206
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        raw = response.raw.read(count+1)
        assert len(raw) == count
    path.write_bytes(raw)
    return raw


def fragment(path, raw, h, records):
    # Preserve first records, changing only advertised record count for MNE.
    buf = bytearray(raw[:h['size']+records*h['record_bytes']])
    buf[236:244] = str(records).ljust(8).encode()
    path.write_bytes(buf)


def verify_fragments(out, sessions):
    """Check the manual decoder with MNE, including unknown native units."""
    checks = []
    for session in sessions:
        s = session['native_session']
        for kind in ['native', 'converted']:
            path = out/f'{kind}_{s}_fragment.bdf'
            raw = path.read_bytes()
            h = header(raw)
            values = decode(raw, h, session['records'])
            reader = mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
            # MNE leaves unrecognized '?V' numerical values unscaled. It
            # converts recognized uV to V. This verifies software, not units.
            scale = 1 if kind == 'native' else 1e-6
            difference = max(float(np.max(np.abs(
                reader.get_data(picks=[name])[0]-value*scale)))
                for name, value in values.items())
            assert difference < (1e-9 if kind == 'native' else 1e-15)
            checks.append(dict(session=s, kind=kind, channels=len(values),
                samples=int(reader.n_times), max_abs_difference=difference,
                mne_scale_from_header_numbers=scale,
                fragment_sha256=hashlib.sha256(raw).hexdigest()))
    result = dict(mne_version=mne.__version__, checks=checks,
        scope='Independent MNE decoder versus manual decoder on first records; physical unit truth not inferred')
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(args.inventory.read_text())
    archive = Archive(inv, args.out/'ranges')
    results = []
    for session in [1, 2, 3]:
        entry, = [e for e in inv['entries']
                  if f'/sourcedata/2C dataset/sub-001/ses-{session:02d}/' in e['path']
                  and e['path'].endswith('/data.bdf')]
        start, _ = archive.header(entry)
        compressed = archive.read(start, 262144)
        decoder = zlib.decompressobj(-15)
        native = decoder.decompress(compressed, 4*1024*1024)
        assert not decoder.eof and not decoder.unconsumed_tail
        nh = header(native)
        nrecords = min(2, (len(native)-nh['size'])//nh['record_bytes'])
        assert nrecords >= 1
        url = ('https://data.nemar.org/nm000348/v1.0.5/'
               f'sub-1/ses-{session-1}/eeg/sub-1_ses-{session-1}_task-imagery_run-0_eeg.bdf')
        ch = header(range_read(url, 16896, args.out/f'converted_{session}_header.bin'))
        assert nh['duration'] == ch['duration']
        converted = range_read(url, ch['size']+nrecords*ch['record_bytes'],
                               args.out/f'converted_{session}_prefix.bin')
        a, b = decode(native, nh, nrecords), decode(converted, ch, nrecords)
        assert set(a) == set(b)
        channels = []
        for name in a:
            ni, ci = nh['labels'].index(name), ch['labels'].index(name)
            assert nh['samples'][ni] == ch['samples'][ci]
            x, y = a[name], b[name]
            xc, yc = x-x.mean(), y-y.mean()
            slope = float(xc@yc/(xc@xc)) if xc@xc else None
            channels.append(dict(channel=name, native_unit_field=nh['units'][ni],
                converted_unit_field=ch['units'][ci],
                native_range=[float(x.min()), float(x.max())],
                converted_range=[float(y.min()), float(y.max())],
                slope=slope, intercept=float(y.mean()-slope*x.mean()) if slope is not None else None,
                correlation=float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None,
                max_abs_numeric_difference=float(np.max(np.abs(y-x))),
                max_abs_affine_residual=float(np.max(np.abs(y-(slope*x+y.mean()-slope*x.mean())))) if slope is not None else None,
                native_quantization=float((nh['pmax'][ni]-nh['pmin'][ni])/(nh['dmax'][ni]-nh['dmin'][ni])),
                converted_quantization=float((ch['pmax'][ci]-ch['pmin'][ci])/(ch['dmax'][ci]-ch['dmin'][ci]))))
        for label, raw, h in [('native', native, nh), ('converted', converted, ch)]:
            fragment(args.out/f'{label}_{session}_fragment.bdf', raw, h, nrecords)
        # The mirror also carries native sourcedata. Check its signal prefix
        # against the author ZIP, then sample fixed record positions throughout.
        source_url = 'https://data.nemar.org/nm000348/v1.0.5/'+quote(
            f'sourcedata/2C dataset/sub-001/ses-{session:02d}/eeg/data.bdf')
        mh_raw = range_read(source_url, nh['size'], args.out/f'mirror_{session}_header.bin')
        mh = header(mh_raw)
        for field in ['labels', 'units', 'samples', 'pmin', 'pmax', 'dmin', 'dmax']:
            assert np.array_equal(nh[field], mh[field])
        assert nh['records'] == mh['records'] == ch['records']
        sparse_checks = []
        for record in [0, nh['records']//4, nh['records']//2,
                       3*nh['records']//4, nh['records']-nrecords]:
            mr = range_read(source_url, nrecords*mh['record_bytes'],
                args.out/f'mirror_{session}_record_{record}.bin',
                mh['size']+record*mh['record_bytes'])
            cr = range_read(url, nrecords*ch['record_bytes'],
                args.out/f'converted_{session}_record_{record}.bin',
                ch['size']+record*ch['record_bytes'])
            if record == 0:
                assert mr == native[nh['size']:nh['size']+len(mr)]
            ma = decode(mh_raw+mr, mh, nrecords)
            ca = decode(converted[:ch['size']]+cr, ch, nrecords)
            errors = []
            for name, x in ma.items():
                i = ch['labels'].index(name)
                q = (ch['pmax'][i]-ch['pmin'][i])/(ch['dmax'][i]-ch['dmin'][i])
                errors.append(float(np.max(np.abs(ca[name]-.001*x)))/q)
            assert max(errors) <= 1.0
            sparse_checks.append(dict(first_record=record, records=nrecords,
                channels=len(ma), max_error_in_converted_quantization_steps=max(errors),
                mirror_sha256=hashlib.sha256(mr).hexdigest(),
                converted_sha256=hashlib.sha256(cr).hexdigest()))
        results.append(dict(subject=1, native_session=session, converted_url=url,
            source_mirror_url=source_url, mirror_prefix_matches_author_signal=True,
            sparse_checks=sparse_checks,
            archive_path=entry['path'], compressed_bytes=len(compressed), records=nrecords,
            duration_seconds=nrecords*nh['duration'], whole_native_crc_verified=False,
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            converted_sha256=hashlib.sha256(converted).hexdigest(), channels=channels))
        print(json.dumps(dict(session=session, seconds=nrecords*nh['duration'],
            channels=len(channels), slope_range=[min(c['slope'] for c in channels if c['slope'] is not None), max(c['slope'] for c in channels if c['slope'] is not None)])))
    result = dict(scope='Subject 1 three sessions: author ZIP prefix and five fixed record positions via native mirror; no trial or model outcomes',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sessions=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    verify_fragments(args.out, results)


if __name__ == '__main__':
    main()
