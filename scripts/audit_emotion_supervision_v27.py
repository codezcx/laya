"""Source-only audit of stimulus labels versus available subjective ratings."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from eeg_tasks.faced_prepare import read_table, video_events


def main():
    root = Path('artifacts/emotion_supervision_v27')
    root.mkdir(exist_ok=False)
    source = Path('artifacts/faced_full_v22')
    plan = json.loads((source / 'cohort.json').read_text())
    rows = [json.loads(s) for s in Path('artifacts/faced_scale_v22/rows.jsonl').read_text().splitlines()]
    records, files, subjects = [], {}, []
    for subject in sorted(plan['train'] + plan['validation']):
        split = 'train' if subject in plan['train'] else 'validation'
        path = source / 'raw' / f'sub-{subject:03d}' / f'sub-{subject:03d}_task-watchingVideoClips_events.tsv'
        events = video_events(read_table(path))
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        by_video = {str(e['video_index']): e for e in events}
        rr = [r for r in rows if r['dataset'] == 'FACED-NEMAR' and r['subject'] == str(subject)]
        assert {r['trial'] for r in rr} == set(by_video)
        for r in rr:
            e = by_video[r['trial']]
            assert r['class_id'] == {'negative': 0, 'neutral': 1, 'positive': 2}[e['binary_label']]
            assert r['event_onset_seconds'] == float(e['onset']) and r['split'] == split
        ratings = []
        for e in events:
            values = {}
            for field in ['Valence', 'Arousal', 'Joy', 'Tenderness', 'Inspiration', 'Amusement', 'Anger', 'Disgust', 'Fear', 'Sadness']:
                try:
                    v = float(e[field])
                except (ValueError, TypeError):
                    v = None
                values[field] = v if v is not None and np.isfinite(v) else None
            record = {'subject': subject, 'split': split, 'video': int(e['video_index']),
                      'stimulus_label': e['binary_label'], 'target_emotion': e['emotion_label'],
                      'ratings': values}
            records.append(record)
            ratings.append(record)
        means = {c: float(np.mean([r['ratings']['Valence'] for r in ratings
                                  if r['stimulus_label'] == c and r['ratings']['Valence'] is not None]))
                 for c in ['negative', 'neutral', 'positive']}
        subjects.append({'subject': subject, 'split': split, 'valence_means': means,
                         'positive_minus_negative': means['positive'] - means['negative']})
    summary = {}
    for split in ['train', 'validation']:
        rr = [r for r in records if r['split'] == split]
        ss = [s for s in subjects if s['split'] == split]
        summary[split] = {'subjects': len(ss), 'trials': len(rr),
            'missing_valence': sum(r['ratings']['Valence'] is None for r in rr),
            'valence_outside_0_7': sum(r['ratings']['Valence'] is not None and not 0 <= r['ratings']['Valence'] <= 7 for r in rr),
            'subjects_positive_mean_not_above_negative': [s['subject'] for s in ss if s['positive_minus_negative'] <= 0],
            'by_stimulus_class': {}}
        for c in ['negative', 'neutral', 'positive']:
            v = [r['ratings']['Valence'] for r in rr if r['stimulus_label'] == c and r['ratings']['Valence'] is not None]
            summary[split]['by_stimulus_class'][c] = {'n': len(v), 'mean_valence': float(np.mean(v)),
                'valence_10_50_90_percentiles': np.quantile(v, [.1, .5, .9]).tolist()}
    overlaps = {}
    for ds in ['SEED', 'FACED-NEMAR']:
        # Trial index is only a stimulus proxy for SEED; do not assert cross-session clip identity.
        sets = {split: {r['trial'] for r in rows if r['dataset'] == ds and r['split'] == split}
                for split in ['train', 'validation']}
        overlaps[ds] = {'train_trial_ids': len(sets['train']), 'validation_trial_ids': len(sets['validation']),
                        'overlap_trial_ids': len(sets['train'] & sets['validation'])}
    result = {'source_revision': plan['source_revision'], 'summary': summary, 'per_subject': subjects,
              'trial_id_overlap': overlaps, 'event_file_hashes': files,
              'note': 'Descriptive source metadata audit. No relabeling, subject exclusion, threshold selection or target label access. Rating semantics require provenance review before training.'}
    (root / 'ratings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (root / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'summary': summary, 'trial_id_overlap': overlaps}, indent=2))


if __name__ == '__main__':
    main()
