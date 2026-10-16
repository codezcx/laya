from eeg_tasks.emotion_rating_v28 import split_indices


def test_subject_and_stimulus_holdout_and_source_test_exclusion():
    rows = [dict(dataset=d, subject=s, split=split, trial=str(v))
            for d in ['SEED', 'FACED-NEMAR']
            for s, split in [('1', 'train'), ('2', 'validation'), ('3', 'test')]
            for v in [1, 2]]
    train, val = split_indices(rows, {1})
    assert all(rows[i]['split'] == 'train' for i in train)
    assert all(rows[i]['trial'] == '2' for i in train if rows[i]['dataset'] == 'FACED-NEMAR')
    assert all(rows[i]['subject'] == '2' and rows[i]['trial'] == '1' for i in val['FACED_unseen'])
    assert len(val['SEED']) == 2 and len(val['FACED_seen']) == 1
