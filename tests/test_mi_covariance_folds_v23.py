from eeg_tasks.mi_covariance_fit_v23 import DOMAINS, fold_indices


def test_whole_domain_exclusion_and_no_source_tests():
    rows = [dict(dataset=d, subject=s, split=split)
            for d in DOMAINS
            for s, split in [('1', 'train'), ('2', 'validation'), ('3', 'test')]]
    for held in DOMAINS:
        train, validation = fold_indices(rows, held)
        assert len(train) == 2 and len(validation) == 1
        assert all(rows[i]['dataset'] != held and rows[i]['split'] == 'train' for i in train)
        assert all(rows[i]['dataset'] == held and rows[i]['split'] == 'validation' for i in validation)
