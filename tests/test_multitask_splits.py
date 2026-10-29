import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    'prepare_multitask_splits', Path(__file__).resolve().parents[1] / 'scripts/prepare_multitask_splits.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestMultitaskSplits(unittest.TestCase):
    def rows(self, subjects):
        return [dict(subject=s, record_id=f'record-{i}') for i, s in enumerate(subjects)]

    def test_bcic_fixed_subjects_and_all_sessions(self):
        df = MODULE.allocate('bcic_2a', self.rows(list(range(1, 10)) * 2))
        self.assertEqual(set(df[df.split == 'train'].subject), set(range(1, 6)))
        self.assertEqual(set(df[df.split == 'valid'].subject), {6, 7})
        self.assertEqual(set(df[df.split == 'test'].subject), {8, 9})

    def test_physionet_fixed_subjects(self):
        df = MODULE.allocate('motor_mv_img', self.rows(range(1, 110)))
        self.assertEqual(df.groupby('split').size().to_dict(), dict(train=69, valid=19, test=21))

    def test_seed_reproducible_and_rng_not_changed(self):
        rows = self.rows(list(range(1, 16)) * 3)
        np.random.seed(102)
        state = np.random.get_state()
        a = MODULE.allocate('seed', rows)
        observed = np.random.rand()
        np.random.set_state(state)
        self.assertEqual(observed, np.random.rand())
        b = MODULE.allocate('seed', rows)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.groupby('split').subject.nunique().to_dict(), dict(train=13, valid=1, test=1))

    def test_subject_leak_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Subject'):
            MODULE.validate(pd.DataFrame(dict(subject=['a','a'], record_id=['1','2'], split=['train','test'])))

    def test_duplicate_record_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            MODULE.allocate('bcic_2a', [dict(subject=1, record_id='x')] * 2)

    def test_extra_data_cannot_contain_eval(self):
        with self.assertRaisesRegex(ValueError, 'train-only'):
            MODULE.validate(pd.DataFrame(dict(subject=['a'], record_id=['1'], split=['test'])), train_only=True)

    def test_cross_dataset_names_are_flagged_not_silently_reassigned(self):
        tables = {d: pd.DataFrame([dict(subject='p', record_id=d, split=s, patient_key='')])
                  for d, s in [('tuab', 'train'), ('tuev', 'test')]}
        report = MODULE.cross_tuh_audit(tables)
        self.assertEqual(len(report['candidate_filename_role_conflicts']), 1)
        self.assertEqual(report['records_without_usable_code'], 2)
        self.assertEqual(tables['tuab'].split.iloc[0], 'train')

    def test_all_pinned_split_methods_extract(self):
        for dataset in MODULE.DEFINITIONS:
            self.assertTrue(callable(MODULE.splitter(dataset)._divide_split))

    def test_edf_placeholder_is_not_a_patient(self):
        header = bytearray(b' ' * 256)
        header[8:11] = b'/XX'
        header[236:244] = b'10      '
        header[244:252] = b'1       '
        header[252:256] = b'1   '
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.edf'
            path.write_bytes(header + b'C3              ')
            result = MODULE.edf_metadata(path)
        self.assertEqual(result['patient_key'], '')
        self.assertEqual(result['time'], 10)


if __name__ == '__main__':
    unittest.main()
