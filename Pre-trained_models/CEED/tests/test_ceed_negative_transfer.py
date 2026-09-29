import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FOLDER = Path(__file__).resolve().parents[1] / 'train_picker'
spec = importlib.util.spec_from_file_location(
    'transfer', Path(__file__).resolve().parents[1] / 'helpers/ceed_negative_transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TransferTests(unittest.TestCase):
    def test_helpers_live_in_ceed(self):
        for name in ('ceed_data_pipeline.py', 'ceed_negative_transfer.py'):
            self.assertFalse((FOLDER / name).exists())
            self.assertTrue((Path(__file__).resolve().parents[1] / 'helpers' / name).exists())
            self.assertFalse((Path(__file__).resolve().parents[3] / 'PAL_src' / name).exists())
        for name in ('1_cut_train-samples_ceed.py',
                     '2.1_build_pos_zarr.py', '3_train_ceed.py'):
            self.assertIn('sys.path.insert(0, str(AI_PAL_ROOT /',
                          (FOLDER / name).read_text())

    def make_store(self, root, prefix, counts):
        for split, count in zip(('train', 'valid'), counts):
            for name, tail in [('data', [3, 2500]), ('target_frame', [246]),
                               ('target_sample', [3, 2500])]:
                array = root / split / (prefix + '_' + name)
                array.mkdir(parents=True)
                (array / '.zarray').write_text(json.dumps({'shape': [count] + tail}))
                (array / '0').write_text(split + ':' + prefix)

    def test_transfer_preserves_splits_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'local', Path(tmp) / 'ceed'
            self.make_store(src, 'negative', [10, 3])
            self.make_store(dst, 'positive', [20, 4])
            models = ['SAR', 'FT', 'PHN', 'RUN']
            transfer.copy_local_negatives(src, dst, models)
            transfer.validate_mixed_store(dst, models)
            for split in ('train', 'valid'):
                self.assertEqual((dst / split / 'negative_data' / '0').read_text(), split + ':negative')
                self.assertEqual((dst / split / 'positive_data' / '0').read_text(), split + ':positive')
                self.assertTrue((src / split / 'negative_data').exists())
            with self.assertRaises(FileExistsError):
                transfer.copy_local_negatives(src, dst, models)

    def test_mismatch_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'local', Path(tmp) / 'ceed'
            self.make_store(src, 'negative', [10, 3])
            self.make_store(dst, 'positive', [20, 4])
            (dst / 'valid' / 'positive_data' / '.zarray').write_text('{"shape": [4, 3, 100]}')
            with self.assertRaises(ValueError):
                transfer.copy_local_negatives(src, dst, ['SAR', 'PHN'])
            self.assertFalse((dst / 'train' / 'negative_data').exists())

    def test_launchers_are_mixed(self):
        text = (FOLDER / '3_train_ceed.py').read_text()
        self.assertNotIn('--positive_only', text)
        self.assertIn('.negative_transfer_in_progress', text)
        for model, negative in [('sar', 16), ('ft', 16), ('phn', 4), ('run', 4)]:
            self.assertIn('self.batch_size = [128, {}]'.format(negative),
                          (FOLDER / ('config_' + model + '_global_ceed.py')).read_text())


if __name__ == '__main__':
    unittest.main()
