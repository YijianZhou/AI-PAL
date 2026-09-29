import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import zarr
except ImportError:
    zarr = None

spec = importlib.util.spec_from_file_location(
    'transfer', Path(__file__).resolve().parents[1] / 'helpers/ceed_negative_transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class LauncherPlacementTests(unittest.TestCase):
    def test_special_launcher_is_in_socal_ceed_training(self):
        root = Path(__file__).resolve().parents[3]
        name = '2.2_copy_neg_zarr_ceed.py'
        launcher = root.parent / 'SoCal_workdir/train_picker_CEED' / name
        self.assertTrue(launcher.is_file())
        self.assertIn('Pre-trained_models/CEED/helpers', launcher.read_text())
        self.assertFalse((root / '2_train_picker' / 'train_picker_local' / name).exists())


@unittest.skipIf(zarr is None, 'Zarr is required for integration tests')
class AnnualTransferTests(unittest.TestCase):
    def build(self, root, prefix, count, value, fmt):
        for split in ('train', 'valid'):
            for name, tail, dtype in [('data', (3, 8), 'float32'),
                                      ('target_frame', (4,), 'int32'),
                                      ('target_sample', (3, 8), 'float32')]:
                array = zarr.open(str(root / split / (prefix + '_' + name)),
                                  mode='w', shape=(count,) + tail,
                                  chunks=(4,) + tail, dtype=dtype, zarr_format=fmt)
                array[:] = value + (10 if split == 'valid' else 0)

    def test_concat_preserves_values_splits_and_positives(self):
        for fmt in (2, 3):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as temp:
                source, dest = Path(temp) / 'annual', Path(temp) / 'ceed'
                self.build(source / '2020.zarr', 'negative', 3, 1, fmt)
                self.build(source / '2021.zarr', 'negative', 5, 2, fmt)
                self.build(dest, 'positive', 2, 9, fmt)
                before = {str(p.relative_to(dest)): p.read_bytes()
                          for p in dest.rglob('*') if p.is_file()}
                models = ['SAR', 'FT', 'PHN', 'RUN']
                transfer.copy_annual_negatives(source, dest, (2020, 2021), models, 2)
                for split in ('train', 'valid'):
                    delta = 10 if split == 'valid' else 0
                    for name in transfer.array_names(models):
                        array = zarr.open(str(dest / split / ('negative_' + name)), mode='r')
                        self.assertEqual(array.shape[0], 8)
                        self.assertTrue(np.all(array[:3] == 1 + delta))
                        self.assertTrue(np.all(array[3:] == 2 + delta))
                for name, content in before.items():
                    self.assertEqual((dest / name).read_bytes(), content)
                self.assertEqual(json.loads((dest / 'local_negative_source.json').read_text())
                                 ['negative_counts'], {'train': 8, 'valid': 8})
                transfer.validate_mixed_store(dest, models)
                transfer.copy_annual_negatives(source, dest, (2020, 2021), models, 2)

    def test_resume_after_copy_or_publication_interruption(self):
        for fmt in (2, 3):
            for failure in ('copy', 'partial', 'publish'):
                with self.subTest(fmt=fmt, failure=failure), tempfile.TemporaryDirectory() as temp:
                    source, dest = Path(temp) / 'annual', Path(temp) / 'ceed'
                    self.build(source / '2020.zarr', 'negative', 3, 1, fmt)
                    self.build(source / '2021.zarr', 'negative', 9, 2, fmt)
                    self.build(dest, 'positive', 2, 9, fmt)
                    original_save = transfer._save_transfer_json
                    original_rename = Path.rename
                    def save(path, value):
                        if (failure == 'partial' and path.name == '.negative_transfer_state.json'
                                and value['rows'].get('train/negative_data') == 8):
                            chunk = dest / '.negative_transfer_stage/train/negative_data'
                            chunk = chunk / ('1.0.0' if fmt == 2 else 'c/1/0/0')
                            chunk.write_bytes(b'interrupted write')
                            raise RuntimeError('disconnect')
                        original_save(path, value)
                        if failure == 'copy' and path.name == '.negative_transfer_state.json' and value['rows'].get('train/negative_data') == 4:
                            raise RuntimeError('disconnect')
                    def rename(path, target):
                        result = original_rename(path, target)
                        if path.name == 'negative_data':
                            raise RuntimeError('disconnect')
                        return result
                    manager = (patch.object(transfer, '_save_transfer_json', side_effect=save)
                               if failure != 'publish' else patch.object(Path, 'rename', rename))
                    with manager, self.assertRaisesRegex(RuntimeError, 'disconnect'):
                        transfer.copy_annual_negatives(source, dest, (2020, 2021), ['SAR'], 2)
                    with self.assertRaises(RuntimeError):
                        transfer.validate_mixed_store(dest, ['SAR'])
                    with self.assertRaises(ValueError):
                        transfer.copy_annual_negatives(source, dest, (2020,), ['SAR'], 2)
                    transfer.copy_annual_negatives(source, dest, (2020, 2021), ['SAR'], 3)
                    for split in ('train', 'valid'):
                        for name in transfer.array_names(['SAR']):
                            arr = zarr.open(str(dest / split / ('negative_' + name)), mode='r')
                            delta = 10 if split == 'valid' else 0
                            self.assertTrue(np.all(arr[:3] == 1 + delta))
                            self.assertTrue(np.all(arr[3:] == 2 + delta))
                    self.assertFalse((dest / '.negative_transfer_in_progress').exists())

    def test_legacy_restart_requires_opt_in_and_lock_excludes_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp) / 'annual', Path(temp) / 'ceed'
            self.build(source / '2020.zarr', 'negative', 3, 1, 2)
            self.build(dest, 'positive', 2, 9, 2)
            (dest / '.negative_transfer_in_progress').touch()
            with self.assertRaises(FileExistsError):
                transfer.copy_annual_negatives(source, dest, (2020,), ['SAR'])
            with transfer._transfer_guard(dest), self.assertRaises(RuntimeError):
                transfer.copy_annual_negatives(source, dest, (2020,), ['SAR'], restart_legacy=True)
            transfer.copy_annual_negatives(source, dest, (2020,), ['SAR'], restart_legacy=True)

    def test_missing_year_does_not_modify_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp) / 'annual', Path(temp) / 'ceed'
            self.build(source / '2020.zarr', 'negative', 3, 1, 2)
            self.build(dest, 'positive', 2, 9, 2)
            with self.assertRaises(FileNotFoundError):
                transfer.copy_annual_negatives(source, dest, (2020, 2021), ['SAR'])
            self.assertFalse((dest / 'train' / 'negative_data').exists())
            self.assertFalse((dest / '.negative_transfer_in_progress').exists())

    def test_incomplete_transfer_blocks_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '.negative_transfer_in_progress').touch()
            with self.assertRaises(RuntimeError):
                transfer.validate_mixed_store(root, ['SAR'])


if __name__ == '__main__':
    unittest.main()
