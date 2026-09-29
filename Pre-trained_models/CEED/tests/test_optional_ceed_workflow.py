"""Optional CEED branch layout, shared settings and placeholder safety."""
import ast
from pathlib import Path
import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest
import numpy as np  # Load before patch.dict restores the module cache.

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / 'Pre-trained_models/CEED/preprocess'
TRAIN = ROOT / 'Pre-trained_models/CEED/train_picker'
LOCAL = ROOT / '2_train_picker/train_picker_local'
sys.path.insert(0, str(ROOT / 'Pre-trained_models/CEED/helpers'))
from ceed_workflow_config import load_config


def test_numbered_workflows_and_syntax():
    for folder, names in [
        (PRE, ['0_download_ceed_raw.py', '1_extract_ceed_phase.py', '2_build_ceed_fixed_window_npy.py']),
        (TRAIN, ['0_analyze_phase_rarity_ceed.py', '1_cut_train-samples_ceed.py', '2.1_build_pos_zarr.py', '2.2_copy_neg_zarr_ceed.py', '3_train_ceed.py']),
        (LOCAL, ['0_analyze_phase_rarity_eg.py', '1_cut_train-samples_eg.py', '2_npy2zarr_eg.py', '3_train_eg.py']),
    ]:
        for name in names:
            assert (folder / name).is_file()
        for path in folder.glob('*.py'):
            ast.parse(path.read_text())
    assert not list(LOCAL.glob('*ceed*'))
    assert not list(TRAIN.glob('config_ai_pal*.py'))
    assert (PRE / 'config_ai_pal_ceed.py').is_file()


def test_dataset_helpers_are_outside_common_source():
    helpers = PRE.parent / 'helpers'
    for name in ('ceed_data_pipeline.py', 'ceed_negative_transfer.py', 'ceed_workflow_config.py'):
        assert (helpers / name).is_file()
        assert not (ROOT / 'PAL_src' / name).exists()
        ast.parse((helpers / name).read_text())
    for script in (TRAIN / '0_analyze_phase_rarity_ceed.py', TRAIN / '1_cut_train-samples_ceed.py',
                   TRAIN / '2.1_build_pos_zarr.py', PRE / '2_build_ceed_fixed_window_npy.py'):
        assert "Path(__file__).resolve().parents[1] / 'helpers'" in script.read_text()


@pytest.mark.parametrize('script', ['2.1_build_pos_zarr.py', '2.2_copy_neg_zarr_ceed.py', '3_train_ceed.py'])
def test_placeholder_guard_precedes_source_staging(script):
    with patch('shutil.copyfile') as copy, patch('subprocess.check_call') as run:
        with pytest.raises(ValueError, match='DATASET_SETTINGS_CONFIRMED'):
            runpy.run_path(str(TRAIN / script), run_name='__main__')
    copy.assert_not_called()
    run.assert_not_called()


def test_split_zarr_steps_and_annual_transfer_parity():
    assert not (TRAIN / '2_npy2zarr_ceed.py').exists()
    positive = (TRAIN / '2.1_build_pos_zarr.py').read_text()
    assert 'copy_local_negatives' not in positive
    assert 'validate_source' not in positive
    workdir = ROOT.parent / 'SoCal_workdir/train_picker_CEED'
    def transfer_call(path):
        tree = ast.parse(path.read_text())
        return next(ast.dump(node) for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'copy_annual_negatives')
    name = '2.2_copy_neg_zarr_ceed.py'
    assert transfer_call(TRAIN / name) == transfer_call(workdir / name)


def test_shared_config_and_cutter_bindings():
    with patch.dict(sys.modules, {'data_pipeline': MagicMock()}):
        cfg = load_config(PRE / 'config_ai_pal_ceed.py')
    assert cfg.win_len == 25 and cfg.samp_rate == 100
    cfg.win_len, cfg.samp_rate, cfg.freq_band = 40, 80, [2, 15]
    tree = ast.parse((TRAIN / '1_cut_train-samples_ceed.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id == 'cfg' for n in ast.walk(node))]
    ns = {'cfg': cfg}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'bindings', 'exec'), ns)
    assert (ns['WINDOW_LENGTH'], ns['SAMPLE_RATE'], ns['FREQMIN'], ns['FREQMAX']) == (40, 80, 2, 15)
    for name in ['0_analyze_phase_rarity_ceed.py', '1_cut_train-samples_ceed.py', '2.1_build_pos_zarr.py', '3_train_ceed.py']:
        assert 'preprocess/config_ai_pal_ceed.py' in (TRAIN / name).read_text()
