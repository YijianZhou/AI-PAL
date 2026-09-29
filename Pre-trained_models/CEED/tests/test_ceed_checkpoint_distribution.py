"""Packaged inference copies must track the supplied CEED release."""
import ast
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'AI-PAL/Pre-trained_models/CEED/CEED_ckpt'
WORKFLOWS = [
    ROOT / 'AI-PAL/3_run_ai_pal' / ('run_ai_pal_' + mode)
    for mode in ('local', 'aws', 'realtime')
] + [ROOT / 'SoCal_workdir/run_ai_pal_realtime', ROOT / 'SoCal_workdir/AWS/3_run_ai_pal']
MODELS = ('sar', 'ft', 'phn', 'run')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@pytest.mark.parametrize('workflow', WORKFLOWS)
@pytest.mark.parametrize('model', MODELS)
def test_checkpoint_and_config(workflow, model):
    name = 'ceed_{}_best.ckpt'.format(model)
    assert digest(workflow / 'input/CEED_ckpt' / name) == digest(SOURCE / name)
    assert not list((workflow / 'input/CEED_ckpt').glob('ceed_pos_*.ckpt'))
    config = workflow / 'config_{}_global_ceed.py'.format(model)
    assert config.is_file()
    # Compare the architecture section without overwriting case training settings.
    def architecture(path):
        tree = ast.parse(path.read_text().split('# Training.')[0])
        return {n.targets[0].attr: ast.dump(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
                and n.targets[0].attr != 'rnn_num_steps'}
    assert architecture(config) == architecture(SOURCE / config.name)


def test_active_references_and_syntax():
    for folder in WORKFLOWS:
        for path in folder.rglob('*.py'):
            if 'output' in path.relative_to(folder).parts:
                continue
            text = path.read_text(encoding='utf-8')
            ast.parse(text, filename=str(path))
            assert 'ceed_pos_' not in text, path
            assert 'global_case.py' not in text, path
        for path in folder.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith('input/CEED_ckpt/') and value.endswith('.ckpt'):
                        assert (folder / value).is_file(), (path, value)


def test_training_config_names():
    folder = SOURCE.parent / 'train_picker'
    for model in MODELS:
        assert (folder / 'config_{}_global_ceed.py'.format(model)).is_file()
        assert not (folder / 'config_{}_ceed.py'.format(model)).exists()
    assert 'config_{}_global_ceed.py' in (folder / '3_train_ceed.py').read_text()
