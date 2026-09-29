"""Offline CEED augmentation parity and reproducibility."""
import ast
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

CEED = Path(__file__).resolve().parents[1]
ROOT = CEED.parents[2]
sys.path.insert(0, str(CEED / 'helpers'))
try:
    import h5py
except ImportError:
    # These tests supply in-memory streams; HDF5 I/O is not exercised.
    with patch.dict(sys.modules, {'h5py': MagicMock()}):
        import ceed_data_pipeline as ceed
else:
    import ceed_data_pipeline as ceed


def setup_window(**overrides):
    args = SimpleNamespace(window_length=5., sample_rate=100., phase_margin=.2,
                           max_noise=.5, global_max_norm=False, to_filter=True,
                           freqmin=1., freqmax=20., filter_corners=4,
                           taper_max_percentage=.05, taper_max_length=1.,
                           integrate_acceleration=False, verbose_errors=True)
    vars(args).update(overrides)
    t = np.arange(2001) / 100.
    stream = Stream([Trace((i + 1) * np.sin(2 * np.pi * 3 * t),
                           header={'sampling_rate': 100., 'starttime': UTCDateTime(0)})
                     for i in range(3)])
    sample = dict(tp=UTCDateTime(8), ts=UTCDateTime(10), split='train', num_aug=3)
    return stream, sample, args


@pytest.mark.parametrize('global_norm', [False, True])
def test_copy_policy_and_reproducibility(global_norm):
    stream, sample, args = setup_window(global_max_norm=global_norm)
    bounds = ceed.valid_window_start_range(stream, sample['tp'], sample['ts'], args)
    windows = list(ceed.training_windows(stream, sample, args, np.random.default_rng(4), bounds))
    again = list(ceed.training_windows(stream, sample, args, np.random.default_rng(4), bounds))
    assert [w[2] for w in windows] == [False, True, True]
    for (cut, start, noisy), (repeat, start2, _) in zip(windows, again):
        assert start == start2
        clean = ceed.cut_and_normalize(stream, start, args)
        assert sample['tp'] - start >= args.phase_margin
        assert start + args.window_length - sample['ts'] >= args.phase_margin
        for tr, tr2, base in zip(cut, repeat, clean):
            assert tr.data.dtype == np.float32 and len(tr.data) == 500
            np.testing.assert_array_equal(tr.data, tr2.data)
            assert np.isfinite(tr.data).all()
            assert abs(tr.data.mean()) < 1e-6
            assert np.array_equal(tr.data, base.data) != noisy
        assert max(np.max(abs(tr.data)) for tr in cut) == pytest.approx(1.)
    sample['split'] = 'valid'
    valid = list(ceed.training_windows(stream, sample, args, np.random.default_rng(4), bounds))
    assert len(valid) == 1 and not valid[0][2]
    for a, b in zip(valid[0][0], windows[0][0]):
        np.testing.assert_array_equal(a.data, b.data)


def test_random_level_matches_local_scaling():
    stream, sample, args = setup_window(to_filter=False)
    cut = ceed.cut_and_normalize(stream, UTCDateTime(7), args)
    seed = 12
    rng = np.random.default_rng(seed)
    pad = int(args.window_length * args.sample_rate / 2)
    noise = cut.copy()
    for tr in noise:
        tr.data = rng.standard_normal(500 + 2 * pad)
        tr.stats.starttime -= pad / args.sample_rate
    noise = ceed.preprocess_stream(noise, args)
    for tr in noise:
        tr.data = tr.data[pad:-pad].copy()
        tr.stats.starttime += pad / args.sample_rate
    noise.normalize(global_max=False)
    alpha = rng.random() * args.max_noise
    expected = cut.copy()
    for tr, n in zip(expected, noise):
        peak = np.max(abs(tr.slice(sample['tp'], sample['ts']).data))
        tr.data = tr.data + alpha * peak * n.data
    expected = ceed.normalize_window(expected, args)
    result = ceed.add_gaussian_noise(cut, sample['tp'], sample['ts'], args,
                                     np.random.default_rng(seed))
    for a, b in zip(result, expected):
        np.testing.assert_array_equal(a.data, b.data)


def test_disabled_and_silent_noise():
    stream, sample, args = setup_window(max_noise=0.)
    bounds = ceed.valid_window_start_range(stream, sample['tp'], sample['ts'], args)
    assert not any(w[2] for w in ceed.training_windows(stream, sample, args,
                                                     np.random.default_rng(5), bounds))
    args.max_noise = .5
    for tr in stream:
        tr.data[:] = 0.
    with np.errstate(all='raise'), pytest.warns(UserWarning, match='dividing through zero'):
        result = ceed.add_gaussian_noise(stream, sample['tp'], sample['ts'], args,
                                        np.random.default_rng(5))
    assert all(np.all(tr.data == 0.) for tr in result)


@pytest.mark.parametrize('script', [
    CEED / 'train_picker/1_cut_train-samples_ceed.py',
    ROOT / 'SoCal_workdir/train_picker_CEED/1_cut_ceed_train_npy.py',
])
def test_both_cutters_use_recipe(script):
    tree = ast.parse(script.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('process_sample', 'make_npy_sample')]
    ns = dict(ceed=ceed, np=np, Counter=Counter)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(script), 'exec'), ns)
    stream, sample, args = setup_window()
    with patch.object(ceed, 'make_stream', return_value=stream):
        counts, arrays = ns['process_sample'](None, sample, args, np.random.default_rng(5))
    assert counts['clean_copies'] == 1 and counts['gaussian_noise_copies'] == 2
    assert np.stack(arrays).shape == (3, 3, 502)
    sample['split'] = 'valid'
    with patch.object(ceed, 'make_stream', return_value=stream):
        counts, arrays = ns['process_sample'](None, sample, args, np.random.default_rng(5))
    assert len(arrays) == 1 and counts['gaussian_noise_copies'] == 0


def test_workdir_helper_matches_packaged_recipe():
    assert (CEED / 'helpers/ceed_data_pipeline.py').read_bytes() == (
        ROOT / 'SoCal_workdir/train_picker_CEED/ceed_data_pipeline.py').read_bytes()
