"""Load the explicit shared config used by optional CEED preparation/training."""
import importlib.util
from pathlib import Path


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('ceed_workflow_settings', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.Config()
    if cfg.win_len <= 0 or cfg.samp_rate <= 0 or cfg.num_chn != 3:
        raise ValueError('CEED requires positive win_len/samp_rate and num_chn=3')
    if cfg.to_filter and not 0 < cfg.freq_band[0] < cfg.freq_band[1] < cfg.samp_rate / 2:
        raise ValueError('CEED freq_band must be below the configured Nyquist frequency')
    return cfg
