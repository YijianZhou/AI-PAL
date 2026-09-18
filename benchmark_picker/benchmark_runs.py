"""User-editable model-run registry shared by benchmark executables."""
from pathlib import Path


# =============================================================================
# USER SETTINGS: ENABLED MODEL VERSIONS
# =============================================================================
# checkpoint may point to a directory (best.ckpt preferred) or to one
# exact checkpoint file. marker must be unique and filesystem-safe.
MODEL_RUNS = [
    {
        'model': 'SAR',
        'checkpoint': Path('/nas/zhouyj/AI_ckpt/ceed_pos/SAR/best.ckpt'),
        'marker': 'ceed_pos_sar_v7',
        'label': 'SAR CEED-pos 0.1s',
        'color': '#4477AA',
    },
    {
        'model': 'PHN',
        'checkpoint': Path('/nas/zhouyj/AI_ckpt/ceed_pos/PHN/best.ckpt'),
        'marker': 'ceed_pos_phn_v7',
        'label': 'PHN CEED-pos 1M',
        'color': '#CC6677',
    },
    {
        'model': 'RUN',
        'checkpoint': Path('/nas/zhouyj/AI_ckpt/ceed_pos/RUN/best.ckpt'),
        'marker': 'ceed_pos_run_v7',
        'label': 'RUN CEED-pos',
        'color': '#228833',
    },
    {
        'model': 'FT',
        'checkpoint': Path('/nas/zhouyj/AI_ckpt/ceed_pos/FT/best.ckpt'),
        'marker': 'ceed_pos_ft_v7',
        'label': 'FT CEED-pos',
        'color': '#AA4499',
    },
    {
        'model': 'SAR',
        'checkpoint': Path('/nas/zhouyj/eh_ckpt/SAR'),
        'marker': 'centcal_sar',
        'label': 'SAR CentCal',
        'color': '#88AADD',
    },
    {
        'model': 'PHN',
        'checkpoint': Path('/nas/zhouyj/eh_ckpt/PHN'),
        'marker': 'centcal_phn',
        'label': 'PHN CentCal',
        'color': '#EE99AA',
    },
    {
        'model': 'RUN',
        'checkpoint': Path('/nas/zhouyj/eh_ckpt/RUN'),
        'marker': 'centcal_run',
        'label': 'RUN CentCal',
        'color': '#66AA77',
    },
    {
        'model': 'FT',
        'checkpoint': Path('/nas/zhouyj/eh_ckpt/FT'),
        'marker': 'centcal_ft',
        'label': 'FT CentCal',
        'color': '#CC99CC',
    },
]

# Select any subset without editing MODEL_RUNS. Use [] to enable every entry.
ENABLED_RUN_MARKERS = [
    'ceed_pos_sar_v7', 'ceed_pos_phn_v7', 'ceed_pos_run_v7', 'ceed_pos_ft_v7',
]

# CEED is an in-domain positive reference. It has no benchmark noise subset.
POSITIVE_DATASETS = ['CEED', 'INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
NOISE_DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'OBST2024']
SAMPLE_TYPES = ['positive', 'noise']

# Name of the combined plot directory and figure titles.
COMPARISON_MARKER = 'ceed_pos_4models_v7'


def enabled_runs():
    selected = set(ENABLED_RUN_MARKERS)
    runs = [run for run in MODEL_RUNS if not selected or run['marker'] in selected]
    markers = [run['marker'] for run in runs]
    if len(markers) != len(set(markers)):
        raise ValueError('MODEL_RUNS contains duplicate markers')
    missing = selected - set(markers)
    if missing:
        raise KeyError('Unknown ENABLED_RUN_MARKERS: {}'.format(sorted(missing)))
    return runs
