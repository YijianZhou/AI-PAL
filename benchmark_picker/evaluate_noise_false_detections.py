"""Evaluate false detections on fixed-window SeisBench noise shards.

A false positive event-window detection is defined here as a noise net.sta trace
for which the picker reports at least one P pick and at least one S pick.  The
script reports that ratio for the emitted picks and also as a function of an
extra post-hoc pick probability threshold.
"""
from pathlib import Path
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# I/O paths and evaluation settings
# -----------------------------------------------------------------------------
DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
DATASET_NAME = ['INSTANCE','CWA','PNW','STEAD','OBST2024'][4]
SHARD_SUBDIR = 'npy_noise_shards_fixed50'
NOISE_INDEX = DATA_ROOT / DATASET_NAME / SHARD_SUBDIR / 'noise_sample_index.csv'

PREDICTION_FILES = {
    'SAR': Path('output/INSTANCE_sar_noise_predictions.csv'),
    'PHN': Path('output/INSTANCE_phn_noise_predictions.csv'),
}

OUT_DIR = Path('output') / 'picker_eval_noise' / DATASET_NAME
PROB_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 3)
CSV_CHUNKSIZE = 300000
SAVE_DPI = 600
MODEL_COLORS = {
    'SAR': '#4477AA',
    'PHN': '#CC6677',
    'RUN': '#228833',
    'FT': '#AA4499',
}
FILTER_PREDICTIONS_TO_NOISE_INDEX = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def finite_numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError('{} missing columns {}'.format(label, missing))


def normalize_shard_id(series):
    return series.astype(str).str.replace('\\', '/', regex=False).str.rstrip('/').str.split('/').str[-1]


def make_sample_key(df):
    require_columns(df, ['shard_path', 'row_in_shard'], 'table')
    shard = normalize_shard_id(df['shard_path'])
    row = finite_numeric(df['row_in_shard']).fillna(-1).astype(np.int64).astype(str)
    return shard + '::' + row


def load_noise_window_keys(path):
    if not path.exists():
        raise FileNotFoundError('Missing noise sample index: {}'.format(path))
    keys = []
    for chunk in pd.read_csv(path, usecols=['shard_path', 'row_in_shard'], chunksize=CSV_CHUNKSIZE):
        keys.extend(make_sample_key(chunk).tolist())
    key_index = pd.Index(keys).drop_duplicates()
    return set(key_index), int(len(key_index))


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError('Missing prediction file: {}'.format(path))
    pred = pd.read_csv(path, low_memory=False)
    require_columns(pred, ['phase', 'pick_time', 'pick_prob', 'shard_path', 'row_in_shard'], str(path))
    pred = pred.copy()
    pred['phase'] = pred['phase'].astype(str).str.upper().str[0]
    pred = pred[pred['phase'].isin(['P', 'S'])].copy()
    pred['pick_time'] = finite_numeric(pred['pick_time'])
    pred['pick_prob'] = finite_numeric(pred['pick_prob']).fillna(0.0)
    pred = pred[pred['pick_time'].notna()].copy()
    if pred.empty:
        pred['sample_key'] = []
        return pred
    pred['sample_key'] = make_sample_key(pred)
    return pred


def summarize_prediction_windows(model_name, pred, tested_window_keys, filter_to_noise_index=False):
    total_noise_windows = len(tested_window_keys)
    num_pred_rows = int(len(pred))
    num_pred_windows = int(pred['sample_key'].nunique()) if 'sample_key' in pred.columns else 0
    num_rows_outside = 0
    num_windows_outside = 0

    if not pred.empty:
        in_test = pred['sample_key'].isin(tested_window_keys)
        num_rows_outside = int((~in_test).sum())
        num_windows_outside = int(pred.loc[~in_test, 'sample_key'].nunique())
        if num_rows_outside:
            action = 'ignored' if filter_to_noise_index else 'kept for ratio calculation'
            print(
                '  warning: {} prediction rows ({:,} windows) are outside the tested noise index and will be {}'.format(
                    num_rows_outside, num_windows_outside, action
                ),
                flush=True,
            )
        if filter_to_noise_index:
            pred = pred.loc[in_test].copy()

    if pred.empty:
        table = pd.DataFrame(columns=[
            'model', 'sample_key', 'has_p', 'has_s', 'false_detection',
            'num_p_picks', 'num_s_picks', 'max_p_prob', 'max_s_prob',
            'first_p_time', 'first_s_time', 'station_id', 'trace_name', 'sb_idx',
            'shard_path', 'row_in_shard'
        ])
        table.attrs['num_pred_rows'] = num_pred_rows
        table.attrs['num_pred_rows_used'] = 0
        table.attrs['num_pred_rows_outside_tested_windows'] = num_rows_outside
        table.attrs['num_pred_windows_in_file'] = num_pred_windows
        table.attrs['num_pred_windows_used'] = 0
        table.attrs['num_pred_windows_outside_tested_windows'] = num_windows_outside
        return table

    base_cols = ['sample_key', 'station_id', 'trace_name', 'sb_idx', 'shard_path', 'row_in_shard']
    for col in base_cols:
        if col not in pred.columns:
            pred[col] = ''
    meta = pred.sort_values('pick_prob', ascending=False).drop_duplicates('sample_key')[base_cols]

    rows = []
    for phase in ['P', 'S']:
        part = pred[pred['phase'] == phase]
        if part.empty:
            continue
        agg = part.groupby('sample_key').agg(
            **{
                'num_{}_picks'.format(phase.lower()): ('pick_time', 'size'),
                'max_{}_prob'.format(phase.lower()): ('pick_prob', 'max'),
                'first_{}_time'.format(phase.lower()): ('pick_time', 'min'),
            }
        )
        rows.append(agg)
    if rows:
        table = pd.concat(rows, axis=1).reset_index()
    else:
        table = pd.DataFrame({'sample_key': []})
    table = table.merge(meta, on='sample_key', how='left')

    for col in ['num_p_picks', 'num_s_picks']:
        if col not in table.columns:
            table[col] = 0
        table[col] = table[col].fillna(0).astype(int)
    for col in ['max_p_prob', 'max_s_prob', 'first_p_time', 'first_s_time']:
        if col not in table.columns:
            table[col] = np.nan
    table['has_p'] = table['num_p_picks'] > 0
    table['has_s'] = table['num_s_picks'] > 0
    table['false_detection'] = table['has_p'] & table['has_s']
    table.insert(0, 'model', model_name)
    table['total_noise_windows'] = int(total_noise_windows)
    table.attrs['num_pred_rows'] = num_pred_rows
    table.attrs['num_pred_rows_used'] = int(len(pred))
    table.attrs['num_pred_rows_outside_tested_windows'] = num_rows_outside
    table.attrs['num_pred_windows_in_file'] = num_pred_windows
    table.attrs['num_pred_windows_used'] = int(table['sample_key'].nunique())
    table.attrs['num_pred_windows_outside_tested_windows'] = num_windows_outside
    if total_noise_windows > 0 and table.attrs['num_pred_windows_used'] > total_noise_windows:
        print(
            '  warning: unique predicted windows ({:,}) exceed tested noise windows ({:,}); check shard/sample index consistency'.format(
                table.attrs['num_pred_windows_used'], total_noise_windows
            ),
            flush=True,
        )
    return table


def threshold_summary(model_name, window_table, total_noise_windows):
    rows = []
    p_prob = finite_numeric(window_table.get('max_p_prob', pd.Series(dtype=float))).to_numpy(dtype=float)
    s_prob = finite_numeric(window_table.get('max_s_prob', pd.Series(dtype=float))).to_numpy(dtype=float)
    for threshold in PROB_THRESHOLDS:
        has_p = np.isfinite(p_prob) & (p_prob >= threshold)
        has_s = np.isfinite(s_prob) & (s_prob >= threshold)
        both = has_p & has_s
        p_only = has_p
        s_only = has_s
        rows.append({
            'model': model_name,
            'prob_threshold': float(threshold),
            'num_noise_windows': int(total_noise_windows),
            'num_windows_with_p': int(np.sum(has_p)),
            'num_windows_with_s': int(np.sum(has_s)),
            'num_windows_p_only': int(np.sum(p_only)),
            'num_windows_s_only': int(np.sum(s_only)),
            'num_false_detections_ps': int(np.sum(both)),
            'p_pick_window_ratio': safe_div(np.sum(has_p), total_noise_windows),
            's_pick_window_ratio': safe_div(np.sum(has_s), total_noise_windows),
            'p_only_window_ratio': safe_div(np.sum(p_only), total_noise_windows),
            's_only_window_ratio': safe_div(np.sum(s_only), total_noise_windows),
            'false_positive_ratio_ps': safe_div(np.sum(both), total_noise_windows),
        })
    return pd.DataFrame(rows)


def default_summary(model_name, window_table, total_noise_windows):
    has_p = window_table['has_p'] if 'has_p' in window_table.columns else pd.Series(dtype=bool)
    has_s = window_table['has_s'] if 'has_s' in window_table.columns else pd.Series(dtype=bool)
    both = window_table['false_detection'] if 'false_detection' in window_table.columns else pd.Series(dtype=bool)
    p_only = has_p
    s_only = has_s
    return {
        'dataset': DATASET_NAME,
        'model': model_name,
        'num_noise_windows': int(total_noise_windows),
        'num_prediction_rows': int(window_table.attrs.get('num_pred_rows', 0)),
        'num_prediction_rows_used': int(window_table.attrs.get('num_pred_rows_used', 0)),
        'num_prediction_rows_outside_tested_windows': int(window_table.attrs.get('num_pred_rows_outside_tested_windows', 0)),
        'num_prediction_windows_in_file': int(window_table.attrs.get('num_pred_windows_in_file', 0)),
        'num_prediction_windows_used': int(window_table.attrs.get('num_pred_windows_used', len(window_table))),
        'num_prediction_windows_outside_tested_windows': int(window_table.attrs.get('num_pred_windows_outside_tested_windows', 0)),
        'num_windows_with_any_pick': int(len(window_table)),
        'num_windows_with_p': int(has_p.sum()),
        'num_windows_with_s': int(has_s.sum()),
        'num_windows_p_only': int(p_only.sum()),
        'num_windows_s_only': int(s_only.sum()),
        'num_false_detections_ps': int(both.sum()),
        'any_pick_window_ratio': safe_div(len(window_table), total_noise_windows),
        'p_pick_window_ratio': safe_div(has_p.sum(), total_noise_windows),
        's_pick_window_ratio': safe_div(has_s.sum(), total_noise_windows),
        'p_only_window_ratio': safe_div(p_only.sum(), total_noise_windows),
        's_only_window_ratio': safe_div(s_only.sum(), total_noise_windows),
        'false_positive_ratio_ps': safe_div(both.sum(), total_noise_windows),
    }


def safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_threshold_curves(curves, out_dir):
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model_name in sorted(curves['model'].dropna().unique()):
        part = curves[curves['model'] == model_name].sort_values('prob_threshold')
        color = MODEL_COLORS.get(model_name, None)
        ax.plot(part['prob_threshold'], part['false_positive_ratio_ps'], color=color, lw=2.0, label='{} P&S'.format(model_name))
        ax.plot(part['prob_threshold'], part['p_only_window_ratio'], color=color, lw=1.0, ls='--', alpha=0.8, label='{} P only'.format(model_name))
        ax.plot(part['prob_threshold'], part['s_only_window_ratio'], color=color, lw=1.0, ls=':', alpha=0.8, label='{} S only'.format(model_name))
    ax.set_xlabel('Post-hoc pick probability threshold')
    ax.set_ylabel('Noise-window false positive ratio')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title('{} noise false pick ratios'.format(DATASET_NAME))
    fig.tight_layout()
    fig.savefig(out_dir / 'noise_false_positive_ratio_thresholds.jpg', dpi=SAVE_DPI)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def evaluate_one_model(model_name, pred_path, tested_window_keys, out_dir):
    total_noise_windows = len(tested_window_keys)
    print('evaluating {} noise predictions: {}'.format(model_name, pred_path), flush=True)
    pred = load_predictions(pred_path)
    window_table = summarize_prediction_windows(model_name, pred, tested_window_keys, FILTER_PREDICTIONS_TO_NOISE_INDEX)
    detail_path = out_dir / '{}_noise_detected_windows.csv'.format(model_name.lower())
    window_table.to_csv(detail_path, index=False)
    curves = threshold_summary(model_name, window_table, total_noise_windows)
    curves_path = out_dir / '{}_noise_threshold_metrics.csv'.format(model_name.lower())
    curves.to_csv(curves_path, index=False)
    summary = default_summary(model_name, window_table, total_noise_windows)
    print('  false positive ratio P&S: {:.6f} ({:,}/{:,})'.format(
        summary['false_positive_ratio_ps'], summary['num_false_detections_ps'], summary['num_noise_windows']
    ), flush=True)
    return summary, curves


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('noise sample index: {}'.format(NOISE_INDEX), flush=True)
    tested_window_keys, total_noise_windows = load_noise_window_keys(NOISE_INDEX)
    print('noise windows: {:,}'.format(total_noise_windows), flush=True)

    summaries = []
    curves = []
    for model_name, pred_path in PREDICTION_FILES.items():
        if not pred_path.exists():
            print('skip {}: missing {}'.format(model_name, pred_path), flush=True)
            continue
        summary, model_curves = evaluate_one_model(model_name, pred_path, tested_window_keys, OUT_DIR)
        summaries.append(summary)
        curves.append(model_curves)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = OUT_DIR / 'noise_false_positive_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print('summary: {}'.format(summary_path), flush=True)
        print(summary_df.to_string(index=False), flush=True)
    if curves:
        curves_df = pd.concat(curves, ignore_index=True)
        curves_path = OUT_DIR / 'all_models_noise_threshold_metrics.csv'
        curves_df.to_csv(curves_path, index=False)
        plot_threshold_curves(curves_df, OUT_DIR)
        print('threshold metrics: {}'.format(curves_path), flush=True)
    print('output dir: {}'.format(OUT_DIR), flush=True)


if __name__ == '__main__':
    main()
