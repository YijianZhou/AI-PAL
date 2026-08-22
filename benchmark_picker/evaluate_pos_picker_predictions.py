"""Evaluate positive-window SAR/PhaseNet picker predictions against cleaned labels.

The positive benchmark datasets may have incomplete manual S labels, so the main
metrics are label-conditioned:
  1. phase detection rate: fraction of labeled phases with closest prediction
     within DETECTION_TOL_SEC;
  2. picking residual statistics and histograms for the closest matched picks.

For comparison with common picker literature, the script also reports
precision/recall/F1 versus prediction-probability threshold. Precision is
computed only within labeled sample/phase groups, so it does not treat picks for
unlabeled S arrivals as false positives.
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
DATASET_NAME = ['CEED','INSTANCE','CWA','PNW','STEAD','piSDL','OBST2024'][0]
SHARD_SUBDIR = 'npy_shards_fixed50'  # SeisBench positives use fixed40; CEED reference currently uses fixed50.
# Optional experiment tags.  Example: PREDICTION_SUFFIX = '_stride-01'
# reads output/CEED_sar_pos_predictions_stride-01.csv and writes evaluation
# into output/picker_eval/CEED_stride-01.
PREDICTION_SUFFIX = ''
EVAL_SUFFIX = PREDICTION_SUFFIX
GT_INDEX = DATA_ROOT / DATASET_NAME / SHARD_SUBDIR / 'cleaned_phase_index.csv'

PREDICTION_FILES = {
    'SAR': Path('output/{}_sar_pos_predictions{}.csv'.format(DATASET_NAME, PREDICTION_SUFFIX)),
    'PHN': Path('output/{}_phn_pos_predictions{}.csv'.format(DATASET_NAME, PREDICTION_SUFFIX)),
}

OUT_DIR = Path('output') / 'picker_eval' / '{}{}'.format(DATASET_NAME, EVAL_SUFFIX)
DETECTION_TOL_SEC = 1.0
PROB_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 3)
PHASES = ['P', 'S']
HIST_RANGE_SEC = (-1.0, 1.0)
HIST_BINS = 100
SAVE_DPI = 600
USE_EXISTING_EVAL = True
MODEL_COLORS = {'SAR': '#4477AA', 'PHN': '#CC6677', 'RUN': '#228833', 'FT': '#AA4499'}

def validate_prediction_paths():
    expected = DATASET_NAME.lower()
    for model_name, path in PREDICTION_FILES.items():
        name = path.name.lower()
        if expected not in name:
            print(
                'warning: {} prediction path does not contain active DATASET_NAME {}: {}'.format(
                    model_name, DATASET_NAME, path
                ),
                flush=True,
            )


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------
def finite_numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def normalize_shard_id(series):
    return series.astype(str).str.replace('\\', '/', regex=False).str.rstrip('/').str.split('/').str[-1]


def make_sample_key(df):
    shard = normalize_shard_id(df.get('shard_path', ''))
    row = finite_numeric(df.get('row_in_shard', np.nan)).fillna(-1).astype(np.int64).astype(str)
    return shard + '::' + row


def first_existing(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def load_ground_truth(path):
    if not path.exists():
        raise FileNotFoundError('Missing ground-truth cleaned index: {}'.format(path))
    df = pd.read_csv(path, low_memory=False)
    if 'shard_path' not in df.columns or 'row_in_shard' not in df.columns:
        raise ValueError('Ground truth must contain shard_path and row_in_shard: {}'.format(path))
    df['sample_key'] = make_sample_key(df)

    station_col = first_existing(df.columns, ['station_key', 'station_id', 'trace_name'])
    if station_col is None:
        df['station_id'] = ''
    else:
        df['station_id'] = df[station_col].fillna('').astype(str)

    rows = []
    for phase, col in [('P', 'p_rel_sec'), ('S', 's_rel_sec')]:
        if col not in df.columns:
            continue
        target = finite_numeric(df[col])
        keep = target.notna()
        if not keep.any():
            continue
        part = df.loc[keep].copy()
        part['phase'] = phase
        part['target_time'] = target.loc[keep].astype(float)
        rows.append(part)
    if not rows:
        raise ValueError('No finite P/S labels found in {}'.format(path))
    gt = pd.concat(rows, ignore_index=True)
    gt['target_id'] = np.arange(len(gt), dtype=np.int64)
    return gt


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError('Missing prediction file: {}'.format(path))
    pred = pd.read_csv(path, low_memory=False)
    required = ['phase', 'pick_time', 'pick_prob', 'shard_path', 'row_in_shard']
    missing = [col for col in required if col not in pred.columns]
    if missing:
        raise ValueError('Prediction file {} missing columns {}'.format(path, missing))
    pred = pred.copy()
    pred['phase'] = pred['phase'].astype(str).str.upper().str[0]
    pred = pred[pred['phase'].isin(PHASES)].copy()
    pred['pick_time'] = finite_numeric(pred['pick_time'])
    pred['pick_prob'] = finite_numeric(pred['pick_prob']).fillna(0.0)
    pred['sample_key'] = make_sample_key(pred)
    pred = pred[pred['pick_time'].notna()].copy()
    pred['pred_id'] = np.arange(len(pred), dtype=np.int64)
    return pred


def report_identity_coverage(model_name, gt, pred):
    gt_keys = pd.Index(gt['sample_key'].drop_duplicates())
    pred_keys = pd.Index(pred['sample_key'].drop_duplicates())
    overlap = pred_keys.intersection(gt_keys)
    gt_shards = normalize_shard_id(gt['shard_path']).nunique()
    pred_shards = normalize_shard_id(pred['shard_path']).nunique()
    coverage = len(overlap) / max(len(gt_keys), 1)
    print(
        '  identity coverage {}: prediction rows {:,} | unique samples {:,} | '
        'matched GT samples {:,}/{:,} ({:.2%}) | shards {:,}/{:,}'.format(
            model_name, len(pred), len(pred_keys), len(overlap), len(gt_keys),
            coverage, pred_shards, gt_shards
        ),
        flush=True,
    )
    if coverage < 0.5:
        print(
            '  ERROR: prediction identities cover less than 50% of ground truth; '
            'check the installed run_picker_pos.py metadata-mapping version.',
            flush=True,
        )


# -----------------------------------------------------------------------------
# Matching and metrics
# -----------------------------------------------------------------------------
def closest_match_for_phase(gt_phase, pred_phase):
    keep_cols = [
        'target_id', 'sample_key', 'phase', 'target_time', 'station_id', 'event_id',
        'trace_name', 'station_key', 'sb_idx', 'shard_path', 'row_in_shard',
        'window_start_sec', 'distance_km'
    ]
    keep_cols = [col for col in keep_cols if col in gt_phase.columns]
    pred_cols = ['sample_key', 'pick_time', 'pick_prob', 'pick_time_std', 'pick_prob_std', 'num_votes', 'pred_id']
    pred_cols = [col for col in pred_cols if col in pred_phase.columns]
    merged = gt_phase[keep_cols].merge(pred_phase[pred_cols], on='sample_key', how='left')
    merged['dt'] = merged['pick_time'] - merged['target_time']
    merged['abs_dt'] = merged['dt'].abs()
    merged['_sort_abs_dt'] = merged['abs_dt'].fillna(np.inf)
    merged['_sort_prob'] = -merged.get('pick_prob', pd.Series(0.0, index=merged.index)).fillna(0.0)
    matched = (
        merged.sort_values(['target_id', '_sort_abs_dt', '_sort_prob'])
        .groupby('target_id', as_index=False)
        .head(1)
        .drop(columns=['_sort_abs_dt', '_sort_prob'])
        .sort_values('target_id')
    )
    matched['detected'] = matched['abs_dt'] <= DETECTION_TOL_SEC
    return matched


def closest_matches(gt, pred):
    rows = []
    for phase in PHASES:
        gt_phase = gt[gt['phase'] == phase].copy()
        pred_phase = pred[pred['phase'] == phase].copy()
        if gt_phase.empty:
            continue
        rows.append(closest_match_for_phase(gt_phase, pred_phase))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    rename = {
        'pick_time': 'pred_time',
        'pick_prob': 'pred_prob',
        'pick_time_std': 'pred_time_std',
        'pick_prob_std': 'pred_prob_std',
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def summarize_matches(model_name, matches):
    rows = []
    for phase in PHASES:
        part = matches[matches['phase'] == phase].copy()
        if part.empty:
            continue
        detected = part[part['detected']].copy()
        dt = finite_numeric(detected['dt']).dropna()
        abs_dt = dt.abs()
        rows.append({
            'model': model_name,
            'phase': phase,
            'dt_tolerance_sec': DETECTION_TOL_SEC,
            'num_targets': int(len(part)),
            'num_with_prediction': int(part['pred_time'].notna().sum()),
            'num_detected': int(part['detected'].sum()),
            'detection_rate_recall': safe_div(part['detected'].sum(), len(part)),
            'outlier_rate_out': 1.0 - safe_div(part['detected'].sum(), len(part)),
            'mean_dt_sec': safe_stat(dt, np.mean),
            'std_dt_sec': safe_stat(dt, np.std),
            'median_dt_sec': safe_stat(dt, np.median),
            'mae_sec': safe_stat(abs_dt, np.mean),
            'rmse_sec': safe_rmse(dt),
            'p90_abs_dt_sec': safe_quantile(abs_dt, 0.90),
            'p95_abs_dt_sec': safe_quantile(abs_dt, 0.95),
        })
    return pd.DataFrame(rows)


def threshold_metrics(model_name, gt, pred):
    rows = []
    for phase in PHASES:
        gt_phase = gt[gt['phase'] == phase].copy()
        pred_phase_all = pred[pred['phase'] == phase].copy()
        if gt_phase.empty:
            continue

        labeled_keys = set(gt_phase['sample_key'].astype(str))
        pred_labeled = pred_phase_all[pred_phase_all['sample_key'].astype(str).isin(labeled_keys)].copy()

        # One merge per phase is enough for the full threshold sweep.  For each
        # target, keep the highest probability among predictions inside the dt
        # tolerance; this gives TP(threshold) without re-running matching 101x.
        gt_small = gt_phase[['target_id', 'sample_key', 'target_time']].copy()
        if pred_labeled.empty:
            best_tp_prob = pd.Series(dtype=float)
            pred_probs = np.asarray([], dtype=float)
        else:
            merged = gt_small.merge(pred_labeled[['sample_key', 'pick_time', 'pick_prob']], on='sample_key', how='left')
            merged['abs_dt'] = (merged['pick_time'] - merged['target_time']).abs()
            in_tol = merged[merged['abs_dt'] <= DETECTION_TOL_SEC]
            best_tp_prob = in_tol.groupby('target_id')['pick_prob'].max()
            pred_probs = pred_labeled['pick_prob'].to_numpy(dtype=float)

        for threshold in PROB_THRESHOLDS:
            tp = int((best_tp_prob >= threshold).sum())
            pred_count = int(np.sum(pred_probs >= threshold))
            fn = int(len(gt_phase) - tp)
            fp_labeled = int(max(pred_count - tp, 0))
            precision = safe_div(tp, tp + fp_labeled)
            recall = safe_div(tp, tp + fn)
            rows.append({
                'model': model_name,
                'phase': phase,
                'prob_threshold': float(threshold),
                'dt_tolerance_sec': DETECTION_TOL_SEC,
                'tp': tp,
                'fp_labeled': fp_labeled,
                'fn': fn,
                'num_targets': int(len(gt_phase)),
                'num_predictions_labeled_groups': pred_count,
                'precision_labeled': precision,
                'recall': recall,
                'f1_labeled': safe_f1(precision, recall),
                'false_picks_per_labeled_trace': safe_div(fp_labeled, len(gt_phase)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = append_auc_metrics(out)
    return out


def append_auc_metrics(curves):
    curves = curves.copy()
    curves['pr_auc_labeled'] = np.nan
    for (model, phase), part in curves.groupby(['model', 'phase']):
        x = part['recall'].to_numpy(dtype=float)
        y = part['precision_labeled'].fillna(1.0).to_numpy(dtype=float)
        order = np.argsort(x)
        auc = float(trapezoid_integral(y[order], x[order])) if len(part) > 1 else np.nan
        mask = (curves['model'] == model) & (curves['phase'] == phase)
        curves.loc[mask, 'pr_auc_labeled'] = auc
    return curves



def trapezoid_integral(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return np.nan
    return float(num) / den


def safe_f1(precision, recall):
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return np.nan
    return 2.0 * precision * recall / (precision + recall)


def safe_stat(values, func):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    return float(func(values.to_numpy(dtype=float)))


def safe_rmse(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def safe_quantile(values, q):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    return float(np.quantile(values.to_numpy(dtype=float), q))


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_dt_histograms(model_name, matches, summary, out_dir):
    """Plot P and S residual histograms for one model in a shared 1x2 figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=False)
    for ax, phase in zip(axes, PHASES):
        part = matches[(matches['phase'] == phase) & (matches['pred_time'].notna())].copy()
        row = summary[(summary['model'] == model_name) & (summary['phase'] == phase)]
        det_rate = float(row['detection_rate_recall'].iloc[0]) if not row.empty else np.nan
        out_rate = 1.0 - det_rate if np.isfinite(det_rate) else np.nan
        mae = float(row['mae_sec'].iloc[0]) if not row.empty else np.nan
        rmse = float(row['rmse_sec'].iloc[0]) if not row.empty else np.nan
        mean_dt = float(row['mean_dt_sec'].iloc[0]) if not row.empty else np.nan
        std_dt = float(row['std_dt_sec'].iloc[0]) if not row.empty else np.nan

        if part.empty:
            ax.text(0.5, 0.5, 'no predictions', transform=ax.transAxes, ha='center', va='center')
        else:
            detected_dt = finite_numeric(part.loc[part['detected'], 'dt']).dropna()
            all_dt = finite_numeric(part['dt']).dropna()
            ax.hist(all_dt, bins=HIST_BINS, range=HIST_RANGE_SEC, color='0.82', edgecolor='none', label='closest prediction')
            if not detected_dt.empty:
                ax.hist(detected_dt, bins=HIST_BINS, range=HIST_RANGE_SEC, color=MODEL_COLORS.get(model_name, '#4477AA'), alpha=0.85, edgecolor='none', label='|dt| <= {:.1f}s'.format(DETECTION_TOL_SEC))
        ax.axvline(0.0, color='k', lw=1.0)
        ax.set_xlim(HIST_RANGE_SEC)
        ax.set_xlabel('dt = predicted - target (s)')
        ax.set_title('{} phase'.format(phase))
        text_main = 'det. = {:.2%}\nmean = {:.3f}s\nstd = {:.3f}s'.format(det_rate, mean_dt, std_dt)
        ax.text(0.98, 0.95, text_main, transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9))
        text_paper = 'OUT {:.3f}\nMAE {:.3f}s\nRMSE {:.3f}s'.format(out_rate, mae, rmse)
        ax.text(0.98, 0.55, text_paper, transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9))
    axes[0].set_ylabel('Count')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc='upper left', frameon=False)
    fig.suptitle('{} residual histograms, {}'.format(model_name, DATASET_NAME), y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / '{}_dt_hist_ps.jpg'.format(model_name.lower()), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def best_f1_row(part):
    if part.empty or 'f1_labeled' not in part.columns:
        return None
    values = finite_numeric(part['f1_labeled'])
    if values.notna().sum() == 0:
        return None
    return part.loc[values.idxmax()]


def plot_all_threshold_curves(curves, out_dir):
    """One ROC/PR-style figure for all models and phases.

    True ROC false-positive rate is not well defined for these positive-only
    windows with incomplete S labels, so the ROC-like left column uses false
    picks per labeled trace on the x axis and recall/TPR on the y axis.
    """
    if curves.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), sharex=False, sharey=False)
    for row_idx, phase in enumerate(PHASES):
        phase_curves = curves[curves['phase'] == phase].copy()
        for model_name in sorted(phase_curves['model'].dropna().unique()):
            part = phase_curves[phase_curves['model'] == model_name].sort_values('prob_threshold')
            if part.empty:
                continue
            color = MODEL_COLORS.get(model_name, None)
            label = model_name
            axes[row_idx, 0].plot(part['false_picks_per_labeled_trace'], part['recall'], color=color, label=label)
            axes[row_idx, 1].plot(part['recall'], part['precision_labeled'], color=color, label=label)
            axes[row_idx, 2].plot(part['prob_threshold'], part['f1_labeled'], color=color, label=label)
            best = best_f1_row(part)
            if best is not None:
                axes[row_idx, 0].plot(best['false_picks_per_labeled_trace'], best['recall'], marker='D', ms=5, color=color)
                axes[row_idx, 1].plot(best['recall'], best['precision_labeled'], marker='D', ms=5, color=color)
                axes[row_idx, 2].plot(best['prob_threshold'], best['f1_labeled'], marker='D', ms=5, color=color)
                auc = best.get('pr_auc_labeled', np.nan)
                txt = '{} F1 {:.3f}'.format(model_name, best['f1_labeled'])
                if np.isfinite(auc):
                    txt += ' AUC {:.3f}'.format(auc)
                axes[row_idx, 1].text(0.03, 0.08 + 0.09 * len(axes[row_idx, 1].texts), txt,
                                      transform=axes[row_idx, 1].transAxes, color=color, fontsize=9)
        axes[row_idx, 0].set_ylabel('{} true positive rate'.format(phase))
        axes[row_idx, 0].set_xlabel('false picks per labeled trace')
        axes[row_idx, 1].set_xlabel('true positive rate / recall')
        axes[row_idx, 1].set_ylabel('precision, labeled groups only')
        axes[row_idx, 2].set_xlabel('probability threshold')
        axes[row_idx, 2].set_ylabel('F1, labeled groups only')
        axes[row_idx, 0].set_ylim(-0.02, 1.02)
        axes[row_idx, 1].set_xlim(-0.02, 1.02)
        axes[row_idx, 1].set_ylim(-0.02, 1.02)
        axes[row_idx, 2].set_xlim(-0.02, 1.02)
        axes[row_idx, 2].set_ylim(-0.02, 1.02)
        axes[row_idx, 0].grid(alpha=0.2)
        axes[row_idx, 1].grid(alpha=0.2)
        axes[row_idx, 2].grid(alpha=0.2)
    axes[0, 0].set_title('ROC-like curve')
    axes[0, 1].set_title('Precision-recall curve')
    axes[0, 2].set_title('F1-threshold curve')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, loc='lower right', frameon=False, fontsize=9)
    fig.suptitle('{} positive-window picker threshold metrics'.format(DATASET_NAME), y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / 'all_models_threshold_curves.jpg', dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def load_existing_eval(model_name, out_dir, pred_path=None):
    comparison_path = out_dir / '{}_pick_comparison.csv'.format(model_name.lower())
    summary_path = out_dir / '{}_metric_summary.csv'.format(model_name.lower())
    curves_path = out_dir / '{}_threshold_metrics.csv'.format(model_name.lower())
    if not (comparison_path.exists() and summary_path.exists() and curves_path.exists()):
        return None
    if pred_path is not None and Path(pred_path).exists():
        pred_mtime = Path(pred_path).stat().st_mtime
        oldest_eval_mtime = min(comparison_path.stat().st_mtime, summary_path.stat().st_mtime, curves_path.stat().st_mtime)
        if pred_mtime > oldest_eval_mtime:
            print(
                'cached evaluation for {} is older than {}; recomputing'.format(model_name, pred_path),
                flush=True,
            )
            return None

    print('loading existing evaluation for {} from {}'.format(model_name, out_dir), flush=True)
    matches = pd.read_csv(comparison_path, low_memory=False)
    summary = pd.read_csv(summary_path, low_memory=False)
    curves = pd.read_csv(curves_path, low_memory=False)
    if pred_path is not None and Path(pred_path).exists():
        has_cached_predictions = 'pred_time' in matches.columns and finite_numeric(matches['pred_time']).notna().any()
        if not has_cached_predictions:
            try:
                pred_header = pd.read_csv(pred_path, nrows=1)
                if len(pred_header) > 0:
                    print(
                        '  cached comparison for {} has no predictions; recomputing from {}'.format(model_name, pred_path),
                        flush=True,
                    )
                    return None
            except Exception as exc:
                print('  warning: could not validate prediction cache {}: {}'.format(pred_path, exc), flush=True)
    return matches, summary, curves


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def evaluate_one_model(model_name, pred_path, gt, out_dir):
    existing = load_existing_eval(model_name, out_dir, pred_path) if USE_EXISTING_EVAL else None
    if existing is not None:
        matches, _, curves = existing
        summary = summarize_matches(model_name, matches)
        summary.to_csv(out_dir / '{}_metric_summary.csv'.format(model_name.lower()), index=False)
        plot_dt_histograms(model_name, matches, summary, out_dir)
        print('  refreshed plots from existing comparison: {}'.format(model_name), flush=True)
        return matches, summary, curves

    if gt is None or gt.empty:
        print('ground truth: {}'.format(GT_INDEX), flush=True)
        gt = load_ground_truth(GT_INDEX)
    print('evaluating {}: {}'.format(model_name, pred_path), flush=True)
    pred = load_predictions(pred_path)
    report_identity_coverage(model_name, gt, pred)
    matches = closest_matches(gt, pred)
    if matches.empty:
        raise RuntimeError('No matches produced for {}'.format(model_name))
    matches.insert(0, 'model', model_name)
    ordered_cols = [
        'model', 'station_id', 'pred_time', 'target_time', 'phase', 'dt', 'abs_dt', 'detected',
        'pred_prob', 'pred_time_std', 'pred_prob_std', 'num_votes', 'event_id', 'trace_name',
        'station_key', 'sb_idx', 'shard_path', 'row_in_shard', 'window_start_sec', 'distance_km'
    ]
    ordered_cols = [col for col in ordered_cols if col in matches.columns]
    matches = matches[ordered_cols + [col for col in matches.columns if col not in ordered_cols]]

    comparison_path = out_dir / '{}_pick_comparison.csv'.format(model_name.lower())
    matches.to_csv(comparison_path, index=False)

    summary = summarize_matches(model_name, matches)
    curves = threshold_metrics(model_name, gt, pred)
    summary_path = out_dir / '{}_metric_summary.csv'.format(model_name.lower())
    curves_path = out_dir / '{}_threshold_metrics.csv'.format(model_name.lower())
    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)
    plot_dt_histograms(model_name, matches, summary, out_dir)
    print('  comparison: {}'.format(comparison_path), flush=True)
    print('  summary: {}'.format(summary_path), flush=True)
    print(summary.to_string(index=False), flush=True)
    return matches, summary, curves


def main():
    validate_prediction_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    need_gt = not USE_EXISTING_EVAL or any(
        not (OUT_DIR / '{}_pick_comparison.csv'.format(model.lower())).exists()
        for model in PREDICTION_FILES
        if PREDICTION_FILES[model].exists()
    )
    gt = None
    if need_gt:
        print('ground truth: {}'.format(GT_INDEX), flush=True)
        gt = load_ground_truth(GT_INDEX)
    else:
        print('using existing comparison files in {}'.format(OUT_DIR), flush=True)

    all_matches = []
    all_summaries = []
    all_curves = []
    for model_name, pred_path in PREDICTION_FILES.items():
        if not pred_path.exists() and not (OUT_DIR / '{}_pick_comparison.csv'.format(model_name.lower())).exists():
            print('skip {}: missing {}'.format(model_name, pred_path), flush=True)
            continue
        matches, summary, curves = evaluate_one_model(model_name, pred_path, gt, OUT_DIR)
        all_matches.append(matches)
        all_summaries.append(summary)
        all_curves.append(curves)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(OUT_DIR / 'all_models_metric_summary.csv', index=False)
    if all_curves:
        all_curves_df = pd.concat(all_curves, ignore_index=True)
        all_curves_df.to_csv(OUT_DIR / 'all_models_threshold_metrics.csv', index=False)
        plot_all_threshold_curves(all_curves_df, OUT_DIR)
    if all_matches:
        pd.concat(all_matches, ignore_index=True).to_csv(OUT_DIR / 'all_models_pick_comparison.csv', index=False)
    print('output dir: {}'.format(OUT_DIR), flush=True)


if __name__ == '__main__':
    main()
