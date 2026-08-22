"""Plot positive accuracy and noise stability for configured benchmark runs."""
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
from benchmark_runs import COMPARISON_MARKER, NOISE_DATASETS, POSITIVE_DATASETS, enabled_runs


# -----------------------------------------------------------------------------
# I/O and active comparison
# -----------------------------------------------------------------------------
EVAL_ROOT = Path('output') / 'picker_eval'
NOISE_EVAL_ROOT = Path('output') / 'picker_eval_noise'
OUT_ROOT = Path('output') / 'picker_eval_comparison'
PHASES = ['P', 'S']
HIST_RANGE_SEC = (-1.0, 1.0)
KDE_POINTS = 600
KDE_MAX_SAMPLES = 200000
KDE_RANDOM_SEED = 20250711
SAVE_DPI = 600
_MISSING_PATHS_REPORTED = set()
COMPARISON_NAME = COMPARISON_MARKER
VARIANTS = [
    {
        'label': run['label'], 'model': run['marker'],
        'eval_suffix': run['marker'], 'color': run['color'],
    }
    for run in enabled_runs()
]
ACTIVE_DATASETS = POSITIVE_DATASETS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def finite_numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def dataset_eval_dir(dataset, variant):
    return EVAL_ROOT / dataset / variant['eval_suffix']


def comparison_path(dataset, variant):
    model = variant['model'].lower()
    return dataset_eval_dir(dataset, variant) / '{}_pick_comparison.csv'.format(model)


def metric_summary_path(dataset, variant):
    model = variant['model'].lower()
    return dataset_eval_dir(dataset, variant) / '{}_metric_summary.csv'.format(model)


def threshold_metrics_path(dataset, variant):
    model = variant['model'].lower()
    return dataset_eval_dir(dataset, variant) / '{}_threshold_metrics.csv'.format(model)


def load_csv_or_none(path):
    if not path.exists():
        path_key = str(path)
        if path_key not in _MISSING_PATHS_REPORTED:
            print('missing {}'.format(path), flush=True)
            _MISSING_PATHS_REPORTED.add(path_key)
        return None
    return pd.read_csv(path, low_memory=False)


def gaussian_kde_1d(values, grid):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if values.size > KDE_MAX_SAMPLES:
        rng = np.random.default_rng(KDE_RANDOM_SEED)
        values = rng.choice(values, size=KDE_MAX_SAMPLES, replace=False)
    std = np.std(values)
    if not np.isfinite(std) or std <= 0.0:
        std = max((HIST_RANGE_SEC[1] - HIST_RANGE_SEC[0]) / 200.0, 1e-3)
    bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
    bandwidth = float(np.clip(bandwidth, 0.003, 0.25))
    density = np.zeros_like(grid, dtype=np.float64)
    chunk_size = 20000
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth * values.size)
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        z = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * z * z).sum(axis=1)
    return density * norm


def auc_for_phase(curves, phase):
    part = curves[curves['phase'] == phase].copy()
    if part.empty:
        return np.nan
    auc_series = finite_numeric(part.get('pr_auc_labeled', pd.Series(dtype=float))).dropna()
    return float(auc_series.iloc[0]) if not auc_series.empty else np.nan



def preflight_inputs():
    print('comparison: {}'.format(COMPARISON_NAME), flush=True)
    total_missing = 0
    for variant in VARIANTS:
        suffix = variant.get('eval_suffix', '')
        expected = []
        for dataset in ACTIVE_DATASETS:
            expected.extend([
                comparison_path(dataset, variant),
                metric_summary_path(dataset, variant),
                threshold_metrics_path(dataset, variant),
            ])
        missing = [path for path in expected if not path.exists()]
        total_missing += len(missing)
        print('  {}: model={} eval_suffix="{}" missing {}/{} files'.format(
            variant['label'], variant['model'], suffix, len(missing), len(expected)
        ), flush=True)
        if missing:
            example_dataset = ACTIVE_DATASETS[0]
            print('    expected example: {}'.format(comparison_path(example_dataset, variant)), flush=True)
            if suffix:
                print('    evaluate raw prediction CSVs for marker {}'.format(suffix), flush=True)
    if total_missing:
        print('  note: raw prediction CSVs can stay in output/. This script needs evaluated CSVs under output/picker_eval/.', flush=True)
        print('  note: run evaluate_picker_benchmark.py before plotting.', flush=True)
    return total_missing
# -----------------------------------------------------------------------------
# Plot 1: per-dataset dt KDE comparisons
# -----------------------------------------------------------------------------
def plot_dataset_dt_kde(dataset, out_dir):
    grid = np.linspace(HIST_RANGE_SEC[0], HIST_RANGE_SEC[1], KDE_POINTS)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=False)
    any_curve = False

    for ax, phase in zip(axes, PHASES):
        y_max = 0.0
        for variant in VARIANTS:
            path = comparison_path(dataset, variant)
            df = load_csv_or_none(path)
            if df is None or df.empty or 'dt' not in df.columns:
                continue
            part = df[(df.get('phase', '').astype(str) == phase) & df.get('pred_time', pd.Series(np.nan, index=df.index)).notna()].copy()
            dt = finite_numeric(part['dt']).dropna()
            dt = dt[(dt >= HIST_RANGE_SEC[0]) & (dt <= HIST_RANGE_SEC[1])]
            if dt.empty:
                continue
            density = gaussian_kde_1d(dt.to_numpy(dtype=float), grid)
            if np.all(~np.isfinite(density)):
                continue
            any_curve = True
            y_max = max(y_max, float(np.nanmax(density)))
            det = finite_numeric(part.get('detected', pd.Series(dtype=float))).fillna(False).astype(bool)
            det_rate = float(det.sum()) / max(len(df[df.get('phase', '').astype(str) == phase]), 1)
            mean_dt = float(np.mean(dt))
            std_dt = float(np.std(dt))
            label = '{}  det={:.1%}, mean={:.3f}s, std={:.3f}s'.format(
                variant['label'], det_rate, mean_dt, std_dt
            )
            ax.plot(grid, density, color=variant['color'], lw=2.0, label=label)
        ax.axvline(0.0, color='k', lw=1.0)
        ax.set_xlim(HIST_RANGE_SEC)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.08)
        ax.set_xlabel('dt = predicted - target (s)')
        ax.set_title('{} phase'.format(phase))
        ax.grid(alpha=0.18)
        if phase == 'P':
            ax.set_ylabel('KDE density')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper left', frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'no finite residuals', transform=ax.transAxes, ha='center', va='center')

    fig.suptitle('{} positive-sample residual KDE, {}'.format(COMPARISON_NAME, dataset), y=1.02)
    fig.tight_layout()
    out_path = out_dir / '{}_dt_kde_{}.jpg'.format(dataset, COMPARISON_NAME)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    return any_curve


# -----------------------------------------------------------------------------
# Plot 2: cross-dataset metrics, 2x4 layout
# -----------------------------------------------------------------------------
def collect_metric_table():
    rows = []
    for dataset in ACTIVE_DATASETS:
        for variant in VARIANTS:
            summary = load_csv_or_none(metric_summary_path(dataset, variant))
            curves = load_csv_or_none(threshold_metrics_path(dataset, variant))
            if summary is None:
                continue
            for phase in PHASES:
                row = summary[summary['phase'].astype(str) == phase]
                if row.empty:
                    continue
                auc = np.nan
                if curves is not None:
                    auc = auc_for_phase(curves, phase)
                rows.append({
                    'dataset': dataset,
                    'variant': variant['label'],
                    'phase': phase,
                    'mean_dt_sec': float(finite_numeric(row['mean_dt_sec']).iloc[0]) if 'mean_dt_sec' in row else np.nan,
                    'std_dt_sec': float(finite_numeric(row['std_dt_sec']).iloc[0]) if 'std_dt_sec' in row else np.nan,
                    'detection_rate': float(finite_numeric(row['detection_rate_recall']).iloc[0]) if 'detection_rate_recall' in row else np.nan,
                    'pr_auc': auc,
                })
    return pd.DataFrame(rows)


def plot_metric_summary(metrics, out_dir):
    if metrics.empty:
        print('no metric rows available', flush=True)
        return
    metric_defs = [
        ('mean_dt_sec', 'dt mean (s)'),
        ('std_dt_sec', 'dt std (s)'),
        ('detection_rate', 'detection rate'),
        ('pr_auc', 'PR AUC'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2), sharex=True)
    x = np.arange(len(ACTIVE_DATASETS), dtype=float)
    offset_values = np.linspace(-0.18, 0.18, max(len(VARIANTS), 1))

    for row_idx, phase in enumerate(PHASES):
        phase_df = metrics[metrics['phase'] == phase]
        for col_idx, (metric_col, title) in enumerate(metric_defs):
            ax = axes[row_idx, col_idx]
            for variant_idx, variant in enumerate(VARIANTS):
                vals = []
                for dataset in ACTIVE_DATASETS:
                    part = phase_df[(phase_df['dataset'] == dataset) & (phase_df['variant'] == variant['label'])]
                    vals.append(float(part[metric_col].iloc[0]) if not part.empty and metric_col in part else np.nan)
                xs = x + offset_values[variant_idx]
                ax.plot(xs, vals, marker='o', ms=4, lw=1.5, color=variant['color'], label=variant['label'])
            ax.set_title('{} {}'.format(phase, title))
            ax.grid(axis='y', alpha=0.22)
            ax.set_xticks(x)
            ax.set_xticklabels(ACTIVE_DATASETS, rotation=35, ha='right')
            if metric_col in ['detection_rate', 'pr_auc']:
                ax.set_ylim(-0.02, 1.02)
            if col_idx == 0:
                ax.set_ylabel('{} phase'.format(phase))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 3].legend(handles, labels, loc='best', frameon=False, fontsize=9)
    fig.suptitle('{} positive-sample metrics across datasets'.format(COMPARISON_NAME), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / '{}_metrics_2x4.jpg'.format(COMPARISON_NAME)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def load_noise_summaries():
    rows = []
    for dataset in NOISE_DATASETS:
        for variant in VARIANTS:
            path = (
                NOISE_EVAL_ROOT / dataset / variant['eval_suffix']
                / 'noise_false_positive_summary.csv'
            )
            part = load_csv_or_none(path)
            if part is None:
                continue
            part['dataset'] = dataset
            part['variant'] = variant['label']
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_noise_summary(summary, out_dir):
    if summary.empty:
        print('no noise metric rows available', flush=True)
        return
    metric_defs = [
        ('false_positive_ratio_ps', 'P&S false positive ratio'),
        ('p_pick_window_ratio', 'P pick window ratio'),
        ('s_pick_window_ratio', 'S pick window ratio'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    x = np.arange(len(NOISE_DATASETS), dtype=float)
    offsets = np.linspace(-0.18, 0.18, max(len(VARIANTS), 1))
    for ax, (metric, title) in zip(axes, metric_defs):
        for offset, variant in zip(offsets, VARIANTS):
            values = []
            for dataset in NOISE_DATASETS:
                part = summary[
                    (summary['dataset'] == dataset)
                    & (summary['variant'] == variant['label'])
                ]
                values.append(
                    float(finite_numeric(part[metric]).iloc[0])
                    if not part.empty and metric in part else np.nan
                )
            ax.plot(
                x + offset, values, marker='o', ms=4, lw=1.5,
                color=variant['color'], label=variant['label'],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(NOISE_DATASETS, rotation=35, ha='right')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis='y', alpha=0.22)
    axes[0].set_ylabel('fraction of tested noise windows')
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc='best', frameon=False, fontsize=8)
    fig.suptitle('{} noise stability'.format(COMPARISON_NAME), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / '{}_noise_metrics.jpg'.format(COMPARISON_NAME)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print('noise figure: {}'.format(path), flush=True)


def plot_noise_thresholds(out_dir):
    for dataset in NOISE_DATASETS:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        plotted = False
        for variant in VARIANTS:
            path = (
                NOISE_EVAL_ROOT / dataset / variant['eval_suffix']
                / '{}_noise_threshold_metrics.csv'.format(variant['model'].lower())
            )
            part = load_csv_or_none(path)
            if part is None or part.empty:
                continue
            part = part.sort_values('prob_threshold')
            color = variant['color']
            ax.plot(
                part['prob_threshold'], part['false_positive_ratio_ps'],
                color=color, lw=2.0, label='{} P&S'.format(variant['label']),
            )
            ax.plot(
                part['prob_threshold'], part['p_only_window_ratio'],
                color=color, lw=1.0, ls='--', alpha=0.8,
                label='{} P'.format(variant['label']),
            )
            ax.plot(
                part['prob_threshold'], part['s_only_window_ratio'],
                color=color, lw=1.0, ls=':', alpha=0.8,
                label='{} S'.format(variant['label']),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel('Post-hoc pick probability threshold')
        ax.set_ylabel('Noise-window false positive ratio')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=2, fontsize=7)
        ax.set_title('{} noise false pick ratios'.format(dataset))
        fig.tight_layout()
        path = out_dir / '{}_noise_false_positive_ratio_thresholds.jpg'.format(dataset)
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight')
        plt.close(fig)
        print('noise threshold figure: {}'.format(path), flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    out_dir = OUT_ROOT / COMPARISON_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print('comparison: {}'.format(COMPARISON_NAME), flush=True)
    print('output: {}'.format(out_dir), flush=True)
    preflight_inputs()

    for dataset in ACTIVE_DATASETS:
        ok = plot_dataset_dt_kde(dataset, out_dir)
        print('{} KDE {}'.format(dataset, 'ok' if ok else 'empty'), flush=True)

    metrics = collect_metric_table()
    metrics_path = out_dir / '{}_metric_values.csv'.format(COMPARISON_NAME)
    metrics.to_csv(metrics_path, index=False)
    print('metrics: {}'.format(metrics_path), flush=True)
    plot_metric_summary(metrics, out_dir)

    noise_metrics = load_noise_summaries()
    noise_path = out_dir / '{}_noise_metric_values.csv'.format(COMPARISON_NAME)
    noise_metrics.to_csv(noise_path, index=False)
    print('noise metrics: {}'.format(noise_path), flush=True)
    plot_noise_summary(noise_metrics, out_dir)
    plot_noise_thresholds(out_dir)


if __name__ == '__main__':
    main()
