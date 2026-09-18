"""Write lightweight CSV and PNG summaries alongside training checkpoints."""

import csv
import re
from pathlib import Path


class TrainingMonitor:
  def __init__(self, log_dir, title):
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.title = str(title)
    model_name = re.sub(r'[^a-z0-9]+', '_', self.title.lower()).strip('_')
    if not model_name:
      model_name = 'model'
    self.rows = {}
    self.csv_path = self.log_dir / '{}_training_metrics.csv'.format(model_name)
    self.accuracy_figure_path = self.log_dir / (
      '{}_training_progress.png'.format(model_name)
    )
    self.diagnostics_figure_path = self.log_dir / (
      '{}_training_diagnostics.png'.format(model_name)
    )
    self._plot_warning_reported = False
    self._plot_disabled = False

  def update(self, step, metrics):
    step = int(step)
    row = self.rows.setdefault(step, {})
    for name, value in metrics.items():
      if value is not None:
        row[str(name)] = float(value)
    self._write_csv()
    if self._plot_disabled:
      return
    try:
      self._write_figure(step)
    except Exception as exc:
      if not self._plot_warning_reported:
        print('warning: live training figure could not be updated: {}'.format(exc),
              flush=True)
        self._plot_warning_reported = True
      self._plot_disabled = True

  def _write_csv(self):
    metric_names = sorted({
      name for row in self.rows.values() for name in row
    })
    temp_path = self.csv_path.with_suffix('.csv.tmp')
    with temp_path.open('w', newline='', encoding='utf-8') as fp:
      writer = csv.DictWriter(fp, fieldnames=['step', *metric_names])
      writer.writeheader()
      for step in sorted(self.rows):
        writer.writerow({'step': step, **self.rows[step]})
    temp_path.replace(self.csv_path)

  def _write_figure(self, latest_step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    all_names = sorted({name for row in self.rows.values() for name in row})

    def names_for(prefix):
      return [name for name in all_names if name.startswith(prefix)]

    def plot_metrics(axis, names):
      plotted_values = []
      for name in names:
        points = [
          (step, self.rows[step][name])
          for step in sorted(self.rows) if name in self.rows[step]
        ]
        if not points:
          continue
        steps, values = zip(*points)
        plotted_values.extend(values)
        label = name.split('/', 1)[-1].replace('_', ' ')
        axis.plot(steps, values, marker='o', markersize=2.5,
                  linewidth=1.4, label=label)
      axis.set_xlabel('Training step')
      axis.grid(True, alpha=0.25)
      if plotted_values:
        axis.legend(fontsize=8)
      else:
        axis.text(0.5, 0.5, 'Not reported', ha='center', va='center',
                  transform=axis.transAxes, color='0.5')
      return np.asarray(plotted_values, dtype=float)

    def set_metric_ylim(axis, values, percentile_zoom=False, bounds=None):
      values = values[np.isfinite(values)]
      if not values.size:
        return
      if percentile_zoom:
        lower, upper = np.percentile(values, [5.0, 95.0])
      else:
        lower, upper = float(values.min()), float(values.max())
      span = upper - lower
      padding = max(0.2, 0.08 * span)
      if span <= 1e-9:
        padding = max(0.5, 0.01 * max(abs(lower), 1.0))
      lower -= padding
      upper += padding
      if bounds is not None:
        bound_lower, bound_upper = bounds
        if bound_lower is not None:
          lower = max(bound_lower, lower)
        if bound_upper is not None:
          upper = min(bound_upper, upper)
      axis.set_ylim(lower, upper)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True,
                             sharex='col')
    for column, (label, prefix) in enumerate((
      ('Positive accuracy', 'pos_acc/'),
      ('Negative accuracy', 'neg_acc/'),
    )):
      names = names_for(prefix)
      full_values = plot_metrics(axes[0, column], names)
      zoom_values = plot_metrics(axes[1, column], names)
      axes[0, column].set_title('{}: full range'.format(label))
      axes[1, column].set_title('{}: 5th-95th percentile zoom'.format(label))
      axes[0, column].set_ylabel('Accuracy (%)')
      axes[1, column].set_ylabel('Accuracy (%)')
      set_metric_ylim(axes[0, column], full_values, bounds=(0.0, 100.0))
      set_metric_ylim(
        axes[1, column], zoom_values, percentile_zoom=True,
        bounds=(0.0, 100.0),
      )
    fig.suptitle('{} detection accuracy | latest step {:,}'.format(
      self.title, int(latest_step)
    ))
    temp_path = self.accuracy_figure_path.with_name(
      '{}.tmp.png'.format(self.accuracy_figure_path.stem)
    )
    fig.savefig(temp_path, dpi=160)
    plt.close(fig)
    temp_path.replace(self.accuracy_figure_path)

    frame_names = names_for('frame_acc/')
    if frame_names:
      accuracy_label = 'Frame accuracy'
      accuracy_names = frame_names
    else:
      accuracy_label = 'Sample accuracy'
      accuracy_names = names_for('sample_acc/')

    fig, axes = plt.subplots(
      2, 2, figsize=(12, 8), constrained_layout=True, sharex='col'
    )
    for column, (label, names, ylabel, bounds) in enumerate((
      ('Loss', names_for('loss/'), 'Loss', (0.0, None)),
      (accuracy_label, accuracy_names, 'Accuracy (%)', (0.0, 100.0)),
    )):
      full_values = plot_metrics(axes[0, column], names)
      zoom_values = plot_metrics(axes[1, column], names)
      axes[0, column].set_title('{}: full range'.format(label))
      axes[1, column].set_title(
        '{}: 5th-95th percentile zoom'.format(label)
      )
      axes[0, column].set_ylabel(ylabel)
      axes[1, column].set_ylabel(ylabel)
      set_metric_ylim(axes[0, column], full_values, bounds=bounds)
      set_metric_ylim(
        axes[1, column], zoom_values, percentile_zoom=True, bounds=bounds
      )
    fig.suptitle('{} training diagnostics | latest step {:,}'.format(
      self.title, int(latest_step)
    ))
    temp_path = self.diagnostics_figure_path.with_name(
      '{}.tmp.png'.format(self.diagnostics_figure_path.stem)
    )
    fig.savefig(temp_path, dpi=160)
    plt.close(fig)
    temp_path.replace(self.diagnostics_figure_path)
