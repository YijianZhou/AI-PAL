"""Plot a detailed AI-PAL realtime timing record."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_timing(path):
    with open(path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    if len(rows) != 1:
        raise ValueError("Expected one timing row in {}".format(path))
    return rows[0]


def get_float(record, key):
    value = record.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def clean_label(value):
    return (
        value.replace("_", " ")
        .replace("AI PAL", "AI-PAL")
        .replace("PHN SB", "PHN-SB")
    )


def bar_panel(axis, labels, values, title, xlabel, color):
    if not labels:
        axis.text(
            0.5, 0.5, "No timing values", ha="center", va="center",
            transform=axis.transAxes, color="#666666",
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_yticks([])
        return
    bars = axis.barh(labels, values, color=color)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    maximum = max(values) if values else 0.0
    axis.set_xlim(0, maximum * 1.22 if maximum > 0 else 1.0)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + max(maximum * 0.015, 0.001),
            bar.get_y() + bar.get_height() / 2,
            "{:.2f}".format(value),
            va="center",
            fontsize=8,
        )
    axis.grid(axis="x", alpha=0.18)


def initial_picker_timings(record):
    suffix = "_station_sum_sec"
    prefix = "picking_"
    keys = sorted(
        key for key in record
        if key.startswith(prefix) and key.endswith(suffix)
    )
    labels = [
        clean_label(key[len(prefix):-len(suffix)]) for key in keys
    ]
    values = [get_float(record, key) for key in keys]
    prep = get_float(record, "preprocess_sec")
    write = get_float(record, "pick_write_sec")
    if prep or write:
        labels = ["Station preprocessing", "Pick-file writing"] + labels
        values = [prep, write] + values
    return labels, values


def repicker_timings(record):
    stage_items = (
        ("Repicking wall", "event_repick_sec"),
        ("Job/stream setup", "event_repick_job_build_sec"),
        ("Shared window prep", "event_repick_window_prepare_sec"),
        ("Device inference wall", "event_repick_device_inference_wall_sec"),
        ("Device transfer (sum)", "event_repick_device_transfer_sec"),
        ("Picker ensemble", "event_repick_result_merge_sec"),
        ("PAL reassociation", "event_repick_reassociation_sec"),
        ("Final amp + P SNR", "event_repick_waveform_qc_measurement_sec"),
        ("Phase-file writing", "event_repick_output_write_sec"),
        ("Event waveform plotting", "event_repick_plot_sec"),
    )
    labels = [label for label, key in stage_items if key in record]
    values = [get_float(record, key) for label, key in stage_items if key in record]
    prefix = "event_repick_picker_"
    suffix = "_sec"
    keys = sorted(
        key for key in record
        if key.startswith(prefix) and key.endswith(suffix)
    )
    labels.extend(
        clean_label(key[len(prefix):-len(suffix)]) + " model (sum)"
        for key in keys
    )
    values.extend(get_float(record, key) for key in keys)
    return labels, values


def association_timings(record):
    labels = []
    values = []
    wall_suffix = "_wall_sec"
    for key in sorted(record):
        if (
            key.startswith("assoc_") and key.endswith(wall_suffix)
            and key != "assoc_wall_sec"
        ):
            labels.append(
                clean_label(key[len("assoc_"):-len(wall_suffix)]) + " wall"
            )
            values.append(get_float(record, key))

    reserved = ("_wall_sec", "_station_sum_sec")
    for key in sorted(record):
        if not key.startswith("assoc_") or not key.endswith("_sec"):
            continue
        if key in ("assoc_sec", "assoc_wall_sec") or key.endswith(reserved):
            continue
        label = clean_label(key[len("assoc_"):-len("_sec")])
        labels.append(label + " worker")
        values.append(get_float(record, key))
    return labels, values


def merge_timings(record):
    labels = []
    values = []
    for prefix, suffix, label_suffix in (
        ("merge_", "_sec", " subnet merge"),
        ("time_segment_merge_", "_sec", " segment finalize"),
    ):
        for key in sorted(record):
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            if key in ("merge_sec", "time_segment_merge_sec"):
                continue
            middle = key[len(prefix):-len(suffix)]
            if middle.startswith("failed_"):
                continue
            labels.append(clean_label(middle) + label_suffix)
            values.append(get_float(record, key))
    return labels, values


def qc_values(record):
    items = [
        ("Raw traces", "num_raw_traces"),
        ("Bad-rate traces", "num_bad_sampling_rate_traces"),
        ("Prepared stations", "num_station_streams"),
        ("Final intervals", "num_final_intervals_written"),
        ("Repick station-event jobs", "num_event_repick_attempts"),
        ("Repick windows", "num_event_repick_windows"),
    ]
    return (
        [label for label, key in items if key in record],
        [get_float(record, key) for label, key in items if key in record],
    )


def pick_count_values(record):
    items = [
        (clean_label(key[len("num_picks_"):]) + " picks", key)
        for key in sorted(record) if key.startswith("num_picks_")
    ]
    return (
        [label for label, key in items if key in record],
        [get_float(record, key) for label, key in items if key in record],
    )


def associated_phase_count_values(record):
    items = [
        (
            clean_label(key[len("num_associated_picks_"):])
            + " associated P/S pairs",
            key,
        )
        for key in sorted(record) if key.startswith("num_associated_picks_")
    ]
    items += [
        ("POS_NEG + POS pairs", "num_event_repick_pairs_both_groups"),
        ("POS_NEG-only pairs", "num_event_repick_pairs_pos_neg_only"),
        ("POS-only pairs", "num_event_repick_pairs_pos_only"),
        ("Generated repicker pairs", "num_event_repicker_pairs_generated"),
        (
            "Glitch-rejected repicker pairs",
            "num_event_repicker_pairs_glitch_rejected",
        ),
        ("Pairs rejected by reassociation",
         "num_event_repick_picks_reassoc_rejected"),
    ]
    return (
        [label for label, key in items if key in record],
        [get_float(record, key) for label, key in items if key in record],
    )


def event_count_values(record):
    items = [
        (clean_label(key[len("merged_events_"):]) + " segment events", key)
        for key in sorted(record) if key.startswith("merged_events_")
    ]
    items += [
        (clean_label(key[len("events_"):]) + " subnet events", key)
        for key in sorted(record) if key.startswith("events_")
    ]
    items += [
        (clean_label(key[len("final_events_"):]) + " published events", key)
        for key in sorted(record) if key.startswith("final_events_")
    ]
    items += [
        (
            clean_label(key[len("event_waveform_final_plots_"):])
            + " waveform PNGs",
            key,
        )
        for key in sorted(record)
        if key.startswith("event_waveform_final_plots_")
    ]
    items += [
        (
            clean_label(key[len("initial_events_qc_rejected_"):])
            + " initial events rejected by waveform QC",
            key,
        )
        for key in sorted(record)
        if key.startswith("initial_events_qc_rejected_")
    ]
    items += [
        ("Repick intervals", "num_event_repick_intervals"),
        ("Late-S events skipped", "num_event_repick_late_s_skipped"),
        ("Events reassociated", "num_event_repick_reassociated"),
        ("Events rejected by reassociation",
         "num_event_repick_reassociation_rejected"),
    ]
    return (
        [label for label, key in items if key in record],
        [get_float(record, key) for label, key in items if key in record],
    )


def waveform_plot_qc_values(record):
    items = []
    for prefix, suffix in (
        ("event_waveform_final_missing_snapshot_", "missing snapshots"),
        ("event_waveform_final_unusable_waveform_", "unusable waveforms"),
    ):
        items.extend(
            (clean_label(key[len(prefix):]) + " " + suffix, key)
            for key in sorted(record) if key.startswith(prefix)
        )
    return (
        [label for label, key in items if get_float(record, key) > 0],
        [get_float(record, key) for label, key in items
         if get_float(record, key) > 0],
    )


def association_ratio_values(record):
    keys = sorted(key for key in record if key.startswith("assoc_ratio_"))
    return (
        [clean_label(key[len("assoc_ratio_"):]) for key in keys],
        [100.0 * get_float(record, key) for key in keys],
    )


def plot_timing(timing_csv, output_png):
    record = read_timing(timing_csv)
    segment = record.get("segment", os.path.basename(timing_csv))

    wall_items = [
        ("Read MiniSEED", "data_read_sec"),
        ("Sampling-rate QC", "sampling_rate_cleanup_sec"),
        ("Merge traces", "data_merge_sec"),
        ("Prep + initial picking", "picking_sec"),
        ("Picker ensemble", "picker_ensemble_sec"),
        ("PAL association", "assoc_wall_sec"),
        ("Subnet event merge", "merge_sec"),
        ("Initial amp + glitch QC", "initial_waveform_qc_sec"),
        ("Segment merge + OT filter", "time_segment_merge_sec"),
        ("Event repicking", "event_repick_sec"),
    ]
    wall_labels = [label for label, key in wall_items if key in record]
    wall_values = [get_float(record, key) for label, key in wall_items if key in record]
    total = get_float(record, "total_sec")
    accounted = sum(wall_values)
    if total > accounted + 0.005:
        wall_labels.append("Other overhead")
        wall_values.append(total - accounted)

    initial_labels, initial_values = initial_picker_timings(record)
    repick_labels, repick_values = repicker_timings(record)
    assoc_labels, assoc_values = association_timings(record)
    merge_labels, merge_values = merge_timings(record)
    pick_labels, pick_counts = pick_count_values(record)
    phase_labels, phase_counts = associated_phase_count_values(record)
    event_labels, event_counts = event_count_values(record)
    ratio_labels, ratio_values = association_ratio_values(record)
    qc_labels, qc_counts = qc_values(record)
    waveform_qc_labels, waveform_qc_counts = waveform_plot_qc_values(record)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
    fig.suptitle(
        "AI-PAL realtime performance: {}".format(segment), fontsize=15
    )
    panels = (
        (axes[0, 0], wall_labels, wall_values, "End-to-end stages", "Wall time (s)", "#2878B5"),
        (axes[0, 1], initial_labels, initial_values, "Initial picker detail", "Accumulated time (s)", "#E18727"),
        (axes[0, 2], repick_labels, repick_values, "Dual repicker-group detail", "Time (s)", "#6F4E9C"),
        (axes[1, 0], assoc_labels, assoc_values, "PAL association detail", "Time (s)", "#D65F5F"),
        (axes[1, 1], merge_labels, merge_values, "Event merging detail", "Wall time (s)", "#4C9F70"),
        (axes[0, 3], pick_labels, pick_counts, "Pick counts", "Count", "#C57B24"),
        (axes[1, 2], event_labels, event_counts, "Event counts", "Count", "#2E8B8B"),
        (axes[1, 3], phase_labels, phase_counts, "Associated phase-pair counts", "Station P/S pairs", "#777777"),
    )
    for panel in panels:
        bar_panel(*panel)

    if qc_labels:
        qc_text = " | ".join(
            "{}: {:g}".format(label, value)
            for label, value in zip(qc_labels, qc_counts)
        )
    else:
        qc_text = ""

    ratio_text = " | ".join(
        "{} association ratio: {:.2f}%".format(label, value)
        for label, value in zip(ratio_labels, ratio_values)
    )
    primary_annotation = [
        "Total segment wall time: {:.2f} s".format(total),
        qc_text,
    ]
    secondary_annotation = [
        ratio_text,
        " | ".join(
            "{}: {:g}".format(label, value)
            for label, value in zip(
                waveform_qc_labels, waveform_qc_counts
            )
        ),
        "Accumulated picker times may overlap because stations/devices run concurrently.",
    ]

    fig.text(
        0.5, 0.027,
        " | ".join(part for part in primary_annotation if part),
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5, 0.009,
        " | ".join(part for part in secondary_annotation if part),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.95])

    out_dir = os.path.dirname(output_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_csv")
    parser.add_argument("output_png")
    args = parser.parse_args()
    plot_timing(args.timing_csv, args.output_png)
