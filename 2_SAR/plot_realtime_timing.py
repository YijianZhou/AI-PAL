"""Plot one AI-PAL realtime timing record."""
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


def plot_timing(timing_csv, output_png):
    record = read_timing(timing_csv)
    segment = record.get("segment", os.path.basename(timing_csv))

    step_keys = [
        ("Read", "data_read_sec"),
        ("Rate cleanup", "sampling_rate_cleanup_sec"),
        ("Merge traces", "data_merge_sec"),
        ("Station prep", "preprocess_sec"),
        ("Phase picking", "picking_sec"),
        ("Association", "assoc_wall_sec"),
        ("Merge events", "merge_sec"),
    ]
    step_labels = [item[0] for item in step_keys]
    step_values = [get_float(record, item[1]) for item in step_keys]

    assoc_keys = sorted(
        key for key in record
        if key.startswith("assoc_") and key.endswith("_wall_sec")
        and key != "assoc_wall_sec"
    )
    assoc_labels = [
        key[len("assoc_"):-len("_wall_sec")] for key in assoc_keys
    ]
    assoc_values = [get_float(record, key) for key in assoc_keys]

    count_keys = [
        ("Raw traces", "num_raw_traces"),
        ("Dropped rate", "num_bad_sampling_rate_traces"),
        ("Stations", "num_station_streams"),
    ]
    count_keys += [
        (key[len("num_picks_"):] + " picks", key)
        for key in sorted(record) if key.startswith("num_picks_")
    ]
    count_keys += [
        (key[len("merged_events_"):] + " events", key)
        for key in sorted(record) if key.startswith("merged_events_")
    ]
    count_labels = [item[0] for item in count_keys]
    count_values = [get_float(record, item[1]) for item in count_keys]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("AI-PAL realtime performance: {}".format(segment), fontsize=13)

    axes[0].barh(step_labels, step_values, color="#2878B5")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Wall time (s)")
    axes[0].set_title("Workflow timing")

    axes[1].barh(assoc_labels, assoc_values, color="#D65F5F")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Worker time (s)")
    axes[1].set_title("Subnet association")

    axes[2].barh(count_labels, count_values, color="#3A923A")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Count")
    axes[2].set_title("Processing counts")

    total = get_float(record, "total_sec")
    fig.text(0.5, 0.01, "Total segment wall time: {:.2f} s".format(total), ha="center")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    out_dir = os.path.dirname(output_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_csv")
    parser.add_argument("output_png")
    args = parser.parse_args()
    plot_timing(args.timing_csv, args.output_png)
