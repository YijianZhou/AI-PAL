"""Plot a detailed AI-PAL realtime timing record."""
import argparse
import csv
import json
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14


def read_timing(path):
    with open(path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    if len(rows) != 1:
        raise ValueError("Expected one timing row in {}".format(path))
    record = rows[0]
    json_fields = ("reference_branch_keys", "reference_picker_keys")
    try:
        if None in record:
            raise ValueError("Extra CSV columns")
        for key in json_fields:
            if record.get(key):
                json.loads(record[key])
        return record
    except ValueError:
        # Older writers emitted unquoted JSON arrays in comma-separated rows.
        # Decode those arrays intact rather than shifting every later column.
        with open(path, newline="") as fp:
            keys = next(csv.reader([fp.readline()]))
            raw = fp.read().rstrip("\r\n")
        decoder = json.JSONDecoder()
        recovered = {}
        for index, key in enumerate(keys):
            if key in json_fields and raw.lstrip().startswith("["):
                raw = raw.lstrip()
                value, end = decoder.raw_decode(raw)
                recovered[key] = json.dumps(value)
                raw = raw[end:]
                if index < len(keys) - 1:
                    if not raw.startswith(","):
                        raise ValueError("Invalid legacy timing row in {}".format(path))
                    raw = raw[1:]
            else:
                value, separator, raw = raw.partition(",")
                recovered[key] = value
                if index < len(keys) - 1 and not separator:
                    raise ValueError("Incomplete legacy timing row in {}".format(path))
        if raw:
            raise ValueError("Unexpected trailing timing fields in {}".format(path))
        return recovered


def get_float(record, key):
    value = record.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def clean_label(value):
    return (
        value.replace("_", " ")
        .replace("AI PAL", "AI-PAL")
        .replace("PHN SB", "PHN-SB")
    )


def subnet_label(value):
    # Retain the workflow prefix, but omit station-file metadata around rN.
    match = re.fullmatch(r"(.+?)_station_.*?_(r\d+)(?:_.*)?", value, re.IGNORECASE)
    if match:
        value = "{}_{}".format(match.group(1), match.group(2).lower())
    return clean_label(value)


def is_published_label(label):
    return label.endswith((" published events", " waveform PNGs"))


def short_tick_label(label):
    replacements = (
        ("initial events rejected by waveform QC", "init. QC rej."),
        ("Glitch-rejected repicker pairs", "Glitch-rej. pairs"),
        ("Generated repicker pairs", "Generated pairs"),
        ("Pairs rejected by reassociation", "Reassoc.-rej. pairs"),
        ("Events rejected by reassociation", "Reassoc.-rej. events"),
        ("Events reassociated", "Reassoc. events"),
        ("associated P/S pairs", "assoc. pairs"),
        ("subnet events", "events"),
        ("segment events", "seg. events"),
        ("published events", "published"),
        ("waveform PNGs", "waveform plots"),
        ("Station preprocessing", "Station preproc."),
        ("Device inference wall", "Inference wall"),
        ("Event waveform plotting", "Waveform plotting"),
        ("PAL reassociation", "PAL reassoc."),
        ("Association (all workflows)", "Assoc. (all)"),
        ("Segment merge + OT filter", "Seg. merge + OT filter"),
        ("waveform QC", "wave QC"),
        ("segment finalize", "seg. finalize"),
    )
    for source, target in replacements:
        label = label.replace(source, target)
    return label


def bar_panel(axis, labels, values, title, xlabel, color, counts=False):
    axis.tick_params(axis="both", labelsize=LABEL_FONTSIZE)
    axis.set_title(title, fontsize=TITLE_FONTSIZE)
    axis.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if not counts:
        keep = [index for index, value in enumerate(values) if value >= 1.0]
        labels = [labels[index] for index in keep]
        values = [values[index] for index in keep]
        if not isinstance(color, str):
            color = [color[index] for index in keep]
    if counts:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not labels:
        axis.text(
            0.5, 0.5, "No counts" if counts else "No timings >= 1 s", ha="center", va="center",
            transform=axis.transAxes, color="#666666", fontsize=LABEL_FONTSIZE,
        )
        axis.set_yticks([])
        return
    positions = list(range(len(labels)))
    bars = axis.barh(positions, values, color=color)
    axis.set_yticks(positions, [short_tick_label(label) for label in labels])
    for bar, label in zip(bars, labels):
        if is_published_label(label):
            bar.set_edgecolor("k")
            bar.set_linewidth(1.3)
    axis.invert_yaxis()
    maximum = max(values) if values else 0.0
    axis.set_xlim(0, maximum * 1.22 if maximum > 0 else 1.0)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + max(maximum * 0.015, 0.001),
            bar.get_y() + bar.get_height() / 2,
            "{:,d}".format(int(value)) if counts else "{:.2f}".format(value),
            va="center",
            fontsize=LABEL_FONTSIZE,
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
    def group_order(key):
        name = key[len(prefix):-len(suffix)]
        if name.startswith("POS_NEG_"):
            return (1, name[len("POS_NEG_"):])
        if name.startswith("POS_"):
            return (0, name[len("POS_"):])
        return (2, name)
    keys = sorted(
        (key for key in record
         if key.startswith(prefix) and key.endswith(suffix)),
        key=group_order,
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
        label = subnet_label(key[len("assoc_"):-len("_sec")])
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
        (subnet_label(key[len("events_"):]) + " subnet events", key)
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
    reference_keys = json.loads(record.get("reference_branch_keys") or '["PHN_SB"]')
    reference_pickers = json.loads(record.get("reference_picker_keys") or '["PHN_SB"]')
    tokens = set(reference_keys + reference_pickers)
    def is_reference(key):
        return any(("_" + token + "_") in key or key.endswith("_" + token)
                   for token in tokens)
    preferred = {key: value for key, value in record.items() if not is_reference(key)}
    references = {key: value for key, value in record.items() if is_reference(key)}

    wall_items = [
        ("Read MiniSEED", "data_read_sec"),
        ("Sampling-rate QC", "sampling_rate_cleanup_sec"),
        ("Merge traces", "data_merge_sec"),
        ("Prep + initial picking", "picking_sec"),
        ("Picker ensemble", "picker_ensemble_sec"),
        ("Association (all workflows)", "assoc_wall_sec"),
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

    def branch_labels(labels):
        return [
            label.replace("PHN-SB", "PHN-SB PAL", 1)
            if label.startswith("PHN-SB") and not label.startswith("PHN-SB PAL")
            and "GaMMA" not in label else label
            for label in labels
        ]

    def combined(function, branch=True):
        labels, values = function(preferred)
        ref_labels, ref_values = function(references)
        if branch:
            ref_labels = branch_labels(ref_labels)
        return labels + ref_labels, values + ref_values

    def colors(labels):
        return [
            "#29966F" if "GaMMA" in label else
            "#DE8F32" if any(label.startswith(clean_label(key)) for key in reference_pickers) else
            "#888888" if label in ("Station preprocessing", "Pick-file writing") else
            "#2878B5"
            for label in labels
        ]

    initial = combined(initial_picker_timings, branch=False)
    repick = repicker_timings(record)
    assoc = combined(association_timings)
    merging = combined(merge_timings)
    for key in sorted(record):
        for prefix, suffix in (("initial_waveform_qc_", " waveform QC"),
                               ("event_waveform_plot_", " plots")):
            if key.startswith(prefix) and key.endswith("_sec") and key != prefix + "sec":
                merging[0].extend(branch_labels([clean_label(key[len(prefix):-4]) + suffix]))
                merging[1].append(get_float(record, key))
    # Reference picks shared by several associators are counted only once.
    pick_record = dict(preferred)
    pick_record.update({key: value for key, value in references.items()
                       if key in {"num_picks_" + name for name in reference_pickers}})
    picks = pick_count_values(pick_record)
    associated_record = {key: value for key, value in record.items()
                         if key.startswith("num_associated_picks_")
                         or key.startswith("num_event_repick_pairs_")}
    associated = associated_phase_count_values(associated_record)
    associated = (branch_labels(associated[0]), associated[1])
    pair_qc = associated_phase_count_values({
        key: value for key, value in record.items()
        if key in ("num_event_repicker_pairs_generated",
                   "num_event_repicker_pairs_glitch_rejected",
                   "num_event_repick_picks_reassoc_rejected")
    })
    events = combined(event_count_values)
    ratio_labels, ratio_values = association_ratio_values(record)
    qc_labels, qc_counts = qc_values(record)
    waveform_qc_labels, waveform_qc_counts = waveform_plot_qc_values(record)

    fig, axes = plt.subplots(3, 3, figsize=(25, 22))
    fig.suptitle("AI-PAL realtime performance: {}".format(segment), fontsize=TITLE_FONTSIZE)
    fig.legend(handles=[
        Patch(color="#2878B5", label="Preferred AI-PAL"),
        Patch(color="#DE8F32", label="Reference picker / PAL"),
        Patch(color="#29966F", label="Reference GaMMA"),
        Patch(color="#888888", label="Shared preprocessing"),
        Patch(facecolor="white", edgecolor="k", linewidth=1.3,
              label="Published outputs"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=5, frameon=False,
       fontsize=LABEL_FONTSIZE)
    panels = (
        (axes[0, 0], (wall_labels, wall_values), "End-to-end stages", "Wall time (s)", False),
        (axes[0, 1], initial, "Picking + shared preprocessing", "Accumulated time (s)", False),
        (axes[0, 2], repick, "Postprocessing timing", "Time (s)", False),
        (axes[1, 0], assoc, "Association timing", "Time (s)", False),
        (axes[1, 1], merging,
         "Merging + waveform QC + plotting" if any(
             value >= 1.0 and ("merge" in label or "finalize" in label)
             for label, value in zip(*merging)
         ) else "Waveform QC + plotting", "Time (s)", False),
        (axes[1, 2], picks, "Generated pick counts", "Station P/S pairs", True),
        (axes[2, 0], events, "Event counts", "Events / outputs", True),
        (axes[2, 1], associated, "Associated pick counts", "Station P/S pairs", True),
        (axes[2, 2], pair_qc, "Repicking pair QC", "Station P/S pairs", True),
    )
    for axis, (labels, values), title, xlabel, counts in panels:
        palette = colors(labels)
        # Stable grouping preserves metric/model order within each workflow.
        priority = {"#2878B5": 0, "#888888": 1, "#DE8F32": 2, "#29966F": 3}
        order = sorted(range(len(labels)), key=lambda index: priority[palette[index]])
        bar_panel(axis, [labels[index] for index in order],
                  [values[index] for index in order], title, xlabel,
                  [palette[index] for index in order], counts=counts)

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
        "Timing bars < 1 s omitted. Accumulated picker times may overlap across stations/devices.",
    ]

    footer = "\n".join(
        textwrap.fill(" | ".join(part for part in parts if part), width=180,
                      break_long_words=False, break_on_hyphens=False)
        for parts in (primary_annotation, secondary_annotation)
    )
    fig.text(0.5, 0.015, footer, ha="center", va="bottom",
             fontsize=LABEL_FONTSIZE, linespacing=1.5)
    footer_height = (footer.count("\n") + 1) * LABEL_FONTSIZE * 1.5 / (22 * 72)
    fig.tight_layout(rect=[0, max(0.065, footer_height + 0.035), 1, 0.95])

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
