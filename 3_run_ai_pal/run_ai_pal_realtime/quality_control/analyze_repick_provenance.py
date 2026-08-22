#!/usr/bin/env python3
"""Developer check of dual-repicker-group outcomes by event magnitude."""

import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# SETTINGS
# ============================================================================
OUT_ROOT = Path("/app/aqms/ai_pal/OUT/v7_bak3")
FINAL_PHASE_DIR = OUT_ROOT / "3.1_phase_final_AI-PAL"
PHASE_GLOB = "phase_final_*.dat"
STATISTICS_DIR = OUT_ROOT / "repick_statistics"
MAGNITUDE_BIN_WIDTH = 0.5
FIGURE_DPI = 180


CATEGORIES = ("both_groups", "pos_neg_only", "pos_only")
LABELS = {
    "both_groups": "POS_NEG + POS",
    "pos_neg_only": "POS_NEG only",
    "pos_only": "POS only",
}
COLORS = {
    "both_groups": "#2878B5",
    "pos_neg_only": "#7A5195",
    "pos_only": "#3A923A",
}


def is_event_header(codes):
    if len(codes) < 5 or "T" not in codes[0]:
        return False
    try:
        [float(value) for value in codes[1:5]]
        return True
    except ValueError:
        return False


def classify_provenance(value):
    tokens = {
        token.strip().lower()
        for token in str(value or "initial").split("|")
        if token.strip()
    }
    if "both_groups" in tokens or "repicked" in tokens:
        category = "both_groups"
    elif "pos_neg_only" in tokens:
        category = "pos_neg_only"
    elif "pos_only" in tokens or "supplemented" in tokens:
        category = "pos_only"
    else:
        category = None
    return category, "|".join(sorted(tokens or {"initial"}))


def read_picks(phase_files):
    rows = []
    event_count = 0
    composite_count = 0
    for phase_path in phase_files:
        event = None
        with phase_path.open(encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                codes = [value.strip() for value in line.strip().split(",")]
                if not codes or not codes[0]:
                    continue
                if is_event_header(codes):
                    event = {"time": codes[0], "magnitude": float(codes[4])}
                    event_count += 1
                    continue
                if event is None or len(codes) < 4:
                    raise ValueError(
                        "{}:{} invalid phase row".format(phase_path, line_number)
                    )
                provenance = codes[14] if len(codes) > 14 else "initial"
                category, provenance = classify_provenance(provenance)
                if category is None:
                    continue
                composite_count += int("|" in provenance)
                rows.append({
                    "phase_file": str(phase_path),
                    "event_time": event["time"],
                    "event_magnitude": event["magnitude"],
                    "station": codes[0],
                    "p_time": codes[1],
                    "s_time": codes[2],
                    "provenance": provenance,
                    "category": category,
                    "repick_status": (
                        codes[15] if len(codes) > 15 else "unknown"
                    ),
                    "repick_support": (
                        int(codes[16])
                        if len(codes) > 16 and codes[16] else -1
                    ),
                    "repick_sources": codes[17] if len(codes) > 17 else "",
                    "repick_required_support": (
                        int(codes[18])
                        if len(codes) > 18 and codes[18] else -1
                    ),
                })
    return rows, event_count, composite_count


def magnitude_edges(rows):
    if MAGNITUDE_BIN_WIDTH <= 0:
        raise ValueError("MAGNITUDE_BIN_WIDTH must be positive")
    magnitudes = [row["event_magnitude"] for row in rows]
    lower = math.floor(min(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    upper = math.ceil(max(magnitudes) / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH
    if upper <= lower:
        upper = lower + MAGNITUDE_BIN_WIDTH
    count = int(round((upper - lower) / MAGNITUDE_BIN_WIDTH))
    return [lower + index * MAGNITUDE_BIN_WIDTH for index in range(count + 1)]


def bin_index(value, edges):
    if value == edges[-1]:
        return len(edges) - 2
    index = int((value - edges[0]) // MAGNITUDE_BIN_WIDTH)
    return min(max(index, 0), len(edges) - 2)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    phase_files = sorted(FINAL_PHASE_DIR.glob(PHASE_GLOB))
    if not phase_files:
        raise FileNotFoundError(
            "no phase files matched {} in {}".format(
                PHASE_GLOB, FINAL_PHASE_DIR
            )
        )
    rows, event_count, composite_count = read_picks(phase_files)
    if not rows:
        raise ValueError("final phase files contain no station picks")

    edges = magnitude_edges(rows)
    bin_counts = [Counter() for _ in edges[:-1]]
    for row in rows:
        bin_counts[bin_index(row["event_magnitude"], edges)][row["category"]] += 1

    binned_rows = []
    for index, counts in enumerate(bin_counts):
        total = sum(counts.values())
        row = {
            "magnitude_min": "{:.2f}".format(edges[index]),
            "magnitude_max": "{:.2f}".format(edges[index + 1]),
            "magnitude_bin": (
                "[{:.2f}, {:.2f}]" if index == len(bin_counts) - 1
                else "[{:.2f}, {:.2f})"
            ).format(edges[index], edges[index + 1]),
            "total": total,
        }
        for category in CATEGORIES:
            row[category] = counts[category]
            row[category + "_percent"] = (
                100.0 * counts[category] / total if total else 0.0
            )
        binned_rows.append(row)

    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        STATISTICS_DIR / "repick_provenance_picks.csv",
        list(rows[0]),
        rows,
    )
    write_csv(
        STATISTICS_DIR / "repick_provenance_by_magnitude.csv",
        list(binned_rows[0]),
        binned_rows,
    )

    totals = Counter(row["category"] for row in rows)
    summary_rows = [
        {
            "metric": category,
            "count": totals[category],
            "percent_of_final_picks": 100.0 * totals[category] / len(rows),
        }
        for category in CATEGORIES
    ]
    summary_rows.extend([
        {"metric": "total_picks", "count": len(rows), "percent_of_final_picks": 100.0},
        {"metric": "events", "count": event_count, "percent_of_final_picks": ""},
        {"metric": "phase_files", "count": len(phase_files), "percent_of_final_picks": ""},
        {"metric": "composite_provenance_rows", "count": composite_count, "percent_of_final_picks": ""},
    ])
    write_csv(
        STATISTICS_DIR / "repick_provenance_summary.csv",
        ["metric", "count", "percent_of_final_picks"],
        summary_rows,
    )
    x = np.arange(len(binned_rows), dtype=float)
    width = 0.25
    labels = [row["magnitude_bin"] for row in binned_rows]
    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(labels) * 0.8), 9))
    fig.suptitle(
        "Overall repicker pairs: both groups {:,} ({:.1f}%) | "
        "POS_NEG only {:,} ({:.1f}%) | POS only {:,} ({:.1f}%)".format(
            totals["both_groups"],
            100.0 * totals["both_groups"] / len(rows),
            totals["pos_neg_only"],
            100.0 * totals["pos_neg_only"] / len(rows),
            totals["pos_only"],
            100.0 * totals["pos_only"] / len(rows),
        ),
        fontsize=12,
    )
    for offset, category in enumerate(CATEGORIES):
        axes[0].bar(
            x + (offset - 1) * width,
            [row[category] for row in binned_rows],
            width=width,
            label=LABELS[category],
            color=COLORS[category],
        )
    axes[0].set_ylabel("Number of station P/S pairs")
    axes[0].set_title("Dual-repicker-group outcomes by event magnitude")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    bottom = np.zeros(len(binned_rows), dtype=float)
    for category in CATEGORIES:
        values = np.asarray([
            row[category + "_percent"] for row in binned_rows
        ])
        axes[1].bar(
            x, values, bottom=bottom,
            label=LABELS[category], color=COLORS[category],
        )
        bottom += values
    axes[1].set_ylabel("Fraction of picks (%)")
    axes[1].set_xlabel("Event magnitude range")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    figure_path = STATISTICS_DIR / "repick_provenance_by_magnitude.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print("repick provenance analysis complete")
    print("  phase files: {}".format(len(phase_files)))
    print("  events: {}".format(event_count))
    print("  total picks: {}".format(len(rows)))
    for category in CATEGORIES:
        print("  {}: {} ({:.2f}%)".format(
            category,
            totals[category],
            100.0 * totals[category] / len(rows),
        ))
    print("  output: {}".format(STATISTICS_DIR))


if __name__ == "__main__":
    main()
