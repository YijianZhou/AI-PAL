#!/usr/bin/env python3
"""Container entry point for one yearly SCEDC NPY cutting job."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SOURCE_DIR = Path("/opt/ml/processing/source")
INPUT_DIR = Path("/opt/ml/processing/input")
PHASE_DIR = INPUT_DIR / "phase"
ASSOC_RATE_DIR = INPUT_DIR / "association_rate"
STATION_DIR = INPUT_DIR / "station"
OUTPUT_DIR = Path("/opt/ml/processing/output/npy")
REQUIREMENTS = SOURCE_DIR / "requirements.txt"


def one_file(root, suffix):
    matches = sorted(path for path in root.rglob("*" + suffix) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one {} file under {}, found {}".format(
                suffix, root, len(matches)
            )
        )
    return matches[0]


def run(command, label, timings):
    print("{}: {}".format(label, " ".join(map(str, command))), flush=True)
    started = time.perf_counter()
    subprocess.check_call([str(item) for item in command], cwd=str(SOURCE_DIR))
    timings[label] = time.perf_counter() - started


def index_summary(path):
    rows = np.load(path, allow_pickle=False)
    if rows.size == 0:
        return {"shards": 0, "samples": 0, "bytes": path.stat().st_size}
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("bad shard index shape for {}: {}".format(path, rows.shape))
    return {
        "shards": int(rows.shape[0]),
        "samples": int(np.asarray(rows[:, 1], dtype=np.int64).sum()),
        "bytes": path.stat().st_size,
    }


def main():
    if REQUIREMENTS.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )

    training_year = int(os.environ["TRAINING_YEAR"])
    phase_file = one_file(PHASE_DIR, ".pha")
    station_file = one_file(STATION_DIR, ".csv")
    rate_file = one_file(ASSOC_RATE_DIR, ".csv")
    rate_rows = 0
    rate_dates = set()
    with rate_file.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required_columns = {
            "date", "net_sta", "num_picks",
            "num_associated_picks", "num_unassociated_picks",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "{} missing columns: {}".format(
                    rate_file, ", ".join(sorted(missing_columns))
                )
            )
        for row in reader:
            if row["date"].strip().lower() == "date":
                continue
            rate_rows += 1
            rate_dates.add(row["date"].strip())
    wrong_year_dates = sorted(
        value for value in rate_dates
        if not value.startswith(str(training_year) + "-")
    )
    if not rate_rows:
        raise RuntimeError("{} contains no station-date rows".format(rate_file))
    if wrong_year_dates:
        raise RuntimeError(
            "{} contains {} dates outside {}".format(
                rate_file, len(wrong_year_dates), training_year
            )
        )
    workers = int(os.environ.get("NUM_WORKERS", "10"))
    shard_size = int(os.environ.get("SHARD_SIZE", "1024"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = {}
    common = [
        "--data_dir", station_file,
        "--fpha", phase_file,
        "--out_root", OUTPUT_DIR,
        "--num_workers", workers,
        "--shard_size", shard_size,
    ]
    run(
        [sys.executable, SOURCE_DIR / "cut_positive_npy.py", *common],
        "positive_sec",
        timings,
    )
    run(
        [
            sys.executable,
            SOURCE_DIR / "cut_negative_npy.py",
            *common,
            "--fassoc_rate",
            rate_file,
        ],
        "negative_sec",
        timings,
    )

    indexes = {}
    for name in (
        "train_pos.npy",
        "valid_pos.npy",
        "train_neg.npy",
        "valid_neg.npy",
    ):
        path = OUTPUT_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        indexes[name] = index_summary(path)

    manifest = {
        "sample_run": os.environ.get("SAMPLE_RUN"),
        "training_year": training_year,
        "phase_file": phase_file.name,
        "station_file": station_file.name,
        "association_rate_file": rate_file.name,
        "association_rate_rows": rate_rows,
        "association_rate_dates": len(rate_dates),
        "num_workers": workers,
        "shard_size": shard_size,
        "indexes": indexes,
        "timing_sec": timings,
    }
    (OUTPUT_DIR / "cut_samples_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
