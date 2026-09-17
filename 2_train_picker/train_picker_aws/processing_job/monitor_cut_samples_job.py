#!/usr/bin/env python3
"""Report annual SCEDC sample-cutting jobs and NPY progress."""

import calendar
import io
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath

import boto3
import numpy as np
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session

from job_common import (
    human_bytes,
    latest_processing_job,
    print_job_status,
    summarize_s3_prefix,
)


# ============================================================================
# USER SETTINGS: edit paths, case identity, AWS resources, and runtime here
# ============================================================================
CASE_CODE = "eg"  # Must match the AWS training submission scripts.
region = "us-west-2"
bucket = None
SAMPLE_RUN = "%s-2020-2025-rarity-v1" % CASE_CODE
years = (2020, 2021, 2022, 2023, 2024, 2025)
artifact_prefix = "sagemaker/ai-pal/training/" + SAMPLE_RUN
audit_completed_indexes = True
required_phase_suffix = "_pal_rarity.pha"

REQUIRED_INDEXES = (
    "train_pos.npy",
    "valid_pos.npy",
    "train_neg.npy",
    "valid_neg.npy",
)


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def read_json(s3, bucket, key):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except s3.exceptions.NoSuchKey:
        return None


def ratio_percent(value, total):
    return 100.0 * value / total if total else 0.0


def read_index_summary(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    rows = np.load(io.BytesIO(body), allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("expected shape (n_shards, 2), found {}".format(rows.shape))

    samples = 0
    bad_paths = []
    for stored_path, count in rows:
        text = str(stored_path)
        path = PurePosixPath(text.replace("\\", "/"))
        if (
            path.is_absolute()
            or PureWindowsPath(text).is_absolute()
            or ".." in path.parts
        ):
            if len(bad_paths) < 3:
                bad_paths.append(text)
        samples += int(count)
    return {
        "shards": int(rows.shape[0]),
        "samples": samples,
        "bad_paths": bad_paths,
    }


def audit_annual_output(s3, bucket, prefix, year, manifest, object_count):
    issues = []
    notes = []
    summaries = {}

    manifest_year = manifest.get("training_year")
    if manifest_year is not None and int(manifest_year) != int(year):
        issues.append(
            "manifest training_year {} does not match {}".format(
                manifest_year, year
            )
        )

    phase_file = str(manifest.get("phase_file", ""))
    if required_phase_suffix and not phase_file.endswith(required_phase_suffix):
        issues.append(
            "phase file {!r} is not rarity-aware (expected suffix {!r})".format(
                phase_file, required_phase_suffix
            )
        )

    expected_dates = 366 if calendar.isleap(int(year)) else 365
    rate_dates = manifest.get("association_rate_dates")
    if rate_dates is None or int(rate_dates) != expected_dates:
        issues.append(
            "association-rate date count {} does not match expected {}".format(
                rate_dates, expected_dates
            )
        )

    recorded_run = manifest.get("sample_run")
    if not recorded_run:
        issues.append("manifest is missing sample_run")
    elif recorded_run != SAMPLE_RUN:
        issues.append(
            "manifest records sample_run {!r}, current library is {!r}".format(
                recorded_run, SAMPLE_RUN
            )
        )

    manifest_indexes = manifest.get("indexes", {})
    for name in REQUIRED_INDEXES:
        key = prefix + "/" + name
        try:
            summary = read_index_summary(s3, bucket, key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                issues.append("missing {}".format(name))
                continue
            raise
        except Exception as exc:
            issues.append("unreadable {}: {}".format(name, exc))
            continue

        summaries[name] = summary
        if summary["bad_paths"]:
            issues.append(
                "{} contains non-portable path(s): {}".format(
                    name, ", ".join(summary["bad_paths"])
                )
            )
        recorded = manifest_indexes.get(name)
        if recorded is None:
            issues.append("manifest has no summary for {}".format(name))
            continue
        for field in ("shards", "samples"):
            if int(recorded.get(field, -1)) != int(summary[field]):
                issues.append(
                    "{} {} mismatch: manifest {}, index {}".format(
                        name, field, recorded.get(field), summary[field]
                    )
                )

    if len(summaries) == len(REQUIRED_INDEXES):
        indexed_shards = sum(item["shards"] for item in summaries.values())
        metadata_objects = object_count - indexed_shards
        if metadata_objects < len(REQUIRED_INDEXES) + 1:
            issues.append(
                "S3 object count is too small for indexed shards and metadata"
            )
        else:
            notes.append(
                "{} indexed shards plus {} metadata/progress objects".format(
                    indexed_shards, metadata_objects
                )
            )
    return not issues, summaries, issues, notes


def print_cut_progress(progresses):
    if not progresses:
        print("  progress:   waiting for first continuous-output sync")
        return

    processed_total = sum(
        int(item.get("processed_attempts", 0)) for item in progresses
    )
    planned_total = sum(
        int(item.get("planned_attempts", 0)) for item in progresses
    )
    generated_total = sum(
        int(item.get("generated_samples", 0)) for item in progresses
    )
    print(
        "  sample work:{:>11,d} / {:,d} attempts ({:.1f}%)".format(
            processed_total,
            planned_total,
            ratio_percent(processed_total, planned_total),
        )
    )
    current_yield = (
        float(generated_total) / processed_total if processed_total else 0.0
    )
    print(
        "  generated:  {:>11,d} samples ({:.1f}% yield)".format(
            generated_total,
            100.0 * current_yield,
        )
    )
    bytes_per_sample = max(
        int(item.get("bytes_per_sample", 0)) for item in progresses
    )
    if bytes_per_sample and processed_total:
        projected_samples = int(round(planned_total * current_yield))
        projected_bytes = projected_samples * bytes_per_sample
        print(
            "  projection: {:>11,d} samples, about {} for known stages".format(
                projected_samples, human_bytes(projected_bytes)
            )
        )
    for item in progresses:
        completed = int(item.get("completed_items", 0))
        total = int(item.get("total_items", 0))
        processed = int(item.get("processed_attempts", 0))
        planned = int(item.get("planned_attempts", 0))
        print(
            "  {:9s} {:>11,d} / {:,d} attempts ({:5.1f}%); "
            "{:,d}/{:,d} station-dates{}".format(
                item.get("stage", "unknown") + ":",
                processed,
                planned,
                ratio_percent(processed, planned),
                completed,
                total,
                " [done]" if item.get("finished") else "",
            )
        )

    latest = max(progresses, key=lambda item: item.get("updated_at", ""))
    disk_total = int(latest.get("disk_total_bytes", 0))
    disk_used = int(latest.get("disk_used_bytes", 0))
    disk_free = int(latest.get("disk_free_bytes", 0))
    if disk_total:
        print(
            "  local disk: {} / {} used ({:.1f}%); {} free".format(
                human_bytes(disk_used),
                human_bytes(disk_total),
                ratio_percent(disk_used, disk_total),
                human_bytes(disk_free),
            )
        )
        print("  disk check: {}".format(latest.get("updated_at", "unknown")))


def main():
    session = Session(boto_session=boto3.Session(region_name=region))
    resolved_bucket = bucket or session.default_bucket()
    sagemaker = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    now = datetime.now(timezone.utc)

    print("AI-PAL yearly sample-cutting jobs")
    print("Checked: {}\n".format(now.strftime("%Y-%m-%d %H:%M:%S UTC")))
    states = {}
    for year in years:
        job_code = "ai-pal-cut-{}-{}".format(SAMPLE_RUN, year)
        latest = latest_processing_job(sagemaker, job_code)
        output_prefix = artifact_prefix + "/01_npy/{}".format(year)
        count, total_bytes, modified = summarize_s3_prefix(
            s3, resolved_bucket, output_prefix
        )
        print("{}:".format(year))
        print_job_status(sagemaker, latest, now)
        print(
            "  S3 stored:  {} objects, {}".format(
                count, human_bytes(total_bytes)
            )
        )

        progresses = []
        for stage in ("positive", "negative"):
            progress = read_json(
                s3,
                resolved_bucket,
                output_prefix + "/cut_{}_progress.json".format(stage),
            )
            if progress is not None:
                progresses.append(progress)
        print_cut_progress(progresses)
        if modified is not None:
            print("  latest S3:  {}".format(modified.isoformat()))

        manifest_key = output_prefix + "/cut_samples_manifest.json"
        try:
            body = s3.get_object(
                Bucket=resolved_bucket, Key=manifest_key
            )["Body"].read()
            manifest = json.loads(body)
        except s3.exceptions.NoSuchKey:
            manifest = None
        except Exception as exc:
            print("  manifest:   unreadable ({})".format(exc))
            manifest = None

        if manifest:
            recorded_run = manifest.get("sample_run")
            print("  sample run: {}".format(
                recorded_run or "MISSING"
            ))
            print("  phase file: {}".format(manifest.get("phase_file", "unknown")))
            print(
                "  rate CSV:   {} ({} dates, {} station-date rows)".format(
                    manifest.get("association_rate_file"),
                    manifest.get("association_rate_dates"),
                    manifest.get("association_rate_rows"),
                )
            )
            display_indexes = manifest.get("indexes", {})
            audit_result = None
            if audit_completed_indexes:
                audit_result = audit_annual_output(
                    s3, resolved_bucket, output_prefix, year, manifest, count
                )
                ready, audited_indexes, issues, notes = audit_result
                if audited_indexes:
                    display_indexes = audited_indexes
                for note in notes:
                    print("  audit note: {}".format(note))
                for issue in issues:
                    print("  AUDIT ERROR: {}".format(issue))
                states[year] = "ready" if ready else "invalid"
                print("  NPY state:  {}".format(
                    "READY FOR ZARR" if ready else "INVALID/INCOMPLETE"
                ))
            else:
                states[year] = "ready"
            for name, summary in sorted(display_indexes.items()):
                print(
                    "  {:14s} {:8d} samples in {:6d} shards".format(
                        name + ":",
                        int(summary.get("samples", 0)),
                        int(summary.get("shards", 0)),
                    )
                )
        else:
            print("  manifest:   not available; stage is incomplete")
            latest_status = (
                latest.get("ProcessingJobStatus") if latest is not None else None
            )
            if count == 0:
                states[year] = "missing"
            elif latest_status in ("InProgress", "Starting", "Stopping"):
                states[year] = "building"
            else:
                states[year] = "incomplete"
            print("  NPY state:  {}".format(states[year].upper()))
        print(
            "  output:     s3://{}/{}/\n".format(
                resolved_bucket, output_prefix
            )
        )

    ready_years = [year for year in years if states.get(year) == "ready"]
    building_years = [year for year in years if states.get(year) == "building"]
    missing_years = [year for year in years if states.get(year) == "missing"]
    invalid_years = [year for year in years if states.get(year) == "invalid"]
    incomplete_years = [
        year for year in years if states.get(year) == "incomplete"
    ]
    print("Annual NPY readiness")
    print("  sample run: {}".format(SAMPLE_RUN))
    print("  ready:      {}".format(
        ", ".join(map(str, ready_years)) or "none"
    ))
    print("  building:   {}".format(
        ", ".join(map(str, building_years)) or "none"
    ))
    print("  missing:    {}".format(
        ", ".join(map(str, missing_years)) or "none"
    ))
    print("  incomplete: {}".format(
        ", ".join(map(str, incomplete_years)) or "none"
    ))
    print("  invalid:    {}".format(
        ", ".join(map(str, invalid_years)) or "none"
    ))
    all_ready = len(ready_years) == len(years)
    print("  selected years Zarr ready: {}".format(
        "YES" if all_ready else "NO"
    ))


if __name__ == "__main__":
    main()
