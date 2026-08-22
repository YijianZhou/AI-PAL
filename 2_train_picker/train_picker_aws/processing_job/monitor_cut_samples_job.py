#!/usr/bin/env python3
"""Report annual SCEDC sample-cutting jobs and NPY progress."""

import json
from datetime import datetime, timezone

import boto3
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
SAMPLE_RUN = "%s-2020-2025-v1" % CASE_CODE
years = (2020, 2021, 2022, 2023, 2024, 2025)
artifact_prefix = "sagemaker/ai-pal/training/" + SAMPLE_RUN


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


def print_cut_progress(progresses):
    if not progresses:
        print("  progress:   waiting for first sync (or unavailable for a legacy job)")
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
    for year in years:
        job_code = "ai-pal-cut-{}-{}".format(SAMPLE_RUN, year)
        latest = latest_processing_job(sagemaker, job_code)
        output_prefix = artifact_prefix + "/01_npy/{}".format(year)
        count, total_bytes, modified = summarize_s3_prefix(
            s3, resolved_bucket, output_prefix
        )
        if latest is None and count == 0:
            continue

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
            print("  sample run: {}".format(
                manifest.get("sample_run", "legacy/unrecorded")
            ))
            print(
                "  rate CSV:   {} ({} dates, {} station-date rows)".format(
                    manifest.get("association_rate_file"),
                    manifest.get("association_rate_dates"),
                    manifest.get("association_rate_rows"),
                )
            )
            for name, summary in sorted(
                manifest.get("indexes", {}).items()
            ):
                print(
                    "  {:14s} {:8d} samples in {:6d} shards".format(
                        name + ":",
                        int(summary.get("samples", 0)),
                        int(summary.get("shards", 0)),
                    )
                )
        else:
            if total_bytes and not progresses:
                legacy_sample_bytes = 3 * (2500 + 2) * 4
                approximate_samples = total_bytes // legacy_sample_bytes
                print(
                    "  legacy est: ~{:,d} samples currently in S3; "
                    "final size unavailable".format(approximate_samples)
                )
            print("  manifest:   not available; stage is incomplete")
        print(
            "  output:     s3://{}/{}/\n".format(
                resolved_bucket, output_prefix
            )
        )


if __name__ == "__main__":
    main()
