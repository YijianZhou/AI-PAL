#!/usr/bin/env python3
"""Report annual NPY-to-Zarr jobs and reusable archive state."""

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
SAMPLE_RUN = "%s-2020-2025-rarity-v1" % CASE_CODE
archive_years = (2020, 2021, 2022, 2023, 2024, 2025)
ZARR_ARCHIVE = "%s-rarity-v1" % CASE_CODE
artifact_prefix = "sagemaker/ai-pal/zarr-archives/" + ZARR_ARCHIVE
job_code = "ai-pal-zarr-" + ZARR_ARCHIVE


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def main():
    session = Session(boto_session=boto3.Session(region_name=region))
    resolved_bucket = bucket or session.default_bucket()
    sagemaker = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    now = datetime.now(timezone.utc)
    output_prefix = artifact_prefix

    print("AI-PAL NPY-to-Zarr job")
    print("Checked: {}\n".format(now.strftime("%Y-%m-%d %H:%M:%S UTC")))
    latest = latest_processing_job(sagemaker, job_code)
    print_job_status(sagemaker, latest, now)

    archive_count = 0
    archive_bytes = 0
    print("  annual stores:")
    for year in archive_years:
        year_prefix = "{}/{}.zarr".format(output_prefix, year)
        count, total_bytes, modified = summarize_s3_prefix(
            s3, resolved_bucket, year_prefix
        )
        archive_count += count
        archive_bytes += total_bytes
        latest_text = modified.isoformat() if modified is not None else "none"
        print("    {}: {:8d} objects, {:>10s}, latest {}".format(
            year, count, human_bytes(total_bytes), latest_text
        ))
    print("  archive total: {:,} objects, {}".format(
        archive_count, human_bytes(archive_bytes)
    ))

    manifest_key = output_prefix + "/archive_manifest.json"
    try:
        body = s3.get_object(Bucket=resolved_bucket, Key=manifest_key)["Body"].read()
        manifest = json.loads(body)
    except s3.exceptions.NoSuchKey:
        manifest = None
    except Exception as exc:
        print("  manifest: unreadable ({})".format(exc))
        manifest = None
    if manifest:
        recorded_years = manifest.get("years", {})
        print("  archive name: {}".format(manifest.get("archive_name")))
        print("  published years: {}".format(
            ", ".join(sorted(recorded_years)) or "none"
        ))
        for year in sorted(recorded_years):
            row = recorded_years[year]
            print("    {} sample run: {}".format(year, row.get("sample_run")))
    else:
        print("  manifest: not available; stage is incomplete or failed")
    failure_key = output_prefix + "/npy2zarr_failure.txt"
    try:
        failure = s3.get_object(
            Bucket=resolved_bucket, Key=failure_key
        )["Body"].read().decode("utf-8", errors="replace")
    except s3.exceptions.NoSuchKey:
        failure = None
    if failure:
        print("  container traceback:")
        for line in failure.rstrip().splitlines():
            print("    " + line)
    log_response = s3.list_objects_v2(
        Bucket=resolved_bucket, Prefix=output_prefix + "/logs/"
    )
    logs = sorted(
        log_response.get("Contents", []),
        key=lambda row: row.get("LastModified"),
    )
    if logs:
        latest_log = logs[-1]["Key"]
        log_text = s3.get_object(
            Bucket=resolved_bucket, Key=latest_log
        )["Body"].read().decode("utf-8", errors="replace")
        print("  latest converter log: {}".format(latest_log.rsplit("/", 1)[-1]))
        for line in log_text.rstrip().splitlines()[-40:]:
            print("    " + line)
    print("  output: s3://{}/{}/".format(resolved_bucket, output_prefix))


if __name__ == "__main__":
    main()
