#!/usr/bin/env python3
"""Report SageMaker and direct-S3 progress for one AI-PAL inference run."""

import json
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session

from job_common import human_bytes, latest_processing_job, print_job_status


# Keep these values aligned with 1_run_ai_pal_pick_assoc_aws_<case>.py.
INFERENCE_RUN = "eg-ai-pal-20190704-20190707-v1"
TIME_RANGE = "20190704-20190707"
region = "us-west-2"


def parse_range(value):
    start_text, end_text = value.split("-", 1)
    return (
        datetime.strptime(start_text, "%Y%m%d").date(),
        datetime.strptime(end_text, "%Y%m%d").date(),
    )


def list_summary(s3, bucket, prefix):
    count = 0
    size = 0
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for row in page.get("Contents", []):
            count += 1
            size += int(row.get("Size", 0))
            keys.append(row["Key"])
    return count, size, keys


def read_json(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "404", "NoSuchKey", "NotFound"
        ):
            return None
        raise


def main():
    session = Session(boto_session=boto3.Session(region_name=region))
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)
    now = datetime.now(timezone.utc)
    job_code = "ai-pal-run-" + INFERENCE_RUN
    output_prefix = "sagemaker/ai-pal/inference/{}/output".format(
        INFERENCE_RUN
    )
    start, end = parse_range(TIME_RANGE)
    target_days = (end - start).days
    expected_pick_days = target_days + 2
    expected_hours = target_days * 24

    print("AI-PAL AWS inference")
    print("Checked: {} UTC\n".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    latest = latest_processing_job(sm, job_code)
    print_job_status(sm, latest, now)
    count, size, keys = list_summary(s3, bucket, output_prefix + "/")
    pick_done = sum("/_status/pick_" in key for key in keys)
    assoc_done = sum("/_status/assoc_" in key for key in keys)
    ensemble_picks = sum(
        "/1.2_picks_AI-PAL-ENSEMBLE/" in key and key.endswith(".pick")
        for key in keys
    )
    final_phases = sum(
        "/3.1_phase_final_AI-PAL/phase_" in key and key.endswith(".dat")
        for key in keys
    )
    print("  stored:     {} objects, {}".format(count, human_bytes(size)))
    print("  pick days:  {} / {} ({:.1f}%)".format(
        pick_done, expected_pick_days,
        100.0 * min(pick_done, expected_pick_days) / expected_pick_days,
    ))
    print("  pick files: {}".format(ensemble_picks))
    print("  assoc hours:{} / {} ({:.1f}%)".format(
        assoc_done, expected_hours,
        100.0 * min(assoc_done, expected_hours) / expected_hours,
    ))
    print("  phase files:{}".format(final_phases))
    manifest = read_json(s3, bucket, output_prefix + "/ai_pal_manifest.json")
    failure = read_json(s3, bucket, output_prefix + "/ai_pal_failure.json")
    if manifest:
        print("  manifest:   complete ({:.2f} h)".format(
            float(manifest.get("elapsed_sec", 0)) / 3600.0
        ))
    else:
        print("  manifest:   not available")
    if failure:
        print("  pipeline failure: {}".format(failure.get("error")))
        trace = failure.get("traceback", "").strip().splitlines()
        for line in trace[-8:]:
            print("    " + line)
    print("  output:     s3://{}/{}/".format(bucket, output_prefix))


if __name__ == "__main__":
    main()
