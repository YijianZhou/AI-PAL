#!/usr/bin/env python3
"""Report independent GPU training jobs and checkpoint artifacts by model."""

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
training_years = (2020, 2021, 2022, 2023, 2024, 2025)
RUN_VERSION = "v1"
TRAINING_RUN = "%s-%d-%d-rarity-%s" % (
    CASE_CODE, training_years[0], training_years[-1], RUN_VERSION
)
enabled_models = ("SAR", "FT", "PHN", "RUN")
artifact_prefix = "sagemaker/ai-pal/training/" + TRAINING_RUN


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def main():
    session = Session(boto_session=boto3.Session(region_name=region))
    resolved_bucket = bucket or session.default_bucket()
    sagemaker = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    now = datetime.now(timezone.utc)
    print("AI-PAL GPU training jobs")
    print("Checked: {}\n".format(now.strftime("%Y-%m-%d %H:%M:%S UTC")))

    for model in enabled_models:
        job_code = "ai-pal-train-{}-{}".format(model.lower(), TRAINING_RUN)
        output_prefix = artifact_prefix + "/03_checkpoints/" + model
        print(model)
        latest = latest_processing_job(sagemaker, job_code)
        print_job_status(sagemaker, latest, now)
        count, total_bytes, modified = summarize_s3_prefix(
            s3, resolved_bucket, output_prefix
        )
        checkpoint_prefix = output_prefix.rstrip("/") + "/"
        checkpoint_count = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=resolved_bucket, Prefix=checkpoint_prefix):
            checkpoint_count += sum(
                item["Key"].endswith(".ckpt") for item in page.get("Contents", [])
            )
        print(
            "  artifacts:  {} objects, {} ({} checkpoints)".format(
                count, human_bytes(total_bytes), checkpoint_count
            )
        )
        if modified is not None:
            print("  latest S3:  {}".format(modified.isoformat()))
        storage_key = output_prefix + "/training_storage_status.json"
        try:
            response = s3.get_object(Bucket=resolved_bucket, Key=storage_key)
            storage = json.loads(response["Body"].read())
            storage_stale = bool(
                latest and response["LastModified"] < latest["CreationTime"]
            )
        except s3.exceptions.NoSuchKey:
            storage = None
            storage_stale = False
        if storage:
            if storage_stale:
                print("  stage:      waiting for current job (prior-job status is stale)")
            else:
                print("  stage:      {} (checked {})".format(
                    storage.get("stage", "unknown"),
                    storage.get("checked_utc", "unknown"),
                ))
                print("  job disk:   {:.2f} / {:.2f} GiB used; {:.2f} GiB free".format(
                    float(storage["filesystem_used_gib"]),
                    float(storage["filesystem_total_gib"]),
                    float(storage["filesystem_free_gib"]),
                ))
        manifest_key = output_prefix + "/training_manifest.json"
        try:
            response = s3.get_object(Bucket=resolved_bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read())
            if latest and response["LastModified"] < latest["CreationTime"]:
                manifest = None
        except s3.exceptions.NoSuchKey:
            manifest = None
        except Exception as exc:
            print("  manifest:   unreadable ({})".format(exc))
            manifest = None
        if manifest:
            print("  train time: {:.1f}s".format(float(manifest["training_sec"])))
        else:
            print("  manifest:   not available; job may still be running")
        failure_key = output_prefix + "/training_failure.txt"
        try:
            response = s3.get_object(Bucket=resolved_bucket, Key=failure_key)
            failure = response["Body"].read().decode("utf-8", errors="replace")
            if latest and response["LastModified"] < latest["CreationTime"]:
                failure = None
        except s3.exceptions.NoSuchKey:
            failure = None
        if failure:
            print("  container traceback:")
            for line in failure.rstrip().splitlines():
                print("    " + line)
        print("  output:     s3://{}/{}/\n".format(resolved_bucket, output_prefix))


if __name__ == "__main__":
    main()
