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
TRAINING_RUN = "%s-2020-2025-v1" % CASE_CODE
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
        manifest_key = output_prefix + "/training_manifest.json"
        try:
            body = s3.get_object(Bucket=resolved_bucket, Key=manifest_key)["Body"].read()
            manifest = json.loads(body)
        except s3.exceptions.NoSuchKey:
            manifest = None
        except Exception as exc:
            print("  manifest:   unreadable ({})".format(exc))
            manifest = None
        if manifest:
            print("  train time: {:.1f}s".format(float(manifest["training_sec"])))
        else:
            print("  manifest:   not available; job may still be running")
        print("  output:     s3://{}/{}/\n".format(resolved_bucket, output_prefix))


if __name__ == "__main__":
    main()
