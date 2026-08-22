#!/usr/bin/env python3
"""Report shared NPY-to-Zarr job state, target builders, and artifact size."""

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
artifact_prefix = "sagemaker/ai-pal/training/" + TRAINING_RUN
job_code = "ai-pal-zarr-" + TRAINING_RUN


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def main():
    session = Session(boto_session=boto3.Session(region_name=region))
    resolved_bucket = bucket or session.default_bucket()
    sagemaker = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    now = datetime.now(timezone.utc)
    output_prefix = artifact_prefix + "/02_zarr"

    print("AI-PAL NPY-to-Zarr job")
    print("Checked: {}\n".format(now.strftime("%Y-%m-%d %H:%M:%S UTC")))
    latest = latest_processing_job(sagemaker, job_code)
    print_job_status(sagemaker, latest, now)

    shared_prefix = output_prefix + "/shared.zarr"
    count, total_bytes, modified = summarize_s3_prefix(
        s3, resolved_bucket, shared_prefix
    )
    latest_text = modified.isoformat() if modified is not None else "none"
    print(
        "  shared.zarr: {:8d} objects, {:>10s}, latest {}".format(
            count, human_bytes(total_bytes), latest_text
        )
    )

    manifest_key = output_prefix + "/npy2zarr_manifest.json"
    try:
        body = s3.get_object(Bucket=resolved_bucket, Key=manifest_key)["Body"].read()
        manifest = json.loads(body)
    except s3.exceptions.NoSuchKey:
        manifest = None
    except Exception as exc:
        print("  manifest: unreadable ({})".format(exc))
        manifest = None
    if manifest:
        print("  sample run:     {}".format(
            manifest.get("sample_run", manifest.get("cut_run", "unknown"))
        ))
        print("  annual inputs:  {}".format(
            ", ".join(manifest.get("annual_inputs", [])) or "none"
        ))
        print("  training mode:  {}".format(
            manifest.get("training_mode", "unknown")
        ))
        print("  target builders: {}".format(manifest.get("target_builders", {})))
        print("  validated arrays:")
        for name, shape in sorted(manifest.get("array_shapes", {}).items()):
            print("    {}: {}".format(name, shape))
        for target_type, seconds in sorted(manifest.get("timing_sec", {}).items()):
            print("  {} target conversion: {:.1f}s".format(
                target_type, float(seconds)
            ))
    else:
        print("  manifest: not available; stage is incomplete or failed")
    print("  output: s3://{}/{}/".format(resolved_bucket, output_prefix))


if __name__ == "__main__":
    main()
