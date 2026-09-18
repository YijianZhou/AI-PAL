#!/usr/bin/env python3
"""Monitor the independent 2.1, 2.2, and 2.3 AWS inference stages."""

import json
import hashlib
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session

from job_common import human_bytes, latest_processing_job, print_job_status


# Keep these aligned with the three staged launchers.
CASE_CODE = "eg"
TIME_RANGE = "20190704-20190707"
INFERENCE_RUN = "eg-ai-pal-staged-20190704-20190707-v1"
region = "us-west-2"


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
    prefix = "sagemaker/ai-pal/inference/{}/output/".format(INFERENCE_RUN)
    keys = []
    total_bytes = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for row in page.get("Contents", []):
            keys.append(row["Key"])
            total_bytes += int(row.get("Size", 0))

    start_text, end_text = TIME_RANGE.split("-", 1)
    start = datetime.strptime(start_text, "%Y%m%d").date()
    end = datetime.strptime(end_text, "%Y%m%d").date()
    target_days = (end - start).days
    case_prefix = prefix + CASE_CODE + "/"
    pick_days = sum(
        key.startswith(case_prefix)
        and "/1.2_picks_AI-PAL-ENSEMBLE/" in key
        and key.endswith(".pick") for key in keys
    )
    assoc_done = sum(
        "/2.1.0_phase_init_AI-PAL/daily_assoc/assoc_status/" in key
        and key.endswith(".done.json") for key in keys
    )
    assoc_failed = sum(
        "/2.1.0_phase_init_AI-PAL/daily_assoc/assoc_status/" in key
        and key.endswith(".failed.json") for key in keys
    )
    post_done = sum(
        "/3.1_phase_final_AI-PAL/postprocess_status/" in key
        and key.endswith(".json") for key in keys
    )
    sac_files = sum(
        "/3.1_phase_final_AI-PAL/event_waveforms/" in key
        and key.endswith(".sac") for key in keys
    )

    print("AI-PAL staged AWS inference")
    print("Checked: {} UTC".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    print("Stored: {} objects, {}\n".format(len(keys), human_bytes(total_bytes)))
    stages = (
        ("2.1 picking", "2-1-pick", "2.1_pick_manifest.json",
         "2.1_pick_failure.json"),
        ("2.2 association", "2-2-assoc", "2.2_assoc_manifest.json",
         "2.2_assoc_failure.json"),
        ("2.3 postprocess", "2-3-postprocess",
         "2.3_postprocess_manifest.json", "2.3_postprocess_failure.json"),
    )
    for label, stage_code, manifest_name, failure_name in stages:
        print(label)
        run_digest = hashlib.sha1(
            INFERENCE_RUN.encode("utf-8")
        ).hexdigest()[:8]
        latest = latest_processing_job(
            sm, "ai-pal-{}-{}-{}".format(
                stage_code, CASE_CODE, run_digest
            )
        )
        print_job_status(sm, latest, now)
        manifest = read_json(s3, bucket, prefix + manifest_name)
        failure = read_json(s3, bucket, prefix + failure_name)
        print("  manifest:   {}".format(
            "complete" if manifest else "not available"
        ))
        if failure:
            print("  pipeline failure: {}".format(failure.get("error")))
        if stage_code == "2-1-pick":
            expected = target_days + 2
            print("  pick days:  {} / {} ({:.1f}%)".format(
                pick_days, expected, 100.0 * min(pick_days, expected) / expected
            ))
        elif stage_code == "2-2-assoc":
            print("  completed:  {} / {} days ({:.1f}%)".format(
                assoc_done, target_days,
                100.0 * min(assoc_done, target_days) / target_days,
            ))
            print("  failed:     {} days".format(assoc_failed))
        else:
            print("  completed:  {} / {} days ({:.1f}%)".format(
                post_done, target_days,
                100.0 * min(post_done, target_days) / target_days,
            ))
            print("  SAC files:  {}".format(sac_files))
        print()
    print("Output: s3://{}/{}".format(bucket, prefix))


if __name__ == "__main__":
    main()
