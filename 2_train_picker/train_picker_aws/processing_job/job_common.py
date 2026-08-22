"""Shared helpers for AI-PAL SageMaker training Processing Jobs."""

from pathlib import Path


def upload_tree(s3_client, local_root, bucket, key_prefix):
    local_root = Path(local_root)
    for path in sorted(local_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(local_root).as_posix()
            key = key_prefix.rstrip("/") + "/" + relative
            s3_client.upload_file(str(path), bucket, key)


def delete_s3_prefix(s3_client, bucket, prefix):
    """Delete every object under an explicitly selected workflow prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    batch = []
    deleted = 0
    for page in paginator.paginate(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/"
    ):
        for item in page.get("Contents", []):
            batch.append({"Key": item["Key"]})
            if len(batch) == 1000:
                s3_client.delete_objects(
                    Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
                )
                deleted += len(batch)
                batch = []
    if batch:
        s3_client.delete_objects(
            Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
        )
        deleted += len(batch)
    return deleted


def prefix_has_objects(s3_client, bucket, prefix):
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1
    )
    return bool(response.get("KeyCount", 0))


def latest_processing_job(sagemaker_client, job_code):
    response = sagemaker_client.list_processing_jobs(
        NameContains=job_code,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    )
    matches = [
        row for row in response.get("ProcessingJobSummaries", [])
        if row["ProcessingJobName"].startswith(job_code + "-")
    ]
    return matches[0] if matches else None


def summarize_s3_prefix(s3_client, bucket, prefix):
    count = 0
    total_bytes = 0
    latest = None
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            count += 1
            total_bytes += int(item.get("Size", 0))
            modified = item.get("LastModified")
            if modified is not None and (latest is None or modified > latest):
                latest = modified
    return count, total_bytes, latest


def human_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0


def print_job_status(sagemaker_client, latest, now):
    if latest is None:
        print("  latest job: not submitted")
        return
    name = latest["ProcessingJobName"]
    details = sagemaker_client.describe_processing_job(ProcessingJobName=name)
    status = details["ProcessingJobStatus"]
    started = details.get("ProcessingStartTime")
    ended = details.get("ProcessingEndTime")
    print("  latest job: {}".format(name))
    print("  status:     {}".format(status))
    if started is not None:
        elapsed = (ended or now) - started
        seconds = max(0, int(elapsed.total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        print("  runtime:    {}h {:02d}m".format(hours, remainder // 60))
    if details.get("FailureReason"):
        print("  failure:    {}".format(details["FailureReason"]))
    if details.get("ExitMessage"):
        print("  exit:       {}".format(details["ExitMessage"]))

