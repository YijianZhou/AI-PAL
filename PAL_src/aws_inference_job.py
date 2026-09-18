"""Small direct-S3 runtime helpers for resumable AWS inference stages."""

import json
import os
from pathlib import Path
import shutil
import multiprocessing

import boto3


def split_s3_uri(uri):
    bucket, _, prefix = uri[5:].partition("/")
    if not uri.startswith("s3://") or not bucket:
        raise ValueError("invalid S3 URI: {}".format(uri))
    return bucket, prefix.rstrip("/")


class IncrementalS3Writer(object):
    """Upload complete files whose size or modification time changed."""

    def __init__(self, local_root, output_uri):
        self.local_root = Path(local_root)
        self.output_uri = output_uri
        self.bucket, self.prefix = split_s3_uri(output_uri)
        self.s3 = boto3.client(
            "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )
        self.signatures = {}
        self.excluded_sync_roots = set()
        for path in self.local_root.rglob("*"):
            if path.is_file():
                self.signatures[self._relative(path)] = self._signature(path)

    def _relative(self, path):
        return Path(path).relative_to(self.local_root).as_posix()

    @staticmethod
    def _signature(path):
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns

    def upload_file(self, path):
        path = Path(path)
        if not path.is_file() or path.name.endswith(".partial"):
            return False
        relative = self._relative(path)
        signature = self._signature(path)
        if self.signatures.get(relative) == signature:
            return False
        key = self.prefix + "/" + relative if self.prefix else relative
        self.s3.upload_file(str(path), self.bucket, key)
        self.signatures[relative] = signature
        return True

    def sync(self):
        uploaded = 0
        for root, directories, filenames in os.walk(self.local_root):
            root = Path(root)
            kept = []
            for name in directories:
                child = root / name
                relative = self._relative(child)
                if relative not in self.excluded_sync_roots:
                    kept.append(name)
            directories[:] = kept
            for name in sorted(filenames):
                uploaded += int(self.upload_file(root / name))
        print("S3 sync: {} changed file(s)".format(uploaded), flush=True)
        return uploaded

    def exclude_from_sync(self, path):
        """Exclude a directory uploaded through a more specific callback."""
        self.excluded_sync_roots.add(self._relative(path))

    def write_json(self, relative, payload):
        path = self.local_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self.upload_file(path)

    def delete(self, relative):
        path = self.local_root / relative
        if path.exists():
            path.unlink()
        key = self.prefix + "/" + relative if self.prefix else relative
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        self.signatures.pop(relative, None)


def _periodic_sync_process(local_root, output_uri, interval_seconds, stop_event):
    writer = IncrementalS3Writer(local_root, output_uri)
    while not stop_event.wait(interval_seconds):
        writer.sync()


class PeriodicS3Sync(object):
    """Upload completed artifacts periodically while a stage is running."""

    def __init__(self, writer, interval_seconds=300):
        self.writer = writer
        self.interval_seconds = max(30, int(interval_seconds))
        self.context = multiprocessing.get_context("spawn")
        self.stop_event = self.context.Event()
        self.process = None

    def __enter__(self):
        self.process = self.context.Process(
            target=_periodic_sync_process,
            args=(
                str(self.writer.local_root), self.writer.output_uri,
                self.interval_seconds, self.stop_event,
            ),
            name="s3-output-sync",
            daemon=True,
        )
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.process is not None:
            self.process.join()
        self.writer.sync()


def prepare_output(output_root, resume_root, output_uri, stale_files=()):
    output_root = Path(output_root)
    resume_root = Path(resume_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if resume_root.exists():
        shutil.copytree(resume_root, output_root, dirs_exist_ok=True)
    writer = IncrementalS3Writer(output_root, output_uri)
    for relative in stale_files:
        writer.delete(relative)
    return writer


def checkpoint_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("checkpoint file not found: {}".format(path))
    return path
