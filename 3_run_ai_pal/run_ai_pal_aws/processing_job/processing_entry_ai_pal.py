#!/usr/bin/env python3
"""Container entry point for SCEDC AI-PAL picking and association."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import boto3


SOURCE_ROOT = Path("/opt/ml/processing/source")
PAL_SRC = SOURCE_ROOT / "PAL_src"
WORKFLOW = SOURCE_ROOT / "workflow"
CHECKPOINT_ROOT = Path("/opt/ml/processing/checkpoints")
RESUME_ROOT = Path("/opt/ml/processing/resume")
OUTPUT_ROOT = Path("/opt/ml/processing/output")


def split_s3_uri(uri):
    bucket, _, prefix = uri[5:].partition("/")
    if not uri.startswith("s3://") or not bucket:
        raise ValueError("invalid S3 URI: {}".format(uri))
    return bucket, prefix.rstrip("/")


class IncrementalS3Writer(object):
    """Upload only files whose size or modification time changed."""

    def __init__(self, local_root, output_uri):
        self.local_root = Path(local_root)
        self.bucket, self.prefix = split_s3_uri(output_uri)
        self.s3 = boto3.client(
            "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )
        self.signatures = {}
        for path in self.local_root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                self.signatures[path.relative_to(self.local_root).as_posix()] = (
                    stat.st_size, stat.st_mtime_ns
                )

    def upload_file(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        relative = path.relative_to(self.local_root).as_posix()
        if self.signatures.get(relative) == signature:
            return False
        key = self.prefix + "/" + relative if self.prefix else relative
        self.s3.upload_file(str(path), self.bucket, key)
        self.signatures[relative] = signature
        return True

    def sync(self):
        uploaded = 0
        for path in sorted(self.local_root.rglob("*")):
            uploaded += int(self.upload_file(path))
        print("S3 sync: {} changed file(s)".format(uploaded), flush=True)
        return uploaded

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


def checkpoint_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("checkpoint file not found: {}".format(path))
    return path


def picker_registries(gpu_map):
    pos_neg = {}
    positive = {}
    positive_names = {
        "SAR": "ceed_pos_sar_best.ckpt",
        "FT": "ceed_pos_ft_best.ckpt",
        "PHN": "ceed_pos_phn_best.ckpt",
        "RUN": "ceed_pos_run_best.ckpt",
    }
    for model in ("SAR", "FT", "PHN", "RUN"):
        lower = model.lower()
        pos_neg[model] = {
            "config": WORKFLOW / "config_{}_case.py".format(lower),
            "gpu_idx": int(gpu_map.get(model, 0)),
            "ckpt": checkpoint_file(CHECKPOINT_ROOT / model / "best.ckpt"),
        }
        positive[model] = {
            "config": WORKFLOW / "config_{}_pos_ceed.py".format(lower),
            "gpu_idx": int(gpu_map.get(model, 0)),
            "ckpt": WORKFLOW / "input" / "CEED_ckpt" / positive_names[model],
        }
    return pos_neg, positive


def selected_specs(cfg, pos_neg, positive):
    continuous = {}
    for model in dict.fromkeys(cfg.picker_pos_neg_group):
        continuous[model] = dict(
            pos_neg[model], group="POS_NEG", model=model
        )
    repick_pos_neg = {
        model: pos_neg[model]
        for model in dict.fromkeys(cfg.repicker_pos_neg_group)
    }
    repick_positive = {
        model: positive[model]
        for model in dict.fromkeys(cfg.repicker_pos_group)
    }
    return continuous, repick_pos_neg, repick_positive


def main():
    requirements = SOURCE_ROOT / "requirements.txt"
    if requirements.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )
    for path in (SOURCE_ROOT, PAL_SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    case_code = os.environ["CASE_CODE"]
    time_range = os.environ["TIME_RANGE"]
    full_station_file = WORKFLOW / "input" / os.environ["FULL_STATION_FILE"]
    subnet_names = json.loads(os.environ.get("SUBNET_STATION_FILES", "{}"))
    subnet_files = {
        name: WORKFLOW / "input" / filename
        for name, filename in subnet_names.items()
    }
    if not subnet_files:
        subnet_files = {"full": full_station_file}
    for path in (full_station_file, *subnet_files.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if RESUME_ROOT.exists():
        shutil.copytree(RESUME_ROOT, OUTPUT_ROOT, dirs_exist_ok=True)
    writer = IncrementalS3Writer(OUTPUT_ROOT, os.environ["OUTPUT_S3_URI"])
    writer.delete("ai_pal_failure.json")
    writer.delete("ai_pal_manifest.json")
    started = time.time()

    import config_ai_pal as config_module
    from offline_pick_assoc_runner import run_offline_pick_assoc

    cfg = config_module.Config()
    cfg.data_buffer_sec = float(os.environ.get(
        "DATA_BUFFER_SEC", cfg.data_buffer_sec
    ))
    gpu_map = json.loads(os.environ.get("MODEL_GPU_MAP", "{}"))
    pos_neg, positive = picker_registries(gpu_map)
    picker_specs, repick_pos_neg, repick_positive = selected_specs(
        cfg, pos_neg, positive
    )

    case_root = OUTPUT_ROOT / case_code
    ensemble_dir = case_root / "1.2_picks_AI-PAL-ENSEMBLE"
    phase_root = case_root / "2.1.0_phase_init_AI-PAL"
    final_root = case_root / "3.1_phase_final_AI-PAL"
    output_catalog = final_root / "catalog_{}.dat".format(time_range)
    output_phase = final_root / "phase_{}.dat".format(time_range)

    def day_complete(date, pick_path, summary):
        writer.sync()
        writer.write_json(
            "_status/pick_{}.json".format(date.date),
            {"date": str(date.date), "pick_file": str(pick_path), **summary},
        )

    def hour_complete(
        interval_start, interval_end, phase_path, catalog_path,
        _waveform_context, summary,
    ):
        stem = "{}_{}".format(
            interval_start.strftime("%Y%m%dT%H%M%SZ"),
            interval_end.strftime("%Y%m%dT%H%M%SZ"),
        )
        for path in (
            Path(phase_path),
            Path(catalog_path),
            Path(phase_path).with_name("event_groups_" + stem + ".csv"),
        ):
            writer.upload_file(path)
        for subnet in subnet_files:
            raw_root = phase_root / "hourly_assoc" / "subnets" / subnet
            writer.upload_file(raw_root / ("phase_" + stem + ".dat"))
            writer.upload_file(raw_root / ("catalog_" + stem + ".dat"))
        writer.write_json(
            "_status/assoc_{}.json".format(
                interval_start.strftime("%Y%m%dT%H%M%S")
            ),
            {
                "interval_start": str(interval_start),
                "interval_end": str(interval_end),
                "phase_file": str(phase_path),
                "catalog_file": str(catalog_path),
                "association": summary,
            },
        )

    writer.write_json("_status/job.json", {
        "status": "running", "case_code": case_code,
        "time_range": time_range, "started_epoch": started,
    })
    try:
        run_offline_pick_assoc(
            ai_pal_root=SOURCE_ROOT,
            cfg=cfg,
            picker_specs=picker_specs,
            data_dir=full_station_file,
            full_station_file=full_station_file,
            subnet_station_files=subnet_files,
            target_time_range=time_range,
            individual_pick_root=case_root,
            ensemble_pick_dir=ensemble_dir,
            assoc_root=phase_root / "hourly_assoc",
            final_root=final_root,
            output_catalog=output_catalog,
            output_phase=output_phase,
            num_pick_workers=int(os.environ.get("NUM_PICK_WORKERS", "8")),
            num_assoc_workers=int(os.environ.get("NUM_ASSOC_WORKERS", "8")),
            day_complete_callback=day_complete,
            hour_complete_callback=hour_complete,
            repicker_pos_neg_specs=repick_pos_neg,
            repicker_pos_specs=repick_positive,
            overwrite_picks=os.environ.get("OVERWRITE_PICKS", "0") == "1",
        )
        writer.sync()
        writer.write_json("ai_pal_manifest.json", {
            "status": "complete", "case_code": case_code,
            "time_range": time_range, "elapsed_sec": time.time() - started,
            "output_phase": str(output_phase.relative_to(OUTPUT_ROOT)),
            "output_catalog": str(output_catalog.relative_to(OUTPUT_ROOT)),
        })
    except Exception as exc:
        writer.write_json("ai_pal_failure.json", {
            "status": "failed", "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        writer.sync()
        raise


if __name__ == "__main__":
    main()
