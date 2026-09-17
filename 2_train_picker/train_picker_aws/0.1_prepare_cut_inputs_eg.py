#!/usr/bin/env python3
"""Prepare one phase file and one association-rate CSV per training year."""

import csv
import io
import shutil
from datetime import date, timedelta
from pathlib import Path

import boto3
from sagemaker.core.helper.session_helper import Session


# ============================================================================
# USER SETTINGS: YEARS, PAL RESULTS, S3, AND LOCAL OUTPUT
# ============================================================================
CASE_CODE = "eg"
years = (2020, 2021, 2022)

region = "us-west-2"
bucket = None  # None uses the active SageMaker default bucket.
results_prefix = "sagemaker/scsn-pal/results"
run_pal_dir = Path.home() / "shared" / "run_pal"
workflow_dir = Path(__file__).resolve().parent
station_file = "station_scedc_aws_selected_20200101_20260701_pal.csv"
overwrite = False

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
ASSOCIATION_RATE_FIELDS = [
    "date",
    "net_sta",
    "num_picks",
    "num_associated_picks",
    "num_unassociated_picks",
    "association_ratio",
]


def dates_in_year(year):
    current = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(days=1)


def list_keys(s3, resolved_bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=resolved_bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return sorted(keys)


def require_daily_keys(keys, expected_names, label):
    by_name = {key.rsplit("/", 1)[-1]: key for key in keys}
    missing = sorted(set(expected_names) - set(by_name))
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += ", ..."
        raise FileNotFoundError(
            "{} is missing {} daily file(s): {}".format(
                label, len(missing), preview
            )
        )
    return [by_name[name] for name in expected_names]


def concatenate_phase_files(s3, resolved_bucket, keys, output_path):
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as output:
        for key in keys:
            body = s3.get_object(Bucket=resolved_bucket, Key=key)["Body"]
            payload = body.read()
            output.write(payload)
            if payload and not payload.endswith(b"\n"):
                output.write(b"\n")
    partial.replace(output_path)


def download_object(s3, resolved_bucket, key, output_path):
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    s3.download_file(resolved_bucket, key, str(partial))
    partial.replace(output_path)


def write_association_rates(payloads, output_path):
    """Write daily association-rate payloads as one validated annual CSV."""
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    num_rows = 0
    seen_station_dates = set()
    with partial.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ASSOCIATION_RATE_FIELDS)
        writer.writeheader()
        for source_name, payload in payloads:
            reader = csv.DictReader(io.StringIO(payload))
            missing = set(ASSOCIATION_RATE_FIELDS) - set(
                reader.fieldnames or []
            )
            if missing:
                raise ValueError(
                    "{} missing association-rate columns: {}".format(
                        source_name, ", ".join(sorted(missing))
                    )
                )
            for row in reader:
                if row["date"].strip().lower() == "date":
                    continue
                station_date = (
                    row["date"].strip(), row["net_sta"].strip()
                )
                if station_date in seen_station_dates:
                    raise ValueError(
                        "duplicate station-date {} {} while reading {}".format(
                            station_date[0], station_date[1], source_name
                        )
                    )
                seen_station_dates.add(station_date)
                writer.writerow({
                    name: row[name] for name in ASSOCIATION_RATE_FIELDS
                })
                num_rows += 1
    partial.replace(output_path)
    return num_rows


def concatenate_s3_association_rates(
    s3, resolved_bucket, keys, output_path
):
    def payloads():
        for key in keys:
            payload = s3.get_object(
                Bucket=resolved_bucket, Key=key
            )["Body"].read().decode("utf-8-sig")
            yield key, payload

    return write_association_rates(payloads(), output_path)


def count_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as fp:
        return sum(1 for _ in csv.DictReader(fp))


def main():
    print("Preparing annual phase and association-rate inputs.")
    s3 = None
    resolved_bucket = None

    def get_s3():
        nonlocal s3, resolved_bucket
        if s3 is None:
            session = Session(boto_session=boto3.Session(region_name=region))
            resolved_bucket = bucket or session.default_bucket()
            s3 = boto3.client("s3", region_name=region)
        return s3, resolved_bucket

    source_station = run_pal_dir / "input" / station_file
    target_station = workflow_dir / "input" / station_file
    if not source_station.exists() and not target_station.exists():
        raise FileNotFoundError(source_station)
    target_station.parent.mkdir(parents=True, exist_ok=True)
    if source_station.exists() and (overwrite or not target_station.exists()):
        shutil.copy2(source_station, target_station)

    for year in years:
        expected_dates = [value.isoformat() for value in dates_in_year(year)]
        phase_names = [
            "phase_{}.dat".format(value) for value in expected_dates
        ]
        rate_names = [
            "association_rate_{}.csv".format(value)
            for value in expected_dates
        ]

        year_dir = workflow_dir / "input" / str(year)
        phase_file = year_dir / (
            "%s_assoc_%d_pal.pha" % (CASE_CODE, year)
        )
        rate_file = year_dir / (
            "%s_assoc_%d_association_rates.csv" % (CASE_CODE, year)
        )
        year_dir.mkdir(parents=True, exist_ok=True)

        if overwrite or not phase_file.exists():
            s3, resolved_bucket = get_s3()
            result_root = (
                results_prefix
                + "/%s-assoc-%d/output/%s" % (CASE_CODE, year, CASE_CODE)
            )
            range_code = "{}0101-{}0101".format(year, year + 1)
            direct_phase_key = result_root + "/phase_{}.dat".format(range_code)
            if direct_phase_key in list_keys(
                s3, resolved_bucket, direct_phase_key
            ):
                download_object(
                    s3, resolved_bucket, direct_phase_key, phase_file
                )
            else:
                legacy_root = (
                    results_prefix
                    + "/%s-assoc-%d/output/assoc" % (CASE_CODE, year)
                )
                phase_prefixes = (
                    result_root + "/association/merged/phase_{}-".format(year),
                    legacy_root + "/merged/phase_{}-".format(year),
                )
                phase_keys = []
                for phase_prefix in phase_prefixes:
                    phase_keys = list_keys(s3, resolved_bucket, phase_prefix)
                    if phase_keys:
                        break
                phase_keys = require_daily_keys(
                    phase_keys, phase_names, "{} merged phases".format(year)
                )
                concatenate_phase_files(
                    s3, resolved_bucket, phase_keys, phase_file
                )

        if overwrite or not rate_file.exists():
            s3, resolved_bucket = get_s3()
            result_root = (
                results_prefix
                + "/%s-assoc-%d/output/%s" % (CASE_CODE, year, CASE_CODE)
            )
            legacy_root = (
                results_prefix
                + "/%s-assoc-%d/output/assoc" % (CASE_CODE, year)
            )
            rate_prefixes = (
                result_root
                + "/association/association_rates/association_rate_{}-".format(year),
                legacy_root
                + "/association_rates/association_rate_{}-".format(year),
            )
            rate_keys = []
            for rate_prefix in rate_prefixes:
                rate_keys = list_keys(s3, resolved_bucket, rate_prefix)
                if rate_keys:
                    break
            rate_keys = require_daily_keys(
                rate_keys, rate_names, "{} association rates".format(year)
            )
            num_rate_rows = concatenate_s3_association_rates(
                s3, resolved_bucket, rate_keys, rate_file
            )
            rate_source = "S3 daily CSVs"
        else:
            num_rate_rows = count_csv_rows(rate_file)
            rate_source = "existing annual CSV"

        print(
            "{} ready: {}, {} ({} station-date rows from {})".format(
                year,
                phase_file,
                rate_file,
                num_rate_rows,
                rate_source,
            )
        )


if __name__ == "__main__":
    main()
