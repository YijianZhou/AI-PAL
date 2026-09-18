"""Explicit realtime reference workflows and isolated GaMMA execution."""
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def require_gamma_runtime():
    instruction = ('GaMMA reference requires: python -m pip install '
                   '"GMMA==1.2.12" "scikit-learn==1.6.1"')
    try:
        gamma_version = version("GMMA")
        sklearn_version = tuple(int(part) for part in version("scikit-learn").split(".")[:2])
    except PackageNotFoundError as exc:
        raise RuntimeError(instruction) from exc
    if gamma_version != "1.2.12" or sklearn_version >= (1, 7):
        raise RuntimeError(instruction)


def reference_workflows(cfg):
    configured = getattr(cfg, "reference_workflows", [])
    workflows = {}
    for item in configured:
        picker = item["picker"]
        method = item["associator"]
        if not isinstance(picker, str) or not picker.strip():
            raise ValueError("reference workflow requires a picker name")
        if method not in ("PAL", "GaMMA"):
            raise ValueError("unsupported reference associator: {}".format(method))
        key = picker if method == "PAL" else "{}_{}".format(picker, method)
        if key in workflows:
            raise ValueError("duplicate reference workflow: {}".format(key))
        workflows[key] = dict(item)
    return workflows


def load_reference_configs(cfg, base_dir, associators):
    settings = {}
    for key, item in reference_workflows(cfg).items():
        if item["associator"] == "PAL":
            continue
        method = item["associator"]
        specification = associators.get(method, {})
        if not specification.get("config"):
            raise ValueError("missing reference associator config: {}".format(method))
        path = Path(specification["config"])
        if not path.is_absolute():
            path = Path(base_dir) / path
        spec = importlib.util.spec_from_file_location("reference_" + key, path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load reference config: {}".format(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        settings[key] = vars(module.Config())
    return settings


def run_gamma(picks, stations, segment, branch):
    """Pass small pick/metadata tables to a fresh process, never GPU state."""
    root = Path(branch["subnet_phase_dir"])
    root.mkdir(parents=True, exist_ok=True)
    phase_path = root / ("phase_{}_{}_full.dat".format(segment, branch["result_name"]))
    payload = {
        "picks": [{name: (str(row[name]) if name in ("net_sta", "tp", "ts", "sources")
                          else float(row[name]))
                   for name in ("net_sta", "tp", "ts", "p_prob", "s_prob", "sources",
                                "tp_std", "ts_std", "p_prob_std", "s_prob_std", "num_support")}
                  for row in picks],
        "stations": {name: list(info[:3]) for name, info in stations.items()},
        "config": branch["associator_config"],
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gamma_", dir=root) as scratch:
        input_path = Path(scratch) / "input.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        log_path = root / ("gamma_{}.log".format(segment))
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("associator_gamma.py")),
                 str(input_path), str(phase_path)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
        if result.returncode:
            raise RuntimeError("GaMMA exited {}. See {}".format(result.returncode, log_path))
    return {"full": {"pha_path": str(phase_path), "num_picks": len(picks),
                     "assoc_sec": time.perf_counter() - started}}
