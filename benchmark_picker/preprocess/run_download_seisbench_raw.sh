#!/usr/bin/env bash
set -euo pipefail

export AI_DATA_ROOT=/nas/zhouyj/AI_datasets
export TMPDIR=${AI_DATA_ROOT}/_tmp
export TEMP=${TMPDIR}
export TMP=${TMPDIR}
export SEISBENCH_CACHE_ROOT=${AI_DATA_ROOT}/_cache/seisbench
export XDG_CACHE_HOME=${AI_DATA_ROOT}/_cache/xdg
export POOCH_HOME=${AI_DATA_ROOT}/_cache/pooch
export MPLCONFIGDIR=${AI_DATA_ROOT}/_cache/matplotlib
export NUMBA_CACHE_DIR=${AI_DATA_ROOT}/_cache/numba

mkdir -p "${TMPDIR}" "${SEISBENCH_CACHE_ROOT}" "${XDG_CACHE_HOME}" \
         "${POOCH_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/download_seisbench_raw.py"