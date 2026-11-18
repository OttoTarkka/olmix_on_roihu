#!/bin/bash
# Runs `python` from the olmix eval venv (EVAL_VENV in env.sh), so that scripts
# can call "$OLMIX_PYTHON" exactly as they would a venv's python.
#
# Usage: hpc/roihu/eval_python.sh [python arguments]
#   hpc/roihu/eval_python.sh scripts/generate_variants.py configs/x.yaml
#   hpc/roihu/eval_python.sh -m olmix.cli fit --config ... --output-dir ...
#
# The venv is built for the ARM GPU nodes, so this only runs on the Roihu-GPU
# login node or a GPU node. The module set of the calling shell is replaced in
# this process only.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../env.sh"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "[error] $EVAL_VENV is built for the ARM GPU nodes, but this is $(uname -m)." >&2
    echo "        Run this on the Roihu-GPU login node or in a GPU job." >&2
    exit 1
fi
if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
    echo "[error] no venv at EVAL_VENV=$EVAL_VENV" >&2
    echo "        Build it with: sbatch hpc/roihu/prepare_eval_venv.sh" >&2
    exit 1
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[error] the 'module' command is not available in this shell." >&2
    exit 1
fi

module purge >/dev/null 2>&1
module load "$PYTORCH_MODULE"
source "$EVAL_VENV/bin/activate"

exec python "$@"
