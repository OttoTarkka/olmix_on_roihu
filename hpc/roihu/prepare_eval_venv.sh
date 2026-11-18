#!/bin/bash
#SBATCH --job-name=prepare-eval-venv
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Builds the olmix eval venv (EVAL_VENV in env.sh): olmix, lm-eval and their
# dependencies on top of the python-pytorch/2.13 module, then checks it.
#
# Run from the repository root:
#
#     mkdir -p logs && sbatch hpc/roihu/prepare_eval_venv.sh
#
# It runs as a GPU job because the venv must be built on the ARM GPU nodes it
# will be used on, and because the final check needs a GPU.
#
# This is a separate venv from the Megatron training one (TRAIN_VENV): olmix
# pins datasets<4 and pyarrow<21 and pulls in OLMo-core, any of which could
# replace packages the training venv relies on.
#
# torch and numpy are pinned to the module's versions with a constraints file,
# so pip fails instead of quietly installing a different torch. The file is
# kept as $EVAL_VENV/constraints.txt for any later `pip install -c`, next to
# pip-freeze.txt, the exact package list.

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Run this script as a slurm job: sbatch $0" >&2
    exit 1
fi

# -E: the ERR trap below also fires inside functions and subshells.
set -Eeuo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f env.sh ]]; then
    echo "[error] submit from the olmix_on_roihu repository root (no env.sh in $PWD)" >&2
    exit 1
fi
source env.sh

OLMIX_SRC="$(cd "$OLMIX_REPO/../olmix" && pwd)"
LM_EVAL_SRC="$(cd "$OLMIX_REPO/../lm-evaluation-harness" && pwd)"
REQUIREMENTS="$OLMIX_REPO/hpc/roihu/eval_venv_requirements.txt"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Venv        : $EVAL_VENV"
echo "Module      : $PYTORCH_MODULE"
echo "olmix       : $OLMIX_SRC"
echo "lm-eval     : $LM_EVAL_SRC"
echo "Requirements: $REQUIREMENTS"
echo

if [ -e "$EVAL_VENV" ]; then
    echo "[error] $EVAL_VENV exists, not clobbering. Remove it first to rebuild." >&2
    exit 1
fi

unset LD_PRELOAD
module purge
module load "$PYTORCH_MODULE"
module list 2>&1

# Remove the half-built venv if any step fails, so that a rerun can start
# clean. Only this job's venv is removed, and only on failure.
venv_created=false
remove_failed_venv() {
    if $venv_created; then
        echo "[error] build failed; removing the incomplete $EVAL_VENV" >&2
        rm -rf "$EVAL_VENV"
    fi
}
trap remove_failed_venv ERR

mkdir -p "$(dirname "$EVAL_VENV")"
python -m venv --system-site-packages "$EVAL_VENV"
venv_created=true
source "$EVAL_VENV/bin/activate"

# Pin what the module's compiled stack depends on.
python - > "$EVAL_VENV/constraints.txt" <<'PY'
import importlib.metadata

for package in ("torch", "numpy"):
    print(f"{package}=={importlib.metadata.version(package)}")
PY
echo "Constraints:"
cat "$EVAL_VENV/constraints.txt"

# Temporary build files on node-local disk, deleted on exit.
mkdir -p "/tmp/$USER"
export TMPDIR="$(mktemp -d "/tmp/$USER/eval-venv.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

echo "START pip install: $(date)"
pip install --upgrade pip
pip install -c "$EVAL_VENV/constraints.txt" -r "$REQUIREMENTS"
pip install -c "$EVAL_VENV/constraints.txt" --no-deps "$OLMIX_SRC"
pip install -c "$EVAL_VENV/constraints.txt" "${LM_EVAL_SRC}[hf]"
echo "DONE pip install: $(date)"

pip freeze > "$EVAL_VENV/pip-freeze.txt"
# pip check also sees the module's own packages (--system-site-packages), whose
# conflicts are not this venv's doing, so a report here is a warning to read,
# not a failure. The checks below are what decide whether the venv works.
pip check || echo "[warning] pip check reported dependency conflicts (see above)" >&2

echo
echo "Checking the venv"
python scripts/check_eval_env.py --task-dir "$EVAL_TASK_DIR" --require-gpu
python -m pytest -p no:cacheprovider tests/test_eval_tasks

echo
echo "Eval venv ready: $EVAL_VENV"
