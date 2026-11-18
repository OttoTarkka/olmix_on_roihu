#!/bin/bash
#SBATCH --job-name=olmix-convert
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Converts every finished variant of a swarm to HuggingFace format with
# scripts/megatron_to_hf.py, one after another in one job (each takes well
# under a minute). The conversion itself needs no GPU, but the Python
# environment is built for the ARM GPU nodes.
#
# Usage (from the repository root):
#   sbatch scripts/convert_to_hf.sh <run_prefix>
#
# Reads   $OUTPUT_ROOT/<run_prefix>-NNNN/checkpoints   (the latest checkpoint)
# Writes  $OUTPUT_ROOT/hf_checkpoints/<run_prefix>-NNNN/<run_prefix>-NNNN_hf_iter_NNNNNNN
#
# Only final checkpoints (iteration == train_iters) are converted; a variant
# that has not finished counts as a failure. Variants already converted are
# skipped, so the job can be resubmitted. Every variant is attempted; the job
# fails at the end if any did.
#
# Optional check against Megatron's validation loss (single-dataset runs only):
#   CHECK_DATA_PREFIX    the Megatron dataset prefix the run trained on
#   CHECK_EXPECTED_LOSS  the "validation loss at iteration N" value from its log
#
# Smoke test, on the run kept in output/tests/smoke:
#   OUTPUT_ROOT=$PWD/output/tests/smoke \
#   CHECK_DATA_PREFIX=/scratch/project_2020507/preprocessed/wikipesto/wikipesto \
#   CHECK_EXPECTED_LOSS=3.913925 \
#       sbatch --partition=gputest --time=00:15:00 scripts/convert_to_hf.sh smoke-wikipesto

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit with sbatch (see the comment at the top of $0)." >&2
    exit 1
fi

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f env.sh ]]; then
    echo "[error] submit from the olmix_on_roihu repository root (no env.sh in $PWD)" >&2
    exit 1
fi
source env.sh

RUN_PREFIX=${1:?"usage: sbatch $0 <run_prefix>"}
if [[ -n "${CHECK_DATA_PREFIX:-}" || -n "${CHECK_EXPECTED_LOSS:-}" ]]; then
    : "${CHECK_DATA_PREFIX:?set both CHECK_DATA_PREFIX and CHECK_EXPECTED_LOSS, or neither}"
    : "${CHECK_EXPECTED_LOSS:?set both CHECK_DATA_PREFIX and CHECK_EXPECTED_LOSS, or neither}"
fi

shopt -s nullglob
RUN_DIRS=("$OUTPUT_ROOT/${RUN_PREFIX}-"[0-9][0-9][0-9][0-9])
shopt -u nullglob
(( ${#RUN_DIRS[@]} > 0 )) || { echo "[error] no variants $OUTPUT_ROOT/${RUN_PREFIX}-NNNN" >&2; exit 1; }

unset LD_PRELOAD
module purge
module load "$PYTORCH_MODULE"
source "$EVAL_VENV/bin/activate"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Run         : $RUN_PREFIX (${#RUN_DIRS[@]} variants in $OUTPUT_ROOT)"
echo "Output      : $OUTPUT_ROOT/hf_checkpoints"
if [[ -n "${CHECK_EXPECTED_LOSS:-}" ]]; then
    echo "Check       : Megatron validation loss $CHECK_EXPECTED_LOSS on $CHECK_DATA_PREFIX"
else
    echo "Check       : none"
fi
python -c "import importlib.metadata as m; print('torch', m.version('torch'), '| transformers', m.version('transformers'))"
echo

converted=0
skipped=0
failed=()
for run_dir in "${RUN_DIRS[@]}"; do
    exp_name="$(basename "$run_dir")"
    tracker="$run_dir/checkpoints/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
        echo "[error] $exp_name: no checkpoint ($tracker missing)" >&2
        failed+=("$exp_name")
        continue
    fi
    iter_name=$(printf "iter_%07d" "$(tr -d '[:space:]' < "$tracker")")
    hf_dir="$OUTPUT_ROOT/hf_checkpoints/$exp_name/${exp_name}_hf_${iter_name}"
    if [[ -d "$hf_dir" ]]; then
        echo "$exp_name: already converted ($hf_dir)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "=== $exp_name $iter_name ($(date))"
    if ! python scripts/megatron_to_hf.py --checkpoint "$run_dir/checkpoints" --output-dir "$hf_dir" --require-final; then
        echo "[error] $exp_name: conversion failed" >&2
        failed+=("$exp_name")
        continue
    fi
    if [[ -n "${CHECK_EXPECTED_LOSS:-}" ]]; then
        if ! python scripts/check_hf_conversion.py --model-dir "$hf_dir" \
                --data-prefix "$CHECK_DATA_PREFIX" --expected-loss "$CHECK_EXPECTED_LOSS"; then
            echo "[error] $exp_name: converted model failed the loss check" >&2
            failed+=("$exp_name")
            continue
        fi
    fi
    converted=$((converted + 1))
done

echo
echo "Converted: $converted   Already done: $skipped   Failed: ${#failed[@]}"
if (( ${#failed[@]} > 0 )); then
    echo "[error] failed variants: ${failed[*]}" >&2
    exit 1
fi
echo "END $SLURM_JOB_ID: $(date)"
