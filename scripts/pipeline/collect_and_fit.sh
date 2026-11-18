#!/bin/bash
#SBATCH --job-name=olmix-collect-fit
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Join BPB scores to mixtures, then fit the regression if asked to.
# Submitted by scripts/run_pipeline.sh; runs as a GPU job only because the
# eval venv is built for the ARM GPU nodes (no GPU is used).
#
# Environment (set by run_pipeline.sh):
#   RUN_PREFIX   the swarm
#   LANGUAGE     deu, spa or hun: results are read from $EVAL_OUTPUT_DIR/<LANGUAGE>
#   FIT          true to run `olmix fit` with data/runs/<run>/fit.yaml

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh

RUN_PREFIX=${RUN_PREFIX:?"set RUN_PREFIX"}
LANGUAGE=${LANGUAGE:?"set LANGUAGE"}
FIT=${FIT:-false}

RUN_DIR="data/runs/${RUN_PREFIX}"
RESULTS_DIR="$EVAL_OUTPUT_DIR/$LANGUAGE"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Swarm       : $RUN_PREFIX"
echo "Results     : $RESULTS_DIR"
echo

"$OLMIX_PYTHON" scripts/collect_bpb.py \
    --results-dir "$RESULTS_DIR" \
    --variants-json "$RUN_DIR/variants.json" \
    --output-dir "$RUN_DIR/olmix_fit"

if [[ "$FIT" != "true" ]]; then
    echo
    echo "No fit requested. Edit $RUN_DIR/fit.yaml if needed, then fit with:"
    echo "  \$OLMIX_PYTHON -m olmix.cli fit --config $RUN_DIR/fit.yaml --output-dir $RUN_DIR/fit"
    exit 0
fi

# olmix keeps its regression cache in ./cache relative to the working directory,
# which is the repository root here.
"$OLMIX_PYTHON" -m olmix.cli fit --config "$RUN_DIR/fit.yaml" --output-dir "$RUN_DIR/fit"
echo "Fit written to $RUN_DIR/fit"
