#!/bin/bash
#SBATCH --job-name=olmix-eval-bpb
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# BPB evaluation of one or more HuggingFace models on one language's Olmix fit
# targets (eval_tasks/bpb/olmix_bpb_<language>.yaml), with lm-eval in the eval
# venv. The models are evaluated one after another on one GPU.
#
# Usage (from the repository root):
#   sbatch scripts/eval_bpb.sh <deu|spa|hun> <model> [<model> ...]
#
# <model> is a directory holding a HuggingFace model (config.json and
# *.safetensors), or a Hub model ID.
#
# Environment:
#   EVAL_NUM_FEWSHOT   few-shot examples (env.sh: 5)
#   EVAL_OUTPUT_DIR    results go to <EVAL_OUTPUT_DIR>/<language>/<model name>/
#                      (env.sh: output/eval in the repository)
#   LIMIT              documents per task, for a quick test (default: all)
#   HF_HOME            dataset cache; must be set (normally in ~/.bashrc)
#   HF_TOKEN           for gated datasets, from the environment or ~/.hpc_secrets
#
# Example quick test on a small public model:
#   LIMIT=20 sbatch scripts/eval_bpb.sh hun HuggingFaceTB/SmolLM2-135M
#
# Note: lm-eval cuts contexts longer than the model's maximum length from the
# left. The proxies have 2,048 positions, so 5-shot prompts with long
# passages (Belebele) may lose part of their first example.

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

if (( $# < 2 )); then
    echo "usage: sbatch $0 <deu|spa|hun> <model> [<model> ...]" >&2
    exit 1
fi
LANGUAGE="$1"
shift
GROUP="olmix_bpb_${LANGUAGE}"
[[ -f "$EVAL_TASK_DIR/$GROUP.yaml" ]] || { echo "[error] no task group $EVAL_TASK_DIR/$GROUP.yaml (language: deu, spa or hun)" >&2; exit 1; }

if [[ -z "${HF_HOME:-}" ]]; then
    echo "[error] HF_HOME is not set; point it at a dataset cache you can write to." >&2
    exit 1
fi

# A local model must be a complete HuggingFace directory. Arguments starting
# with / or . are always local paths (a missing one is an error, not a Hub ID
# to download); any other argument that is not an existing path is a Hub ID.
for model in "$@"; do
    if [[ "$model" == /* || "$model" == .* ]] && [[ ! -e "$model" ]]; then
        echo "[error] no such model directory: $model" >&2
        exit 1
    fi
    if [[ -e "$model" ]]; then
        [[ -f "$model/config.json" ]] || { echo "[error] no config.json in $model" >&2; exit 1; }
        ls "$model"/*.safetensors >/dev/null 2>&1 || { echo "[error] no *.safetensors in $model" >&2; exit 1; }
    fi
done

if [[ -f ~/.hpc_secrets ]]; then
    source ~/.hpc_secrets
fi

unset LD_PRELOAD
module purge
module load "$PYTORCH_MODULE"
source "$EVAL_VENV/bin/activate"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Group       : $GROUP ($EVAL_TASK_DIR)"
echo "Few-shot    : $EVAL_NUM_FEWSHOT"
echo "Limit       : ${LIMIT:-none}"
echo "Models      : $#"
echo "Output      : $EVAL_OUTPUT_DIR/$LANGUAGE"
echo "HF_HOME     : $HF_HOME"
python -c "import importlib.metadata as m; print('lm-eval', m.version('lm_eval'), '| torch', m.version('torch'), '| transformers', m.version('transformers'), '| datasets', m.version('datasets'))"
echo

for model in "$@"; do
    name="$(basename "$model")"
    output_path="$EVAL_OUTPUT_DIR/$LANGUAGE/$name"
    # Already evaluated (lm-eval writes <output_path>/<model>/results_<time>.json)?
    # Then skip it, so a resubmitted batch does not redo finished models. Only
    # a full run counts: lm-eval records "limit": null in its config then; the
    # results of a --limit test run are incomplete.
    if [[ -z "${LIMIT:-}" ]] && compgen -G "$output_path/*/results_*.json" >/dev/null \
        && grep -q '"limit": null' "$output_path"/*/results_*.json; then
        echo "=== $model: already evaluated in $output_path, skipping"
        continue
    fi
    echo "=== $model -> $output_path ($(date))"
    srun --ntasks=1 lm-eval run \
        --model hf \
        --model_args "pretrained=$model,dtype=bfloat16" \
        --include_path "$EVAL_TASK_DIR" \
        --tasks "$GROUP" \
        --num_fewshot "$EVAL_NUM_FEWSHOT" \
        --batch_size auto \
        --output_path "$output_path" \
        "${LIMIT_ARGS[@]}"
done

echo "END $SLURM_JOB_ID: $(date)"
