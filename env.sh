# Shared settings for olmix on Roihu. Every script sources this file, and so
# should your shell, from the repository root:
#
#     source env.sh
#
# Override any of the values below by exporting it before sourcing.
#
# Roihu has two CPU architectures. The GPU nodes and the Roihu-GPU login node
# are ARM (aarch64); the CPU nodes and the Roihu-CPU login node are x86. Both
# Python environments below are built for the GPU side, so everything that
# runs Python (mixture generation, training, evaluation, collection, the fit)
# must run on the Roihu-GPU login node or a GPU node.

OLMIX_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OLMIX_REPO

export PROJECT_ALLOCATION="${PROJECT_ALLOCATION:-project_2020507}"

# --- Python environments ------------------------------------------------------
# Both are venvs on top of the same module, with --system-site-packages.
export PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.13}"
# Megatron-LM training: TransformerEngine, apex, Megatron Core
# (roihu-megatron-gpt/prepare_venv.sh).
export TRAIN_VENV="${TRAIN_VENV:-/scratch/$PROJECT_ALLOCATION/venv/python-pytorch2.13-megatron}"
# Mixture generation, evaluation and the fit: olmix, lm-eval
# (hpc/roihu/prepare_eval_venv.sh).
export EVAL_VENV="${EVAL_VENV:-/scratch/$PROJECT_ALLOCATION/venv/python-pytorch2.13-olmix-eval}"
# `python` from EVAL_VENV, usable from any shell: "$OLMIX_PYTHON" script.py ...
export OLMIX_PYTHON="$OLMIX_REPO/hpc/roihu/eval_python.sh"

# --- Training -----------------------------------------------------------------
# The Megatron-LM checkout that prepare_venv.sh installed into TRAIN_VENV
# (tag core_v0.16.1). pretrain_gpt.py imports megatron.core from its own
# directory, so this must be the same checkout as the installed one.
# (Resolved without a failing command: env.sh is sourced by `set -e` scripts,
# which would otherwise exit silently when the directory is missing.
# train-0.05B.sh reports a missing pretrain_gpt.py itself.)
if [[ -z "${MEGATRON_DIR:-}" ]]; then
    MEGATRON_DIR="$OLMIX_REPO/../roihu-megatron-gpt/Megatron-LM"
    if [[ -d "$MEGATRON_DIR" ]]; then
        MEGATRON_DIR="$(cd "$MEGATRON_DIR" && pwd)"
    fi
fi
export MEGATRON_DIR
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-openeurollm/tokenizer-256k}"
# Outputs live under output/ in this repository, in subdirectories: output/
# itself holds the LUMI fit results (4917f21a31ecad77, english_removed).
#   output/runs/   real swarms (OUTPUT_ROOT below)
#   output/tests/  one directory per test run: set OUTPUT_ROOT=$OLMIX_REPO/output/tests/<name>
#   output/eval/   evaluation results (EVAL_OUTPUT_DIR below)
# One directory per variant under OUTPUT_ROOT: checkpoints, TensorBoard, W&B,
# data index cache; HuggingFace conversions go to OUTPUT_ROOT/hf_checkpoints.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$OLMIX_REPO/output/runs}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmix-roihu}"

# --- Evaluation ---------------------------------------------------------------
export EVAL_TASK_DIR="$OLMIX_REPO/eval_tasks/bpb"
export EVAL_NUM_FEWSHOT="${EVAL_NUM_FEWSHOT:-5}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OLMIX_REPO/output/eval}"
