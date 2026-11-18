#!/bin/bash
#SBATCH --job-name=olmix-proxy
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# A requeued task (e.g. after a node failure) appends to its logs instead of
# overwriting them.
#SBATCH --open-mode=append

# Trains one 30M-parameter proxy model of an Olmix swarm on one GH200 GPU.
# Submitted as an array, one task per mixture; the array index selects the
# mixture. launch_all_swarms.sh (written by scripts/generate_variants.py) does
# this:
#
#     sbatch --array=0-<n-1> --export=ALL,MIX_PREFIX=<run> scripts/train-0.05B.sh
#
# Run from the repository root. The model and optimiser settings are those of
# the LUMI swarms (Olmix paper, Table 6); the Roihu environment and launch
# follow roihu-megatron-gpt/train-gpt.sh.
#
# Environment overrides:
#   TRAIN_TOKENS  tokens to train on (default 3_000_000_000). For a short test,
#                 also set OUTPUT_ROOT to a test directory: a run resumes
#                 from its checkpoints, and the schedule depends on this value.
#   OUTPUT_ROOT   where each variant's directory goes (env.sh: output/runs).
#                 Test runs use output/tests/<name>, e.g.
#                 OUTPUT_ROOT=$PWD/output/tests/smoke from the repository root.
#   MICRO_BATCH_SIZE  per-step batch on the GPU (default 16; must divide 64).
#                 Changes speed and memory only: the global batch stays 64.
#   EXIT_AFTER_ITERS  stop (and save) after this many iterations, keeping the
#                 full schedule. For benchmarks; use OUTPUT_ROOT=output/tests/<name>.
#
# A variant whose checkpoint already reached the last iteration exits at once,
# so the whole array can be resubmitted to fill in failed tasks.

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

TASK_ID=${SLURM_ARRAY_TASK_ID:?"submit as an array job (--array), one task per mixture"}
MIX_PREFIX=${MIX_PREFIX:?"set MIX_PREFIX (launch_all_swarms.sh does this)"}
EXP_NAME="${MIX_PREFIX}-$(printf "%04d" "$TASK_ID")"
MIX_FILE="$OLMIX_REPO/data/runs/${MIX_PREFIX}/mixes/${EXP_NAME}.txt"

echo "Job ${SLURM_ARRAY_JOB_ID}_${TASK_ID} (${SLURM_JOB_ID}) on $(hostname) at $(date)"
echo "Variant     : $EXP_NAME"
echo "Mix file    : $MIX_FILE"
echo "Megatron    : $MEGATRON_DIR"
echo "Venv        : $TRAIN_VENV"

[[ -f "$MIX_FILE" ]] || { echo "[error] no mix file: $MIX_FILE" >&2; exit 1; }
[[ -f "$MEGATRON_DIR/pretrain_gpt.py" ]] || { echo "[error] no pretrain_gpt.py in MEGATRON_DIR=$MEGATRON_DIR" >&2; exit 1; }

# --- Environment ------------------------------------------------------------
unset LD_PRELOAD
module purge
module load "$PYTORCH_MODULE"
source "$TRAIN_VENV/bin/activate"
module list 2>&1

# W&B (and HF_TOKEN for the tokenizer, if needed) come from ~/.hpc_secrets.
if [[ -f ~/.hpc_secrets ]]; then
    source ~/.hpc_secrets
fi

# --- Paths --------------------------------------------------------------------
OUTPUT_DIR="$OUTPUT_ROOT/$EXP_NAME"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
WANDB_DIR="$OUTPUT_DIR/wandb"
DATA_CACHE_PATH="$OUTPUT_DIR/data_cache"
LAUNCH_SCRIPT="$OLMIX_REPO/launch.sh"

# Single process on one GPU: torch.distributed still needs a rendezvous
# address. The port depends on the job ID so that array tasks sharing a node
# do not collide.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 10000 + (SLURM_JOB_ID % 20000) ))
export WORLD_SIZE=$SLURM_NTASKS
# Needed for sequence parallelism
# (see https://github.com/NVIDIA/Megatron-LM/issues/533)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2
export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# ====================================================================
# MODEL AND PRETRAINING CONFIGURATION (30M Olmix setup)
# ====================================================================

# --- 30M architecture (Olmix paper, Table 6) ---
NUM_LAYERS=4
HIDDEN_SIZE=256
FFN_HIDDEN_SIZE=768          # Scaled down for SwiGLU (approx 8/3 * 256)
NUM_ATTENTION_HEADS=8
NUM_QUERY_GROUPS=8           # MHA (matches attention heads)
KV_CHANNELS=32               # head_dim (256 / 8 = 32)
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=2048              # per olmix paper
ROTARY_BASE=10000

# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=0.007
MIN_LR=0
LR_WARMUP_ITERS=500
COOLDOWN_FRACTION=1/5
CLIP_GRAD=1.0
WEIGHT_DECAY=0.1    # match scaling law setup

# PARALLELISM (single GPU)
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

# TRAINING
# Same global batch as the LUMI swarms (64 x 2048 tokens). On one GPU, a micro
# batch of 16 means 4 gradient-accumulation steps per iteration. The micro
# batch only changes speed and memory, not what is trained, so it can be
# overridden (MICRO_BATCH_SIZE in the environment); it must divide the global
# batch. Benchmark on one GH200 (2026-11-18, 300 iterations): micro batch 8
# took 925 ms per iteration (27 GB peak), 16 took 830 ms (52 GB peak, 84 GB
# reserved of 95), and 32 ran out of memory: the fp32 logits of the 262k
# vocabulary alone need 64 GiB.
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
if (( MICRO_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "[error] MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE must divide GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE" >&2
    exit 1
fi
TRAIN_TOKENS=${TRAIN_TOKENS:-3_000_000_000}

# SAVING AND EVALUATION
LOG_INTERVAL=20
SAVE_INTERVAL=4000
EVAL_INTERVAL=1000
EVAL_ITERS=100

# ====================================================================
# DERIVED SETTINGS
# ====================================================================

confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
        echo "Error: variable '$varname' should not be set." >&2
        exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up "$TRAIN_TOKENS" "$ITER_TOKENS")
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))
LR_DECAY_ITERS=$TRAIN_ITERS

# Megatron asserts this deep inside its setup; say it plainly instead.
if (( TRAIN_ITERS <= LR_WARMUP_ITERS )); then
    echo "[error] TRAIN_TOKENS=$TRAIN_TOKENS gives $TRAIN_ITERS iterations, not more than the" \
         "$LR_WARMUP_ITERS warmup iterations. For a short test, keep TRAIN_TOKENS and set EXIT_AFTER_ITERS." >&2
    exit 1
fi

EXIT_ARGS=()
if [[ -n "${EXIT_AFTER_ITERS:-}" ]]; then
    EXIT_ARGS=(--exit-interval "$EXIT_AFTER_ITERS")
    echo "Exit after  : $EXIT_AFTER_ITERS iterations (benchmark/test run)"
fi

echo "Train tokens: $TRAIN_TOKENS ($TRAIN_ITERS iterations)"
echo "Batch       : global $GLOBAL_BATCH_SIZE, micro $MICRO_BATCH_SIZE ($((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE)) accumulation steps)"
echo "Output      : $OUTPUT_DIR"

# --- Skip finished variants -----------------------------------------------------
# Megatron writes the tracker file only after a checkpoint is fully saved.
TRACKER="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
if [[ -f "$TRACKER" ]]; then
    last_iter=$(tr -d '[:space:]' < "$TRACKER")
    if [[ "$last_iter" == "$TRAIN_ITERS" ]]; then
        echo "$EXP_NAME already finished (checkpoint at iteration $last_iter); nothing to do."
        exit 0
    fi
    echo "Resuming $EXP_NAME from iteration $last_iter"
fi

# --- Data mixture -----------------------------------------------------------------
# Each line of the mix file is "<weight> <prefix>"; every prefix needs its .bin
# and .idx. Check them all now rather than fail after the job has started.
DATA_PATH_ARGS=()
line_number=0
# (The || keeps a last line that has no trailing newline.)
while read -r weight prefix extra || [[ -n "${weight:-}" ]]; do
    line_number=$((line_number + 1))
    if [[ -z "$weight" || -z "$prefix" || -n "${extra:-}" ]]; then
        echo "[error] $MIX_FILE line $line_number is not '<weight> <prefix>'" >&2
        exit 1
    fi
    for suffix in bin idx; do
        [[ -f "${prefix}.${suffix}" ]] || { echo "[error] missing ${prefix}.${suffix} ($MIX_FILE line $line_number)" >&2; exit 1; }
    done
    DATA_PATH_ARGS+=("$weight" "$prefix")
done < "$MIX_FILE"
(( ${#DATA_PATH_ARGS[@]} > 0 )) || { echo "[error] empty mix file: $MIX_FILE" >&2; exit 1; }
echo "Data        : $(( ${#DATA_PATH_ARGS[@]} / 2 )) prefixes"

mkdir -p "$CHECKPOINT_PATH"

# ====================================================================
# COMMAND-LINE ARGUMENTS
# ====================================================================

DATA_ARGS=(
    --data-path "${DATA_PATH_ARGS[@]}"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 99,1,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type cyclic
    --num-workers 2
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --use-flash-attn
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --kv-channels $KV_CHANNELS
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --qk-layernorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --bf16
    --swiglu
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(
        --untie-embeddings-and-output-weights
    )
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --context-parallel-size $CONTEXT_PARALLEL_SIZE
    --sequence-parallel
    --use-distributed-optimizer
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-interval $LOG_INTERVAL
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    OUTPUT_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "${EXP_NAME}-mix"
        --wandb-save-dir "$WANDB_DIR"
    )
    echo "W&B         : project $WANDB_PROJECT"
else
    echo "W&B         : off (no WANDB_API_KEY in the environment or ~/.hpc_secrets)"
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch_dist
    --load "$CHECKPOINT_PATH"
    --save "$CHECKPOINT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND=(
    "$MEGATRON_DIR/pretrain_gpt.py"
    "${MODEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OUTPUT_ARGS[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${EXIT_ARGS[@]}"
)

echo '============= COMMAND: ============='
printf '%q ' "${COMMAND[@]}"
echo
echo '===================================='

echo "START $SLURM_JOB_ID: $(date)"

srun \
    --label \
    "$LAUNCH_SCRIPT" \
    "${COMMAND[@]}"

echo "END $SLURM_JOB_ID: $(date)"
