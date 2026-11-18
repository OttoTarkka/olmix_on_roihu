#!/bin/bash
# Run everything after training for one swarm: convert -> evaluate -> collect -> fit.
#
#   ./scripts/run_pipeline.sh <run_prefix> <deu|spa|hun> [--fit]
#
# Run from the repository root (on the Roihu-GPU login node), once the training
# array has finished. It submits:
#
#   1. one conversion job (scripts/convert_to_hf.sh), for every variant;
#   2. evaluation jobs (scripts/eval_bpb.sh), EVAL_BATCH models each, after 1;
#   3. one collect job (scripts/pipeline/collect_and_fit.sh), after all of 2,
#      which also runs `olmix fit` with data/runs/<run>/fit.yaml if --fit is given.
#
# Unlike the LUMI version, everything is submitted at once: training has
# finished, so the final iteration of every variant, and therefore the path of
# its HF model, is already known.
#
# Dependencies are afterok: if a conversion or an evaluation fails, the later
# jobs are cancelled rather than run on part of the swarm. Fix the cause and
# run this again: finished conversions and evaluations are skipped.
#
# Environment:
#   EVAL_BATCH   models per evaluation job (default 8; one model takes about
#                7.5 min for the six tasks of one language)
#   EVAL_TIME    time limit of an evaluation job (default 02:00:00)

set -euo pipefail

if [[ ! -f env.sh ]]; then
    echo "[error] run from the olmix_on_roihu repository root (no env.sh in $PWD)" >&2
    exit 1
fi
source env.sh

usage="usage: $0 <run_prefix> <deu|spa|hun> [--fit]"
RUN_PREFIX=${1:?$usage}
LANGUAGE=${2:?$usage}
FIT=false
if [[ "${3:-}" == "--fit" ]]; then
    FIT=true
elif [[ -n "${3:-}" ]]; then
    echo "$usage" >&2
    exit 1
fi
EVAL_BATCH=${EVAL_BATCH:-8}
EVAL_TIME=${EVAL_TIME:-02:00:00}

RUN_DIR="data/runs/${RUN_PREFIX}"
[[ -f "$RUN_DIR/variants.json" ]] || { echo "[error] no such run: $RUN_DIR/variants.json" >&2; exit 1; }
[[ -f "$EVAL_TASK_DIR/olmix_bpb_${LANGUAGE}.yaml" ]] || { echo "[error] no task group for language '$LANGUAGE'" >&2; exit 1; }
if $FIT; then
    [[ -f "$RUN_DIR/fit.yaml" ]] || { echo "[error] --fit, but no $RUN_DIR/fit.yaml" >&2; exit 1; }
fi
(( EVAL_BATCH >= 1 )) || { echo "[error] EVAL_BATCH must be at least 1" >&2; exit 1; }

# Every variant must have finished training: its tracker names the iteration
# that convert_to_hf.sh will convert (it also checks that this is the last one).
mapfile -t VARIANTS < <(python3 -c "import json, sys; print('\n'.join(v['variant_id'] for v in json.load(open(sys.argv[1]))))" "$RUN_DIR/variants.json")
(( ${#VARIANTS[@]} > 0 )) || { echo "[error] no variants in $RUN_DIR/variants.json" >&2; exit 1; }
MODELS=()
not_trained=()
for variant in "${VARIANTS[@]}"; do
    tracker="$OUTPUT_ROOT/$variant/checkpoints/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
        not_trained+=("$variant")
        continue
    fi
    iter_name=$(printf "iter_%07d" "$(tr -d '[:space:]' < "$tracker")")
    MODELS+=("$OUTPUT_ROOT/hf_checkpoints/$variant/${variant}_hf_${iter_name}")
done
if (( ${#not_trained[@]} > 0 )); then
    echo "[error] ${#not_trained[@]} variant(s) have no checkpoint yet: ${not_trained[*]:0:8}" >&2
    echo "        Wait for the training array, or resubmit it for the failed tasks." >&2
    exit 1
fi

mkdir -p logs
echo "Swarm        : $RUN_PREFIX (${#VARIANTS[@]} variants, $OUTPUT_ROOT)"
echo "Language     : $LANGUAGE"
echo "Fit          : $($FIT && echo "yes, $RUN_DIR/fit.yaml" || echo "no, stopping after collect")"
echo

# --- 1. conversion ------------------------------------------------------------
convert_id=$(sbatch --parsable scripts/convert_to_hf.sh "$RUN_PREFIX")
echo "convert      : job $convert_id"

# --- 2. evaluation, EVAL_BATCH models per job ----------------------------------
eval_ids=()
for (( i = 0; i < ${#MODELS[@]}; i += EVAL_BATCH )); do
    batch=("${MODELS[@]:i:EVAL_BATCH}")
    eval_ids+=("$(sbatch --parsable --time="$EVAL_TIME" --dependency="afterok:$convert_id" \
        scripts/eval_bpb.sh "$LANGUAGE" "${batch[@]}")")
done
echo "evaluate     : ${#eval_ids[@]} job(s) of up to $EVAL_BATCH models: ${eval_ids[*]}"

# --- 3. collect (and fit) -------------------------------------------------------
dependency="afterok:$(IFS=:; echo "${eval_ids[*]}")"
collect_id=$(sbatch --parsable --dependency="$dependency" \
    --export="ALL,RUN_PREFIX=$RUN_PREFIX,LANGUAGE=$LANGUAGE,FIT=$FIT" \
    scripts/pipeline/collect_and_fit.sh)
printf "%-13s: job %s\n" "$($FIT && echo "collect+fit" || echo "collect")" "$collect_id"
echo
echo "Follow with: squeue --me; logs in logs/. If a job fails, the later ones stay"
echo "pending with reason DependencyNeverSatisfied: scancel them, fix, and rerun this."
