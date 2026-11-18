# olmix on Roihu

Finds a good pretraining data mixture by training many small proxy models on
different mixtures and fitting a regression to their evaluation scores. The
mixture sampling and the regression come from
[olmix](https://github.com/allenai/olmix)
([paper](https://arxiv.org/abs/2602.12237)); everything else here adapts it to
CSC's Roihu supercomputer. It is a port of the LUMI version of this repository,
which is kept separately.

The pipeline for one swarm:

```
configs/<experiment>.yaml
  └─ scripts/generate_variants.py  → data/runs/<run>/{config.yaml, variants.json, fit.yaml, mixes/}
  └─ launch_all_swarms.sh          → one proxy training run per mixture (scripts/train-0.05B.sh)
  └─ scripts/convert_to_hf.sh      → HuggingFace models     (scripts/megatron_to_hf.py)
  └─ scripts/eval_bpb.sh           → 5-shot BPB scores      (lm-evaluation-harness, eval_tasks/bpb)
  └─ scripts/collect_bpb.py        → data/runs/<run>/olmix_fit/{ratios,metrics}.csv
  └─ olmix fit                     → proposed mixture
```

## Status

| Step | Roihu version | Tested on Roihu |
| --- | --- | --- |
| Environment | `env.sh`; eval venv from `hpc/roihu/prepare_eval_venv.sh` | yes |
| Proxy training | `scripts/train-0.05B.sh`: 30M-parameter model, one GH200 per variant | smoke run (200M tokens), micro-batch benchmark |
| Conversion | `scripts/convert_to_hf.sh` + `scripts/megatron_to_hf.py`: direct Megatron → Qwen3 conversion | locally only; Roihu check pending |
| Evaluation | `scripts/eval_bpb.sh`: lm-evaluation-harness, 5-shot BPB, one task group per language | on a public model (20 documents per task) |
| Collection and fit | `scripts/collect_bpb.py`, `olmix fit` | collection on synthetic results; no fit yet |
| One-command pipeline | `scripts/run_pipeline.sh`, `scripts/pipeline/*` | **not ported**: still calls the LUMI conversion and oellm-eval. Run the steps by hand (below). |

There is no Roihu experiment config yet: the data (HPLT web documents labelled
with WebOrganizer topics and formats) is still being prepared. The configs in
`configs/` are LUMI examples, with LUMI paths.

Measured on one GH200 with the 256k-vocabulary tokenizer:

| | |
| --- | --- |
| Training speed | 830 ms per iteration (64 × 2,048 tokens), micro batch 16 |
| Full proxy run | about 5.7 h for 3B tokens (22,889 iterations, including in-training evaluation) |
| GPU memory | 52 GB allocated, 84 of 95 GB reserved at micro batch 16; micro batch 32 runs out of memory |
| Conversion | under a minute per model |

## Repository layout

| Path | What it is |
| --- | --- |
| `env.sh` | Shared settings; every script sources it |
| `configs/*.yaml` | Experiment definitions (LUMI examples for now) |
| `data/runs/<run>/` | One swarm: its config, `variants.json`, `fit.yaml` and mixes |
| `scripts/generate_variants.py` | Samples the mixtures with olmix and writes the fit config |
| `scripts/train-0.05B.sh` | One proxy training run; submitted as an array |
| `scripts/convert_to_hf.sh`, `scripts/megatron_to_hf.py` | Megatron checkpoint → HuggingFace Qwen3 model |
| `scripts/check_hf_conversion.py` | Checks a converted model against Megatron's validation loss |
| `scripts/megatron_indexed.py` | Reads Megatron `.idx`/`.bin` datasets (numpy only) |
| `scripts/eval_bpb.sh` | BPB evaluation with lm-eval |
| `eval_tasks/bpb/` | The lm-eval task definitions; see [`eval_tasks/README.md`](eval_tasks/README.md) |
| `scripts/collect_bpb.py` | Evaluation results + `variants.json` → `ratios.csv`, `metrics.csv` |
| `scripts/make_fake_fit_inputs.py` | Fake scores from a known rule, to check the fit recovers it |
| `scripts/check_eval_env.py` | Checks the eval venv |
| `hpc/roihu/` | Eval venv build, its requirements and the `$OLMIX_PYTHON` wrapper |
| `tests/` | pytest tests |
| `output/` | Outputs; see below |

`output/` holds the LUMI fit results (`4917f21a31ecad77`, `english_removed`)
directly, and the Roihu outputs in subdirectories that git ignores:

| Path | What |
| --- | --- |
| `output/runs/` | Real swarms (`OUTPUT_ROOT`): one directory per variant, HF models in `hf_checkpoints/` |
| `output/tests/<name>/` | Test runs: set `OUTPUT_ROOT=$PWD/output/tests/<name>` |
| `output/eval/` | Evaluation results (`EVAL_OUTPUT_DIR`) |

## Setup

Roihu's GPU nodes are ARM (aarch64) and its CPU nodes x86, and Python
environments built on one side do not run on the other. Both environments here
are built for the GPU side, so run everything from the **Roihu-GPU login node**
or in GPU jobs.

### Expected checkouts

This repository expects three sibling directories:

```
olmix-repo/
├── olmix_on_roihu/          this repository
├── roihu-megatron-gpt/      Megatron-LM training setup, with Megatron-LM/ (tag core_v0.16.1)
├── olmix/                   https://github.com/allenai/olmix
└── lm-evaluation-harness/   https://github.com/EleutherAI/lm-evaluation-harness
```

### Training venv

Built by `roihu-megatron-gpt/prepare_venv.sh` (TransformerEngine, apex and
Megatron Core on `python-pytorch/2.13`), at
`/scratch/project_2020507/venv/python-pytorch2.13-megatron`. Two things that
setup needs:

- TransformerEngine's `stable` branch builds an NCCL extension (NCCL EP) that
  requires NCCL 2.30.7, newer than the module's 2.29.7. `prepare_venv.sh` must
  set `NVTE_WITH_NCCL_EP=0`.
- Train with the `Megatron-LM` checkout that `prepare_venv.sh` installs.
  `pretrain_gpt.py` imports `megatron.core` from its own directory, so running
  it from a different Megatron-LM checkout silently uses different code.
  `env.sh` sets `MEGATRON_DIR` to `../roihu-megatron-gpt/Megatron-LM`.

### Eval venv

```bash
cd /scratch/project_2020507/users/$USER/olmix-repo/olmix_on_roihu
mkdir -p logs
sbatch hpc/roihu/prepare_eval_venv.sh
```

It builds `/scratch/project_2020507/venv/python-pytorch2.13-olmix-eval`: olmix,
lm-eval (from the checkout, with the `hf` extra) and their dependencies on top
of `python-pytorch/2.13`. torch and numpy are pinned to the module's versions.
olmix is installed without its dependencies, which are listed in
`hpc/roihu/eval_venv_requirements.txt` instead: its `pyproject.toml` takes
OLMo-core from a git branch that no longer exists, so OLMo-core is pinned to the
commit in olmix's `uv.lock`. The job ends by running `scripts/check_eval_env.py`
and the tests; `logs/prepare-eval-venv_<jobid>.out` should end with
"Eval venv ready". A failed build removes its half-built venv.

`$OLMIX_PYTHON` (`hpc/roihu/eval_python.sh`) runs this venv's python from any
shell on the GPU side.

### Settings and secrets

`env.sh` is the one place for paths and settings. Source it from the repository
root, and override a value by exporting it first:

| Variable | Default |
| --- | --- |
| `PROJECT_ALLOCATION` | `project_2020507` |
| `PYTORCH_MODULE` | `python-pytorch/2.13` |
| `TRAIN_VENV`, `EVAL_VENV` | the two venvs above |
| `MEGATRON_DIR` | `../roihu-megatron-gpt/Megatron-LM` |
| `TOKENIZER_MODEL` | `openeurollm/tokenizer-256k` |
| `OUTPUT_ROOT` | `output/runs` |
| `EVAL_OUTPUT_DIR` | `output/eval` |
| `EVAL_NUM_FEWSHOT` | `5` |
| `WANDB_PROJECT` | `olmix-roihu` |

Put tokens in `~/.hpc_secrets`; training and evaluation source it if it exists.
Without `WANDB_API_KEY`, training runs without W&B and says so in its log.

```bash
export WANDB_API_KEY=""
export HF_TOKEN=""
```

Evaluation also needs `HF_HOME` set to a dataset cache you can write to
(normally in `~/.bashrc`). Roihu's compute nodes have internet access, so
datasets and tokenizers download on first use.

### Smoke tests

From the repository root:

```bash
# Evaluation on a public model, 20 documents per task
LIMIT=20 sbatch scripts/eval_bpb.sh hun HuggingFaceTB/SmolLM2-135M

# Training: 200M tokens on wikipesto (see data/runs/smoke-wikipesto/README.md)
TRAIN_TOKENS=200_000_000 OUTPUT_ROOT=$PWD/output/tests/smoke \
    sbatch --array=0 --export=ALL,MIX_PREFIX=smoke-wikipesto scripts/train-0.05B.sh

# Conversion of that run, checked against Megatron's logged validation loss
OUTPUT_ROOT=$PWD/output/tests/smoke \
CHECK_DATA_PREFIX=/scratch/project_2020507/preprocessed/wikipesto/wikipesto \
CHECK_EXPECTED_LOSS=3.913925 \
    sbatch --partition=gputest --time=00:15:00 scripts/convert_to_hf.sh smoke-wikipesto
```

The test queues (`gputest`, `test`) allow two jobs per user at a time.

## Running a swarm

### 1. Describe the experiment

A config has three blocks: `settings`, `swarm` and `datasets`.

```yaml
settings:
  max_tokens: 3000000000      # size of the run the mixture is being chosen for
  repetition_factor: 5.0      # how many times a dataset may be repeated
  seed: 42                    # same seed + same config = same swarm

datasets:
  open_web_math:
    multiplier: 1             # >1 if the path is a sample of a larger dataset
    paths:
      - /scratch/.../open-web-math/open-web-math    # no .bin suffix
```

A dataset may instead have `quality:` buckets, which become separate leaves
named `source:bucket`. A dataset with several paths is one leaf; its weight is
split across the paths in proportion to their token counts. That is how
WebOrganizer topic × format cells can be grouped into topic or format domains
without re-tokenizing.

### 2. Generate the mixtures

```bash
$OLMIX_PYTHON scripts/generate_variants.py configs/<experiment>.yaml
# or: ./generate_mixes.sh configs/<experiment>.yaml
```

This writes `data/runs/<run>/` (the run is named `stage2_mix_<date>_<time>`):
the mixes, `variants.json` mapping each variant to its weights, a copy of the
config, and `fit.yaml`, a fit config with the priors filled in. It also rewrites
`launch_all_swarms.sh` for this run.

Keep `variants.json`. It is the only record of which mixture a model saw, and
the fit needs it.

#### Mixture reuse

To keep an existing mixture fixed and search only over what you add to it,
point `existing_mix_file` at a JSON of `{dataset_name: weight}`:

```yaml
swarm:
  existing_mix_file: "data/flag_frozen.json"
  manual_prior:
    existing: 0.95            # share of the simplex the frozen block occupies
    open_web_math: 0.05
  nonzero_weight:             # source names; stops sparse variants dropping the block
    - existing
    - open_web_math
```

Datasets named in that file are locked into one source called `existing`,
keeping their ratios to each other. Datasets in the config but *not* in that
file are what the swarm varies, so which dataset you leave out decides what the
experiment answers.

Without `manual_prior`, the split between frozen and new comes from token
counts, which usually leaves a new dataset with roughly the prior it already
has, and the swarm explores nothing. Set it explicitly.

Two settings need different values than a full swarm: use
`min_source_strength: 1` / `max_source_strength: 20` (low strengths make
degenerate mixes when there are only a few sources), and keep
`minimum_source_weight` at `0.0001`. The frozen block's weight is split among
its members, so a coarse grid rounds the smaller ones to zero.

#### Checking a swarm before spending compute

```bash
$OLMIX_PYTHON -c "
import json
v = json.load(open('data/runs/<run>/variants.json'))
print(list(v[0]['mix']))                       # leaf names, alphabetical
print(sorted(round(x['mix']['<dataset>']['weight'], 4) for x in v))
"
```

Weights should span a range that brackets the value you are questioning. If
every variant sits near the same number, the prior or the strengths need
adjusting.

### 3. Train

```bash
./launch_all_swarms.sh
```

This submits one array task per mixture to `gpumedium`, each on one GH200. Each
task reads `data/runs/<run>/mixes/<variant>.txt` and writes to
`$OUTPUT_ROOT/<variant>/` (checkpoints, TensorBoard, W&B, data index cache).
Model, optimizer and token budget are in `scripts/train-0.05B.sh`; the global
batch is 64 × 2,048 tokens and `TRAIN_TOKENS` defaults to 3B.

Overrides, set in the environment when submitting:

| Variable | Use |
| --- | --- |
| `TRAIN_TOKENS` | Shorter runs for tests. Must give more than the 500 warmup iterations; use a separate `OUTPUT_ROOT`, because a run resumes from its checkpoints. |
| `EXIT_AFTER_ITERS` | Stop (and save) after N iterations while keeping the full schedule; for benchmarks. |
| `MICRO_BATCH_SIZE` | Default 16; must divide 64. Changes speed and memory only. |

A variant that already reached its last iteration exits at once, so the array
can be resubmitted to fill in failed tasks. Requeued tasks append to their logs.

### 4. Convert to HuggingFace

When the training array has finished:

```bash
sbatch scripts/convert_to_hf.sh <run>
```

This converts every variant's final checkpoint to
`$OUTPUT_ROOT/hf_checkpoints/<variant>/<variant>_hf_iter_<N>`, one after
another in one job. It refuses unfinished checkpoints, skips variants already
converted, and fails at the end if any variant failed.

`megatron_to_hf.py` maps the weights directly to a Qwen3 model, without
Megatron-Bridge. It reads every architecture setting from the training
arguments saved in the checkpoint and rejects configurations it does not
support. The vocabulary keeps Megatron's padding (262,272 rows), so the
probabilities match training. On LUMI, the Bridge template set the
normalization epsilon to 1e-6, while the proxies trained with 1e-5; the
converter now uses the trained value.

### 5. Evaluate

```bash
models=(output/runs/hf_checkpoints/<run>-*/*_hf_iter_*)
for ((i = 0; i < ${#models[@]}; i += 8)); do
    sbatch scripts/eval_bpb.sh hun "${models[@]:i:8}"
done
```

`eval_bpb.sh <deu|spa|hun> <model> [...]` evaluates its models one after
another on one GPU, 5-shot, on the language's six BPB tasks: Okapi HellaSwag,
ARC and MMLU, Belebele, Global PIQA and INCLUDE. Results go to
`$EVAL_OUTPUT_DIR/<language>/<model name>/`. The batch of 8 above is a guess:
the time per model has not been measured yet. Time one full evaluation before
choosing. The tasks, their few-shot examples and their origin are described in
[`eval_tasks/README.md`](eval_tasks/README.md).

### 6. Collect

```bash
$OLMIX_PYTHON scripts/collect_bpb.py \
    --results-dir output/eval/hun \
    --variants-json data/runs/<run>/variants.json \
    --output-dir data/runs/<run>/olmix_fit
```

This writes `ratios.csv` (mixture weights) and `metrics.csv` (one BPB column per
task) for the variants it finds, matching each result to a variant by the model
path recorded in it. It warns about results that match no variant (such as the
smoke-test models), variants without results, and implausible values, and it
stops if a variant has results from two different checkpoints.

Columns with "mmlu" in their name are written with "MMLU" (for example
`okapi_MMLU_hu_bpb`). `olmix fit` would otherwise apply its English-MMLU
aggregation to them and stop with an error.

### 7. Fit

```bash
$OLMIX_PYTHON -m olmix.cli fit --config data/runs/<run>/fit.yaml --output-dir data/runs/<run>/fit
```

`fit.yaml` comes from step 2, with the priors filled in from the same token
counts the sampler used; its CSV paths point at step 6's output. Edit the
regression, proposer and constraints for the experiment:

- `regression.aggregate_task_families` must stay `false` unless you also supply
  an eval config defining the families.
- `constraints.enabled: true` caps each weight by its token budget
  (`weight × target_tokens ≤ tokens × repetition_factor`). Set `target_tokens`
  to the size of the real run you are choosing a mixture for, not the proxy's.
- `proposer.kl_reg` pulls the proposal towards the prior. On LUMI, olmix's usual
  0.1 proposed more English than any swarm model had seen; 0.02 kept every
  source inside the range the swarm covered.

To check that the fit works before real results exist,
`scripts/make_fake_fit_inputs.py` writes fake scores that follow a known rule.

There is no batch script for the fit yet. A small fit runs on the login node;
for a large one, use a short GPU job.

#### Reading the result of a reuse swarm

Only the datasets you left *out* of the frozen file carry information.
Everything inside the frozen block moves in fixed proportion across every
variant, so the regression cannot tell which member caused a change. Their
individual weights in the output come from the KL term pulling toward the prior,
not from evidence. Read the varied datasets and the block total, and treat the
rest as unchanged.

This also sets the size of an experiment: one unfrozen dataset answers one
question. To compare several candidates, unfreeze them all so they vary
independently, and expect to need more variants (olmix used 16 for a two-source
round, 64 for six).

## Tests

```bash
$OLMIX_PYTHON -m pytest tests                  # everything that needs no data
$OLMIX_PYTHON -m pytest tests -m data          # also check the configs' .bin files exist
```

`tests/test_scripts/test_megatron_indexed.py` compares the dataset reader with
Megatron's own; that part is skipped unless a Megatron-LM checkout is on
`PYTHONPATH`.

## Known issues and to do

- `scripts/run_pipeline.sh` and `scripts/pipeline/*` still use the LUMI
  conversion and oellm-eval. Run steps 4–7 by hand until they are ported.
- `scripts/generate_variants.py` counts tokens assuming 4 bytes per token
  (right only for vocabularies over 65k, such as the 256k tokenizer), and
  treats a missing `.bin` file as 0 tokens with only a warning.
  `scripts/megatron_indexed.count_tokens` reads the count from the `.idx` file
  instead.
- The configs in `configs/` point at LUMI paths, and `configs/config.yaml` at
  `data/stage_1_optimal.json`, which is not in this repository.
- `requirements.txt` is a freeze of a LUMI environment and is not used; the
  eval venv's requirements are in `hpc/roihu/eval_venv_requirements.txt`.
- The proxies have 2,048 positions, and lm-eval cuts longer prompts from the
  left. Check whether 5-shot Belebele prompts are cut on the first real models.
- Megatron's fused cross-entropy (`--cross-entropy-loss-fusion`) might allow
  micro batch 32 and faster training, but Megatron marks it as
  non-deterministic. Untested.
- The BPB tasks report accuracy too. Comparing it with the upstream lm-eval
  tasks would confirm each task scores the correct answer; this check existed
  on LUMI and has not been ported.

## Notes

The generator calls olmix's `generate_weights_dirichlet` directly rather than
its CLI, because the CLI assumes AI2's launcher. That function returns weights
ordered by source name, so `olmix_leaf_order()` builds the priors in that same
order. Without it, every dataset receives another dataset's weight. Pin the
olmix version when upgrading, and re-check that ordering.

The original LUMI `nested-swarm` results predate that fix: each dataset's prior
was applied to a different dataset. Its mixes and variants file agree with each
other, so the fit describes what was trained, but the exploration was not as
intended.
