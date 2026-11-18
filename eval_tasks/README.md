# BPB evaluation tasks for lm-evaluation-harness

`bpb/` holds lm-eval task definitions that score **bits per byte (BPB) of the
correct answer**, the fit target for the Olmix swarms. Accuracy is near chance
for 30M-parameter proxy models; BPB still moves smoothly with the data mixture.
`bpb/utils.py` explains how the scoring works.

They are plain lm-eval tasks, loaded with `--include_path`; nothing from
oellm-eval needs to be installed.

## Running

`scripts/eval_bpb.sh` runs one language's group on one or more models in the
eval venv, **5-shot** (`EVAL_NUM_FEWSHOT` in `env.sh`):

```bash
sbatch scripts/eval_bpb.sh hun /abs/path/to/hf_model [...]
LIMIT=20 sbatch scripts/eval_bpb.sh hun HuggingFaceTB/SmolLM2-135M   # quick test
```

It amounts to:

```bash
lm-eval run --model hf \
    --model_args pretrained=/abs/path/to/hf_model,dtype=bfloat16 \
    --include_path eval_tasks/bpb \
    --tasks olmix_bpb_hun \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path /abs/path/to/results
```

Few-shot examples are the first five documents of each task's
`fewshot_split` (`sampler: first_n`), so every model sees the same prompts:

| Task | Evaluated on | Examples from |
|---|---|---|
| Okapi HellaSwag | `val` (declared as test split) | `val`, minus the scored document |
| Okapi ARC, Okapi MMLU | `test` | `train` |
| Belebele, Global PIQA | `test` | `test`, minus the scored document |
| INCLUDE | `test` | `validation` |

lm-eval leaves the scored document out of its own examples only when the
few-shot split is the test split, so the Okapi HellaSwag task declares its
only split, `val`, as the test split.

One group per swarm language, with the same six benchmarks in each:

| Group | HellaSwag (Okapi) | ARC-Challenge (Okapi) | MMLU (Okapi) | Belebele | Global PIQA (parallel) | INCLUDE |
|---|---|---|---|---|---|---|
| `olmix_bpb_deu` | `de` | `de` | `de` | `deu_Latn` | `deu_latn` | German |
| `olmix_bpb_spa` | `es` | `es` | `es` | `spa_Latn` | `spa_latn_spai` (Spain) | Spanish |
| `olmix_bpb_hun` | `hu` | `hu` | `hu` | `hun_Latn` | `hun_latn` | Hungarian |

The groups report each task separately and have no group average: olmix fits
each metric on its own. All tasks are in cloze form: the scored continuation is
the answer text, never a letter. Every task also reports `acc` and `acc_norm`.

The other tasks in `bpb/` (English benchmarks, Global-MMLU, FLORES, XCOPA, ...)
are kept as they came, so they can be run as extra checks.

## Where the files come from

`bpb/` is a copy of `oellm/resources/custom_lm_eval_tasks/bpb` from the
`bpb-metrics` branch of <https://github.com/nicher92/oellm-eval>, commit
`e8e3f35` (2026-10-14), the BPB tasks used for the LUMI swarms. It's copied
rather than checked out so that the task definitions stay fixed for a swarm.

Changes from that commit:

- `utils._score` raises `ValueError` on a malformed document. Upstream, it
  returned `(0.0, 1)`: a zero-loss byte that silently lowers the corpus BPB.
- `_include_bpb_template_yaml` and `_global_mmlu_bpb_template_yaml` use the
  current dataset names `CohereLabs/...`. The old `CohereForAI/...` names
  only redirect.
- `_include_bpb_template_yaml` takes few-shot examples from the `validation`
  split; the original named no few-shot split.
- New tasks: `okapi_hellaswag/`, `okapi_arc/`, `okapi_mmlu/` and
  `global_piqa/` (German, Spanish, Hungarian), their `_*_bpb_template_yaml`
  templates, their functions at the end of `utils.py`, and the
  `olmix_bpb_{deu,spa,hun}.yaml` groups.

`tests/test_eval_tasks/test_bpb_tasks.py` checks the new scoring functions,
that every task file's `include` and `!function` references resolve, and that
every task in the three groups names its few-shot split. It does not need
lm-eval.

## Checked on Roihu

- Prompt length (`scripts/check_prompt_lengths.py`, 2026-11-18): no 5-shot
  prompt of the 18 group tasks is cut to the proxies' 2,048 positions.
  The longest is 1,923 tokens (`okapi_mmlu_hu_bpb`); Belebele's longest is 1,272.

  | Documents per task | de | es | hu |
  |---|---:|---:|---:|
  | Okapi HellaSwag | 9,368 | 9,374 | 9,119 |
  | Okapi ARC | 1,169 | 1,170 | 1,168 |
  | Okapi MMLU | 13,258 | 13,334 | 13,024 |
  | Belebele | 900 | 900 | 900 |
  | Global PIQA | 103 | 103 | 103 |
  | INCLUDE | 139 | 550 | 550 |

  The small tasks (Global PIQA, German INCLUDE) give noisier BPB.
- A full 5-shot run of `olmix_bpb_hun` on a converted proxy: 7.5 minutes,
  every task reports `bits_per_byte`.

## Not yet checked

- The tasks outside the three groups (English benchmarks, Global-MMLU, FLORES,
  ...) have not been run on Roihu. Global-MMLU may need an accepted licence on
  the Hub (then set `HF_TOKEN`). FLORES needs `datasets` below 4, because
  `facebook/flores` uses a loading script; the eval venv has 3.6.
