# smoke-wikipesto

A one-variant "swarm" for testing `scripts/train-0.05B.sh` on Roihu before the
real data exists. The single mixture is all of `wikipesto`
(`roihu-megatron-gpt/download_data.sh`), which was tokenized with the
GPT-NeoX tokenizer. The run uses the 256k tokenizer (env.sh) anyway, so that
the output layer and the throughput match the real proxies; the loss means
nothing for language quality.

From the repository root:

    TRAIN_TOKENS=200_000_000 OUTPUT_ROOT=$PWD/output/tests/smoke \
        sbatch --array=0 --export=ALL,MIX_PREFIX=smoke-wikipesto scripts/train-0.05B.sh

The run of 2026-11-18 (job 3198162) is kept in `output/tests/smoke` as the
input for testing the HuggingFace conversion. Megatron logged
"validation loss at iteration 1526 on validation set | lm loss value:
3.913925E+00" for it.

`variants.json` (one variant, all wikipesto) lets `scripts/run_pipeline.sh`
run on this test swarm. With the conversion and evaluation above already done,
both stages skip the model and only the collection does real work:

    OUTPUT_ROOT=$PWD/output/tests/smoke EVAL_OUTPUT_DIR=$PWD/output/tests/smoke/eval \
        ./scripts/run_pipeline.sh smoke-wikipesto hun
