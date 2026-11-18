#!/usr/bin/env python3
"""Count how many evaluation prompts lm-eval would cut to fit a model's context.

lm-eval scores ``context + continuation`` and, when that is longer than the
model's maximum length, cuts tokens from the left of the context
(``HFLM._loglikelihood_tokens``). With 5-shot prompts and the proxies' 2,048
positions, tasks with long passages (Belebele) may lose part of their first
examples. That changes the task, and differently for different languages
because their tokens per word differ.

This builds every request exactly as an evaluation would (the same task
configs, few-shot count and seed) and encodes it with the same model wrapper
and tokenizer, but runs no model. It reports, per task: requests, documents,
the length distribution, and how many requests and documents would be cut.

Run in the eval venv, e.g. in a short GPU job (the venv is built for the GPU
nodes; no GPU is used):

    sbatch -A project_2020507 -p gputest -t 00:15:00 --gres=gpu:gh200:1 -o logs/prompt-lengths_%j.out \\
        --wrap "hpc/roihu/eval_python.sh scripts/check_prompt_lengths.py \\
                --model <hf model dir> --tasks olmix_bpb_deu olmix_bpb_spa olmix_bpb_hun"
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("check_prompt_lengths")
REPO_ROOT = Path(__file__).resolve().parents[1]
# lm-eval's default few-shot seed (lm_eval/defaults.py DEFAULT_OTHER_SEED).
FEWSHOT_SEED = 1234


@dataclass
class TaskLengths:
    """Prompt-length statistics for one task.

    Attributes:
        task: task name.
        requests: scored (context, continuation) pairs; one per answer choice.
        documents: evaluated documents.
        max_length: the model's maximum length.
        median_tokens, p95_tokens, max_tokens: of context + continuation.
        cut_requests: requests longer than ``max_length + 1`` tokens.
        cut_documents: documents with at least one cut request.
        mean_tokens_cut: average tokens removed from a cut request.
    """

    task: str
    requests: int
    documents: int
    max_length: int
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    cut_requests: int
    cut_documents: int
    mean_tokens_cut: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model directory or Hub ID (gives tokenizer and max length).")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task or group names.")
    parser.add_argument("--task-dir", type=Path, default=REPO_ROOT / "eval_tasks" / "bpb", help="lm-eval include path.")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Documents per task (default: all).")
    parser.add_argument("--output-json", type=Path, default=None, help="Also write the statistics here.")
    return parser.parse_args()


def flatten_tasks(task_dict: dict[Any, Any]) -> dict[str, Any]:
    """Task name -> Task object, from lm-eval's possibly nested group dict."""
    from lm_eval.api.task import Task

    tasks: dict[str, Any] = {}
    for key, value in task_dict.items():
        if isinstance(value, dict):
            tasks.update(flatten_tasks(value))
        elif isinstance(value, Task):
            tasks[value.config.task if hasattr(value, "config") else str(key)] = value
        else:
            raise TypeError(f"unexpected entry {key!r}: {type(value).__name__}")
    return tasks


def measure(task: Any, name: str, lm: Any, num_fewshot: int, limit: int | None) -> TaskLengths:
    """Build one task's requests and measure their encoded lengths.

    Args:
        task: the lm-eval Task.
        name: its name, for the report.
        lm: an lm-eval HFLM (only its tokenization and max length are used).
        num_fewshot: few-shot examples per prompt.
        limit: documents to use, or None for all.

    Returns:
        The task's statistics.

    Raises:
        ValueError: if the task produces requests this check does not handle.
    """
    task.set_config(key="num_fewshot", value=num_fewshot)
    task.set_fewshot_seed(seed=FEWSHOT_SEED)
    task.build_all_requests(limit=limit, rank=0, world_size=1)

    lengths: list[int] = []
    cut_docs: set[int] = set()
    docs: set[int] = set()
    cuts: list[int] = []
    for instance in task.instances:
        if instance.request_type not in ("loglikelihood", "multiple_choice"):
            raise ValueError(f"{name}: request type {instance.request_type} is not a loglikelihood request")
        context, continuation = instance.arguments
        if not context:
            raise ValueError(f"{name}: empty context (doc {instance.doc_id}); not handled by this check")
        context_enc, continuation_enc = lm._encode_pair(context, continuation)
        total = len(context_enc) + len(continuation_enc)
        lengths.append(total)
        docs.add(instance.doc_id)
        if total > lm.max_length + 1:
            cuts.append(total - (lm.max_length + 1))
            cut_docs.add(instance.doc_id)

    if not lengths:
        raise ValueError(f"{name}: no requests were built")
    array = np.asarray(lengths)
    return TaskLengths(
        task=name,
        requests=len(lengths),
        documents=len(docs),
        max_length=lm.max_length,
        median_tokens=float(np.median(array)),
        p95_tokens=float(np.percentile(array, 95)),
        max_tokens=int(array.max()),
        cut_requests=len(cuts),
        cut_documents=len(cut_docs),
        mean_tokens_cut=float(np.mean(cuts)) if cuts else 0.0,
    )


def main() -> None:
    """Measure every task and log a table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()

    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict

    lm = HFLM(pretrained=args.model, device="cpu", dtype="float32", batch_size=1)
    LOGGER.info("Model %s: max length %d", args.model, lm.max_length)
    manager = TaskManager(include_path=str(args.task_dir.resolve()))
    tasks = flatten_tasks(get_task_dict(args.tasks, manager))
    LOGGER.info("%d tasks: %s", len(tasks), ", ".join(sorted(tasks)))

    results = [measure(task, name, lm, args.num_fewshot, args.limit) for name, task in sorted(tasks.items())]

    header = f"{'task':40} {'docs':>7} {'median':>7} {'p95':>7} {'max':>7} {'cut req':>8} {'cut docs':>9} {'avg cut':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        share = f"{100 * r.cut_documents / r.documents:.1f}%"
        lines.append(
            f"{r.task:40} {r.documents:7d} {r.median_tokens:7.0f} {r.p95_tokens:7.0f} {r.max_tokens:7d} "
            f"{r.cut_requests:8d} {r.cut_documents:5d} {share:>3} {r.mean_tokens_cut:8.0f}"
        )
    LOGGER.info(
        "Prompt lengths (context + continuation, tokens) with %d-shot prompts; 'cut' = longer than %d + 1:\n%s",
        args.num_fewshot, lm.max_length, "\n".join(lines),
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
        LOGGER.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
