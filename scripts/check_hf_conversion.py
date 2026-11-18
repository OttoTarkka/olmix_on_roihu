#!/usr/bin/env python3
"""Check a converted HF model against the validation loss Megatron logged for its checkpoint.

Computes the HF model's language-modelling loss on the validation part of a
Megatron dataset, the same way Megatron does (windows of ``seq_length + 1``
tokens across document boundaries; ``seq_length`` predictions per window,
softmax over the padded vocabulary), and compares it with the value from the
training log ("validation loss at iteration N ... lm loss value: X").

Megatron evaluates shuffled samples while this reads evenly spaced windows of
the same validation tokens, so the two agree only within sampling noise; a
wrong conversion is off by whole units (a model with its gate and up
projections swapped scored 6.8 instead of 4.2 in a local test).

Only single-prefix data is supported, so use it with a model trained on one
dataset (e.g. the smoke-wikipesto run), not on a blended mixture.

Usage:
    python scripts/check_hf_conversion.py --model-dir <hf_dir> --data-prefix <prefix> --expected-loss 3.9139
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from megatron_indexed import open_dataset, split_range  # noqa: E402

LOGGER = logging.getLogger("check_hf_conversion")
SPLIT_PARTS = {"train": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="Converted HF model directory.")
    parser.add_argument("--data-prefix", type=Path, required=True, help="Megatron dataset prefix (no .bin/.idx).")
    parser.add_argument("--expected-loss", type=float, required=True, help="Megatron's logged validation loss.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Allowed |difference| in nats per token.")
    parser.add_argument("--split", default="99,1,0", help="The --split the model was trained with.")
    parser.add_argument("--part", choices=sorted(SPLIT_PARTS), default="validation")
    parser.add_argument("--num-windows", type=int, default=512, help="Windows to score, spread evenly.")
    parser.add_argument("--seq-length", type=int, default=None, help="Default: the model's max_position_embeddings.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def contiguous_tokens(prefix: Path, split: str, part: str) -> np.ndarray:
    """The tokens of one split part, as one contiguous slice of the .bin file.

    Args:
        prefix: dataset prefix.
        split: Megatron ``--split`` string.
        part: train, validation or test.

    Returns:
        A memory-mapped token array.

    Raises:
        ValueError: if the part is empty or its sequences are not stored back to back.
    """
    dataset = open_dataset(prefix)
    indices = split_range(dataset.num_sequences, split, SPLIT_PARTS[part])
    if len(indices) == 0:
        raise ValueError(f"the {part} part of split {split} is empty for {prefix}")
    lengths = dataset.sequence_lengths[indices.start : indices.stop].astype(np.int64)
    pointers = dataset.sequence_pointers[indices.start : indices.stop]
    item = dataset.dtype.itemsize
    if not np.array_equal(pointers[1:], pointers[:-1] + lengths[:-1] * item):
        raise ValueError(f"the {part} sequences of {prefix} are not stored back to back")
    start = int(pointers[0]) // item
    LOGGER.info(
        "%s: %s sequences %d-%d of %d, %d tokens",
        prefix, part, indices.start, indices.stop - 1, dataset.num_sequences, int(lengths.sum()),
    )
    return dataset.tokens[start : start + int(lengths.sum())]


def window_starts(num_tokens: int, window: int, num_windows: int) -> np.ndarray:
    """Evenly spaced start offsets of ``num_windows`` windows of ``window`` tokens."""
    if num_tokens < window:
        raise ValueError(f"only {num_tokens} tokens, fewer than one window of {window}")
    count = min(num_windows, num_tokens // window)
    return np.linspace(0, num_tokens - window, count).astype(np.int64)


@torch.no_grad()
def mean_loss(model: torch.nn.Module, tokens: np.ndarray, starts: np.ndarray, seq_length: int, batch_size: int,
              device: str) -> float:
    """Token-averaged cross-entropy over the windows, as Megatron computes it.

    Each window has ``seq_length + 1`` tokens: the first ``seq_length`` are the
    input and the last ``seq_length`` the targets.
    """
    total_loss = 0.0
    total_tokens = 0
    for first in range(0, len(starts), batch_size):
        batch = np.stack([np.asarray(tokens[s : s + seq_length + 1], dtype=np.int64) for s in starts[first : first + batch_size]])
        batch_tensor = torch.from_numpy(batch).to(device)
        inputs, targets = batch_tensor[:, :-1], batch_tensor[:, 1:]
        logits = model(input_ids=inputs).logits.float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def main() -> None:
    """Score the model and compare with the expected loss; exit 1 if outside the tolerance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(args.device).eval()
    seq_length = args.seq_length or model.config.max_position_embeddings
    tokens = contiguous_tokens(args.data_prefix, args.split, args.part)
    if int(tokens.max()) >= model.config.vocab_size:
        raise ValueError(f"data has token id {int(tokens.max())}, beyond the model's vocabulary {model.config.vocab_size}")
    starts = window_starts(len(tokens), seq_length + 1, args.num_windows)
    loss = mean_loss(model, tokens, starts, seq_length, args.batch_size, args.device)
    difference = loss - args.expected_loss
    LOGGER.info(
        "HF loss %.4f on %d windows of %d tokens; Megatron %.4f; difference %+.4f (tolerance %.4f)",
        loss, len(starts), seq_length, args.expected_loss, difference, args.tolerance,
    )
    if abs(difference) > args.tolerance:
        LOGGER.error("FAILED: the converted model's loss differs from Megatron's by more than the tolerance")
        sys.exit(1)
    LOGGER.info("PASSED")


if __name__ == "__main__":
    main()
