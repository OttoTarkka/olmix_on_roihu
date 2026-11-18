"""Tests for scripts/megatron_indexed.py against Megatron's own dataset code.

The format tests write datasets with Megatron's IndexedDatasetBuilder and are
skipped when Megatron is not importable; run them with the Megatron-LM checkout
on PYTHONPATH (e.g. roihu-megatron-gpt/Megatron-LM). The split tests compare
with the formula in Megatron's blended_megatron_dataset_builder.py and need
nothing extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import megatron_indexed  # noqa: E402

SEQUENCES = [[5, 6, 7], [8], [9, 10, 11, 12, 13], [14, 15]]


def _megatron_indexed_dataset() -> object:
    try:
        from megatron.core.datasets import indexed_dataset
    except ImportError:
        pytest.skip("Megatron is not importable; put a Megatron-LM checkout on PYTHONPATH")
    return indexed_dataset


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
def test_reader_matches_megatron_builder(tmp_path: Path, dtype: type[np.generic]) -> None:
    indexed_dataset = _megatron_indexed_dataset()
    import torch

    prefix = tmp_path / "data"
    builder = indexed_dataset.IndexedDatasetBuilder(f"{prefix}.bin", dtype=dtype)
    builder.add_document(torch.tensor(SEQUENCES[0] + SEQUENCES[1]), [3, 1])  # one document, two sequences
    builder.add_item(torch.tensor(SEQUENCES[2]))
    builder.end_document()
    builder.add_item(torch.tensor(SEQUENCES[3]))
    builder.end_document()
    builder.finalize(f"{prefix}.idx")

    ours = megatron_indexed.open_dataset(prefix)
    theirs = indexed_dataset.IndexedDataset(str(prefix))
    assert ours.dtype == np.dtype(dtype)
    assert ours.num_sequences == len(theirs) == len(SEQUENCES)
    assert ours.num_tokens == sum(len(s) for s in SEQUENCES)
    assert megatron_indexed.count_tokens(prefix) == ours.num_tokens
    np.testing.assert_array_equal(ours.document_indices, theirs.document_indices)
    for index, expected in enumerate(SEQUENCES):
        np.testing.assert_array_equal(ours.sequence(index), expected)
        np.testing.assert_array_equal(ours.sequence(index), theirs[index])


def test_missing_and_corrupt_files_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        megatron_indexed.read_index(tmp_path / "absent")
    (tmp_path / "bad.idx").write_bytes(b"NOTANIDX" + bytes(40))
    with pytest.raises(ValueError, match="bad header"):
        megatron_indexed.read_index(tmp_path / "bad")


def _megatron_split(num_sequences: int, split: str, which: int) -> range:
    """Megatron's computation: normalise the weights, cumulate, round(bound * n)."""
    weights = [float(part) for part in split.split(",")]
    weights = [weight / sum(weights) for weight in weights]
    bounds = [0.0]
    for weight in weights:
        bounds.append(bounds[-1] + weight)
    return range(int(round(bounds[which] * float(num_sequences))), int(round(bounds[which + 1] * float(num_sequences))))


@pytest.mark.parametrize("num_sequences", [1, 99, 100, 101, 12345, 7_654_321])
@pytest.mark.parametrize("split", ["99,1,0", "969,30,1", "1,1,1"])
def test_split_range_matches_megatron(num_sequences: int, split: str) -> None:
    parts = [megatron_indexed.split_range(num_sequences, split, which) for which in range(3)]
    assert parts == [_megatron_split(num_sequences, split, which) for which in range(3)]
    assert parts[0].start == 0 and parts[2].stop == num_sequences
    assert parts[0].stop == parts[1].start and parts[1].stop == parts[2].start


def test_split_range_rejects_bad_split() -> None:
    with pytest.raises(ValueError):
        megatron_indexed.split_range(10, "99,1", 1)
    with pytest.raises(ValueError):
        megatron_indexed.split_range(10, "a,b,c", 1)
