"""Read Megatron-LM indexed datasets (``<prefix>.idx`` + ``<prefix>.bin``) with numpy only.

The format is that of ``megatron/core/datasets/indexed_dataset.py`` (Megatron
Core 0.16), version 1:

    .idx: b"MMIDIDX\\x00\\x00", version <Q (1), dtype code <B,
          sequence count <Q, document count <Q,
          sequence lengths   int32[sequence count],
          sequence pointers  int64[sequence count]   (byte offsets into .bin),
          document indices   int64[document count]
    .bin: the tokens of every sequence, back to back, in the dtype above.

Megatron splits GPT data by *sequence* index (``--split 99,1,0`` gives the
validation set sequences ``[round(0.99 N), N)``), so this module works in
sequences too. Nothing here depends on Megatron.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODES: dict[int, type[np.generic]] = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


@dataclass(frozen=True)
class IndexedDataset:
    """One Megatron indexed dataset, memory-mapped.

    Attributes:
        prefix: path without the ``.idx``/``.bin`` suffix.
        dtype: token dtype.
        sequence_lengths: tokens per sequence.
        sequence_pointers: byte offset of each sequence in ``.bin``.
        document_indices: sequence index at which each document starts (plus the end).
        tokens: the whole ``.bin`` file as one flat token array.
    """

    prefix: Path
    dtype: np.dtype
    sequence_lengths: np.ndarray
    sequence_pointers: np.ndarray
    document_indices: np.ndarray
    tokens: np.ndarray

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the dataset."""
        return int(self.sequence_lengths.shape[0])

    @property
    def num_tokens(self) -> int:
        """Total number of tokens over all sequences."""
        return int(self.sequence_lengths.sum(dtype=np.int64))

    def sequence(self, index: int) -> np.ndarray:
        """The tokens of one sequence (a view into the memory map)."""
        start = int(self.sequence_pointers[index]) // self.dtype.itemsize
        return self.tokens[start : start + int(self.sequence_lengths[index])]


def read_index(prefix: Path) -> tuple[np.dtype, np.ndarray, np.ndarray, np.ndarray]:
    """Read ``<prefix>.idx``.

    Args:
        prefix: dataset path without suffix.

    Returns:
        The token dtype, the sequence lengths, the sequence pointers and the
        document indices.

    Raises:
        FileNotFoundError: if the ``.idx`` file does not exist.
        ValueError: if the file is not a version-1 Megatron index or is truncated.
    """
    idx_path = Path(f"{prefix}.idx")
    if not idx_path.is_file():
        raise FileNotFoundError(f"no index file: {idx_path}")
    with idx_path.open("rb") as stream:
        header = stream.read(len(INDEX_HEADER))
        if header != INDEX_HEADER:
            raise ValueError(f"not a Megatron index (bad header {header!r}): {idx_path}")
        (version,) = struct.unpack("<Q", stream.read(8))
        if version != 1:
            raise ValueError(f"unsupported index version {version}: {idx_path}")
        (code,) = struct.unpack("<B", stream.read(1))
        if code not in DTYPE_CODES:
            raise ValueError(f"unknown dtype code {code}: {idx_path}")
        sequence_count, document_count = struct.unpack("<QQ", stream.read(16))
        offset = stream.tell()

    index = np.memmap(idx_path, mode="r", dtype=np.uint8)
    lengths_end = offset + 4 * sequence_count
    pointers_end = lengths_end + 8 * sequence_count
    documents_end = pointers_end + 8 * document_count
    if index.shape[0] < documents_end:
        raise ValueError(
            f"index is truncated: {idx_path} has {index.shape[0]} bytes, the header implies at least {documents_end}"
        )
    sequence_lengths = np.frombuffer(index, dtype=np.int32, count=sequence_count, offset=offset)
    sequence_pointers = np.frombuffer(index, dtype=np.int64, count=sequence_count, offset=lengths_end)
    document_indices = np.frombuffer(index, dtype=np.int64, count=document_count, offset=pointers_end)
    return np.dtype(DTYPE_CODES[code]), sequence_lengths, sequence_pointers, document_indices


def open_dataset(prefix: Path) -> IndexedDataset:
    """Memory-map ``<prefix>.idx`` and ``<prefix>.bin``.

    Args:
        prefix: dataset path without suffix.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: if either file is missing.
        ValueError: if the ``.bin`` size does not match what the index describes.
    """
    dtype, lengths, pointers, documents = read_index(prefix)
    bin_path = Path(f"{prefix}.bin")
    if not bin_path.is_file():
        raise FileNotFoundError(f"no data file: {bin_path}")
    tokens = np.memmap(bin_path, mode="r", dtype=dtype)
    if lengths.shape[0]:
        end = int(pointers[-1]) // dtype.itemsize + int(lengths[-1])
        if end > tokens.shape[0]:
            raise ValueError(f"{bin_path} holds {tokens.shape[0]} tokens, the index needs {end}")
    return IndexedDataset(Path(prefix), dtype, lengths, pointers, documents, tokens)


def count_tokens(prefix: Path) -> int:
    """Number of tokens in a dataset, from its index alone.

    Args:
        prefix: dataset path without suffix.

    Returns:
        The total token count over all sequences.
    """
    _, lengths, _, _ = read_index(prefix)
    return int(lengths.sum(dtype=np.int64))


def split_range(num_sequences: int, split: str, which: int) -> range:
    """Sequence indices of one part of a Megatron ``--split``, as Megatron computes them.

    Args:
        num_sequences: number of sequences in the dataset.
        split: the ``--split`` string, e.g. ``"99,1,0"``.
        which: 0 for train, 1 for validation, 2 for test.

    Returns:
        The range of sequence indices in that part.

    Raises:
        ValueError: if the split string is malformed.
    """
    try:
        weights = [float(part) for part in split.split(",")]
    except ValueError as error:
        raise ValueError(f"bad split {split!r}") from error
    if len(weights) != 3 or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"split must be three non-negative numbers, got {split!r}")
    total = sum(weights)
    bounds = [0.0]
    for weight in weights:
        bounds.append(bounds[-1] + weight / total)
    # Megatron (blended_megatron_dataset_builder.py): int(round(bound * num_elements)).
    start = int(round(bounds[which] * float(num_sequences)))
    end = int(round(bounds[which + 1] * float(num_sequences)))
    return range(start, end)
