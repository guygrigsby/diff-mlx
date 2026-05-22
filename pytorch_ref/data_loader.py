"""Mmap-based deterministic loader. Mirrors ../data/loader.py.

Reads uint32 token shards (train-NNN.bin) and a single val.bin.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class ShardLoader:
    shards_dir: Path
    split: str  # "train" or "val"
    _shards: list = field(default=None, repr=False)
    _cumulative: np.ndarray = field(default=None, repr=False)
    total_tokens: int = 0

    def __post_init__(self):
        self.shards_dir = Path(self.shards_dir)
        meta = json.loads((self.shards_dir / "meta.json").read_text())
        if self.split == "train":
            paths = sorted(self.shards_dir.glob("train-*.bin"))
            assert paths, f"no train shards in {self.shards_dir}"
            self._shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        elif self.split == "val":
            p = self.shards_dir / "val.bin"
            assert p.exists(), f"missing {p}"
            self._shards = [np.memmap(p, dtype=np.uint32, mode="r")]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {self.split!r}")
        sizes = np.array([len(s) for s in self._shards], dtype=np.int64)
        self._cumulative = np.cumsum(sizes)
        self.total_tokens = int(self._cumulative[-1])

    def read(self, offset: int, n: int) -> np.ndarray:
        if offset + n > self.total_tokens:
            raise IndexError(f"read past end: offset={offset} n={n} total={self.total_tokens}")
        shard_idx = int(np.searchsorted(self._cumulative, offset, side="right"))
        local_offset = offset - (self._cumulative[shard_idx - 1] if shard_idx > 0 else 0)
        out_parts: list[np.ndarray] = []
        remaining = n
        while remaining > 0:
            shard = self._shards[shard_idx]
            take = min(remaining, len(shard) - local_offset)
            out_parts.append(np.asarray(shard[local_offset:local_offset + take]))
            remaining -= take
            shard_idx += 1
            local_offset = 0
        return np.concatenate(out_parts) if len(out_parts) > 1 else out_parts[0]


def sample_batch(
    loader: ShardLoader,
    block_size: int,
    micro_batch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample micro_batch windows of length (block_size + 1) and return (x, y) int64.

    PyTorch's nn.Embedding requires int64 (long) indices, so we use int64 here
    even though the MLX side uses int32. Same underlying token values.
    """
    max_offset = loader.total_tokens - block_size - 1
    if max_offset <= 0:
        raise ValueError(f"loader has {loader.total_tokens} tokens, need > {block_size + 1}")
    offsets = rng.integers(0, max_offset, size=micro_batch, dtype=np.int64)
    x = np.empty((micro_batch, block_size), dtype=np.int64)
    y = np.empty((micro_batch, block_size), dtype=np.int64)
    for i, off in enumerate(offsets):
        window = loader.read(int(off), block_size + 1)
        x[i] = window[:-1]
        y[i] = window[1:]
    return x, y
