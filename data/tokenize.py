"""Tokenize FineWeb-Edu parquet shards into uint32 mmap-able binaries.

Writes:
- data/shards/train-NNNN.bin    (uint32 token streams, mmap-able)
- data/shards/val.bin           (deterministic val subset)
- data/shards/meta.json         (vocab, version pins, counts)

Val selection is deterministic on a hash of (file_index, doc_index): any doc
whose hash % val_doc_hash_mod < val_doc_hash_keep_below goes to val. With
mod=100 and keep_below=1, ~1% of docs go to val.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import struct

import numpy as np
import pyarrow.parquet as pq

from data.tokenizer import get_tokenizer, EOT_ID, VOCAB_SIZE, tiktoken_version


@dataclass
class ShardMeta:
    vocab_size: int
    eot_id: int
    tiktoken_version: str
    tokenizer_name: str
    train_token_count: int
    val_token_count: int
    n_train_shards: int
    source_files: list[str]


def _doc_hash(file_idx: int, doc_idx: int) -> int:
    h = hashlib.sha256(f"{file_idx}:{doc_idx}".encode()).digest()
    return struct.unpack("<Q", h[:8])[0]


def tokenize_parquet_to_shards(
    input_files: list[Path],
    out_dir: Path,
    train_shard_max_tokens: int = 100_000_000,
    val_tokens: int = 100_000_000,
    val_doc_hash_mod: int = 100,
    val_doc_hash_keep_below: int = 1,
) -> ShardMeta:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = get_tokenizer()

    val_buf: list[np.ndarray] = []
    val_total = 0
    train_buf: list[np.ndarray] = []
    train_buf_tokens = 0
    train_total = 0
    shard_idx = 0

    def flush_train_shard():
        nonlocal shard_idx, train_buf, train_buf_tokens
        if not train_buf:
            return
        out = np.concatenate(train_buf)
        path = out_dir / f"train-{shard_idx:04d}.bin"
        out.astype(np.uint32).tofile(path)
        shard_idx += 1
        train_buf = []
        train_buf_tokens = 0

    for file_idx, p in enumerate(input_files):
        table = pq.read_table(p, columns=["text"])
        texts = table.column("text").to_pylist()
        for doc_idx, text in enumerate(texts):
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(EOT_ID)
            arr = np.asarray(ids, dtype=np.uint32)
            h = _doc_hash(file_idx, doc_idx)
            is_val = (val_total < val_tokens) and (h % val_doc_hash_mod) < val_doc_hash_keep_below
            if is_val:
                val_buf.append(arr)
                val_total += len(arr)
            else:
                train_buf.append(arr)
                train_buf_tokens += len(arr)
                train_total += len(arr)
                if train_buf_tokens >= train_shard_max_tokens:
                    flush_train_shard()

    flush_train_shard()

    if val_buf:
        val_out = np.concatenate(val_buf).astype(np.uint32)
        val_out.tofile(out_dir / "val.bin")
    else:
        np.zeros(0, dtype=np.uint32).tofile(out_dir / "val.bin")

    meta = ShardMeta(
        vocab_size=VOCAB_SIZE,
        eot_id=EOT_ID,
        tiktoken_version=tiktoken_version(),
        tokenizer_name="cl100k_base",
        train_token_count=train_total,
        val_token_count=val_total,
        n_train_shards=shard_idx,
        source_files=[str(p) for p in input_files],
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob", type=str, default="data/raw/*.parquet")
    p.add_argument("--out_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--train_shard_max_tokens", type=int, default=100_000_000)
    p.add_argument("--val_tokens", type=int, default=75_000_000)
    p.add_argument("--val_doc_hash_mod", type=int, default=100)
    p.add_argument("--val_doc_hash_keep_below", type=int, default=1)
    args = p.parse_args()
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise SystemExit(f"No input files matched {args.input_glob}")
    meta = tokenize_parquet_to_shards(
        files, args.out_dir,
        args.train_shard_max_tokens, args.val_tokens,
        args.val_doc_hash_mod, args.val_doc_hash_keep_below,
    )
    print(f"Train tokens: {meta.train_token_count:,} in {meta.n_train_shards} shards")
    print(f"Val tokens:   {meta.val_token_count:,}")


if __name__ == "__main__":
    main()
