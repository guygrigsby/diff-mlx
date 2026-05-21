import json
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from data.tokenize import tokenize_parquet_to_shards, ShardMeta


def _write_tiny_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def test_tokenize_tiny_parquet_writes_uint32_shards(tmp_path):
    input_pq = tmp_path / "input.parquet"
    _write_tiny_parquet(input_pq, ["Hello world.", "Another doc here.", "Third."])
    out_dir = tmp_path / "shards"
    meta = tokenize_parquet_to_shards(
        [input_pq],
        out_dir=out_dir,
        train_shard_max_tokens=1024,
        val_tokens=10,
        val_doc_hash_mod=10,
        val_doc_hash_keep_below=2,
    )
    assert isinstance(meta, ShardMeta)
    assert meta.vocab_size == 100_277
    assert (out_dir / "val.bin").exists()
    assert (out_dir / "meta.json").exists()
    train_files = sorted(out_dir.glob("train-*.bin"))
    assert len(train_files) >= 1
    arr = np.memmap(train_files[0], dtype=np.uint32, mode="r")
    assert arr.dtype == np.uint32
    assert len(arr) > 0
    assert arr.max() < 100_277


def test_meta_json_has_expected_keys(tmp_path):
    input_pq = tmp_path / "input.parquet"
    _write_tiny_parquet(input_pq, ["text"] * 50)
    out_dir = tmp_path / "shards"
    tokenize_parquet_to_shards([input_pq], out_dir=out_dir, train_shard_max_tokens=512, val_tokens=20)
    meta = json.loads((out_dir / "meta.json").read_text())
    for key in ["vocab_size", "eot_id", "tiktoken_version", "tokenizer_name",
                "train_token_count", "val_token_count", "n_train_shards", "source_files"]:
        assert key in meta, f"missing key: {key}"
