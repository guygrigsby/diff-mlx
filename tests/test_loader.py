import numpy as np
from pathlib import Path
from data.loader import ShardLoader, sample_batch


def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 100_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = np.arange(n_tokens, dtype=np.uint32) % 100_277
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:10_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 100000, '
        '"val_token_count": 10000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir


def test_loader_reads_uint32_shards(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    assert loader.total_tokens == 100_000
    arr = loader.read(0, 100)
    assert arr.dtype == np.uint32
    assert len(arr) == 100
    assert arr[0] == 0 and arr[99] == 99


def test_sample_batch_returns_correct_shape(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    x, y = sample_batch(loader, block_size=128, micro_batch=4, rng=np.random.default_rng(0))
    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    np.testing.assert_array_equal(y[0, :-1], x[0, 1:])


def test_sample_batch_determinism(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    x1, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng1)
    x2, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng2)
    np.testing.assert_array_equal(x1, x2)


def test_val_split_deterministic(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader_1 = ShardLoader(shards_dir, split="val")
    val_loader_2 = ShardLoader(shards_dir, split="val")
    assert val_loader_1.total_tokens == val_loader_2.total_tokens == 10_000
    np.testing.assert_array_equal(val_loader_1.read(0, 1000), val_loader_2.read(0, 1000))
