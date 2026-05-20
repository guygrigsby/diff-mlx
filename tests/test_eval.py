import numpy as np
import mlx.core as mx
from pathlib import Path
from eval import compute_val_loss
from model import Transformer
from config import ModelConfig
from data.loader import ShardLoader


def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 50_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = (np.arange(n_tokens, dtype=np.uint32)) % 100_277
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:5_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 50000, '
        '"val_token_count": 5000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir


def test_compute_val_loss_returns_finite_float(tmp_path):
    mx.random.seed(0)
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=2, qk_head_dim=32,
        vocab_size=100_277, mlp_intermediate=128, block_size=64,
    )
    model = Transformer(cfg)
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader = ShardLoader(shards_dir, split="val")
    loss = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    assert isinstance(loss, float)
    assert loss > 0
    assert np.isfinite(loss)


def test_eval_is_deterministic(tmp_path):
    mx.random.seed(0)
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=2, qk_head_dim=32,
        vocab_size=100_277, mlp_intermediate=128, block_size=64,
    )
    model = Transformer(cfg)
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader = ShardLoader(shards_dir, split="val")
    loss1 = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    loss2 = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    assert loss1 == loss2
