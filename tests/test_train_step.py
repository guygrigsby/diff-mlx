import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from train_step import compute_loss_and_grads, train_step
from model import Transformer
from config import ModelConfig


def test_compute_loss_returns_scalar_fp32():
    mx.random.seed(0)
    cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
        vocab_size=100_277, mlp_intermediate=64, block_size=32,
    )
    model = Transformer(cfg)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 32), dtype=np.int32))
    y = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 32), dtype=np.int32))
    loss, _ = compute_loss_and_grads(model, x, y)
    assert loss.dtype == mx.float32
    assert loss.shape == ()
    assert mx.isfinite(loss).item()


def test_train_step_decreases_loss_on_one_batch_overfit():
    """Smoke: model can overfit a single (x, y) batch — proves backward works."""
    mx.random.seed(0)
    cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
        vocab_size=100_277, mlp_intermediate=64, block_size=16,
    )
    model = Transformer(cfg)
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 16), dtype=np.int32))
    y = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 16), dtype=np.int32))
    losses = []
    for _ in range(50):
        loss = train_step(model, opt, x, y, grad_clip=1.0)
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.5, f"did not overfit: start={losses[0]:.3f} end={losses[-1]:.3f}"
