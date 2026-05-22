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


def test_grad_accum_matches_full_batch():
    """grad_accum=2 with micro_batch=B should yield the same update as
    grad_accum=1 with micro_batch=2B (within numerical tolerance), when the
    accumulator divides by N to average.
    """
    import mlx.nn as nn
    from config import ModelConfig
    from model import Transformer
    from optim import make_adamw
    from train_step import train_step, train_step_with_accum

    mx.random.seed(0)
    cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=4, qk_head_dim=8,
        vocab_size=128, mlp_intermediate=64, block_size=16,
    )

    # Reference: one update on a batch of size 4 (=2*2).
    m_ref = Transformer(cfg, variant="vanilla")
    mx.eval(m_ref.parameters())
    opt_ref = make_adamw(lr=1e-3, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    x_full = mx.random.randint(0, 128, shape=(4, 16))
    y_full = mx.random.randint(0, 128, shape=(4, 16))
    train_step(m_ref, opt_ref, x_full, y_full, grad_clip=1e9)

    # Accumulated: two updates of size 2, averaged.
    mx.random.seed(0)
    m_acc = Transformer(cfg, variant="vanilla")
    mx.eval(m_acc.parameters())
    opt_acc = make_adamw(lr=1e-3, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    train_step_with_accum(
        m_acc, opt_acc,
        [(x_full[:2], y_full[:2]), (x_full[2:], y_full[2:])],
        grad_clip=1e9,
    )

    # Sanity: parameters should be very close (tolerance for accumulation order).
    def first_leaf(p):
        if isinstance(p, dict):
            for v in p.values():
                r = first_leaf(v)
                if r is not None:
                    return r
        elif isinstance(p, list):
            for v in p:
                r = first_leaf(v)
                if r is not None:
                    return r
        elif isinstance(p, mx.array):
            return p
        return None

    p_ref = first_leaf(m_ref.parameters())
    p_acc = first_leaf(m_acc.parameters())
    diff = float(mx.max(mx.abs(p_ref - p_acc)).item())
    assert diff < 1e-4, f"grad accum diverged from full-batch: max |Δ| = {diff:.3e}"
