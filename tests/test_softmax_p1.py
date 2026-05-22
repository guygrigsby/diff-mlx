"""P1 softmax kernel acceptance tests per design §5.1.

Forward: matches mx.softmax(x, axis=-1) within 1e-4 fp32 / 1e-2 bf16.
Backward: matches autograd of pure-MLX softmax + finite-difference grad check.
"""
import mlx.core as mx
import numpy as np
import pytest

from kernels.softmax_p1 import softmax_p1, softmax_p1_with_backward


# ---------- Forward ----------

@pytest.mark.parametrize("shape", [
    (8,),
    (16, 32),
    (4, 8, 16),
    (2, 16, 64),
    (4, 8, 1024),
    (1, 1, 2048),
    (32, 12, 2048),  # Stage 1 vanilla attention shape
])
def test_softmax_p1_forward_fp32_matches_reference(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape, dtype=mx.float32)
    y = softmax_p1(x)
    ref = mx.softmax(x, axis=-1)
    mx.eval(y, ref)
    max_diff = float(mx.max(mx.abs(y - ref)).item())
    assert max_diff < 1e-4, f"fp32 softmax mismatch on {shape}: {max_diff:.3e}"


@pytest.mark.parametrize("shape", [(2, 16, 64), (32, 12, 2048)])
def test_softmax_p1_forward_bf16_matches_reference(shape):
    mx.random.seed(0)
    x = (mx.random.normal(shape, dtype=mx.bfloat16) * 2)
    y = softmax_p1(x)
    ref = mx.softmax(x, axis=-1)
    mx.eval(y, ref)
    max_diff = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert max_diff < 1e-2, f"bf16 softmax mismatch on {shape}: {max_diff:.3e}"


def test_softmax_p1_rows_sum_to_one():
    mx.random.seed(0)
    x = mx.random.normal((4, 16, 256), dtype=mx.float32) * 3
    y = softmax_p1(x)
    sums = mx.sum(y, axis=-1)
    mx.eval(sums)
    max_off_one = float(mx.max(mx.abs(sums - 1.0)).item())
    assert max_off_one < 1e-5, f"rows not normalized: {max_off_one:.3e}"


def test_softmax_p1_handles_large_input_range():
    """Max-subtraction stability: x with extreme values must not overflow."""
    x = mx.array([[-100.0, 0.0, 100.0]])
    y = softmax_p1(x)
    ref = mx.softmax(x, axis=-1)
    mx.eval(y, ref)
    assert mx.all(mx.isfinite(y)).item(), "output not finite under extreme input"
    max_diff = float(mx.max(mx.abs(y - ref)).item())
    assert max_diff < 1e-6, f"extreme-input mismatch: {max_diff:.3e}"


# ---------- Backward ----------

def test_softmax_p1_backward_matches_mlx_autograd():
    """Gradient of (softmax(x)**2).sum() w.r.t. x must match between our
    kernel and pure-MLX softmax."""
    mx.random.seed(0)
    x = mx.random.normal((2, 4, 8), dtype=mx.float32)

    def ref_loss(x_in):
        return (mx.softmax(x_in, axis=-1) ** 2).sum()

    def ours_loss(x_in):
        return (softmax_p1_with_backward(x_in) ** 2).sum()

    ref_grad = mx.grad(ref_loss)(x)
    ours_grad = mx.grad(ours_loss)(x)
    mx.eval(ref_grad, ours_grad)
    d = float(mx.max(mx.abs(ref_grad - ours_grad)).item())
    assert d < 1e-5, f"backward mismatch vs mlx autograd: {d:.3e}"


def test_softmax_p1_finite_difference_grad_check():
    """Central-difference numerical gradient vs analytical. Tolerance accounts
    for fp32 + h=1e-3 round-off (~1e-4 expected).
    """
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 4, 8)).astype(np.float64)

    def loss_fn(x_in):
        return (softmax_p1_with_backward(x_in) ** 2).sum()

    analytic = np.array(mx.grad(loss_fn)(mx.array(x_np.astype(np.float32))))
    h = 1e-3
    numeric = np.zeros_like(x_np, dtype=np.float64)
    for idx in np.ndindex(*x_np.shape):
        xp = x_np.copy(); xp[idx] += h
        xm = x_np.copy(); xm[idx] -= h
        lp = float(loss_fn(mx.array(xp.astype(np.float32))).item())
        lm = float(loss_fn(mx.array(xm.astype(np.float32))).item())
        numeric[idx] = (lp - lm) / (2 * h)
    d = float(np.abs(analytic.astype(np.float64) - numeric).max())
    assert d < 1e-3, f"finite-diff vs analytic: {d:.3e}"
