"""Cross-check our MLX DiffAttention against a fixture generated from the
official PyTorch microsoft/unilm/Diff-Transformer reference.

Design §7.4: internal v0/v1 agreement is necessary but not sufficient — both
can share an architecture bug. The cross-check vs the official PyTorch impl
catches shared bugs by going through a fundamentally different codepath.

Important note on streams: MLX's default GPU (Metal) matmul uses reduced fp32
precision (~1e-3 elementwise error per matmul). The cumulative error through
projections + attention + subln + o_proj reaches ~1e-3 against the PyTorch
reference even when the algorithm is bit-equivalent. We therefore force the
forward onto the CPU stream for this comparison: CPU matmul is IEEE fp32 and
gives us a clean ~1e-7 agreement when the algorithm matches the paper.

The fp32-vs-reduced-precision matmul gap is a property of the hardware/runtime,
not of our implementation. Training and inference on GPU are unaffected — we
only need full precision for the cross-check.
"""
from pathlib import Path

import numpy as np
import mlx.core as mx

from model import DiffAttention

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "ref_fixtures" / "diffattn_toy_v1.npz"


def _set_param(params, path, value):
    """Return a new params dict (same nested shape) with the value at `path` replaced."""
    if len(path) == 1:
        out = dict(params)
        out[path[0]] = value
        return out
    head, *tail = path
    out = dict(params)
    out[head] = _set_param(out[head], tail, value)
    return out


def test_mlx_diff_attention_matches_pytorch_reference():
    if not FIXTURE.exists():
        import pytest
        pytest.skip(f"fixture missing: {FIXTURE}. Run scripts/generate_ref_fixture.py once.")

    data = np.load(FIXTURE)
    DIM = int(data["dim"])
    NUM_HEADS_REF = int(data["num_heads_ref"])

    # The reference's num_heads equals our n_heads_diff (because the reference defines
    # head_dim = embed_dim // num_heads // 2). So n_heads_vanilla = 2 * NUM_HEADS_REF.
    n_heads_vanilla = NUM_HEADS_REF * 2
    qk_head_dim = DIM // n_heads_vanilla

    input_x = mx.array(data["input_x"].astype(np.float32))
    expected_out = data["ref_output"].astype(np.float32)

    # Run everything on the CPU stream for IEEE fp32 matmul (see module docstring).
    with mx.stream(mx.cpu):
        # Build MLX module (layer_idx=1 so lambda_init=0.2, matching reference depth=0).
        attn = DiffAttention(
            dim=DIM,
            n_heads_vanilla=n_heads_vanilla,
            qk_head_dim=qk_head_dim,
            layer_idx=1,
        )

        # Copy weights from fixture into MLX module.
        # MLX nn.Linear.weight shape is (out_features, in_features) — same as PyTorch, direct copy.
        params = attn.parameters()
        params = _set_param(params, ["q_proj", "weight"], mx.array(data["weight__q_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["k_proj", "weight"], mx.array(data["weight__k_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["v_proj", "weight"], mx.array(data["weight__v_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["o_proj", "weight"], mx.array(data["weight__out_proj__weight"].astype(np.float32)))
        params["lambda_q1"] = mx.array(data["weight__lambda_q1"].astype(np.float32))
        params["lambda_k1"] = mx.array(data["weight__lambda_k1"].astype(np.float32))
        params["lambda_q2"] = mx.array(data["weight__lambda_q2"].astype(np.float32))
        params["lambda_k2"] = mx.array(data["weight__lambda_k2"].astype(np.float32))
        # subln.weight (ref) -> subln.scale (ours)
        params = _set_param(params, ["subln", "scale"], mx.array(data["weight__subln__weight"].astype(np.float32)))
        attn.update(params)

        actual = attn(input_x)
        mx.eval(actual)

    actual_np = np.array(actual)
    max_diff = float(np.abs(actual_np - expected_out).max())
    mean_diff = float(np.abs(actual_np - expected_out).mean())
    assert max_diff < 1e-3, (
        f"max |diff| = {max_diff:.3e} (mean |diff| = {mean_diff:.3e}); "
        f"expected < 1e-3 per design §7.4. "
        f"actual stats: mean={actual_np.mean():.4f} std={actual_np.std():.4f}; "
        f"expected stats: mean={expected_out.mean():.4f} std={expected_out.std():.4f}"
    )
