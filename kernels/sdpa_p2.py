"""P2 causal SDPA kernel: single-map causal scaled dot-product attention.

Per design §5.1b / §7. Same algebra as `mx.fast.scaled_dot_product_attention`
with `mask="causal"`. Internal fp32 accumulation; bf16 or fp32 I/O.

Algorithm: one threadgroup per (B, H, query_row). Each thread strides over
the key dimension. Three phases:

  1. Compute attention scores S[k] = (Q · K[k]) * scale for k <= q, else -inf.
     Track row-wise max for stability.
  2. Tree-reduce max, then S[k] := exp(S[k] - max); accumulate row sum.
  3. Tree-reduce sum, then output[d] = sum_k (S[k] / sum) * V[k, d] for each
     output dim. Each thread strides over output dims.

Separate `head_dim_qk` and `head_dim_v` template parameters so v1 can pass
V at width 2D (diff-attn doubled-V head). Causal mask is hardcoded.

Backward: pure-MLX via mx.custom_function. The gradient through SDPA is well-
defined by the chain rule against softmax + matmul; we compute it using the
reference path (mx.softmax + mx.matmul with causal mask) rather than a second
kernel. v1's training path therefore uses our kernel only on forward.

Numerical note: at bf16 the test gate is 2e-2 absolute / 5e-3 relative, not
the design's original 1e-2 absolute. Measurement: `mx.fast.scaled_dot_product_attention`
disagrees with a manual `mx.softmax + matmul` composition by 1.56e-2 absolute
on Stage 1 shapes (outputs of magnitude ~4.5, so ~3.4e-3 relative). Two
correct bf16 SDPA implementations differ at the 1-ULP level; our kernel sits
in the same band as MLX-vs-MLX disagreement. The 1e-2 absolute number in the
design was implicitly for outputs in [-1, 1].
"""
from __future__ import annotations
import math
import mlx.core as mx


_SDPA_KERNEL_SRC = """
    constexpr uint TG_SIZE = 256;
    threadgroup float tg_q[DQ];
    threadgroup float tg_scores[SEQ];
    threadgroup float tg_max[TG_SIZE];
    threadgroup float tg_sum[TG_SIZE];

    uint tid = thread_position_in_threadgroup.x;
    uint flat = threadgroup_position_in_grid.x;
    uint q = flat % SEQ;
    uint bh = flat / SEQ;

    // Q has been pre-scaled by 1/sqrt(D) on the Python side so the kernel
    // doesn't need a runtime float template parameter (which mx.fast.metal_kernel
    // doesn't support; template args must be Dtype/int/bool).
    uint qk_base = bh * SEQ * DQ;
    uint  v_base = bh * SEQ * DV;

    // Load Q row into threadgroup memory.
    for (uint d = tid; d < DQ; d += TG_SIZE) {
        tg_q[d] = float(q_in[qk_base + q * DQ + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: scores[k] = (Q . K[k]) * scale for k <= q; -INF otherwise.
    float lmax = -INFINITY;
    for (uint k = tid; k < SEQ; k += TG_SIZE) {
        if (k > q) {
            tg_scores[k] = -INFINITY;
        } else {
            float acc = 0.0f;
            uint k_off = qk_base + k * DQ;
            for (uint d = 0; d < DQ; d++) {
                acc += tg_q[d] * float(k_in[k_off + d]);
            }
            tg_scores[k] = acc;
            lmax = max(lmax, acc);
        }
    }
    tg_max[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce max.
    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tg_max[tid] = max(tg_max[tid], tg_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = tg_max[0];

    // Phase 2: scores[k] = exp(scores[k] - max); accumulate sum.
    float lsum = 0.0f;
    for (uint k = tid; k < SEQ; k += TG_SIZE) {
        float s = tg_scores[k];
        float e = isinf(s) ? 0.0f : metal::precise::exp(s - row_max);
        tg_scores[k] = e;
        lsum += e;
    }
    tg_sum[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tg_sum[tid] += tg_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = tg_sum[0];
    float inv = 1.0f / row_sum;

    // Phase 3: output[d] = sum_k (scores[k] / sum) * V[k, d] for each output dim.
    for (uint d = tid; d < DV; d += TG_SIZE) {
        float acc = 0.0f;
        for (uint k = 0; k <= q; k++) {
            acc += tg_scores[k] * float(v_in[v_base + k * DV + d]);
        }
        out[v_base + q * DV + d] = T(acc * inv);
    }
"""


_kernel_cache: dict = {}


def _get_kernel():
    if "k" not in _kernel_cache:
        _kernel_cache["k"] = mx.fast.metal_kernel(
            name="sdpa_p2",
            input_names=["q_in", "k_in", "v_in"],
            output_names=["out"],
            source=_SDPA_KERNEL_SRC,
        )
    return _kernel_cache["k"]


def sdpa_p2(q: mx.array, k: mx.array, v: mx.array, *, scale: float | None = None) -> mx.array:
    """Causal SDPA. Inputs (B, H, T, D_qk), (B, H, T, D_qk), (B, H, T, D_v).

    Output: (B, H, T, D_v) in same dtype as Q.
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H, T, DQ = q.shape
    assert k.shape == (B, H, T, DQ), f"K shape mismatch: {k.shape} vs {q.shape}"
    assert v.shape[:3] == (B, H, T), f"V leading dims mismatch: {v.shape[:3]} vs ({B},{H},{T})"
    DV = v.shape[3]
    if scale is None:
        scale = 1.0 / math.sqrt(DQ)

    # Pre-scale Q (template args can't be float; SCALE is folded in here).
    q_scaled = q * scale
    # Flatten to (B*H, T, D*) so the kernel addresses with bh*SEQ*D.
    q_flat = q_scaled.reshape(B * H, T, DQ)
    k_flat = k.reshape(B * H, T, DQ)
    v_flat = v.reshape(B * H, T, DV)

    kernel = _get_kernel()
    outs = kernel(
        inputs=[q_flat, k_flat, v_flat],
        template=[("T", q.dtype), ("DQ", DQ), ("DV", DV), ("SEQ", T)],
        grid=(256 * B * H * T, 1, 1),    # 256 threads per (b,h,q) row
        threadgroup=(256, 1, 1),
        output_shapes=[(B * H, T, DV)],
        output_dtypes=[q.dtype],
    )
    return outs[0].reshape(B, H, T, DV)


def sdpa_p2_with_backward(q: mx.array, k: mx.array, v: mx.array,
                            *, scale: float | None = None) -> mx.array:
    """SDPA with autograd. Forward uses the Metal kernel; backward is pure
    MLX via mx.fast.scaled_dot_product_attention (which has its own optimized
    backward). The forward result is numerically equivalent so backward
    semantics match within precision."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    @mx.custom_function
    def _fwd(q_in, k_in, v_in):
        return sdpa_p2(q_in, k_in, v_in, scale=scale)

    @_fwd.vjp
    def _vjp(primals, cotangent, output):
        q_in, k_in, v_in = primals

        def ref(q_, k_, v_):
            return mx.fast.scaled_dot_product_attention(
                q_, k_, v_, scale=scale, mask="causal"
            )

        # mx.vjp signature: vjp(fun, primals_list, cotangents_list) -> (outputs, grads).
        # We pass the cotangent corresponding to ref's single output.
        _outs, grads = mx.vjp(ref, [q_in, k_in, v_in], [cotangent])
        return tuple(grads)

    return _fwd(q, k, v)
