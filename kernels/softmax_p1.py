"""P1 softmax kernel: row-wise softmax over the last dim.

Phase C Stage P1 preflight per design §5.1. Validates the MLX custom-kernel
toolchain (write MSL, wire through mx.fast.metal_kernel, register autograd via
mx.custom_function, gradient-check) before tackling the bigger P2 SDPA kernel.

Numerics: max-subtraction trick for stability. Internal compute in fp32 even
when input is bf16 (softmax over wide vocabs in bf16 overflows easily). Output
matches input dtype.

Backward: pure-MLX softmax Jacobian (no custom backward kernel). The Jacobian
of softmax is `J(x) = diag(y) - y y^T` where `y = softmax(x)`. For a gradient
`g` from upstream, the input gradient is `y * (g - sum(g * y, axis=-1))`.
"""
from __future__ import annotations
import mlx.core as mx


# Kernel body: one threadgroup per row. Each thread strides across the row.
# Uses 256 threads per group. Shared-memory reductions for max and sum.
_SOFTMAX_KERNEL_SRC = """
    constexpr uint TG_SIZE = 256;
    threadgroup float tg_max[TG_SIZE];
    threadgroup float tg_sum[TG_SIZE];

    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint cols = COLS;
    uint base = row * cols;

    // Per-thread max over its stride slice. Use -INF as identity.
    float lmax = -INFINITY;
    for (uint c = tid; c < cols; c += TG_SIZE) {
        float v = float(inp[base + c]);
        lmax = max(lmax, v);
    }
    tg_max[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for max.
    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_max[tid] = max(tg_max[tid], tg_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = tg_max[0];

    // Compute exp(x - max) per element; accumulate per-thread sum; stash exp into out.
    float lsum = 0.0f;
    for (uint c = tid; c < cols; c += TG_SIZE) {
        float e = metal::precise::exp(float(inp[base + c]) - row_max);
        out[base + c] = T(e);
        lsum += e;
    }
    tg_sum[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum.
    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_sum[tid] += tg_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = tg_sum[0];
    float inv = 1.0f / row_sum;

    // Normalize.
    for (uint c = tid; c < cols; c += TG_SIZE) {
        out[base + c] = T(float(out[base + c]) * inv);
    }
"""


def _build_kernel():
    """Construct the JIT-compiled kernel once. Cached on first call."""
    return mx.fast.metal_kernel(
        name="softmax_p1",
        input_names=["inp"],
        output_names=["out"],
        source=_SOFTMAX_KERNEL_SRC,
    )


_kernel_cache: dict = {}


def _get_kernel():
    if "k" not in _kernel_cache:
        _kernel_cache["k"] = _build_kernel()
    return _kernel_cache["k"]


def softmax_p1(x: mx.array) -> mx.array:
    """Row-wise softmax over the last dim. Pure forward (no autograd hook).

    Use `softmax_p1_with_backward` for an autograd-friendly version.
    """
    if x.ndim < 1:
        raise ValueError("softmax_p1 expects an array with at least one dim")
    # Flatten leading dims into rows.
    cols = x.shape[-1]
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    x_flat = x.reshape(rows, cols)

    kernel = _get_kernel()
    outs = kernel(
        inputs=[x_flat],
        template=[("T", x.dtype), ("COLS", cols)],
        grid=(256 * rows, 1, 1),       # 256 threads per row
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[x.dtype],
    )
    return outs[0].reshape(x.shape)


def softmax_p1_with_backward(x: mx.array) -> mx.array:
    """Row-wise softmax with autograd. Forward uses the Metal kernel; backward
    is pure MLX (softmax Jacobian via the y * (g - sum(g*y, axis=-1)) formula).
    """
    @mx.custom_function
    def _fwd(x_inner):
        return softmax_p1(x_inner)

    @_fwd.vjp
    def _vjp(primals, cotangent, output):
        # primals: tuple of inputs that were passed positionally. Here it's just x.
        # cotangent: gradient w.r.t. output (same shape as output).
        # output: the forward result (we use it instead of recomputing softmax).
        # d/dx_i softmax(x)_j = y_j * (delta_ij - y_i)
        # so grad_x = y * (g - sum(g * y, axis=-1, keepdims=True))
        y = output
        g = cotangent
        scaled = mx.sum(g * y, axis=-1, keepdims=True)
        return (y * (g - scaled),)

    return _fwd(x)
