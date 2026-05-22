"""Single training step: forward, CE loss, backward, optimizer step."""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn


def _ce_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Mean cross-entropy. Logits cast to fp32 (design §9.0) for stability."""
    logits = model(x).astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    gathered = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    return -gathered.mean()


class CompiledTrainStep:
    """Wraps a model + optimizer with a compiled forward+backward closure.

    The forward+backward (loss + grads) is compiled once at construction via
    ``mx.compile`` with ``inputs=state, outputs=state`` so MLX can fuse kernel
    dispatch across the computation graph.  The grad-norm walk and grad-clip
    cannot live inside the compiled boundary (pytree traversal is not MLX-
    traceable), so they run in the outer wrapper after the compiled call returns.
    Likewise ``optimizer.update`` runs outside.

    Construct ONCE per training session.  Re-constructing per step rebuilds
    the graph and eliminates the speedup.

    Usage::

        compiled_step = CompiledTrainStep(model, optimizer)
        for x, y in batches:
            loss = compiled_step.step(x, y, grad_clip=1.0)
    """

    def __init__(self, model: nn.Module, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

        # Capture state lists for compile so MLX knows which arrays to treat
        # as mutable in/out state across calls.
        state = [model.state, optimizer.state]

        _lg = nn.value_and_grad(model, _ce_loss)

        def _inner(x: mx.array, y: mx.array):
            return _lg(model, x, y)

        self._compiled_lg = mx.compile(_inner, inputs=state, outputs=state)

    def step(self, x: mx.array, y: mx.array, grad_clip: float = 1.0) -> float:
        """Run one compiled optimizer step. Returns scalar loss as Python float."""
        loss, grads = self._compiled_lg(x, y)
        norm = _global_grad_norm(grads)
        grads = _clip_grads(grads, grad_clip, norm)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return loss.item()


def compute_loss_and_grads(model: nn.Module, x: mx.array, y: mx.array):
    """Returns (loss, grads_pytree)."""
    loss_and_grad = nn.value_and_grad(model, _ce_loss)
    loss, grads = loss_and_grad(model, x, y)
    return loss, grads


def _global_grad_norm(grads) -> mx.array:
    """L2 norm over all gradient tensors (mlx-style nested dict/list pytree)."""
    sq_sum = mx.zeros(())

    def walk(g):
        nonlocal sq_sum
        if isinstance(g, dict):
            for v in g.values():
                walk(v)
        elif isinstance(g, list):
            for v in g:
                walk(v)
        elif isinstance(g, mx.array):
            sq_sum = sq_sum + (g.astype(mx.float32) ** 2).sum()

    walk(grads)
    return mx.sqrt(sq_sum)


def _clip_grads(grads, clip: float, current_norm: mx.array):
    factor = mx.minimum(mx.array(1.0), clip / (current_norm + 1e-8))

    def walk(g):
        if isinstance(g, dict):
            return {k: walk(v) for k, v in g.items()}
        if isinstance(g, list):
            return [walk(v) for v in g]
        if isinstance(g, mx.array):
            return g * factor.astype(g.dtype)
        return g

    return walk(grads)


def train_step(model: nn.Module, optimizer, x: mx.array, y: mx.array,
               grad_clip: float = 1.0) -> float:
    """Run one optimizer step. Returns the scalar loss as a Python float."""
    loss, grads = compute_loss_and_grads(model, x, y)
    norm = _global_grad_norm(grads)
    grads = _clip_grads(grads, grad_clip, norm)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss.item()


def train_step_with_accum(model: nn.Module, optimizer, batches,
                           grad_clip: float = 1.0) -> float:
    """Accumulate gradients over a sequence of (x, y) micro-batches, then
    apply one optimizer step on the averaged grads.

    Returns the mean loss across the micro-batches as a Python float.
    """
    n = len(batches)
    assert n >= 1, "train_step_with_accum requires at least one batch"

    accum_grads = None
    total_loss = mx.zeros(())
    for x, y in batches:
        loss, grads = compute_loss_and_grads(model, x, y)
        total_loss = total_loss + loss

        def add_to(a, b):
            if isinstance(b, dict):
                return {k: add_to(a[k] if a is not None else None, b[k]) for k in b}
            if isinstance(b, list):
                return [add_to(a[i] if a is not None else None, b[i]) for i in range(len(b))]
            if isinstance(b, mx.array):
                return b if a is None else a + b
            return b

        accum_grads = grads if accum_grads is None else add_to(accum_grads, grads)

    def scale(g, s):
        if isinstance(g, dict):
            return {k: scale(v, s) for k, v in g.items()}
        if isinstance(g, list):
            return [scale(v, s) for v in g]
        if isinstance(g, mx.array):
            return g * s
        return g

    accum_grads = scale(accum_grads, 1.0 / n)
    norm = _global_grad_norm(accum_grads)
    accum_grads = _clip_grads(accum_grads, grad_clip, norm)
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), optimizer.state)
    return (total_loss / n).item()
