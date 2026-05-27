# bf16 mixed precision implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Halve forward-activation memory by running matmuls/attention in bf16 while keeping fp32 master params, fp32 logits, fp32 RMSNorm internals, and fp32 lambda math. Phase D prerequisite.

**Architecture:** Option A from `docs/2026-05-21-bf16-mixed-precision-design.md`. Add `LinearAMP` (an `nn.Linear` subclass that casts weight/bias to a configured dtype inside `__call__`) and a `amp_dtype` string field on `ModelConfig` plumbed through `Transformer` → `Block` → attention/MLP modules. No changes to optimizer, checkpointing, or loss path; existing fp32 protections (RMSNorm internals, CE logits cast, grad-norm fp32) are already in place.

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`), pytest. Working directory: `diff-mlx`. Venv at `.venv/`; activate before running.

**Files touched:**
- `model.py`: add `LinearAMP`; replace `nn.Linear` use sites in `VanillaMHA`, `DiffAttention`, `SwiGLU`; thread `amp_dtype` through `Block` and `Transformer`; cast `tok_embed` output to `amp_dtype` at start of `Transformer.__call__`.
- `config.py`: add `amp_dtype: str = "float32"` field to `ModelConfig`; add module-level helper `resolve_amp_dtype(s: str) -> mx.Dtype`; default stage1/stage2 to `"bfloat16"`.
- `tests/test_precision.py` (new): unit tests for `LinearAMP` and end-to-end forward/backward at bf16.
- `tests/test_model.py`: add a parametrized check that the model forward returns the right output dtype at each `amp_dtype`.
- `tests/test_paired_init.py`: extend to verify byte-identical fp32 storage when `amp_dtype="bfloat16"`.
- `tests/test_diff_reference.py`: re-run cross-check with `amp_dtype="bfloat16"` at the bf16 tolerance (1e-2).

---

## Task 1: Add `amp_dtype` to `ModelConfig` and resolver helper

**Why first:** every other task references `cfg.amp_dtype`. Adding the field + resolver unblocks the rest.

**Files:**
- Modify: `config.py:4-15` (add field), `config.py` (add module-level resolver)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
import mlx.core as mx
import pytest
from config import ModelConfig, resolve_amp_dtype


def test_resolve_amp_dtype_float32():
    assert resolve_amp_dtype("float32") is mx.float32


def test_resolve_amp_dtype_bfloat16():
    assert resolve_amp_dtype("bfloat16") is mx.bfloat16


def test_resolve_amp_dtype_unknown_raises():
    with pytest.raises(ValueError, match="amp_dtype"):
        resolve_amp_dtype("float16")


def test_modelconfig_default_amp_dtype_is_float32():
    cfg = ModelConfig.stage0()
    assert cfg.amp_dtype == "float32"


def test_modelconfig_stage1_default_bf16():
    assert ModelConfig.stage1().amp_dtype == "bfloat16"


def test_modelconfig_stage2_default_bf16():
    assert ModelConfig.stage2().amp_dtype == "bfloat16"
```

- [ ] **Step 2: Run to confirm failure**

```
source .venv/bin/activate
pytest tests/test_config.py -v
```

Expected: ImportError on `resolve_amp_dtype`, and AttributeError on `cfg.amp_dtype`.

- [ ] **Step 3: Add the field + resolver**

In `config.py`, add a top-of-file import and a helper, plus the new field on `ModelConfig`:

```python
import mlx.core as mx


def resolve_amp_dtype(s: str) -> "mx.Dtype":
    if s == "float32":
        return mx.float32
    if s == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unknown amp_dtype {s!r}; expected 'float32' or 'bfloat16'")
```

In `ModelConfig` (after `tie_embeddings`):

```python
    amp_dtype: str = "float32"
```

Update `stage1` and `stage2` constructors to pass `amp_dtype="bfloat16"`. Stage0 stays default.

- [ ] **Step 4: Run tests, confirm pass**

```
pytest tests/test_config.py -v
```

Expected: all six new tests pass; pre-existing config tests still pass.

- [ ] **Step 5: Commit**

```
git add config.py tests/test_config.py
git commit -m "config: add amp_dtype field + resolver (defaults stage1/2 to bfloat16)"
```

---

## Task 2: Implement `LinearAMP` (the only structural new code)

**Why:** Single cast point for weight + bias. The risk flagged in the spec is "does grad flow back to fp32 weight through the cast inside `value_and_grad`?" This task's tests verify it.

**Files:**
- Modify: `model.py` (add class, near `RMSNorm`)
- Test: `tests/test_precision.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_precision.py`:

```python
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from model import LinearAMP


def test_linear_amp_fp32_matches_nn_linear():
    """At amp_dtype=float32, LinearAMP must be numerically identical to nn.Linear."""
    mx.random.seed(0)
    lin = nn.Linear(8, 4, bias=False)
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.float32)
    amp.weight = lin.weight  # share weights
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y_lin = lin(x)
    y_amp = amp(x)
    assert mx.allclose(y_lin, y_amp, atol=1e-7).item()


def test_linear_amp_output_dtype_is_amp_dtype():
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y = amp(x)
    assert y.dtype == mx.bfloat16


def test_linear_amp_weight_stays_fp32_after_forward():
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    _ = amp(x)
    assert amp.weight.dtype == mx.float32


def test_linear_amp_grad_flows_to_fp32_weight():
    """value_and_grad with bf16 forward must yield fp32 grads on the fp32 weight."""
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)

    def loss_fn(model, x):
        return (model(x) ** 2).sum()

    x = mx.random.normal((3, 8), dtype=mx.float32)
    loss_and_grad = nn.value_and_grad(amp, loss_fn)
    loss, grads = loss_and_grad(amp, x)
    mx.eval(loss, grads)
    assert grads["weight"].dtype == mx.float32
    assert grads["weight"].shape == amp.weight.shape


def test_linear_amp_with_bias():
    amp = LinearAMP(8, 4, bias=True, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y = amp(x)
    assert y.dtype == mx.bfloat16
    assert amp.bias.dtype == mx.float32
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_precision.py -v
```

Expected: ImportError on `LinearAMP`.

- [ ] **Step 3: Implement `LinearAMP`**

Insert into `model.py` after `RMSNorm` (around line 28):

```python
class LinearAMP(nn.Linear):
    """nn.Linear that casts weight/bias to `amp_dtype` inside forward.

    Storage stays in the dtype of `self.weight` (fp32 by default, set at init).
    Forward output is in `amp_dtype`. Grads through `value_and_grad` accumulate
    in `self.weight.dtype` via the implicit graph cast.

    When `amp_dtype` equals the weight dtype, this is a no-op vs nn.Linear.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 amp_dtype: "mx.Dtype" = mx.float32):
        super().__init__(in_features, out_features, bias=bias)
        self.amp_dtype = amp_dtype

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight.astype(self.amp_dtype)
        x = x.astype(self.amp_dtype)
        out = x @ w.T
        if "bias" in self:
            out = out + self.bias.astype(self.amp_dtype)
        return out
```

Note on the `"bias" in self` check: MLX's `nn.Linear` exposes `self.bias` only when bias was requested; this idiom matches the upstream `Linear.__call__` pattern. Verify by reading `mlx/nn/layers/linear.py` in the installed package if behavior diverges.

- [ ] **Step 4: Run tests**

```
pytest tests/test_precision.py -v
```

Expected: all five tests pass.

- [ ] **Step 5: Run the full suite to confirm no regressions**

```
pytest tests/ -q
```

Expected: 76 (existing) + 5 (new) = 81 passed.

- [ ] **Step 6: Commit**

```
git add model.py tests/test_precision.py
git commit -m "model: add LinearAMP for AMP-style weight cast inside forward"
```

---

## Task 3: Wire `LinearAMP` into `SwiGLU`, `VanillaMHA`, `DiffAttention`

**Why:** Replace all `nn.Linear` instances with `LinearAMP` so the cast happens everywhere a matmul does. Threading `amp_dtype` through constructors.

**Files:**
- Modify: `model.py:30-39` (`SwiGLU`), `model.py:42-70` (`VanillaMHA`), `model.py:73-166` (`DiffAttention`)
- Test: extend `tests/test_precision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_precision.py`:

```python
from model import SwiGLU, VanillaMHA, DiffAttention


def test_swiglu_amp_output_bf16():
    mlp = SwiGLU(dim=32, intermediate=64, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 8, 32), dtype=mx.float32)
    y = mlp(x)
    assert y.dtype == mx.bfloat16
    assert mlp.gate.weight.dtype == mx.float32


def test_vanilla_mha_amp_output_bf16():
    attn = VanillaMHA(dim=64, n_heads=4, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 16, 64), dtype=mx.float32)
    y = attn(x)
    assert y.dtype == mx.bfloat16
    assert attn.q_proj.weight.dtype == mx.float32


def test_diff_attention_amp_output_bf16():
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16,
                         layer_idx=1, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 16, 64), dtype=mx.float32)
    y = attn(x)
    assert y.dtype == mx.bfloat16
    assert attn.q_proj.weight.dtype == mx.float32
    # Lambda vectors must remain fp32 regardless of amp_dtype
    assert attn.lambda_q1.dtype == mx.float32
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_precision.py::test_swiglu_amp_output_bf16 -v
```

Expected: TypeError on unexpected kwarg `amp_dtype` in `SwiGLU.__init__`.

- [ ] **Step 3: Update `SwiGLU`**

Replace lines 30-39 in `model.py`:

```python
class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). All linears bias=False."""
    def __init__(self, dim: int, intermediate: int, amp_dtype: "mx.Dtype" = mx.float32):
        super().__init__()
        self.gate = LinearAMP(dim, intermediate, bias=False, amp_dtype=amp_dtype)
        self.up = LinearAMP(dim, intermediate, bias=False, amp_dtype=amp_dtype)
        self.down = LinearAMP(intermediate, dim, bias=False, amp_dtype=amp_dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 4: Update `VanillaMHA`**

Replace the constructor signature and the projection lines (model.py:48-59):

```python
class VanillaMHA(nn.Module):
    """Standard MHA: q/k/v/o all (dim, dim), bias=False. RoPE on q/k. Causal mask."""
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0,
                 amp_dtype: "mx.Dtype" = mx.float32):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.k_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.v_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.o_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
```

The body (lines 61-70) does not change. RoPE accepts any dtype; SDPA inputs will be in amp_dtype because q/k/v are now bf16-output.

- [ ] **Step 5: Update `DiffAttention`**

Modify the constructor signature (model.py:90-99) to accept `amp_dtype`:

```python
    def __init__(
        self,
        dim: int,
        n_heads_vanilla: int,
        qk_head_dim: int,
        layer_idx: int,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
        amp_dtype: "mx.Dtype" = mx.float32,
    ):
```

Replace the four projection assignments (model.py:113-116) with `LinearAMP`:

```python
        self.q_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.k_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.v_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
        self.o_proj = LinearAMP(dim, dim, bias=False, amp_dtype=amp_dtype)
```

The lambda vectors stay `mx.float32` (lines 119-122 unchanged). The `_compute_lambda` and `__call__` bodies do not change.

- [ ] **Step 6: Run the precision tests**

```
pytest tests/test_precision.py -v
```

Expected: all eight pass.

- [ ] **Step 7: Run the full suite**

```
pytest tests/ -q
```

Expected: 81 + 3 = 84 passed.

- [ ] **Step 8: Commit**

```
git add model.py tests/test_precision.py
git commit -m "model: thread amp_dtype through SwiGLU, VanillaMHA, DiffAttention"
```

---

## Task 4: Thread `amp_dtype` through `Block` and `Transformer`, cast embedding output

**Why:** Top-level entry point. `Block` accepts `amp_dtype`, forwards to attention + MLP. `Transformer` reads `cfg.amp_dtype`, resolves via `resolve_amp_dtype`, casts the token embedding output before the block stack.

**Files:**
- Modify: `model.py:169-214` (`Block`), `model.py:217-252` (`Transformer`)
- Test: extend `tests/test_precision.py` and `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_precision.py`:

```python
from model import Transformer
from config import ModelConfig


def test_transformer_amp_output_bf16():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="vanilla")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    # logits come out of `x @ tok_embed.weight.T`; tok_embed.weight is fp32,
    # x is bf16 (cast at start of forward), so the matmul output dtype depends
    # on MLX broadcast rules. We DO require: forward runs without error and
    # output is one of {bf16, fp32}; CE loss explicitly upcasts later anyway.
    assert logits.dtype in (mx.bfloat16, mx.float32)
    assert logits.shape == (2, 16, 128)


def test_transformer_amp_diff_variant_bf16():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="diff")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 128)


def test_transformer_fp32_default_unchanged():
    """At amp_dtype='float32' (the default), behavior is unchanged."""
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
    )
    model = Transformer(cfg, variant="vanilla")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    assert logits.dtype == mx.float32
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_precision.py::test_transformer_amp_output_bf16 -v
```

Expected: TypeError or AttributeError because `Block`/`Transformer` don't yet accept/use `amp_dtype`.

- [ ] **Step 3: Update `Block.__init__` signature and body**

Modify model.py:178-209. Add `amp_dtype` parameter; pass it to the attention module and to `SwiGLU`:

```python
    def __init__(
        self,
        dim: int,
        n_heads_vanilla: int,
        qk_head_dim: int,
        mlp_intermediate: int,
        variant: str = "vanilla",
        layer_idx: int | None = None,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
        amp_dtype: "mx.Dtype" = mx.float32,
    ):
        super().__init__()
        assert dim == n_heads_vanilla * qk_head_dim, (
            f"dim={dim} != n_heads_vanilla*qk_head_dim={n_heads_vanilla * qk_head_dim}"
        )
        self.norm_attn = RMSNorm(dim, eps=rms_eps)
        if variant == "vanilla":
            self.attn = VanillaMHA(dim, n_heads_vanilla, rope_base=rope_base,
                                   amp_dtype=amp_dtype)
        elif variant == "diff":
            assert layer_idx is not None, "variant='diff' requires layer_idx (1-indexed)"
            self.attn = DiffAttention(
                dim=dim,
                n_heads_vanilla=n_heads_vanilla,
                qk_head_dim=qk_head_dim,
                layer_idx=layer_idx,
                rope_base=rope_base,
                rms_eps=rms_eps,
                amp_dtype=amp_dtype,
            )
        else:
            raise ValueError(f"unknown variant {variant!r}; expected 'vanilla' or 'diff'")
        self.norm_mlp = RMSNorm(dim, eps=rms_eps)
        self.mlp = SwiGLU(dim, mlp_intermediate, amp_dtype=amp_dtype)
```

- [ ] **Step 4: Update `Transformer.__init__` and `__call__`**

Modify model.py:225-252. Add resolved dtype on the model and use it to cast the embedding output:

```python
    def __init__(self, cfg, variant: str = "vanilla"):
        super().__init__()
        from config import resolve_amp_dtype
        self.cfg = cfg
        self.variant = variant
        self._amp_dtype = resolve_amp_dtype(cfg.amp_dtype)
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [
            Block(
                dim=cfg.dim,
                n_heads_vanilla=cfg.n_heads_vanilla,
                qk_head_dim=cfg.qk_head_dim,
                mlp_intermediate=cfg.mlp_intermediate,
                variant=variant,
                layer_idx=(i + 1),
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
                amp_dtype=self._amp_dtype,
            )
            for i in range(cfg.n_layers)
        ]
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.tok_embed(tokens).astype(self._amp_dtype)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_embed.weight.T
        return logits
```

Note: `self.tok_embed.weight` stays fp32 (it's a regular `nn.Embedding`); the cast happens on the output of `self.tok_embed(tokens)`. The matmul against `tok_embed.weight.T` at the end mixes the (post-final_norm) tensor with the fp32 embedding weight; MLX will handle the dtype combination, and the downstream `_ce_loss` already casts to fp32.

- [ ] **Step 5: Run the precision tests**

```
pytest tests/test_precision.py -v
```

Expected: all 11 pass.

- [ ] **Step 6: Run the full suite**

```
pytest tests/ -q
```

Expected: 84 + 3 = 87 passed. If a pre-existing model test fails due to dtype assumptions, investigate before continuing. Most existing tests build configs without `amp_dtype` and inherit the default `"float32"`, so they should be unaffected.

- [ ] **Step 7: Commit**

```
git add model.py tests/test_precision.py
git commit -m "model: thread amp_dtype through Block + Transformer, cast embedding output"
```

---

## Task 5: End-to-end backward at bf16 (no-NaN smoke + grad-dtype invariant)

**Why:** Tasks 1-4 verify forward. This task verifies the full forward + backward + optimizer step survives bf16, with grads on fp32 params, no NaN/Inf.

**Files:**
- Test: extend `tests/test_precision.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_precision.py`:

```python
from train_step import train_step
from optim import make_adamw


def test_bf16_train_step_runs_clean():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="diff")
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    x = mx.random.randint(0, 128, shape=(2, 16))
    y = mx.random.randint(0, 128, shape=(2, 16))
    loss = train_step(model, opt, x, y, grad_clip=1.0)
    assert isinstance(loss, float)
    assert mx.array(loss).dtype == mx.float32 or isinstance(loss, float)
    import math
    assert not math.isnan(loss)
    assert not math.isinf(loss)

    # All params must still be fp32 storage after a step
    def walk(p, names):
        if isinstance(p, dict):
            for k, v in p.items():
                walk(v, names + [k])
        elif isinstance(p, list):
            for i, v in enumerate(p):
                walk(v, names + [str(i)])
        elif isinstance(p, mx.array):
            full = ".".join(names)
            assert p.dtype == mx.float32, f"param {full} drifted to {p.dtype}"
    walk(model.parameters(), [])


def test_fp32_and_bf16_initial_loss_within_tolerance():
    """Same paired init, one step at fp32 vs bf16: loss within design §9.0 tolerance."""
    base_cfg_kwargs = dict(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
    )
    cfg_fp32 = ModelConfig(**base_cfg_kwargs)
    cfg_bf16 = ModelConfig(**base_cfg_kwargs, amp_dtype="bfloat16")

    mx.random.seed(42)
    m_fp32 = Transformer(cfg_fp32, variant="vanilla")
    mx.eval(m_fp32.parameters())

    mx.random.seed(42)
    m_bf16 = Transformer(cfg_bf16, variant="vanilla")
    mx.eval(m_bf16.parameters())

    x = mx.random.randint(0, 128, shape=(2, 16))
    y = mx.random.randint(0, 128, shape=(2, 16))

    from train_step import _ce_loss
    l_fp32 = _ce_loss(m_fp32, x, y).item()
    l_bf16 = _ce_loss(m_bf16, x, y).item()
    assert abs(l_fp32 - l_bf16) < 1e-2, f"fp32={l_fp32} bf16={l_bf16}"
```

- [ ] **Step 2: Run the new tests**

```
pytest tests/test_precision.py::test_bf16_train_step_runs_clean tests/test_precision.py::test_fp32_and_bf16_initial_loss_within_tolerance -v
```

Expected: both pass. If `test_fp32_and_bf16_initial_loss_within_tolerance` fails by a wide margin, investigate which op silently upcasts or which intermediate carries fp32 across a path it shouldn't.

- [ ] **Step 3: Run the full suite**

```
pytest tests/ -q
```

Expected: 87 + 2 = 89 passed.

- [ ] **Step 4: Commit**

```
git add tests/test_precision.py
git commit -m "tests: bf16 end-to-end train_step + paired-init fp32/bf16 loss tolerance"
```

---

## Task 6: Verify PyTorch cross-check still passes at bf16

**Why:** Acceptance criterion #2 from the spec. The existing cross-check (`tests/test_diff_reference.py`) runs at fp32 and gets ~3.58e-7. We need to verify the bf16 path also satisfies the looser 1e-2 tolerance (design §7.4 / §9.0).

**Files:**
- Modify: `tests/test_diff_reference.py`

- [ ] **Step 1: Add the bf16 test, mirroring the fp32 test**

Append to `tests/test_diff_reference.py`:

```python
def test_mlx_diff_attention_matches_pytorch_reference_at_bf16():
    """Cross-check at amp_dtype=bfloat16. Tolerance loosened to 1e-2 per design §7.4 / §9.0.

    Same fixture and weight-copy logic as the fp32 test; only differences are
    amp_dtype=mx.bfloat16 on the module and a looser atol.
    """
    if not FIXTURE.exists():
        import pytest
        pytest.skip(f"fixture missing: {FIXTURE}. Run scripts/generate_ref_fixture.py once.")

    data = np.load(FIXTURE)
    DIM = int(data["dim"])
    NUM_HEADS_REF = int(data["num_heads_ref"])
    n_heads_vanilla = NUM_HEADS_REF * 2
    qk_head_dim = DIM // n_heads_vanilla

    input_x = mx.array(data["input_x"].astype(np.float32))
    expected_out = data["ref_output"].astype(np.float32)

    with mx.stream(mx.cpu):
        attn = DiffAttention(
            dim=DIM,
            n_heads_vanilla=n_heads_vanilla,
            qk_head_dim=qk_head_dim,
            layer_idx=1,
            amp_dtype=mx.bfloat16,
        )

        params = attn.parameters()
        params = _set_param(params, ["q_proj", "weight"], mx.array(data["weight__q_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["k_proj", "weight"], mx.array(data["weight__k_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["v_proj", "weight"], mx.array(data["weight__v_proj__weight"].astype(np.float32)))
        params = _set_param(params, ["o_proj", "weight"], mx.array(data["weight__out_proj__weight"].astype(np.float32)))
        params["lambda_q1"] = mx.array(data["weight__lambda_q1"].astype(np.float32))
        params["lambda_k1"] = mx.array(data["weight__lambda_k1"].astype(np.float32))
        params["lambda_q2"] = mx.array(data["weight__lambda_q2"].astype(np.float32))
        params["lambda_k2"] = mx.array(data["weight__lambda_k2"].astype(np.float32))
        params = _set_param(params, ["subln", "scale"], mx.array(data["weight__subln__weight"].astype(np.float32)))
        attn.update(params)

        actual = attn(input_x)
        mx.eval(actual)

    actual_np = np.array(actual).astype(np.float32)
    max_diff = float(np.abs(actual_np - expected_out).max())
    assert max_diff < 1e-2, (
        f"max |diff| = {max_diff:.3e}; expected < 1e-2 per design §7.4 bf16 tolerance."
    )
```

- [ ] **Step 2: Run**

```
pytest tests/test_diff_reference.py -v
```

Expected: both tests pass; bf16 test reports max |diff| around 1e-3 to 1e-2.

- [ ] **Step 3: Commit**

```
git add tests/test_diff_reference.py
git commit -m "tests: diff-attn cross-check at bf16 (1e-2 tolerance)"
```

---

## Task 7: Paired-init invariant under bf16

**Why:** Acceptance criterion #4 from the spec. The paired-init protocol (design §9.7) must produce byte-identical fp32 weight storage whether built with `amp_dtype="float32"` or `amp_dtype="bfloat16"`. Storage stays fp32; only forward casts.

**Files:**
- Modify: `tests/test_paired_init.py`

- [ ] **Step 1: Identify existing paired-init test**

```
grep -n "def test_\|build_paired_models" tests/test_paired_init.py | head -20
```

- [ ] **Step 2: Add the test**

Append a parametrized check:

```python
import pytest


@pytest.mark.parametrize("amp_dtype", ["float32", "bfloat16"])
def test_paired_init_byte_identical_under_amp(amp_dtype):
    """fp32 storage of shared params must be byte-identical regardless of amp_dtype.

    Storage is fp32 always; amp_dtype only affects forward cast.
    """
    from config import ModelConfig
    from paired_init import build_paired_models
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype=amp_dtype,
    )
    vanilla, diff = build_paired_models(cfg, seed=0)
    mx.eval(vanilla.parameters(), diff.parameters())

    # Shared params: tok_embed, q/k/v/o on first block (per design §9.7)
    assert mx.array_equal(
        vanilla.tok_embed.weight, diff.tok_embed.weight
    ).item()
    # Spot-check a projection
    assert mx.array_equal(
        vanilla.blocks[0].attn.q_proj.weight,
        diff.blocks[0].attn.q_proj.weight,
    ).item()
    assert vanilla.tok_embed.weight.dtype == mx.float32
    assert diff.tok_embed.weight.dtype == mx.float32
```

If the existing paired-init helpers reference an explicit dtype list that excludes `amp_dtype`, this test will pass without code changes (since storage is unaffected). If it fails, the implementation in `paired_init.py` may need an audit; investigate then.

- [ ] **Step 3: Run**

```
pytest tests/test_paired_init.py -v
```

Expected: existing tests pass; new parametrized test passes for both values.

- [ ] **Step 4: Commit**

```
git add tests/test_paired_init.py
git commit -m "tests: paired-init byte-identical fp32 storage under amp_dtype"
```

---

## Task 8: Memory sanity check at Stage 0 diff

**Why:** Acceptance criterion #3 from the spec. Confirm bf16 actually drops the activation peak. Sanity that the memory win is real (not a strict threshold).

**Files:**
- Create: `scripts/measure_peak_bf16.py`

- [ ] **Step 1: Write the script**

```python
"""Quick peak-memory measurement at Stage 0 diff, fp32 vs bf16.

Builds a Transformer + runs N steps at each precision, prints peak
metal memory. Manual sanity check, not a test.

Usage:
    python scripts/measure_peak_bf16.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
import mlx.core as mx
import numpy as np
from config import ModelConfig, TrainConfig
from model import Transformer
from train_step import train_step
from optim import make_adamw


def measure(amp_dtype: str, steps: int = 20) -> float:
    mx.random.seed(0)
    cfg = replace(ModelConfig.stage0(), amp_dtype=amp_dtype)
    model = Transformer(cfg, variant="diff")
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = mx.array(rng.integers(0, cfg.vocab_size, size=(16, cfg.block_size), dtype=np.int32))
        y = mx.array(rng.integers(0, cfg.vocab_size, size=(16, cfg.block_size), dtype=np.int32))
        train_step(model, opt, x, y, grad_clip=1.0)
    return mx.metal.get_peak_memory() / 1e9


if __name__ == "__main__":
    fp32_peak = measure("float32")
    print(f"fp32 peak: {fp32_peak:.2f} GB")
    bf16_peak = measure("bfloat16")
    print(f"bf16 peak: {bf16_peak:.2f} GB")
    print(f"reduction: {(1 - bf16_peak / fp32_peak) * 100:.1f}%")
```

- [ ] **Step 2: Run the script**

```
caffeinate -disu python scripts/measure_peak_bf16.py
```

Expected: bf16 peak noticeably lower than fp32 peak. Concrete threshold is workload-dependent; design says "roughly half" on activations, but optimizer state (fp32) doesn't shrink, so total peak reduction is somewhere in the 20-50% range depending on how much of peak is activations vs static. Capture the printed numbers.

- [ ] **Step 3: Commit the script**

```
git add scripts/measure_peak_bf16.py
git commit -m "scripts: quick peak-memory measurement at Stage 0 fp32 vs bf16"
```

---

## Task 9: Update Phase A retro to mark §9.0 closed

**Why:** Phase A retro listed bf16 mixed precision as a deferred item; design self-review references it. Mark closed and reference this implementation.

**Files:**
- Modify: `docs/2026-05-20-phase-a-retro.md`

- [ ] **Step 1: Edit `docs/2026-05-20-phase-a-retro.md`**

The deferred item is at lines 43-46. Replace that block with:

```markdown
1. **Precision is pure fp32, not bf16-mixed-with-fp32-master as design §9.0 specifies.**
   - **Closed 2026-05-21.** Implemented via `LinearAMP` (option A from
     `docs/2026-05-21-bf16-mixed-precision-design.md`): fp32 params, bf16 cast
     inside forward at op boundaries. Stage 1/2 ModelConfig defaults switched
     to `amp_dtype="bfloat16"`.
```

- [ ] **Step 2: Commit**

```
git add docs/2026-05-20-phase-a-retro.md
git commit -m "phase-a retro: mark bf16 mixed precision item closed"
```

---

## Final acceptance run

After Task 9, run the full suite once more and confirm green:

```
source .venv/bin/activate
pytest tests/ -q
```

Then update the Phase B retro's "Updated Phase D prerequisites" section: cross out item 1 (bf16) or move it to the "done" column if the section structure supports that.

```
git add docs/2026-05-20-phase-b-retro.md
git commit -m "phase-b retro: mark bf16 mixed precision prereq complete"
```

## Out of scope (do not implement here)

- Option B (bf16 storage with parallel fp32 master).
- Loss scaling.
- AMP in checkpoint format.
- Stage 1/2 actual runs. Those happen as part of Phase D, after the other prereqs (#3-5) land.
