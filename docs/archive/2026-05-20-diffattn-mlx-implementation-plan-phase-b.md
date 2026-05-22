# diff-mlx Phase B: Differential Attention v0 + Paired-Seed Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the paper-canonical Differential Attention (v0 path: two MLX SDPA calls + Python subtract), the paired-seed init protocol (§9.7) so vanilla and diff share byte-identical backbone weights, a reference cross-check against the official PyTorch implementation, and a Stage 0 paired smoke run (vanilla seed 0 + diff seed 0) to validate the A/B pipeline end-to-end.

**Architecture:** Extend `model.py` with `DiffAttention` (paper-canonical: H_diff = n_heads_vanilla/2, qk_head_dim = D, v_head_dim = 2D, all four projections dim → dim). Add a `variant` flag to `Block` and `Transformer` to swap attention type. Lambda parameters are per-layer fp32 vectors of shape `(qk_head_dim,)`, init `randn * 0.1`. Lambda scalar is `exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init` (depth-scheduled). subln is RMSNorm over `2D` applied per-head AFTER subtraction. v0 forward uses two `mx.fast.scaled_dot_product_attention` calls (linearity rewrite from design §7.1). Paired-seed init copies vanilla's state-dict into diff's by parameter name (embed, MLPs, RMSNorms, ALL attention projections), so the paired δ measures only the diff-attn vs vanilla mechanism difference.

**Tech Stack:** Same as Phase A (MLX, tiktoken, FineWeb-Edu shards). Adds PyTorch as a one-time dev-only dep for generating a fixture file from the official reference; the fixture is saved as `.npz` and committed, after which torch is not needed at runtime.

**Reference design:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` — read §6.3 (diff-attn arch + forward pseudocode), §7.1 (v1 algebra; v0 follows same shape), §7.4 (reference cross-check), §9.7 (paired-seed init), §5.2 (Stage 0 SDPA oracle definition) before starting.

**Scope:** Phase B produces a working diff-attn v0 path, a paired-seed init protocol, a reference fixture, and one paired Stage 0 smoke run (vanilla seed 0 + diff seed 0). **Phase B does NOT include:** custom Metal kernels (Phase C), Stages 1 and 2 (Phase D), bf16 mixed precision (Phase D prereq), optimizer-state checkpoints (Phase D prereq), grad_accum (Phase D prereq).

---

## File Structure

```
diff-mlx/
  model.py                      # extended: + DiffAttention class, + variant flag on Block/Transformer
  paired_init.py                # NEW: paired-seed init protocol (vanilla → diff weight copy)
  train.py                      # modified: split `seed` → `data_seed` + `model_seed`; add --variant=diff

  scripts/
    generate_ref_fixture.py     # NEW: one-time, uses PyTorch to produce reference outputs at toy shape
    stage0_paired.py            # NEW: build vanilla + diff with paired-seed init; run both to completion

  tests/
    test_diff_attention.py      # NEW: DiffAttention shape, param count, lambda math, causal property, oracle agreement
    test_diff_reference.py      # NEW: load fixture .npz; compare MLX DiffAttention output to PyTorch reference
    test_paired_init.py         # NEW: byte-identity of shared weights after copy
    test_model.py               # extended: + variant="diff" tests on Block and Transformer
    test_train_driver.py        # extended: paired data_seed/model_seed determinism

  data/
    ref_fixtures/
      diffattn_toy_v1.npz       # NEW: committed fixture (inputs, weights, reference output) from PyTorch
```

**Design notes:**

- DiffAttention lives in `model.py` per design §11 ("single source of truth"). No separate file.
- `paired_init.py` is new because the protocol is a distinct concept; keeping it out of `checkpoint.py` keeps both files focused.
- Reference fixture generation requires PyTorch but only once; the resulting `.npz` is committed, so the test depends only on numpy/MLX.
- `Block(variant=...)` and `Transformer(variant=...)` constructor arguments — `variant` is NOT in `ModelConfig` because both variants share the same shape config; variant is an experiment knob, not a model property.

---

## Phase B overview (commit cadence)

- **Tasks 1-2:** Backbone extension for layer_idx + lambda init (preparation for DiffAttention).
- **Tasks 3-4:** DiffAttention class + tests.
- **Tasks 5-6:** Block/Transformer variant flag.
- **Tasks 7-8:** Paired-seed init protocol + tests.
- **Tasks 9-10:** Reference fixture (PyTorch one-shot) + MLX cross-check test.
- **Task 11:** Split seed in train.py.
- **Tasks 12-13:** Stage 0 paired smoke run script + execution.
- **Tasks 14-15:** Phase B retro + tag.

Total: 15 tasks, ~80-100 steps. Implementation ~4-6 hours of focused work + ~2.5h Stage 0 paired run.

---

## Task 1: Add layer_idx + lambda_init schedule helper

DiffAttention needs to know which layer it is (for the lambda_init depth schedule). Adding the helper first so Task 3 can use it cleanly.

**Files:**
- Modify: `model.py` (append helper)
- Create: `tests/test_lambda_init_schedule.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lambda_init_schedule.py
import math
from model import lambda_init_for_layer


def test_layer_1_returns_0_2():
    assert abs(lambda_init_for_layer(1) - 0.2) < 1e-9


def test_layer_2_increases_above_0_2():
    v = lambda_init_for_layer(2)
    assert v > 0.2
    assert v < 0.8


def test_layer_16_approaches_0_8():
    v = lambda_init_for_layer(16)
    assert v > 0.79
    assert v < 0.8


def test_schedule_is_monotonically_increasing():
    vals = [lambda_init_for_layer(i) for i in range(1, 17)]
    for a, b in zip(vals[:-1], vals[1:]):
        assert b > a, f"non-monotonic: {a} → {b}"


def test_paper_formula_exact_layer_1():
    expected = 0.8 - 0.6 * math.exp(-0.3 * (1 - 1))
    assert lambda_init_for_layer(1) == expected
```

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate
pytest tests/test_lambda_init_schedule.py -v
```
Expected: ImportError on `lambda_init_for_layer`.

- [ ] **Step 3: Append to `model.py`**

```python
# Append to model.py (after the existing imports/classes)

def lambda_init_for_layer(layer_idx: int) -> float:
    """Paper-canonical λ_init depth schedule (1-indexed layer).

    λ_init = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))

    Layer 1 → 0.2; approaches 0.8 with depth. Per design §6.3, paper §2.2.
    """
    import math
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_lambda_init_schedule.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_lambda_init_schedule.py
git commit -m "model: lambda_init depth schedule helper (paper §2.2 formula)"
```

---

## Task 2: Implement DiffAttention class

The core experimental arm. Paper-canonical dimensions, two-SDPA forward (no map materialization).

**Files:**
- Modify: `model.py` (append DiffAttention class)
- Create: `tests/test_diff_attention.py`

- [ ] **Step 1: Write failing tests at `tests/test_diff_attention.py`**

```python
import math
import numpy as np
import mlx.core as mx
from model import DiffAttention, VanillaMHA


def _flatten(d, prefix=""):
    if hasattr(d, "items"):
        for k, v in d.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def test_diff_attention_shape_preserved():
    """dim → dim regardless of head config."""
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    x = mx.random.normal((2, 32, 256), dtype=mx.float32)
    y = attn(x)
    assert y.shape == x.shape


def test_diff_attention_param_count_matches_vanilla():
    """Per §6.3: 4·dim² total attention params, same as vanilla MHA."""
    diff = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    vanilla = VanillaMHA(dim=256, n_heads=4)
    diff_params = sum(p.size for _, p in _flatten(diff.parameters()))
    vanilla_params = sum(p.size for _, p in _flatten(vanilla.parameters()))
    # Diff has 4 projection matrices (each dim*dim = 4*dim²) PLUS lambda vectors (4 * qk_head_dim)
    # PLUS subln scale (2 * qk_head_dim). Lambdas + subln are small.
    expected_proj = 4 * 256 * 256
    expected_lambdas = 4 * 64  # 4 vectors of qk_head_dim
    expected_subln = 2 * 64    # RMSNorm scale over 2D
    assert diff_params == expected_proj + expected_lambdas + expected_subln
    assert vanilla_params == expected_proj
    # Diff is slightly larger only due to lambda vectors + subln scale (~320 params at this shape)
    assert diff_params - vanilla_params == expected_lambdas + expected_subln


def test_diff_attention_no_bias_on_projections():
    attn = DiffAttention(dim=128, n_heads_vanilla=4, qk_head_dim=32, layer_idx=1)
    params = attn.parameters()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert "bias" not in params[name], f"{name} should not have bias"


def test_diff_attention_n_heads_diff_is_half_of_vanilla():
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    assert attn.n_heads_diff == 2
    assert attn.v_head_dim == 128  # 2 * qk_head_dim


def test_diff_attention_lambda_at_init_close_to_lambda_init():
    """With λ_q* and λ_k* randn*0.1, exp(dot) values are small.
    Lambda scalar should equal ~λ_init at init.
    """
    mx.random.seed(0)
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    lam = attn._compute_lambda()
    # At init, both exp() terms are very close to 1.0, so lam ≈ 1 - 1 + λ_init = λ_init
    # but randn*0.1 means dot products can be small but non-zero, giving ~0.01-0.05 swing.
    expected_lam_init = 0.8 - 0.6 * math.exp(-0.3 * (1 - 1))  # = 0.2
    assert abs(lam.item() - expected_lam_init) < 0.5  # generous tolerance


def test_diff_attention_causal_property():
    """Future-token perturbations don't affect past-position outputs."""
    mx.random.seed(0)
    attn = DiffAttention(dim=64, n_heads_vanilla=2, qk_head_dim=32, layer_idx=1)
    x1 = mx.random.normal((1, 16, 64), dtype=mx.float32)
    perturb = mx.random.normal((1, 8, 64), dtype=mx.float32)
    x2 = mx.concatenate([x1[:, :8, :], perturb], axis=1)
    y1 = attn(x1)
    y2 = attn(x2)
    assert mx.allclose(y1[0, :8, :], y2[0, :8, :], atol=1e-4).item()


def test_diff_attention_matches_sdpa_oracle():
    """v0 forward must equal the paper's explicit SDPA-composed oracle (design §5.2)."""
    mx.random.seed(42)
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16, layer_idx=2)
    x = mx.random.normal((2, 8, 64), dtype=mx.float32)
    y_attn = attn(x)

    # Re-implement the oracle inline using only public MLX ops:
    H = attn.n_heads_diff
    D = attn.qk_head_dim
    B, T, _ = x.shape
    q = attn.q_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, T, H, 2 * D).transpose(0, 2, 1, 3)
    q1, q2 = q[:, :H, :, :], q[:, H:, :, :]
    k1, k2 = k[:, :H, :, :], k[:, H:, :, :]
    q1 = mx.fast.rope(q1, dims=D, traditional=False, base=10000.0, scale=1.0, offset=0)
    q2 = mx.fast.rope(q2, dims=D, traditional=False, base=10000.0, scale=1.0, offset=0)
    k1 = mx.fast.rope(k1, dims=D, traditional=False, base=10000.0, scale=1.0, offset=0)
    k2 = mx.fast.rope(k2, dims=D, traditional=False, base=10000.0, scale=1.0, offset=0)
    scale = 1.0 / math.sqrt(D)
    out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=scale, mask="causal")
    out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=scale, mask="causal")
    lam = attn._compute_lambda()
    out = out1 - lam * out2
    out = attn.subln(out)
    out = (1 - attn.lambda_init) * out
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * 2 * D)
    y_oracle = attn.o_proj(out)

    assert mx.allclose(y_attn, y_oracle, atol=1e-5).item()
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_diff_attention.py -v
```
Expected: ImportError on `DiffAttention`.

- [ ] **Step 3: Append `DiffAttention` class to `model.py`**

```python
# Append to model.py (after VanillaMHA)

class DiffAttention(nn.Module):
    """Paper-canonical Differential Attention (Ye et al., ICLR 2025).

    Per design §6.3:
    - n_heads_diff = n_heads_vanilla // 2
    - qk_head_dim = D (same as vanilla head_dim)
    - v_head_dim = 2 * D
    - All projections dim → dim (same total widths as vanilla)
    - subln = RMSNorm over 2D applied per-head AFTER differential subtraction
    - lambda = exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init  (scalar, per-forward)
    - Output scaled by (1 - λ_init) before o_proj
    - RoPE via mx.fast.rope(traditional=False) on Q1/K1/Q2/K2 independently
    - SDPA via mx.fast.scaled_dot_product_attention(scale=1/√D, mask="causal")

    v0 forward: two SDPA calls with shared V at width 2D, subtract outputs
    (design §7.1 linearity rewrite — no T×T map materialization).
    """
    def __init__(
        self,
        dim: int,
        n_heads_vanilla: int,
        qk_head_dim: int,
        layer_idx: int,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        assert n_heads_vanilla % 2 == 0, "n_heads_vanilla must be even (paired into diff heads)"
        assert n_heads_vanilla * qk_head_dim == dim, "dim must equal n_heads_vanilla * qk_head_dim"
        self.dim = dim
        self.n_heads_vanilla = n_heads_vanilla
        self.n_heads_diff = n_heads_vanilla // 2
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = 2 * qk_head_dim
        self.layer_idx = layer_idx
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(qk_head_dim)
        self.lambda_init = lambda_init_for_layer(layer_idx)

        # Projections (all dim → dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Lambda vectors (fp32, init randn * 0.1 per design §6.3)
        self.lambda_q1 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_k1 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_q2 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_k2 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1

        # subln: RMSNorm over the V head width (2D), per-head application
        self.subln = RMSNorm(self.v_head_dim, eps=rms_eps)

    def _compute_lambda(self) -> mx.array:
        """λ = exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init  (scalar, fp32)."""
        l1 = mx.exp(mx.sum(self.lambda_q1.astype(mx.float32) * self.lambda_k1.astype(mx.float32)))
        l2 = mx.exp(mx.sum(self.lambda_q2.astype(mx.float32) * self.lambda_k2.astype(mx.float32)))
        return l1 - l2 + self.lambda_init

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        H = self.n_heads_diff
        D = self.qk_head_dim

        q = self.q_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)  # (B, 2H, T, D)
        k = self.k_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)  # (B, 2H, T, D)
        v = self.v_proj(x).reshape(B, T, H, 2 * D).transpose(0, 2, 1, 3)  # (B, H, T, 2D)

        q1, q2 = q[:, :H, :, :], q[:, H:, :, :]
        k1, k2 = k[:, :H, :, :], k[:, H:, :, :]

        q1 = mx.fast.rope(q1, dims=D, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        q2 = mx.fast.rope(q2, dims=D, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k1 = mx.fast.rope(k1, dims=D, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k2 = mx.fast.rope(k2, dims=D, traditional=False, base=self.rope_base, scale=1.0, offset=0)

        out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=self.scale, mask="causal")
        out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=self.scale, mask="causal")

        lam = self._compute_lambda()
        out = out1 - lam.astype(out1.dtype) * out2

        out = self.subln(out)
        out = (1.0 - self.lambda_init) * out

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * 2 * D)  # H*2D = dim
        return self.o_proj(out)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_diff_attention.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_diff_attention.py
git commit -m "model: DiffAttention class (paper-canonical, v0 = two SDPA calls + subtract)"
```

---

## Task 3: Add variant flag to Block

Block currently constructs `VanillaMHA` directly. Make it pick between vanilla and diff based on a variant flag.

**Files:**
- Modify: `model.py` (Block class)
- Modify: `tests/test_model.py` (extend existing block tests)

- [ ] **Step 1: Add failing tests to `tests/test_model.py`**

Append to `tests/test_model.py`:

```python
# Diff-variant Block tests

def test_block_diff_variant_shape():
    from model import Block
    block = Block(
        dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352,
        variant="diff", layer_idx=1,
    )
    x = mx.random.normal((2, 16, 128), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


def test_block_vanilla_variant_still_works():
    """Backward compat: existing Block API still works (defaults to vanilla)."""
    from model import Block
    block = Block(dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352)
    x = mx.random.normal((2, 16, 128), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


def test_block_diff_attn_is_diffattention():
    from model import Block, DiffAttention, VanillaMHA
    block_diff = Block(
        dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352,
        variant="diff", layer_idx=1,
    )
    block_vanilla = Block(dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352)
    assert isinstance(block_diff.attn, DiffAttention)
    assert isinstance(block_vanilla.attn, VanillaMHA)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_model.py -v -k "block"
```
Expected: 3 new tests fail (Block doesn't take `qk_head_dim`, `variant`, or `layer_idx`).

- [ ] **Step 3: Update `Block` in `model.py`**

Replace the existing `Block` class with:

```python
class Block(nn.Module):
    """Pre-norm transformer block. Variant selects attention type.

    variant="vanilla": uses VanillaMHA(dim, n_heads_vanilla)
    variant="diff":    uses DiffAttention(dim, n_heads_vanilla, qk_head_dim, layer_idx)

    qk_head_dim is required and must satisfy dim == n_heads_vanilla * qk_head_dim.
    layer_idx is required for variant="diff" (1-indexed).
    """
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
    ):
        super().__init__()
        assert dim == n_heads_vanilla * qk_head_dim, (
            f"dim={dim} != n_heads_vanilla*qk_head_dim={n_heads_vanilla * qk_head_dim}"
        )
        self.norm_attn = RMSNorm(dim, eps=rms_eps)
        if variant == "vanilla":
            self.attn = VanillaMHA(dim, n_heads_vanilla, rope_base=rope_base)
        elif variant == "diff":
            assert layer_idx is not None, "variant='diff' requires layer_idx (1-indexed)"
            self.attn = DiffAttention(
                dim=dim,
                n_heads_vanilla=n_heads_vanilla,
                qk_head_dim=qk_head_dim,
                layer_idx=layer_idx,
                rope_base=rope_base,
                rms_eps=rms_eps,
            )
        else:
            raise ValueError(f"unknown variant {variant!r}; expected 'vanilla' or 'diff'")
        self.norm_mlp = RMSNorm(dim, eps=rms_eps)
        self.mlp = SwiGLU(dim, mlp_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_model.py -v -k "block"
```
Expected: all block tests pass (existing 2 + new 3 = 5).

- [ ] **Step 5: Run full test suite to ensure nothing else broke**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: Block variant flag (vanilla | diff)"
```

---

## Task 4: Add variant flag to Transformer

Pass variant + layer_idx through to each Block.

**Files:**
- Modify: `model.py` (Transformer class)
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_model.py`:

```python
def test_transformer_diff_variant_forward_shape():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int32))
    logits = model(x)
    assert logits.shape == (2, 64, cfg.vocab_size)


def test_transformer_diff_variant_blocks_are_diff():
    from model import DiffAttention
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    assert all(isinstance(b.attn, DiffAttention) for b in model.blocks)


def test_transformer_diff_layer_idx_is_1_indexed():
    """First block has layer_idx=1, last has layer_idx=n_layers."""
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    assert model.blocks[0].attn.layer_idx == 1
    assert model.blocks[-1].attn.layer_idx == cfg.n_layers


def test_transformer_vanilla_still_works():
    """Default variant still works without the new arg."""
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int32))
    logits = model(x)
    assert logits.shape == (2, 64, cfg.vocab_size)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_model.py -v -k "transformer_diff or transformer_vanilla"
```
Expected: 3 failures (Transformer doesn't take variant), 1 pass (vanilla default).

- [ ] **Step 3: Update `Transformer` in `model.py`**

Replace the existing `Transformer` class with:

```python
class Transformer(nn.Module):
    """Pre-norm LLaMA-style transformer with tied embeddings.

    variant="vanilla" uses VanillaMHA in every block (Phase A baseline).
    variant="diff"    uses DiffAttention in every block (Phase B+).

    forward(token_ids: (B, T)) -> logits: (B, T, vocab_size)
    """
    def __init__(self, cfg, variant: str = "vanilla"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [
            Block(
                dim=cfg.dim,
                n_heads_vanilla=cfg.n_heads_vanilla,
                qk_head_dim=cfg.qk_head_dim,
                mlp_intermediate=cfg.mlp_intermediate,
                variant=variant,
                layer_idx=(i + 1),  # 1-indexed for paper's lambda_init schedule
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
            )
            for i in range(cfg.n_layers)
        ]
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.tok_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_embed.weight.T
        return logits
```

- [ ] **Step 4: Run all model tests**

```bash
pytest tests/test_model.py -v
```
Expected: all passes (Phase A tests still green + new diff-variant tests pass).

- [ ] **Step 5: Run full test suite**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: Transformer variant flag, layer_idx propagation to Block"
```

---

## Task 5: Paired-seed init protocol (`paired_init.py`)

Per design §9.7: build vanilla with seed s, copy backbone + attention projection weights to diff (which uses a separate RNG stream for lambda vectors). Save both state-dicts.

**Files:**
- Create: `paired_init.py`
- Create: `tests/test_paired_init.py`

- [ ] **Step 1: Write failing tests at `tests/test_paired_init.py`**

```python
from pathlib import Path
import mlx.core as mx
from paired_init import build_paired_models, save_paired_init, load_paired_init
from config import ModelConfig


def _flatten(d, prefix=""):
    out = {}
    if hasattr(d, "items"):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}.{i}"))
    else:
        out[prefix] = d
    return out


def test_paired_init_shared_weights_byte_identical():
    cfg = ModelConfig.stage0()
    vanilla, diff = build_paired_models(cfg, seed=0)
    flat_v = _flatten(vanilla.parameters())
    flat_d = _flatten(diff.parameters())
    # Backbone names: tok_embed, MLP, all RMSNorms (norm_attn, norm_mlp, final_norm), ALL attn projections
    shared_names = [n for n in flat_v if n in flat_d]
    assert len(shared_names) > 0
    for name in shared_names:
        # All names that exist in both should match byte-for-byte
        assert mx.array_equal(flat_v[name], flat_d[name]).item(), f"mismatch at {name}"


def test_paired_init_diff_has_lambda_and_subln_extras():
    cfg = ModelConfig.stage0()
    _, diff = build_paired_models(cfg, seed=0)
    flat_d = _flatten(diff.parameters())
    # Lambda vectors and subln scale only exist on diff side
    assert any("lambda_q1" in n for n in flat_d)
    assert any("lambda_k1" in n for n in flat_d)
    assert any("lambda_q2" in n for n in flat_d)
    assert any("lambda_k2" in n for n in flat_d)
    assert any("subln" in n for n in flat_d)


def test_paired_init_same_seed_gives_same_lambdas():
    """Diff-only params should be deterministic given the same seed (separate RNG stream is still seeded)."""
    cfg = ModelConfig.stage0()
    _, diff_a = build_paired_models(cfg, seed=42)
    _, diff_b = build_paired_models(cfg, seed=42)
    flat_a = _flatten(diff_a.parameters())
    flat_b = _flatten(diff_b.parameters())
    for n, v in flat_a.items():
        if "lambda_" in n:
            assert mx.array_equal(v, flat_b[n]).item(), f"non-deterministic lambda at {n}"


def test_save_and_load_paired_init_roundtrip(tmp_path):
    cfg = ModelConfig.stage0()
    vanilla, diff = build_paired_models(cfg, seed=0)
    save_paired_init(vanilla, diff, tmp_path)
    assert (tmp_path / "vanilla.safetensors").exists()
    assert (tmp_path / "diff.safetensors").exists()
    flat_v_before = _flatten(vanilla.parameters())
    flat_d_before = _flatten(diff.parameters())
    v_loaded, d_loaded = load_paired_init(cfg, tmp_path)
    flat_v_after = _flatten(v_loaded.parameters())
    flat_d_after = _flatten(d_loaded.parameters())
    for n, v in flat_v_before.items():
        assert mx.array_equal(v, flat_v_after[n]).item(), f"vanilla mismatch at {n}"
    for n, v in flat_d_before.items():
        assert mx.array_equal(v, flat_d_after[n]).item(), f"diff mismatch at {n}"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_paired_init.py -v
```
Expected: ImportError on `paired_init`.

- [ ] **Step 3: Implement `paired_init.py`**

```python
"""Paired-seed init protocol per design §9.7.

Build vanilla and diff models with byte-identical shared weights (embed, MLPs,
RMSNorms, ALL attention projections). Diff-only params (lambda vectors, subln
scale) use a separate RNG stream so vanilla/diff weight tensors don't interact.

Both state-dicts can then be saved/loaded so paired Stage 0/1/2 runs always
start from byte-identical shared init for clean δ analysis.
"""
from __future__ import annotations
from pathlib import Path
import mlx.core as mx

from config import ModelConfig
from model import Transformer
from checkpoint import save_checkpoint, load_checkpoint, _flatten


def _copy_shared_weights(src: dict, dst: dict) -> dict:
    """Return a new dict shaped like dst, with values from src for any matching
    leaf-name. Leaves with no src counterpart are kept from dst (diff-only params).
    """
    flat_src = _flatten(src)
    flat_dst = _flatten(dst)
    merged = {}
    for name, val in flat_dst.items():
        if name in flat_src and flat_src[name].shape == val.shape:
            merged[name] = flat_src[name]
        else:
            merged[name] = val
    return _unflatten_via_dst(merged, dst)


def _unflatten_via_dst(flat: dict, template: dict) -> dict:
    """Rebuild nested structure matching `template` using values from flat dict (keys are dotted paths)."""
    out = {}
    for k, v in template.items():
        if isinstance(v, dict):
            sub_template = v
            sub_flat = {kk[len(k) + 1:]: vv for kk, vv in flat.items() if kk.startswith(k + ".")}
            out[k] = _unflatten_via_dst(sub_flat, sub_template)
        elif isinstance(v, list):
            sub_list = []
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    prefix = f"{k}.{i}"
                    sub_flat = {kk[len(prefix) + 1:]: vv for kk, vv in flat.items() if kk.startswith(prefix + ".")}
                    sub_list.append(_unflatten_via_dst(sub_flat, item))
                else:
                    sub_list.append(flat.get(f"{k}.{i}", item))
            out[k] = sub_list
        else:
            out[k] = flat.get(k, v)
    return out


def build_paired_models(cfg: ModelConfig, seed: int) -> tuple[Transformer, Transformer]:
    """Build (vanilla, diff) Transformers with byte-identical shared weights.

    Protocol (design §9.7):
    1. Seed MLX random with `seed`, build vanilla. RNG consumed for: embed, MLPs, norms, attn projections.
    2. Re-seed MLX random with `seed` + an offset, build diff. RNG consumed for: same backbone + 4 lambda vectors.
    3. Copy vanilla's backbone + ALL attention projections (which have matching shapes) into diff's state-dict.
       Diff retains its own lambda vectors and subln scale (init from its RNG stream).
    """
    # Step 1: vanilla
    mx.random.seed(seed)
    vanilla = Transformer(cfg, variant="vanilla")

    # Step 2: diff (separate RNG stream by re-seeding with offset)
    # Using a large prime offset keeps the lambda RNG well-separated from the backbone RNG.
    mx.random.seed(seed + 1_000_003)
    diff = Transformer(cfg, variant="diff")

    # Step 3: copy shared weights vanilla → diff
    new_diff_params = _copy_shared_weights(vanilla.parameters(), diff.parameters())
    diff.update(new_diff_params)
    return vanilla, diff


def save_paired_init(vanilla: Transformer, diff: Transformer, out_dir: Path) -> None:
    """Save both state-dicts to `out_dir/{vanilla,diff}.safetensors`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(vanilla.parameters(), step=0, ckpt_path=out_dir / "vanilla.safetensors")
    save_checkpoint(diff.parameters(), step=0, ckpt_path=out_dir / "diff.safetensors")


def load_paired_init(cfg: ModelConfig, in_dir: Path) -> tuple[Transformer, Transformer]:
    """Load both state-dicts and return constructed Transformers."""
    in_dir = Path(in_dir)
    vanilla = Transformer(cfg, variant="vanilla")
    diff = Transformer(cfg, variant="diff")
    v_params, _ = load_checkpoint(in_dir / "vanilla.safetensors")
    d_params, _ = load_checkpoint(in_dir / "diff.safetensors")
    vanilla.update(v_params)
    diff.update(d_params)
    return vanilla, diff
```

- [ ] **Step 4: Run paired_init tests**

```bash
pytest tests/test_paired_init.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add paired_init.py tests/test_paired_init.py
git commit -m "paired_init: byte-identical shared-weight protocol for vanilla/diff (design §9.7)"
```

---

## Task 6: Generate PyTorch reference fixture

One-time script that uses the official PyTorch implementation to produce a reference output for a tiny toy shape. The .npz fixture is committed; the test then needs only numpy + MLX (no torch).

**Files:**
- Create: `scripts/generate_ref_fixture.py`
- Create: `data/ref_fixtures/` (directory)
- Create: `data/ref_fixtures/diffattn_toy_v1.npz` (generated artifact, committed)

- [ ] **Step 1: Install torch (one-time dev dep)**

```bash
source .venv/bin/activate
pip install torch
```

Expected: pytorch installs cleanly with MPS support on Apple Silicon (~600 MB).

- [ ] **Step 2: Vendor the reference impl**

Download the official `multihead_diffattn.py` from microsoft/unilm at a pinned commit and save locally:

```bash
mkdir -p data/ref_fixtures
curl -L -o data/ref_fixtures/multihead_diffattn_reference.py \
  https://raw.githubusercontent.com/microsoft/unilm/master/Diff-Transformer/multihead_diffattn.py
```

Then verify:
```bash
head -5 data/ref_fixtures/multihead_diffattn_reference.py
```
Expected: starts with the file's actual header (imports, class definition for `MultiheadDiffAttn`).

- [ ] **Step 3: Implement `scripts/generate_ref_fixture.py`**

```python
"""Generate a PyTorch reference fixture for the diff-attn cross-check test.

One-time setup. Saves: data/ref_fixtures/diffattn_toy_v1.npz with:
- input_x: (B, T, dim) random tensor
- weights: q_proj, k_proj, v_proj, out_proj weight matrices
- lambda vectors (q1, k1, q2, k2) + λ_init
- output: reference output of the PyTorch MultiheadDiffAttn forward
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "ref_fixtures"))

import math
import numpy as np
import torch

# Import the vendored reference impl
import multihead_diffattn_reference as ref

# Toy shape (design §7.4): (B=2, T=16, dim=64, n_heads_diff=2, qk_head_dim=16).
# Reference takes num_heads (the *baseline* heads); n_heads_diff = num_heads / 2.
# Here num_heads=4 (vanilla baseline), giving n_heads_diff=2.
B, T, DIM = 2, 16, 64
NUM_HEADS_BASELINE = 4
DEPTH = 2  # layer_idx (1-indexed in our code, but reference may be 0-indexed; check)

torch.manual_seed(42)

# Build the reference module
attn = ref.MultiheadDiffAttn(
    embed_dim=DIM,
    depth=DEPTH - 1,        # reference is 0-indexed
    num_heads=NUM_HEADS_BASELINE,
)
attn.eval()  # no dropout

# Random input
x = torch.randn(B, T, DIM, dtype=torch.float32)

# Forward
with torch.no_grad():
    # Reference signature: forward(x, rel_pos=None, attn_mask=None) — check
    output = attn(x)

# Extract weights for the MLX side
state = {k: v.detach().cpu().numpy() for k, v in attn.state_dict().items()}

# Save fixture
out_path = Path(__file__).resolve().parent.parent / "data" / "ref_fixtures" / "diffattn_toy_v1.npz"
np.savez(
    out_path,
    input_x=x.detach().cpu().numpy(),
    output=output.detach().cpu().numpy(),
    B=B, T=T, dim=DIM, num_heads_baseline=NUM_HEADS_BASELINE, depth=DEPTH,
    **{f"weight__{k.replace('.', '__')}": v for k, v in state.items()},
)

print(f"Saved fixture: {out_path}")
print(f"Input shape: {x.shape}; Output shape: {output.shape}")
print(f"Output stats: mean={output.mean():.4f} std={output.std():.4f} "
      f"min={output.min():.4f} max={output.max():.4f}")
print(f"State-dict keys: {sorted(state.keys())}")
```

- [ ] **Step 4: Run the script**

```bash
python scripts/generate_ref_fixture.py
```
Expected: prints "Saved fixture: ...diffattn_toy_v1.npz" plus shape/stats info. The script may fail with a name mismatch (the reference might be named differently); inspect `data/ref_fixtures/multihead_diffattn_reference.py` for the actual class name and adjust import/constructor accordingly.

- [ ] **Step 5: Commit fixture + script + vendored reference**

```bash
git add scripts/generate_ref_fixture.py data/ref_fixtures/multihead_diffattn_reference.py data/ref_fixtures/diffattn_toy_v1.npz
git commit -m "ref_fixtures: vendored PyTorch reference + generated toy-shape output"
```

(Note: the .npz is small — a few hundred KB — fine to commit. Torch can be uninstalled after this step if you want to keep the venv lean.)

---

## Task 7: Reference cross-check test

Load the fixture, copy the PyTorch weights into our MLX DiffAttention, run the forward, compare outputs.

**Files:**
- Create: `tests/test_diff_reference.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_diff_reference.py
"""Cross-check our MLX DiffAttention against a fixture generated from the
official PyTorch microsoft/unilm/Diff-Transformer reference.

The fixture is generated by scripts/generate_ref_fixture.py (one-time) and
contains: input tensor, PyTorch weights, PyTorch reference output.

This test loads those, copies weights into the MLX module, runs forward,
and compares to within 1e-3 (fp32) tolerance per design §7.4.
"""
from pathlib import Path
import math
import numpy as np
import mlx.core as mx
from model import DiffAttention

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "ref_fixtures" / "diffattn_toy_v1.npz"


def test_mlx_diff_attention_matches_pytorch_reference():
    if not FIXTURE.exists():
        import pytest
        pytest.skip(f"fixture missing: {FIXTURE}. Run scripts/generate_ref_fixture.py once.")

    data = np.load(FIXTURE)
    B = int(data["B"])
    T = int(data["T"])
    DIM = int(data["dim"])
    NUM_HEADS_BASELINE = int(data["num_heads_baseline"])
    DEPTH = int(data["depth"])
    qk_head_dim = DIM // NUM_HEADS_BASELINE  # in reference, head_dim = embed_dim // num_heads // 2
    # But wait — reference does head_dim = embed_dim // num_heads // 2. So qk_head_dim there = DIM/NUM_HEADS/2.
    # Our DiffAttention takes (dim, n_heads_vanilla, qk_head_dim) where n_heads_vanilla = NUM_HEADS * 2 and qk_head_dim = embed_dim / (n_heads_vanilla * 2)... wait, this needs care.
    # Adapt this test once fixture is in hand to match the reference's exact naming.

    input_x = mx.array(data["input_x"].astype(np.float32))
    expected_out = data["output"].astype(np.float32)

    attn = DiffAttention(
        dim=DIM,
        n_heads_vanilla=NUM_HEADS_BASELINE,  # reference's num_heads is our n_heads_vanilla
        qk_head_dim=qk_head_dim,
        layer_idx=DEPTH,
    )
    # Copy weights from fixture: q_proj, k_proj, v_proj, out_proj, lambda vectors, subln scale
    # Names in fixture are weight__<dotted_path> (with . → __)
    # Build mapping from reference names to our names. Will need adjustment based on actual fixture keys.
    weights_to_copy = {}
    for k in data.files:
        if k.startswith("weight__"):
            ref_name = k[len("weight__"):].replace("__", ".")
            weights_to_copy[ref_name] = mx.array(data[k].astype(np.float32))

    # Heuristic name mapping; finalize after inspecting fixture keys.
    name_map = {
        "q_proj.weight": "q_proj.weight",
        "k_proj.weight": "k_proj.weight",
        "v_proj.weight": "v_proj.weight",
        "out_proj.weight": "o_proj.weight",
        "lambda_q1": "lambda_q1",
        "lambda_k1": "lambda_k1",
        "lambda_q2": "lambda_q2",
        "lambda_k2": "lambda_k2",
        "subln.weight": "subln.scale",
    }
    target_params = attn.parameters()
    for ref_name, our_name in name_map.items():
        if ref_name in weights_to_copy:
            # Apply to attn.parameters() flat dict; need to find and update
            _set_nested(target_params, our_name.split("."), weights_to_copy[ref_name])
    attn.update(target_params)

    actual_out = attn(input_x)
    actual_np = np.array(actual_out)
    diff = np.abs(actual_np - expected_out).max()
    assert diff < 1e-3, f"max |diff| = {diff:.3e}; expected < 1e-3"


def _set_nested(d, path, value):
    cur = d
    for p in path[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(p, {})
        else:
            raise KeyError(f"cannot descend into {p}")
    cur[path[-1]] = value
```

- [ ] **Step 2: Run test (initially expected to fail)**

```bash
pytest tests/test_diff_reference.py -v
```
Expected: probably fails — the name mapping and possibly the reshape conventions need adjustment based on what the PyTorch reference actually does. **This is a research test**: when it fails, inspect the fixture keys, adapt the mapping, and iterate until the diff is < 1e-3.

- [ ] **Step 3: Iterate on the name mapping until the test passes**

This is the meat of the cross-check. Likely adjustments:
- Linear weight transpose convention (PyTorch stores `Linear.weight` as `(out_features, in_features)`; MLX is the same. Check first.)
- subln name (PyTorch `subln.weight` ↔ our `subln.scale`)
- The reference may flatten heads differently; reshape conventions may need a transpose
- The reference may use `num_kv_heads` (GQA-style); our impl assumes full MHA (n_kv = n_heads). Confirm by reading the vendored reference file.

If the diff stays above tolerance, **document the discrepancy in `data/ref_fixtures/INVESTIGATION.md`** explaining what's different. The cross-check is necessary but not sufficient for the science; if it can't be made to converge perfectly, the v0 oracle agreement (Task 2 test) is the next-best evidence of correctness.

- [ ] **Step 4: Once green, commit**

```bash
git add tests/test_diff_reference.py
git commit -m "test: PyTorch reference cross-check for DiffAttention (design §7.4 gate)"
```

---

## Task 8: Split `seed` into `data_seed` + `model_seed` in train.py

Phase A used one seed for both. Phase B needs them split so paired runs share data ordering while varying model init.

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train_driver.py`

- [ ] **Step 1: Update existing test to use new API**

In `tests/test_train_driver.py`, change the `train_run` call to use explicit `data_seed` and `model_seed`:

```python
def test_train_run_smoke_writes_metrics_and_checkpoint(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    run_dir = tmp_path / "runs" / "smoke"
    model_cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
        vocab_size=100_277, mlp_intermediate=64, block_size=64,
    )
    train_cfg = TrainConfig(
        peak_lr=1e-3, warmup_steps=10, total_tokens=50 * 64 * 2,
        micro_batch=2, eval_every=20, full_eval_every=50,
        monitoring_tokens=500, full_eval_tokens=2000, save_every=25,
    )
    train_run(model_cfg, train_cfg, shards_dir, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")
    assert (run_dir / "metrics.jsonl").exists()
    # ... rest of assertions unchanged
```

Add a new test for separability:

```python
def test_data_seed_determines_batch_order(tmp_path):
    """Same data_seed → same first batch regardless of model_seed."""
    shards_dir = _make_synthetic_shards(tmp_path)
    from data.loader import ShardLoader, sample_batch
    import numpy as np
    loader = ShardLoader(shards_dir, "train")
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    x_a, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng_a)
    x_b, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng_b)
    np.testing.assert_array_equal(x_a, x_b)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_train_driver.py -v
```
Expected: existing test fails (signature change), new test passes (just exercises numpy rng).

- [ ] **Step 3: Update `train.py`**

Replace the `train_run` signature and seed handling:

```python
def train_run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    shards_dir: Path,
    run_dir: Path,
    data_seed: int = 0,
    model_seed: int = 0,
    variant: str = "vanilla",
    init_state_dict: dict | None = None,
) -> None:
    """Run one training stage. Writes metrics, checkpoints, metadata.

    Args:
      data_seed: seeds the data sampler (shared across paired runs for paired δ).
      model_seed: seeds model init (varies per seed in paired analysis).
      variant: "vanilla" or "diff".
      init_state_dict: optional pre-built state dict to override default init (used
        for paired-seed init protocol — pass diff's state-dict copied from vanilla).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(shards_dir)

    # Seed MLX (controls model init) and numpy (controls data sampling — via rng below)
    mx.random.seed(model_seed)
    rng = np.random.default_rng(data_seed)

    train_loader = ShardLoader(shards_dir, "train")
    val_loader = ShardLoader(shards_dir, "val")
    data_meta = json.loads((shards_dir / "meta.json").read_text())

    model = Transformer(model_cfg, variant=variant)
    if init_state_dict is not None:
        model.update(init_state_dict)

    optimizer = make_adamw(
        lr=0.0,
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.adam_beta1,
        beta2=train_cfg.adam_beta2,
        eps=train_cfg.adam_eps,
    )

    save_run_metadata(
        run_dir=run_dir, model_cfg=model_cfg,
        train_cfg_dict=asdict(train_cfg),
        git_hash=current_hash(), git_dirty=is_dirty(),
        mlx_version=_mlx_version(),
        seed=model_seed, data_meta=data_meta,
    )
    (run_dir / "variant.txt").write_text(variant + "\n")
    (run_dir / "seeds.txt").write_text(f"data_seed={data_seed}\nmodel_seed={model_seed}\n")

    # Rest of the loop unchanged
    eff_tokens_per_step = train_cfg.micro_batch * model_cfg.block_size * train_cfg.grad_accum
    total_steps = max(1, train_cfg.total_tokens // eff_tokens_per_step)
    print(f"[train] variant={variant} {total_steps} steps, ~{eff_tokens_per_step} tokens/step, "
          f"{train_cfg.total_tokens / 1e6:.1f}M total tokens")

    logger = MetricsLogger(run_dir / "metrics.jsonl")
    t0 = time.time()
    step = 0
    while step < total_steps:
        lr = cosine_lr_with_warmup(
            step, train_cfg.peak_lr, train_cfg.warmup_steps, total_steps,
            min_lr_frac=0.1,
        )
        optimizer.learning_rate = lr
        x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
        x = mx.array(x_np)
        y = mx.array(y_np)
        loss = train_step(model, optimizer, x, y, grad_clip=train_cfg.grad_clip)
        do_full_eval = (step > 0 and step % train_cfg.full_eval_every == 0)
        do_monitor_eval = (step > 0 and step % train_cfg.eval_every == 0)
        record = {
            "step": step,
            "train_loss": loss,
            "lr": lr,
            "tps": int(eff_tokens_per_step * (step + 1) / max(1e-6, time.time() - t0)),
            "wall": round(time.time() - t0, 1),
        }
        if do_monitor_eval:
            val = compute_val_loss(model, val_loader, model_cfg.block_size,
                                   train_cfg.micro_batch, train_cfg.monitoring_tokens)
            record["val_loss_monitor"] = val
        if do_full_eval:
            val_full = compute_val_loss(model, val_loader, model_cfg.block_size,
                                        train_cfg.micro_batch, train_cfg.full_eval_tokens)
            record["val_loss_full"] = val_full
        logger.log(**record)
        if (step > 0 and step % train_cfg.save_every == 0) or step == total_steps - 1:
            save_checkpoint(model.parameters(), step=step, ckpt_path=run_dir / "latest.safetensors")
        step += 1
    logger.close()
    print(f"[train] done in {time.time() - t0:.1f}s")
```

And update `main()` to accept both seeds:

```python
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    args = p.parse_args()
    model_cfg, train_cfg = _build_cfgs(args.stage)
    train_run(model_cfg, train_cfg, args.shards_dir, args.run_dir,
              data_seed=args.data_seed, model_seed=args.model_seed,
              variant=args.variant)
```

Note: `main()` no longer rejects `--variant=diff` — diff is now supported.

- [ ] **Step 4: Run all tests**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_driver.py
git commit -m "train: split seed → data_seed + model_seed; support variant=diff; init_state_dict arg"
```

---

## Task 9: Stage 0 paired runner script

A script that builds the paired init, then runs vanilla and diff to completion at Stage 0 scale (~73 min each, ~2.5 hours total).

**Files:**
- Create: `scripts/stage0_paired.py`

- [ ] **Step 1: Implement the script**

```python
"""Stage 0 paired smoke run.

Build vanilla + diff with paired-seed init (design §9.7), train both to
completion at Stage 0 scale, save metrics for paired δ analysis.

Run from project root with venv activated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from config import ModelConfig, TrainConfig
from paired_init import build_paired_models, save_paired_init
from train import train_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--out_root", type=Path, default=Path("runs"))
    args = p.parse_args()

    model_cfg = ModelConfig.stage0()
    train_cfg = TrainConfig.stage0()

    # Build paired init and save for reproducibility
    init_dir = args.out_root / f"init-seed{args.model_seed}"
    print(f"[paired] building paired init at {init_dir}")
    vanilla, diff = build_paired_models(model_cfg, seed=args.model_seed)
    save_paired_init(vanilla, diff, init_dir)

    # Vanilla run
    vanilla_dir = args.out_root / f"stage0-paired-vanilla-seed{args.model_seed}"
    print(f"[paired] running vanilla at {vanilla_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, vanilla_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="vanilla",
        init_state_dict=vanilla.parameters(),
    )

    # Diff run
    diff_dir = args.out_root / f"stage0-paired-diff-seed{args.model_seed}"
    print(f"[paired] running diff at {diff_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, diff_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="diff",
        init_state_dict=diff.parameters(),
    )

    print(f"[paired] done. Compare final val losses in:")
    print(f"  {vanilla_dir}/metrics.jsonl")
    print(f"  {diff_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the script with reduced step count to verify the orchestration works**

Edit `scripts/stage0_paired.py` to use a smaller `total_tokens` for the dry-run, OR create a quick `stage0_paired_dryrun.py`:

```python
# scripts/stage0_paired_dryrun.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
from config import ModelConfig, TrainConfig
from paired_init import build_paired_models
from train import train_run


def main():
    model_cfg = ModelConfig.stage0()
    # ~30 steps each
    train_cfg = replace(TrainConfig.stage0(), total_tokens=30 * 16 * 1024)

    vanilla, diff = build_paired_models(model_cfg, seed=0)

    train_run(model_cfg, train_cfg, Path("data/shards"), Path("runs/stage0-paired-dryrun-vanilla"),
              data_seed=0, model_seed=0, variant="vanilla", init_state_dict=vanilla.parameters())
    train_run(model_cfg, train_cfg, Path("data/shards"), Path("runs/stage0-paired-dryrun-diff"),
              data_seed=0, model_seed=0, variant="diff", init_state_dict=diff.parameters())


if __name__ == "__main__":
    main()
```

Run:
```bash
source .venv/bin/activate
time python scripts/stage0_paired_dryrun.py
```
Expected: ~30 seconds total (15s × 2 runs). Both runs produce `metrics.jsonl` with sane loss curves.

- [ ] **Step 3: Verify the dry-run results**

```bash
python -c "
import json
for variant in ['vanilla', 'diff']:
    p = f'runs/stage0-paired-dryrun-{variant}/metrics.jsonl'
    records = [json.loads(l) for l in open(p)]
    print(f'{variant}: step0_loss={records[0][\"train_loss\"]:.3f} final_loss={records[-1][\"train_loss\"]:.3f}')
"
```
Expected output (rough): both variants start at ~11.99 (random init NLL) and descend by step 30. Diff and vanilla should be CLOSE (their backbone is identical, only attention differs).

- [ ] **Step 4: Commit the scripts**

```bash
git add scripts/stage0_paired.py scripts/stage0_paired_dryrun.py
git commit -m "scripts: stage0 paired runner + dry-run validator"
```

---

## Task 10: Execute Stage 0 paired run

The actual experiment. ~2.5 hours wall time (2 × 73 min based on Phase A calibration).

**Files:** None (creates run output in `runs/`)

- [ ] **Step 1: Confirm machine state**

- Close LM Studio / other GPU consumers
- Connect to AC power
- Ensure data/shards/ is populated (from Phase A)

```bash
ls -lh data/shards/ | head -5
```

- [ ] **Step 2: Launch the run**

```bash
source .venv/bin/activate
caffeinate -di python scripts/stage0_paired.py --model_seed 0 --data_seed 0 \
  2>&1 | tee runs/stage0-paired.log
```
Expected wall time: ~2.5 hours (~73 min × 2 + paired-init overhead).

- [ ] **Step 3: Analyze paired δ**

```bash
python -c "
import json

def final_val(path):
    records = [json.loads(l) for l in open(path)]
    # Get last record with a val_loss_full
    fulls = [r for r in records if 'val_loss_full' in r]
    if not fulls:
        # Fall back to monitor
        monitors = [r for r in records if 'val_loss_monitor' in r]
        return monitors[-1]['val_loss_monitor'] if monitors else None
    return fulls[-1]['val_loss_full']

van = final_val('runs/stage0-paired-vanilla-seed0/metrics.jsonl')
dif = final_val('runs/stage0-paired-diff-seed0/metrics.jsonl')
print(f'Vanilla final val: {van:.4f}')
print(f'Diff    final val: {dif:.4f}')
delta = dif - van
print(f'δ = diff - vanilla = {delta:+.4f} ({\"diff wins\" if delta < 0 else \"vanilla wins\"})')
print()
print('Note: Stage 0 (~100M tokens, 30M params) is undertrained; expect both to be similar.')
print('The point of Stage 0 paired is to validate pipeline, not to detect diff signal.')
"
```

- [ ] **Step 4: Write run NOTES.md**

Create `runs/stage0-paired-vanilla-seed0/NOTES.md` and `runs/stage0-paired-diff-seed0/NOTES.md` summarizing each run's final loss, wall time, throughput, NaN/spike status. Format similar to `runs/stage0-vanilla-seed0/NOTES.md` from Phase A.

- [ ] **Step 5: Commit the notes (force-add since runs/ is gitignored)**

```bash
git add -f runs/stage0-paired-vanilla-seed0/NOTES.md \
            runs/stage0-paired-diff-seed0/NOTES.md
git commit -m "stage0-paired: vanilla + diff seed 0 runs complete; notes capture results"
```

---

## Task 11: Phase B retro

**Files:**
- Create: `docs/2026-05-20-phase-b-retro.md`

- [ ] **Step 1: Run all tests once more**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 2: Write the retro**

```markdown
# Phase B retro

**Date:** <today>
**Status:** Complete. Diff-attn v0 implemented, paired-seed init protocol working, Stage 0 paired smoke run done.

## What works
- DiffAttention module (paper-canonical): H_diff = n_heads_vanilla/2, qk_dim = D, v_dim = 2D. v0 = two SDPA calls + Python subtract per design §7.1 linearity rewrite.
- Lambda machinery: per-layer fp32 vectors (4 × qk_head_dim), depth-scheduled λ_init, fp32 lambda scalar broadcast over (B, H, T, 2D).
- subln: per-head RMSNorm over 2D, applied AFTER differential subtraction.
- Block + Transformer variant flag (`variant="vanilla"|"diff"`).
- Paired-seed init protocol (design §9.7): byte-identical backbone + attention projections; diff-only lambda/subln from separate RNG stream.
- Reference cross-check vs PyTorch impl: <fill in tolerance achieved>.
- Stage 0 paired (vanilla + diff at seed 0): <fill in wall time, final losses, δ>.

## Numbers
| | Vanilla seed 0 | Diff seed 0 | δ |
|---|---|---|---|
| Final train_loss | ... | ... | ... |
| Final val_full | ... | ... | ... |
| Wall time | ... | ... | — |
| TPS | ... | ... | — |

## What's brittle (Phase D prereqs)
- Precision is pure fp32 — bf16 mixed precision still not wired in. Affects Stage 1/2 memory + throughput.
- Optimizer state not saved in checkpoints — Stage 1/2 long runs need resume capability.
- `grad_accum` field ignored — Stage 2 needs grad_accum=4.

## Ready for Phase C?
- [ ] All Phase B tests green
- [ ] Stage 0 paired loss curves sane (both descending, no NaN)
- [ ] DiffAttention matches PyTorch reference within design tolerance
- [ ] Paired-seed init verified byte-identical on shared params

## What Phase C adds
- Custom Metal kernels: P1 (softmax preflight), P2 (causal SDPA preflight), v1 (two P2-style calls + subtract)
- Kernel correctness gates (v1 vs v0 numerical agreement)
- Memory gate (full forward+backward+optimizer step at Stage 1/2 shapes)
- Kernel speed eval (vs v0, vs MLX SDPA)
```

- [ ] **Step 3: Commit**

```bash
git add docs/2026-05-20-phase-b-retro.md
git commit -m "docs: Phase B retrospective"
```

---

## Task 12: Tag the Phase B milestone

- [ ] **Step 1: Tag**

```bash
git tag -a phase-b-complete -m "Phase B: diff-attn v0 + paired-seed protocol + reference cross-check + Stage 0 paired smoke run complete"
```

- [ ] **Step 2: Merge to main**

```bash
git checkout main
git merge --no-ff phase-b-diffattn -m "Merge phase-b-diffattn: Phase B complete"
git log --oneline -5
```

Phase B done. Next: write Phase C plan (custom Metal kernels).

---

## Self-review against the design doc

Spec coverage checklist:

- ✅ DiffAttention paper-canonical dims (Task 2)
- ✅ Lambda parameter machinery + depth schedule (Tasks 1, 2)
- ✅ subln per-head RMSNorm over 2D (Task 2)
- ✅ v0 = two SDPA calls + subtract (Task 2)
- ✅ Variant flag (vanilla | diff) on Block (Task 3) and Transformer (Task 4)
- ✅ Paired-seed init protocol (Task 5)
- ✅ Reference cross-check vs PyTorch (Tasks 6-7)
- ✅ Split data_seed + model_seed (Task 8)
- ✅ Stage 0 paired smoke run (Tasks 9-10)

Phase B out-of-scope (deferred to later phases):
- Custom Metal kernels → Phase C
- Stages 1 and 2 full runs → Phase D
- bf16 mixed precision → Phase D prereq
- Optimizer state in checkpoints → Phase D prereq
- grad_accum → Phase D prereq
- AR-hit eval slice → Phase D (optional)
- Final writeup → Phase D
