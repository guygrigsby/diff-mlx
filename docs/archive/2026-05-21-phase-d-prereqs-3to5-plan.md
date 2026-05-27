# Phase D prereqs 3-5 implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement task-by-task.

**Goal:** Close the three remaining Phase D prerequisites from `docs/2026-05-20-phase-a-retro.md`:
- (#3) Save and restore optimizer state in checkpoints; auto-resume from latest checkpoint
- (#4) Honor `TrainConfig.grad_accum` in the training loop (currently a no-op except for token counting)
- (#5) Multi-seed orchestration script for paired runs across N seeds

**Architecture:** All three are local changes against the existing single-process training loop. No new abstractions, no new dependencies. The bf16 work merged on `main`; the test suite sits at 98/98.

**Tech stack:** MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`), numpy, pytest, bash. Working dir `diff-mlx`. Venv at `.venv/`.

**Design notes:**
- **Optimizer state serialization:** safetensors only stores flat string→array dicts. Use a single safetensors file per checkpoint with two namespaces: keys prefixed `model.` for params and `opt.` for optimizer state. Step + variant + amp_dtype go in the safetensors metadata. RNG state is out of scope for this round (sampling RNG is `numpy.random.default_rng(data_seed)`, deterministic from seed; resume picks up at the next step, advancing the sampler past the consumed positions is a separate problem and not load-bearing for crash-recovery).
- **Resume policy:** at the start of `train_run`, look for `latest.safetensors` in `run_dir`. If present and not size-zero, load it, restore params + optimizer state, set `step = saved_step + 1`, log a "resuming from step N" line. If absent, start fresh. This means re-running `python -m train ...` against an existing run dir resumes automatically, which is the desired behavior for both crash recovery and intentional continuation. To start over from scratch, the user deletes the run dir (or `latest.safetensors`). No CLI flag for "force fresh" needed; deleting the file is unambiguous.
- **grad_accum semantics:** sample `grad_accum` micro-batches per outer step. Forward + backward each one, accumulate gradients into a running dict, then call `optimizer.update` once on the averaged accumulated grads. Logged `train_loss` is the mean over the micro-batches in that outer step. The existing `eff_tokens_per_step` math (already accounts for grad_accum) becomes correct automatically.
- **Multi-seed orchestration:** small bash script that loops over N seeds, invokes `scripts/stage0_paired.sh` once per seed with `MODEL_SEED=$i DATA_SEED=$i OUT_ROOT=...-seed$i`. Caffeinate already handled by the per-pair script.

---

## Task 1: Round-trip save/load of optimizer state

**Files:**
- Modify: `checkpoint.py` (`save_checkpoint`, `load_checkpoint`)
- Test: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_checkpoint.py`:

```python
from optim import make_adamw


def test_save_and_load_roundtrips_optimizer_state(tmp_path):
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)

    # Force the optimizer to materialize state by doing one update.
    import mlx.nn as nn

    def loss_fn(m, tokens):
        return (m(tokens) ** 2).sum()

    tokens = mx.array([[0, 1, 2, 3]])
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, tokens)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)

    ckpt = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=7, ckpt_path=ckpt, optim_state=opt.state)

    loaded_params, step, loaded_opt_state = load_checkpoint(ckpt)
    assert step == 7

    # Spot-check: the optimizer's m/v buffers for a known param should round-trip
    def find_first_array(d):
        if isinstance(d, dict):
            for v in d.values():
                r = find_first_array(v)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for v in d:
                r = find_first_array(v)
                if r is not None:
                    return r
        elif isinstance(d, mx.array):
            return d
        return None

    orig_arr = find_first_array(opt.state)
    loaded_arr = find_first_array(loaded_opt_state)
    assert orig_arr is not None and loaded_arr is not None
    assert mx.array_equal(orig_arr, loaded_arr).item()


def test_load_checkpoint_without_optim_state_returns_none(tmp_path):
    """Backward compatibility: old checkpoints without opt state still load."""
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    ckpt = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=3, ckpt_path=ckpt)
    loaded_params, step, opt_state = load_checkpoint(ckpt)
    assert step == 3
    assert opt_state is None
```

- [ ] **Step 2: Run to confirm failure**

```
source .venv/bin/activate
pytest tests/test_checkpoint.py::test_save_and_load_roundtrips_optimizer_state tests/test_checkpoint.py::test_load_checkpoint_without_optim_state_returns_none -v
```

Expected: TypeError or assertion error (load_checkpoint currently returns a 2-tuple).

- [ ] **Step 3: Extend `save_checkpoint`**

Modify `checkpoint.py`. Replace `save_checkpoint`:

```python
def save_checkpoint(params: dict, step: int, ckpt_path: Path,
                    optim_state=None, rng_state=None) -> None:
    """Save params + optional optimizer state to a single safetensors file.

    Keys are namespaced: `model.<path>` for params, `opt.<path>` for optimizer
    state. Step is recorded in safetensors metadata. rng_state is currently
    accepted but not yet persisted (left as a future hook).
    """
    flat_model = {f"model.{k}": v for k, v in _flatten(params).items()}
    bundle = dict(flat_model)
    if optim_state is not None:
        flat_opt = {f"opt.{k}": v for k, v in _flatten(optim_state).items()
                    if isinstance(v, mx.array)}
        bundle.update(flat_opt)
    metadata = {"step": str(step), "has_opt": "1" if optim_state is not None else "0"}
    mx.save_safetensors(str(ckpt_path), bundle, metadata=metadata)
```

Replace `load_checkpoint`:

```python
def load_checkpoint(ckpt_path: Path) -> tuple[dict, int, dict | None]:
    """Load a checkpoint. Returns (params, step, optim_state-or-None).

    Splits the namespaced keys back into the two pytrees. An older checkpoint
    without optimizer state returns None for the third element.
    """
    loaded = mx.load(str(ckpt_path), return_metadata=True)
    tensors, metadata = loaded
    step = int(metadata.get("step", "0"))
    has_opt = metadata.get("has_opt", "0") == "1"

    model_flat = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    if not model_flat:
        # legacy checkpoint: no namespace prefix
        model_flat = dict(tensors)
    params = _unflatten(model_flat)

    optim_state: dict | None = None
    if has_opt:
        opt_flat = {k[len("opt."):]: v for k, v in tensors.items() if k.startswith("opt.")}
        optim_state = _unflatten(opt_flat) if opt_flat else None

    return params, step, optim_state
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_checkpoint.py -v
```

Expected: existing 2 tests pass (the original `test_save_and_load_roundtrip` may need adjustment for the 3-tuple return; if it fails, that's expected and is fixed in Step 5).

- [ ] **Step 5: Update the existing `test_save_and_load_roundtrip` to unpack the 3-tuple**

Edit `tests/test_checkpoint.py`. Find the existing `test_save_and_load_roundtrip`:

```python
def test_save_and_load_roundtrip(tmp_path):
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat_before = _flatten(model.parameters())
    ckpt_path = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=1234, ckpt_path=ckpt_path)
    loaded, step = load_checkpoint(ckpt_path)
    ...
```

Change `loaded, step = load_checkpoint(ckpt_path)` to `loaded, step, _opt = load_checkpoint(ckpt_path)`.

- [ ] **Step 6: Run full suite**

```
pytest tests/ -q
```

Expected: 98 + 2 = 100 passed.

- [ ] **Step 7: Commit**

```bash
git add checkpoint.py tests/test_checkpoint.py
git commit -m "$(cat <<'EOF'
checkpoint: round-trip optimizer state alongside params

Single safetensors file with namespaced keys (model.* / opt.*).
load_checkpoint now returns (params, step, opt_state-or-None).
Legacy checkpoints without opt state load cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire resume into `train_run`

**Files:**
- Modify: `train.py:54-130` (the `train_run` body)
- Test: `tests/test_train_driver.py`

- [ ] **Step 1: Add a resume test**

Append to `tests/test_train_driver.py`. (If that file's existing imports don't include the needed bits, add them at the top.)

```python
def test_train_run_resumes_from_latest_checkpoint(tmp_path, monkeypatch):
    """Run a tiny training for 2 steps, then resume in a second train_run call.
    The second call should pick up at step 2 (not restart from 0).
    """
    from train import train_run
    from config import ModelConfig, TrainConfig
    from dataclasses import replace
    import json

    # Minimal stage0-like config but tiny + short
    model_cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=4, qk_head_dim=8,
        vocab_size=128, mlp_intermediate=64, block_size=16,
    )
    train_cfg = replace(
        TrainConfig.stage0(),
        total_tokens=64,          # micro_batch * block_size * 2 steps
        micro_batch=2,
        warmup_steps=0,
        eval_every=10_000,
        full_eval_every=10_000,
        save_every=1,
    )

    # Need a fake shards dir. The data loader uses ShardLoader + sample_batch;
    # we can stub by writing a minimal meta.json + at least one shard.
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "meta.json").write_text(json.dumps({
        "vocab_size": 128, "tiktoken_version": "0.13.0", "tokenizer_name": "test",
    }))
    # Write a single tiny train + val shard as int32 arrays.
    np.array(list(range(1024)), dtype=np.int32).tofile(shards / "train_000.bin")
    np.array(list(range(1024)), dtype=np.int32).tofile(shards / "val_000.bin")

    run_dir = tmp_path / "run"

    # First call: run from scratch.
    train_run(model_cfg, train_cfg, shards, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")

    metrics_first = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_first) >= 1
    last_step_first = json.loads(metrics_first[-1])["step"]

    # Bump total_tokens so the second call has more work to do.
    train_cfg_resumed = replace(train_cfg, total_tokens=128)
    train_run(model_cfg, train_cfg_resumed, shards, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")

    metrics_second = (run_dir / "metrics.jsonl").read_text().splitlines()
    last_step_second = json.loads(metrics_second[-1])["step"]
    assert last_step_second > last_step_first, (
        f"resume did not advance: first ended at {last_step_first}, second at {last_step_second}"
    )
```

If the data loader / sample_batch primitives expect a different shard format than what's written here, fall back to a simpler approach: monkeypatch `sample_batch` to return random arrays directly. The exact stubbing details depend on `data/loader.py`; investigate before writing.

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_train_driver.py::test_train_run_resumes_from_latest_checkpoint -v
```

Expected: the resume behavior doesn't exist yet, so the second call starts at step 0 and the assertion `last_step_second > last_step_first` fails.

- [ ] **Step 3: Add resume to `train_run`**

In `train.py`, after the optimizer is created (after line 75) and before `save_run_metadata`, add:

```python
    # Resume from latest.safetensors if present.
    latest_ckpt = run_dir / "latest.safetensors"
    start_step = 0
    if latest_ckpt.exists() and latest_ckpt.stat().st_size > 0:
        from checkpoint import load_checkpoint
        loaded_params, saved_step, loaded_opt_state = load_checkpoint(latest_ckpt)
        model.update(loaded_params)
        if loaded_opt_state is not None:
            optimizer.state = loaded_opt_state
        start_step = saved_step + 1
        print(f"[train] resuming from step {saved_step + 1} ({latest_ckpt})")
```

Change the training loop initializer (line 94) from `step = 0` to `step = start_step`.

And change the checkpoint save call (line 127) to pass optimizer state:

```python
            save_checkpoint(model.parameters(), step=step,
                            ckpt_path=run_dir / "latest.safetensors",
                            optim_state=optimizer.state)
```

- [ ] **Step 4: Run the test**

```
pytest tests/test_train_driver.py -v
```

Expected: pass (or pytest.skip if the data stubbing trick doesn't work; in that case revisit Step 1's approach).

- [ ] **Step 5: Run full suite**

```
pytest tests/ -q
```

Expected: 100 + 1 = 101 passed.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train_driver.py
git commit -m "$(cat <<'EOF'
train: auto-resume from latest.safetensors with optimizer state

If run_dir already contains latest.safetensors, load params + optim
state, advance step counter, log a resume line. Re-running against an
existing run_dir continues; delete the file to start fresh.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: grad_accum in the training loop

**Files:**
- Modify: `train_step.py` (add `train_step_with_accum`)
- Modify: `train.py:95-129` (inner loop)
- Test: `tests/test_train_step.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_train_step.py` (read the file first to match its import style):

```python
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
```

- [ ] **Step 2: Add `train_step_with_accum` to `train_step.py`**

Append to `train_step.py`:

```python
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
```

- [ ] **Step 3: Run the new test**

```
pytest tests/test_train_step.py::test_grad_accum_matches_full_batch -v
```

Expected: pass.

- [ ] **Step 4: Wire grad_accum into train.py**

In `train.py`, modify the inner loop (around line 102-105) to use `train_step_with_accum` when `train_cfg.grad_accum > 1`:

```python
        if train_cfg.grad_accum <= 1:
            x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
            x = mx.array(x_np); y = mx.array(y_np)
            loss = train_step(model, optimizer, x, y, grad_clip=train_cfg.grad_clip)
        else:
            from train_step import train_step_with_accum
            batches = []
            for _ in range(train_cfg.grad_accum):
                x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
                batches.append((mx.array(x_np), mx.array(y_np)))
            loss = train_step_with_accum(model, optimizer, batches, grad_clip=train_cfg.grad_clip)
```

- [ ] **Step 5: Run full suite**

```
pytest tests/ -q
```

Expected: 101 + 1 = 102 passed.

- [ ] **Step 6: Commit**

```bash
git add train_step.py train.py tests/test_train_step.py
git commit -m "$(cat <<'EOF'
train: honor TrainConfig.grad_accum (was no-op outside token counting)

train_step_with_accum averages gradients over N micro-batches before
applying the optimizer update. train.py routes to the accumulator
when grad_accum > 1. Test verifies bit-close agreement with a single
full-batch step at the same effective batch size.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Multi-seed orchestration script

**Files:**
- Create: `scripts/multi_seed_paired.sh`

- [ ] **Step 1: Write the script**

Create `scripts/multi_seed_paired.sh`:

```bash
#!/bin/bash
# Multi-seed paired runs. Loops over N seeds, invoking the paired runner once
# per seed. Each pair writes to OUT_ROOT_BASE-seed$i/ so runs don't collide.
#
# Usage:
#   ./scripts/multi_seed_paired.sh                # default: 4 seeds (0..3)
#   N_SEEDS=6 ./scripts/multi_seed_paired.sh      # 6 seeds (0..5)
#   START_SEED=4 N_SEEDS=2 ./scripts/multi_seed_paired.sh
#
# Each pair already runs under caffeinate via stage0_paired.sh, so display
# sleep can't degrade these long runs.
set -euo pipefail

N_SEEDS="${N_SEEDS:-4}"
START_SEED="${START_SEED:-0}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-runs/multi-seed-paired}"

end=$(( START_SEED + N_SEEDS - 1 ))
for i in $(seq "$START_SEED" "$end"); do
  echo "[multi-seed] starting pair seed=$i ($((i - START_SEED + 1))/$N_SEEDS)"
  DATA_SEED="$i" MODEL_SEED="$i" OUT_ROOT="${OUT_ROOT_BASE}-seed${i}" \
    ./scripts/stage0_paired.sh
done

echo "[multi-seed] all $N_SEEDS pairs complete"
```

- [ ] **Step 2: Make executable**

```
chmod +x scripts/multi_seed_paired.sh
```

- [ ] **Step 3: Dry-validate (no execution; just lint)**

```
bash -n scripts/multi_seed_paired.sh
```

Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/multi_seed_paired.sh
git commit -m "$(cat <<'EOF'
scripts: multi-seed paired runner

Loops over N seeds (default 4), invokes stage0_paired.sh once per seed.
Caffeinate is already wrapping each pair.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final acceptance run

```
source .venv/bin/activate
pytest tests/ -q
```

Expected: 102 passed.

Update `docs/2026-05-20-phase-b-retro.md` Phase D prerequisites list: cross out items 2, 3, 4 (the new closures) and commit:

```bash
git add docs/2026-05-20-phase-b-retro.md
git commit -m "phase-b retro: mark optim-state, grad_accum, multi-seed prereqs closed"
```

## Out of scope

- RNG state persistence on resume (data sampler advances arbitrarily; not load-bearing for crash recovery).
- Resume metadata mismatch detection (no check that the loaded checkpoint's config matches the resumed config).
- Multi-machine / multi-GPU.
- Replacing the bash multi-seed script with a Python orchestrator. Bash + env vars is enough.
