# diff-mlx Phase A: Infrastructure + Vanilla Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project, data pipeline, model architecture, training loop, and Stage 0 vanilla smoke test — enough to train a vanilla MHA transformer on FineWeb-Edu at ~30M scale and confirm the infrastructure works end-to-end.

**Architecture:** Pre-norm LLaMA-style transformer in pure MLX (Apple Silicon). cl100k_base tokenizer (paper-canonical), tiktoken. FineWeb-Edu single corpus, uint32 mmap shards. fp32 master weights, bf16 forward, fp32 logits/loss. AdamW with decoupled exclusions. Native `mx.fast.rope` and `mx.fast.scaled_dot_product_attention`. Two-tier eval: small monitoring slice (frequent) + full val (sparse). No custom Metal kernels in Phase A — those land in Phase C.

**Tech Stack:** Python 3.12+, MLX (pinned), tiktoken (pinned), numpy, huggingface_hub (for FineWeb-Edu), pytest, safetensors. macOS / M5 Max.

**Reference design:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` — read §6.1 (backbone), §6.2 (vanilla attn), §8 (data), §9.0–9.6 (training), §11 (file layout) before starting.

**Scope:** Phase A produces a vanilla model trained on Stage 0 config (~30M params, ~100M tokens, T=1024, B=16, single seed) with metrics, checkpoints, and tier-1 eval. **Diff-attn, paired-seed init, reference cross-check, custom kernels, and Stages 1/2 are out of scope** — they land in Phases B, C, D respectively.

---

## File Structure

```
diff-mlx/
  pyproject.toml                # Phase A: mlx, numpy, tiktoken, safetensors, pytest
  .gitignore                    # ignore runs/, data/shards/, .venv/
  README.md                     # one-liner pointing at docs/

  config.py                     # ModelConfig + TrainConfig dataclasses
  model.py                      # RMSNorm, SwiGLU, VanillaMHA, Block, Transformer
  optim.py                      # AdamW wrapper with weight-decay exclusions + fp32 master
  schedule.py                   # cosine LR with warmup
  train.py                      # training loop driver
  eval.py                       # tier-1 (monitoring) and tier-2 (full) eval

  data/
    download.py                 # FineWeb-Edu download via huggingface_hub
    tokenize.py                 # cl100k_base → uint32 shards
    loader.py                   # mmap loader, deterministic sampler

  tests/
    test_data.py                # tokenizer round-trip, shard format, loader determinism
    test_model.py               # forward shape, param count, forward determinism
    test_optim.py               # weight-decay exclusion split, fp32 master roundtrip
    test_schedule.py            # warmup + cosine values at known steps
    test_train_loop.py          # one-step train smoke test on tiny config
    test_eval.py                # tier-1 and tier-2 eval, val determinism

  runs/                         # gitignored
    stage0-vanilla-seed0/       # Stage 0 smoke output
```

**Design notes for the file layout:**
- One file per responsibility. `model.py` will grow to ~300 lines by end of Phase B (diff-attn added); fine.
- `config.py` is the single source of truth for hyperparameters; `train.py` reads it.
- `optim.py` and `schedule.py` are kept separate from `train.py` so they're independently testable.
- `data/tokenize.py` is a one-shot script you run before training; `data/loader.py` is the runtime path.

---

## Phase A overview (commit cadence)

- **Tasks 1-3:** Project scaffold + pinned deps. Commit after each.
- **Tasks 4-7:** Data pipeline (download, tokenize, load, test). Commit per task.
- **Tasks 8-15:** Model components, TDD'd one at a time. Commit per component.
- **Tasks 16-19:** Optimizer + schedule + checkpoint, TDD'd. Commit per component.
- **Tasks 20-22:** Training loop + eval + metrics logging, TDD'd. Commit per component.
- **Task 23:** Stage 0 smoke run. Commit the config + a notes file with the loss curve.

Total: 23 tasks, ~150 steps. Bite-sized; each task takes 30-90 min of focused work.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
*.egg-info/
.pytest_cache/

# Project
runs/
data/shards/
data/raw/

# OS
.DS_Store
```

- [ ] **Step 2: Create `pyproject.toml` (Phase A deps only)**

```toml
[project]
name = "diff-mlx"
version = "0.1.0"
description = "Differential Transformer reproduction in MLX on Apple Silicon"
requires-python = ">=3.12"
dependencies = [
    "mlx>=0.20",          # PIN exact version after install — record below
    "numpy>=1.26",
    "tiktoken>=0.8",      # PIN exact version after install — record below
    "safetensors>=0.4",
    "huggingface_hub>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 3: Create `README.md`**

```markdown
# diff-mlx

Differential Transformer reproduction in MLX on Apple Silicon.

- **Design:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`
- **Implementation plans:** `docs/2026-05-20-diffattn-mlx-implementation-plan-phase-{a,b,c,d}.md`

See the design doc for hardware, hypothesis, scope, and stage gates.
```

- [ ] **Step 4: Initialize venv and install**

Run:
```bash
cd /Users/guygrigsby/projects/diff-mlx
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: clean install, no errors.

- [ ] **Step 5: Pin exact versions in pyproject.toml**

Run:
```bash
python -c "import mlx; print(mlx.__version__)"
python -c "import tiktoken; print(tiktoken.__version__)"
```

Replace `mlx>=0.20` and `tiktoken>=0.8` in `pyproject.toml` with the exact installed versions (e.g. `mlx==0.20.5`). This is the version-pin requirement from the design doc §4.

- [ ] **Step 6: Verify install + commit**

Run:
```bash
python -c "import mlx.core as mx; print(mx.array([1, 2, 3]).sum())"
python -c "import tiktoken; print(len(tiktoken.get_encoding('cl100k_base')._special_tokens) + tiktoken.get_encoding('cl100k_base').n_vocab)"
pytest --collect-only
```

Expected: MLX prints `array(6, dtype=int32)`, tiktoken prints ~100k vocab, pytest reports 0 tests (no tests yet).

Commit:
```bash
git add pyproject.toml .gitignore README.md
git commit -m "scaffold: pyproject + gitignore + README"
```

---

## Task 2: Project config dataclasses

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from config import ModelConfig, TrainConfig

def test_model_config_stage0():
    cfg = ModelConfig.stage0()
    assert cfg.dim == 256
    assert cfg.n_layers == 6
    assert cfg.n_heads_vanilla == 4
    assert cfg.qk_head_dim == 64
    assert cfg.vocab_size == 100_277
    assert cfg.mlp_intermediate == 704  # ceil(8/3 * 256) rounded up to multiple of 32
    assert cfg.block_size == 1024
    assert cfg.rope_base == 10000.0
    assert cfg.rms_eps == 1e-5

def test_train_config_stage0():
    cfg = TrainConfig.stage0()
    assert cfg.peak_lr == 6e-4
    assert cfg.warmup_steps == 500
    assert cfg.weight_decay == 0.1
    assert cfg.adam_beta1 == 0.9
    assert cfg.adam_beta2 == 0.95
    assert cfg.adam_eps == 1e-8
    assert cfg.grad_clip == 1.0
    assert cfg.micro_batch == 16
    assert cfg.grad_accum == 1
    assert cfg.total_tokens == 100_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: ModuleNotFoundError (config doesn't exist yet)

- [ ] **Step 3: Implement `config.py`**

```python
# config.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int
    n_heads_vanilla: int
    qk_head_dim: int
    vocab_size: int
    mlp_intermediate: int
    block_size: int
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True

    @classmethod
    def stage0(cls) -> "ModelConfig":
        return cls(
            dim=256, n_layers=6, n_heads_vanilla=4, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=704, block_size=1024,
        )

    @classmethod
    def stage1(cls) -> "ModelConfig":
        return cls(
            dim=768, n_layers=12, n_heads_vanilla=12, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=2048, block_size=2048,
        )

    @classmethod
    def stage2(cls) -> "ModelConfig":
        return cls(
            dim=1024, n_layers=16, n_heads_vanilla=16, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=2752, block_size=2048,
        )

@dataclass(frozen=True)
class TrainConfig:
    peak_lr: float
    warmup_steps: int
    total_tokens: int
    micro_batch: int
    grad_accum: int = 1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    eval_every: int = 500
    full_eval_every: int = 5000
    monitoring_tokens: int = 2_000_000
    full_eval_tokens: int = 75_000_000
    save_every: int = 1000

    @classmethod
    def stage0(cls) -> "TrainConfig":
        return cls(
            peak_lr=6e-4, warmup_steps=500,
            total_tokens=100_000_000, micro_batch=16, grad_accum=1,
            eval_every=500, full_eval_every=2500,
        )

    @classmethod
    def stage1(cls) -> "TrainConfig":
        return cls(
            peak_lr=4e-4, warmup_steps=1000,
            total_tokens=2_000_000_000, micro_batch=32, grad_accum=1,
            eval_every=1000, full_eval_every=5000,
        )

    @classmethod
    def stage2(cls) -> "TrainConfig":
        return cls(
            peak_lr=3e-4, warmup_steps=2000,
            total_tokens=4_000_000_000, micro_batch=32, grad_accum=4,
            eval_every=1000, full_eval_every=5000,
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "config: ModelConfig and TrainConfig dataclasses for stages 0/1/2"
```

---

## Task 3: Tokenizer wrapper + version pin

**Files:**
- Create: `data/__init__.py` (empty)
- Create: `data/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer.py
from data.tokenizer import get_tokenizer, EOT_ID, VOCAB_SIZE

def test_get_tokenizer_returns_cl100k_base():
    enc = get_tokenizer()
    assert enc.name == "cl100k_base"

def test_vocab_size_constant():
    assert VOCAB_SIZE == 100_277

def test_eot_id_is_endoftext():
    enc = get_tokenizer()
    assert EOT_ID == enc.eot_token

def test_roundtrip_simple_text():
    enc = get_tokenizer()
    text = "Hello, world. This is a test."
    ids = enc.encode(text)
    decoded = enc.decode(ids)
    assert decoded == text

def test_token_max_fits_in_uint32():
    # vocab_size fits in uint32 (4 billion) but not uint16 (65k)
    assert VOCAB_SIZE > 65_535
    assert VOCAB_SIZE < 2**32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenizer.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `data/tokenizer.py`**

```python
# data/tokenizer.py
"""cl100k_base tokenizer wrapper. Pinned via tiktoken version in pyproject.toml."""
import tiktoken

_ENCODING_NAME = "cl100k_base"
_enc = tiktoken.get_encoding(_ENCODING_NAME)

VOCAB_SIZE = _enc.n_vocab  # 100_277 for cl100k_base
EOT_ID = _enc.eot_token

def get_tokenizer() -> tiktoken.Encoding:
    return _enc

def tiktoken_version() -> str:
    import tiktoken as _t
    return _t.__version__
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tokenizer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add data/__init__.py data/tokenizer.py tests/test_tokenizer.py
git commit -m "data: cl100k_base tokenizer wrapper with version pin"
```

---

## Task 4: FineWeb-Edu download utility

**Files:**
- Create: `data/download.py`
- Test: `tests/test_download.py` (only test the API contract, not actual download)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_download.py
from pathlib import Path
from data.download import target_files_for, download_fineweb_edu_sample

def test_target_files_returns_parquet_paths(tmp_path):
    files = target_files_for(out_dir=tmp_path, n_files=3)
    assert len(files) == 3
    assert all(str(f).endswith(".parquet") for f in files)
    assert all(f.parent == tmp_path for f in files)

def test_download_signature_exists():
    # Just verify the function exists with the right signature; do not actually download
    import inspect
    sig = inspect.signature(download_fineweb_edu_sample)
    assert "out_dir" in sig.parameters
    assert "n_files" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `data/download.py`**

```python
# data/download.py
"""Download FineWeb-Edu sample parquet shards from HuggingFace Hub.

Run once per machine: python -m data.download --out_dir data/raw --n_files 8
"""
from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"  # ~10B token sample; we take a slice via n_files

def target_files_for(out_dir: Path, n_files: int) -> list[Path]:
    """Return the local paths we'll download to (does not download)."""
    out_dir = Path(out_dir)
    return [out_dir / f"shard_{i:04d}.parquet" for i in range(n_files)]

def download_fineweb_edu_sample(out_dir: Path, n_files: int) -> list[Path]:
    """Download N parquet shards from the sample-10BT subset.

    Each shard is ~500MB and contains ~500k documents. n_files=4-6 yields
    enough text for Stage 0 (~100M tokens after tokenization at 1.3-1.5
    tokens/word).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_files):
        remote = f"{SUBSET}/{i:03d}_00000.parquet"
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote,
            repo_type="dataset",
            local_dir=out_dir,
        )
        paths.append(Path(local))
    return paths

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--n_files", type=int, default=4)
    args = p.parse_args()
    paths = download_fineweb_edu_sample(args.out_dir, args.n_files)
    print(f"Downloaded {len(paths)} shards to {args.out_dir}")
    for p_ in paths:
        size_mb = p_.stat().st_size / 1e6
        print(f"  {p_.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_download.py -v`
Expected: 2 passed

- [ ] **Step 5: Actually download a small sample (one shard) to validate**

Run:
```bash
mkdir -p data/raw
python -m data.download --out_dir data/raw --n_files 1
```

Expected: ~500MB parquet file in `data/raw/`. Takes 1-3 min on residential connection.

- [ ] **Step 6: Commit**

```bash
git add data/download.py tests/test_download.py
git commit -m "data: FineWeb-Edu download utility"
```

---

## Task 5: Tokenization to uint32 shards

**Files:**
- Create: `data/tokenize.py`
- Test: `tests/test_tokenize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenize.py
import json
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from data.tokenize import tokenize_parquet_to_shards, ShardMeta

def _write_tiny_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path)

def test_tokenize_tiny_parquet_writes_uint32_shards(tmp_path):
    input_pq = tmp_path / "input.parquet"
    _write_tiny_parquet(input_pq, ["Hello world.", "Another doc here.", "Third."])
    out_dir = tmp_path / "shards"
    meta = tokenize_parquet_to_shards(
        [input_pq],
        out_dir=out_dir,
        train_shard_max_tokens=1024,
        val_tokens=10,
        val_doc_hash_mod=10,
        val_doc_hash_keep_below=2,  # keep ~20% as val
    )
    assert isinstance(meta, ShardMeta)
    assert meta.vocab_size == 100_277
    assert (out_dir / "val.bin").exists()
    assert (out_dir / "meta.json").exists()
    train_files = sorted(out_dir.glob("train-*.bin"))
    assert len(train_files) >= 1
    arr = np.memmap(train_files[0], dtype=np.uint32, mode="r")
    assert arr.dtype == np.uint32
    assert len(arr) > 0
    assert arr.max() < 100_277

def test_meta_json_has_expected_keys(tmp_path):
    input_pq = tmp_path / "input.parquet"
    _write_tiny_parquet(input_pq, ["text"] * 50)
    out_dir = tmp_path / "shards"
    tokenize_parquet_to_shards([input_pq], out_dir=out_dir, train_shard_max_tokens=512, val_tokens=20)
    meta = json.loads((out_dir / "meta.json").read_text())
    for key in ["vocab_size", "eot_id", "tiktoken_version", "tokenizer_name",
                "train_token_count", "val_token_count", "n_train_shards", "source_files"]:
        assert key in meta, f"missing key: {key}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenize.py -v`
Expected: ModuleNotFoundError, AND ModuleNotFoundError for pyarrow if not installed yet.

Install pyarrow first:
```bash
pip install pyarrow
```
Add to pyproject.toml `dependencies`: `"pyarrow>=15.0",`

- [ ] **Step 3: Implement `data/tokenize.py`**

```python
# data/tokenize.py
"""Tokenize FineWeb-Edu parquet shards into uint32 mmap-able binaries.

Writes:
- data/shards/train-NNNN.bin    (uint32 token streams, mmap-able)
- data/shards/val.bin           (deterministic val subset)
- data/shards/meta.json         (vocab, version pins, counts)

Val selection is deterministic on a hash of (file_index, doc_index): any doc
whose hash % val_doc_hash_mod < val_doc_hash_keep_below goes to val. With
mod=100 and keep_below=1, ~1% of docs go to val.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import struct

import numpy as np
import pyarrow.parquet as pq

from data.tokenizer import get_tokenizer, EOT_ID, VOCAB_SIZE, tiktoken_version

@dataclass
class ShardMeta:
    vocab_size: int
    eot_id: int
    tiktoken_version: str
    tokenizer_name: str
    train_token_count: int
    val_token_count: int
    n_train_shards: int
    source_files: list[str]

def _doc_hash(file_idx: int, doc_idx: int) -> int:
    h = hashlib.sha256(f"{file_idx}:{doc_idx}".encode()).digest()
    return struct.unpack("<Q", h[:8])[0]

def tokenize_parquet_to_shards(
    input_files: list[Path],
    out_dir: Path,
    train_shard_max_tokens: int = 100_000_000,
    val_tokens: int = 100_000_000,
    val_doc_hash_mod: int = 100,
    val_doc_hash_keep_below: int = 1,
) -> ShardMeta:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = get_tokenizer()

    val_buf: list[np.ndarray] = []
    val_total = 0
    train_buf: list[np.ndarray] = []
    train_buf_tokens = 0
    train_total = 0
    shard_idx = 0

    def flush_train_shard():
        nonlocal shard_idx, train_buf, train_buf_tokens
        if not train_buf:
            return
        out = np.concatenate(train_buf)
        path = out_dir / f"train-{shard_idx:04d}.bin"
        out.astype(np.uint32).tofile(path)
        shard_idx += 1
        train_buf = []
        train_buf_tokens = 0

    for file_idx, p in enumerate(input_files):
        table = pq.read_table(p, columns=["text"])
        texts = table.column("text").to_pylist()
        for doc_idx, text in enumerate(texts):
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(EOT_ID)
            arr = np.asarray(ids, dtype=np.uint32)
            h = _doc_hash(file_idx, doc_idx)
            is_val = (val_total < val_tokens) and (h % val_doc_hash_mod) < val_doc_hash_keep_below
            if is_val:
                val_buf.append(arr)
                val_total += len(arr)
            else:
                train_buf.append(arr)
                train_buf_tokens += len(arr)
                train_total += len(arr)
                if train_buf_tokens >= train_shard_max_tokens:
                    flush_train_shard()

    flush_train_shard()

    if val_buf:
        val_out = np.concatenate(val_buf).astype(np.uint32)
        val_out.tofile(out_dir / "val.bin")
    else:
        # empty val.bin (still create so loader assertions don't fail)
        np.zeros(0, dtype=np.uint32).tofile(out_dir / "val.bin")

    meta = ShardMeta(
        vocab_size=VOCAB_SIZE,
        eot_id=EOT_ID,
        tiktoken_version=tiktoken_version(),
        tokenizer_name="cl100k_base",
        train_token_count=train_total,
        val_token_count=val_total,
        n_train_shards=shard_idx,
        source_files=[str(p) for p in input_files],
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob", type=str, default="data/raw/*.parquet")
    p.add_argument("--out_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--train_shard_max_tokens", type=int, default=100_000_000)
    p.add_argument("--val_tokens", type=int, default=75_000_000)
    p.add_argument("--val_doc_hash_mod", type=int, default=100)
    p.add_argument("--val_doc_hash_keep_below", type=int, default=1)
    args = p.parse_args()
    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise SystemExit(f"No input files matched {args.input_glob}")
    meta = tokenize_parquet_to_shards(
        files, args.out_dir,
        args.train_shard_max_tokens, args.val_tokens,
        args.val_doc_hash_mod, args.val_doc_hash_keep_below,
    )
    print(f"Train tokens: {meta.train_token_count:,} in {meta.n_train_shards} shards")
    print(f"Val tokens:   {meta.val_token_count:,}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tokenize.py -v`
Expected: 2 passed

- [ ] **Step 5: Tokenize the downloaded shard for real**

Run:
```bash
python -m data.tokenize --input_glob "data/raw/*.parquet" --out_dir data/shards --val_tokens 5000000
```

Expected output: prints train tokens (~100M-200M depending on shard) and val tokens (~5M). `data/shards/` contains `train-0000.bin`, `val.bin`, `meta.json`. Takes 1-3 min.

Verify:
```bash
ls -lh data/shards/
cat data/shards/meta.json
```

- [ ] **Step 6: Commit**

```bash
git add data/tokenize.py tests/test_tokenize.py pyproject.toml
git commit -m "data: tokenize parquet to uint32 shards with deterministic val split"
```

---

## Task 6: Deterministic data loader

**Files:**
- Create: `data/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
import numpy as np
from pathlib import Path
from data.loader import ShardLoader, sample_batch

def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 100_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = np.arange(n_tokens, dtype=np.uint32) % 100_277  # deterministic content
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:10_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 100000, '
        '"val_token_count": 10000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir

def test_loader_reads_uint32_shards(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    assert loader.total_tokens == 100_000
    arr = loader.read(0, 100)
    assert arr.dtype == np.uint32
    assert len(arr) == 100
    assert arr[0] == 0 and arr[99] == 99

def test_sample_batch_returns_correct_shape(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    x, y = sample_batch(loader, block_size=128, micro_batch=4, rng=np.random.default_rng(0))
    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    # y is x shifted by one position
    np.testing.assert_array_equal(y[0, :-1], x[0, 1:])

def test_sample_batch_determinism(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    loader = ShardLoader(shards_dir, split="train")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    x1, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng1)
    x2, _ = sample_batch(loader, block_size=64, micro_batch=2, rng=rng2)
    np.testing.assert_array_equal(x1, x2)

def test_val_split_deterministic(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader_1 = ShardLoader(shards_dir, split="val")
    val_loader_2 = ShardLoader(shards_dir, split="val")
    assert val_loader_1.total_tokens == val_loader_2.total_tokens == 10_000
    np.testing.assert_array_equal(val_loader_1.read(0, 1000), val_loader_2.read(0, 1000))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_loader.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `data/loader.py`**

```python
# data/loader.py
"""Mmap-based deterministic loader over uint32 shards."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass
class ShardLoader:
    """Memory-maps train shards or the val file. Allows arbitrary token-offset reads.

    For train (multi-shard), `read(offset, n)` treats all shards as concatenated.
    For val (single file), same interface.
    """
    shards_dir: Path
    split: str  # "train" or "val"
    _shards: list[np.memmap] = None
    _cumulative: np.ndarray = None  # cumulative token counts for shard boundaries
    total_tokens: int = 0

    def __post_init__(self):
        self.shards_dir = Path(self.shards_dir)
        meta = json.loads((self.shards_dir / "meta.json").read_text())
        if self.split == "train":
            paths = sorted(self.shards_dir.glob("train-*.bin"))
            assert paths, f"no train shards in {self.shards_dir}"
            self._shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        elif self.split == "val":
            p = self.shards_dir / "val.bin"
            assert p.exists(), f"missing {p}"
            self._shards = [np.memmap(p, dtype=np.uint32, mode="r")]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {self.split!r}")
        sizes = np.array([len(s) for s in self._shards], dtype=np.int64)
        self._cumulative = np.cumsum(sizes)
        self.total_tokens = int(self._cumulative[-1])

    def read(self, offset: int, n: int) -> np.ndarray:
        """Read n contiguous tokens starting at logical offset (across shard boundaries)."""
        if offset + n > self.total_tokens:
            raise IndexError(f"read past end: offset={offset} n={n} total={self.total_tokens}")
        # find starting shard
        shard_idx = int(np.searchsorted(self._cumulative, offset, side="right"))
        local_offset = offset - (self._cumulative[shard_idx - 1] if shard_idx > 0 else 0)
        out_parts: list[np.ndarray] = []
        remaining = n
        while remaining > 0:
            shard = self._shards[shard_idx]
            take = min(remaining, len(shard) - local_offset)
            out_parts.append(np.asarray(shard[local_offset : local_offset + take]))
            remaining -= take
            shard_idx += 1
            local_offset = 0
        return np.concatenate(out_parts) if len(out_parts) > 1 else out_parts[0]

def sample_batch(
    loader: ShardLoader,
    block_size: int,
    micro_batch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample micro_batch windows of length (block_size + 1) and return (x, y).

    x is the first block_size tokens, y is shifted by one (next-token targets).
    Both are int32 numpy arrays of shape (micro_batch, block_size).
    """
    max_offset = loader.total_tokens - block_size - 1
    if max_offset <= 0:
        raise ValueError(f"loader has {loader.total_tokens} tokens, need > {block_size + 1}")
    offsets = rng.integers(0, max_offset, size=micro_batch, dtype=np.int64)
    x = np.empty((micro_batch, block_size), dtype=np.int32)
    y = np.empty((micro_batch, block_size), dtype=np.int32)
    for i, off in enumerate(offsets):
        window = loader.read(int(off), block_size + 1)
        x[i] = window[:-1]
        y[i] = window[1:]
    return x, y
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loader.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add data/loader.py tests/test_loader.py
git commit -m "data: mmap loader with deterministic batch sampling"
```

---

## Task 7: RMSNorm module

**Files:**
- Create: `model.py` (start the file)
- Test: `tests/test_model.py` (start the file)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import numpy as np
import mlx.core as mx
from model import RMSNorm

def test_rmsnorm_shape_preserved():
    x = mx.random.normal((2, 8, 16), dtype=mx.float32)
    norm = RMSNorm(dim=16)
    y = norm(x)
    assert y.shape == x.shape

def test_rmsnorm_matches_manual_formula():
    """Verify against the formula x * scale / sqrt(mean(x^2) + eps)."""
    x = mx.random.normal((4, 32), dtype=mx.float32)
    norm = RMSNorm(dim=32, eps=1e-5)
    y = norm(x)
    expected = x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    # scale is init to 1.0 so output equals normalized input
    assert mx.allclose(y, expected, atol=1e-6).item()

def test_rmsnorm_scale_is_learnable_and_init_to_one():
    norm = RMSNorm(dim=64)
    # MLX Module: scale should be a parameter
    params = norm.parameters()
    assert "scale" in params
    assert mx.array_equal(params["scale"], mx.ones(64)).item()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py -v`
Expected: ImportError (model.py doesn't have RMSNorm)

- [ ] **Step 3: Create `model.py` with RMSNorm**

```python
# model.py
"""Transformer model: backbone + vanilla MHA (Phase A) + diff-attn (Phase B).

Phase A scope: shared backbone + VanillaMHALayer only.
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm: x * scale / sqrt(mean(x^2) + eps). Learned scale, no bias.

    Normalizes the LAST dimension only. Apply to (B, T, dim) -> normalize over dim,
    or to (B, H, T, head_dim) -> normalize over head_dim (per-head; no cross-head mixing).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Compute in fp32 for stability, cast back to input dtype
        in_dtype = x.dtype
        x_fp32 = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
        out = x_fp32 / rms
        return (out * self.scale.astype(mx.float32)).astype(in_dtype)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: RMSNorm with learned scale (no bias), fp32 internal computation"
```

---

## Task 8: SwiGLU MLP module

**Files:**
- Modify: `model.py` (append SwiGLU class)
- Modify: `tests/test_model.py` (append tests)

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/test_model.py
from model import SwiGLU

def test_swiglu_shape_and_dtype():
    cfg_dim, cfg_intermediate = 256, 704
    mlp = SwiGLU(dim=cfg_dim, intermediate=cfg_intermediate)
    x = mx.random.normal((2, 16, cfg_dim), dtype=mx.float32)
    y = mlp(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_swiglu_no_bias():
    mlp = SwiGLU(dim=128, intermediate=352)
    params = mlp.parameters()
    # gate, up, down all bias=False
    for name in ("gate", "up", "down"):
        assert name in params, f"missing {name}"
        # Each is a Linear module with only "weight" key (no "bias")
        sub = params[name]
        assert "weight" in sub
        assert "bias" not in sub

def test_swiglu_param_count():
    dim, intermediate = 256, 704
    mlp = SwiGLU(dim=dim, intermediate=intermediate)
    total = sum(p.size for _, p in _flatten_params(mlp.parameters()))
    expected = 3 * dim * intermediate  # gate + up + down, all (dim, intermediate)
    assert total == expected

def _flatten_params(d, prefix=""):
    """Tiny helper to flatten nested parameter dicts."""
    if hasattr(d, "items"):
        for k, v in d.items():
            yield from _flatten_params(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten_params(v, f"{prefix}[{i}]")
    else:
        yield prefix, d
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py::test_swiglu_shape_and_dtype -v`
Expected: ImportError on SwiGLU

- [ ] **Step 3: Append SwiGLU to `model.py`**

```python
# Append to model.py
class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). All linears bias=False."""
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 6 passed (3 RMSNorm + 3 SwiGLU)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: SwiGLU MLP, bias-free, gate/up/down at 8/3-rounded intermediate"
```

---

## Task 9: Vanilla MHA attention module

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/test_model.py
from model import VanillaMHA

def test_vanilla_mha_shape():
    attn = VanillaMHA(dim=256, n_heads=4)
    x = mx.random.normal((2, 32, 256), dtype=mx.float32)
    y = attn(x)
    assert y.shape == x.shape

def test_vanilla_mha_param_count_4dim2():
    """Vanilla MHA at dim=256: q,k,v,o each (256, 256) = 4 * 256^2 params."""
    attn = VanillaMHA(dim=256, n_heads=4)
    total = sum(p.size for _, p in _flatten_params(attn.parameters()))
    assert total == 4 * 256 * 256

def test_vanilla_mha_causal_property():
    """Output at position t must not depend on inputs at positions > t."""
    attn = VanillaMHA(dim=64, n_heads=2)
    x1 = mx.random.normal((1, 16, 64), dtype=mx.float32)
    x2 = mx.array(x1)
    x2[0, 8:, :] = mx.random.normal((8, 64))  # modify tokens 8+
    y1 = attn(x1)
    y2 = attn(x2)
    # Outputs at positions 0..7 must be byte-identical
    assert mx.allclose(y1[0, :8, :], y2[0, :8, :], atol=1e-5).item()

def test_vanilla_mha_no_bias():
    attn = VanillaMHA(dim=128, n_heads=4)
    params = attn.parameters()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert "bias" not in params[name]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py::test_vanilla_mha_shape -v`
Expected: ImportError

- [ ] **Step 3: Append `VanillaMHA` to `model.py`**

```python
# Append to model.py
class VanillaMHA(nn.Module):
    """Standard MHA: q/k/v/o all (dim, dim), bias=False. RoPE on q/k. Causal mask.

    Uses mx.fast.rope (traditional=False = LLaMA rotate-halves) and
    mx.fast.scaled_dot_product_attention (mask="causal", scale required kw-only).
    """
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # RoPE: LLaMA rotate-halves convention (traditional=False)
        q = mx.fast.rope(q, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.o_proj(out)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: VanillaMHA with mx.fast.rope (traditional=False) and mx.fast SDPA (mask=causal)"
```

---

## Task 10: Transformer Block

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/test_model.py
from model import Block

def test_block_shape_and_residual():
    block = Block(dim=128, n_heads=4, mlp_intermediate=352)
    x = mx.random.normal((2, 16, 128), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape

def test_block_has_two_norms_attn_mlp():
    block = Block(dim=128, n_heads=4, mlp_intermediate=352)
    params = block.parameters()
    assert "norm_attn" in params
    assert "norm_mlp" in params
    assert "attn" in params
    assert "mlp" in params
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py::test_block_shape_and_residual -v`
Expected: ImportError

- [ ] **Step 3: Append `Block` to `model.py`**

```python
# Append to model.py
class Block(nn.Module):
    """Pre-norm transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""
    def __init__(self, dim: int, n_heads: int, mlp_intermediate: int,
                 rope_base: float = 10000.0, rms_eps: float = 1e-5):
        super().__init__()
        self.norm_attn = RMSNorm(dim, eps=rms_eps)
        self.attn = VanillaMHA(dim, n_heads, rope_base=rope_base)
        self.norm_mlp = RMSNorm(dim, eps=rms_eps)
        self.mlp = SwiGLU(dim, mlp_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: pre-norm Block with VanillaMHA + SwiGLU"
```

---

## Task 11: Transformer model (with tied embeddings)

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/test_model.py
from model import Transformer
from config import ModelConfig

def test_transformer_stage0_forward_shape():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int32))
    logits = model(x)
    assert logits.shape == (2, 64, cfg.vocab_size)

def test_transformer_stage0_param_count_approx():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    total = sum(p.size for _, p in _flatten_params(model.parameters()))
    # Stage 0: embed 100277*256 = 25.67M, transformer body ~4.8M, total ~30.5M
    # Tied: only count embed once (out_proj shares the matrix)
    assert 28_000_000 < total < 32_000_000, f"unexpected param count: {total:,}"

def test_transformer_tied_embeddings():
    """The LM head shares weights with the token embedding."""
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    # When tied, the LM head logits are computed as x @ embed.weight.T
    # We verify by checking the embed matrix is reachable both as embedding and as projection.
    # The simplest check: model has tok_embed but no separate lm_head.weight parameter.
    params = model.parameters()
    assert "tok_embed" in params
    # If tied properly, no separate lm_head with its own weight tensor in parameters
    if "lm_head" in params:
        # If lm_head is a Linear module, its weight should be the SAME object as embed.weight
        assert mx.array_equal(params["lm_head"]["weight"], params["tok_embed"]["weight"]).item()

def test_transformer_final_rmsnorm_exists():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    params = model.parameters()
    assert "final_norm" in params
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py::test_transformer_stage0_forward_shape -v`
Expected: ImportError

- [ ] **Step 3: Append `Transformer` to `model.py`**

```python
# Append to model.py
class Transformer(nn.Module):
    """Pre-norm LLaMA-style transformer with tied embeddings.

    forward(token_ids: (B, T)) -> logits: (B, T, vocab_size)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [
            Block(
                dim=cfg.dim,
                n_heads=cfg.n_heads_vanilla,
                mlp_intermediate=cfg.mlp_intermediate,
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
            )
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        # Tied embeddings: do NOT instantiate a separate lm_head linear.
        # The forward pass uses the embedding matrix directly.

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.tok_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Tied LM head: logits = x @ embed.weight.T
        # nn.Embedding stores weight as (vocab_size, dim); matmul:
        logits = x @ self.tok_embed.weight.T
        return logits
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: full Transformer with tied embeddings, final RMSNorm"
```

---

## Task 12: Cosine LR schedule with warmup

**Files:**
- Create: `schedule.py`
- Create: `tests/test_schedule.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedule.py
from schedule import cosine_lr_with_warmup

def test_warmup_zero_at_step_zero():
    lr = cosine_lr_with_warmup(step=0, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 0.0) < 1e-9

def test_warmup_peak_at_warmup_end():
    lr = cosine_lr_with_warmup(step=100, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-3) < 1e-9

def test_cosine_midway_between_peak_and_min():
    # At halfway through cosine decay (step = warmup + (total - warmup) / 2),
    # cosine factor = 0.5 so lr = peak * (0.5 * (1 - min_frac) + min_frac) = peak * 0.55 when min_frac=0.1
    step = 100 + (1000 - 100) // 2
    lr = cosine_lr_with_warmup(step=step, peak_lr=1.0, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 0.55) < 1e-6

def test_min_lr_at_total_steps():
    lr = cosine_lr_with_warmup(step=1000, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-4) < 1e-9

def test_holds_min_after_total_steps():
    lr = cosine_lr_with_warmup(step=5000, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-4) < 1e-9
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schedule.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `schedule.py`**

```python
# schedule.py
"""Cosine LR schedule with linear warmup. Pure float math (no MLX dependency)."""
import math

def cosine_lr_with_warmup(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_frac: float = 0.1,
) -> float:
    """LR at `step`: linear warmup from 0 to peak over warmup_steps, then cosine
    decay from peak to (peak * min_lr_frac) over (total_steps - warmup_steps).

    Holds at the floor for step > total_steps.
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return peak_lr * min_lr_frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_schedule.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add schedule.py tests/test_schedule.py
git commit -m "schedule: cosine LR with linear warmup, holds at min after total_steps"
```

---

## Task 13: Optimizer with weight-decay exclusions (no master weights yet)

**Files:**
- Create: `optim.py`
- Create: `tests/test_optim.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optim.py
import mlx.core as mx
import mlx.nn as nn
from optim import split_params_for_decay, FLAT_DECAY_NAMES, FLAT_NO_DECAY_NAMES
from model import Transformer
from config import ModelConfig

def test_split_params_categorizes_correctly():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat = _flatten(model.parameters())
    decay, no_decay = split_params_for_decay(flat)

    # Embeddings -> no_decay
    assert any("tok_embed.weight" in name for name in no_decay)
    # RMSNorms -> no_decay
    assert any("norm" in name and "scale" in name for name in no_decay)
    assert any("final_norm.scale" in name for name in no_decay)
    # MLP weights -> decay
    assert any("mlp.gate.weight" in name for name in decay)
    assert any("mlp.up.weight" in name for name in decay)
    assert any("mlp.down.weight" in name for name in decay)
    # Attention projections -> decay
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(f"attn.{proj}.weight" in name for name in decay)

def test_split_params_no_overlap():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat = _flatten(model.parameters())
    decay, no_decay = split_params_for_decay(flat)
    assert set(decay).isdisjoint(set(no_decay))
    assert set(decay) | set(no_decay) == set(flat.keys())

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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_optim.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `optim.py`**

```python
# optim.py
"""AdamW optimizer wrapper with weight-decay exclusion policy.

Per design §9.1:
- Decay: linear projection weights (q/k/v/o, mlp gate/up/down)
- No decay: embeddings, all RMSNorm/subLN scales, lambda vectors (Phase B)
"""
from __future__ import annotations
import mlx.optimizers as optim

# Substrings: any flat param name containing one of these is NO-DECAY.
FLAT_NO_DECAY_NAMES = (
    "tok_embed",       # token embedding matrix
    "norm",            # any RMSNorm (norm_attn, norm_mlp, final_norm, subln)
    "lambda_",         # diff-attn lambda vectors (lambda_q1/k1/q2/k2)
)

# Any param not matching NO_DECAY is DECAY-eligible.
FLAT_DECAY_NAMES = ("weight",)  # informative; actual rule is complement of no-decay

def split_params_for_decay(flat_params: dict) -> tuple[list[str], list[str]]:
    """Given a flat param dict (name -> tensor), return (decay_names, no_decay_names)."""
    decay, no_decay = [], []
    for name in flat_params:
        if any(needle in name for needle in FLAT_NO_DECAY_NAMES):
            no_decay.append(name)
        else:
            decay.append(name)
    return decay, no_decay

def make_adamw(
    *,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> optim.AdamW:
    """Construct an AdamW optimizer.

    Note: weight-decay exclusions are applied at the training step (zero-out the
    decay term for excluded params), not via the optimizer constructor (MLX's
    AdamW applies a single decay value to all params). See train.py.
    """
    return optim.AdamW(
        learning_rate=lr,
        betas=[beta1, beta2],
        eps=eps,
        weight_decay=weight_decay,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_optim.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add optim.py tests/test_optim.py
git commit -m "optim: AdamW factory + weight-decay exclusion split (embed, norms, lambdas)"
```

---

## Task 14: fp32 master weights, bf16 forward cast

**Files:**
- Modify: `optim.py`
- Modify: `tests/test_optim.py`

The simplest path (per design §9.0 option (a)): **keep params as fp32 in MLX, cast to bf16 only for the forward**. The optimizer applies fp32 updates directly. This is option (a) from the design doc. Verify it works for our use case before adding wrappers.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_optim.py
import mlx.core as mx

def test_cast_params_to_bf16_for_forward(tmp_path):
    """Smoke test: model params are fp32, forward dtype is bf16 after cast."""
    from optim import to_bf16_view
    p_fp32 = mx.random.normal((4, 8), dtype=mx.float32)
    p_bf16 = to_bf16_view(p_fp32)
    assert p_bf16.dtype == mx.bfloat16
    assert p_bf16.shape == p_fp32.shape
    # Values close (some loss expected from bf16 quantization)
    assert mx.allclose(p_bf16.astype(mx.float32), p_fp32, atol=1e-2).item()

def test_to_bf16_view_dict_recurses():
    from optim import to_bf16_dict
    d = {
        "a": mx.random.normal((4,), dtype=mx.float32),
        "b": {"c": mx.random.normal((4,), dtype=mx.float32)},
    }
    out = to_bf16_dict(d)
    assert out["a"].dtype == mx.bfloat16
    assert out["b"]["c"].dtype == mx.bfloat16
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_optim.py::test_cast_params_to_bf16_for_forward -v`
Expected: ImportError

- [ ] **Step 3: Append cast helpers to `optim.py`**

```python
# Append to optim.py
import mlx.core as mx

def to_bf16_view(x: mx.array) -> mx.array:
    """Cast an fp32 tensor to bf16. Used to derive forward-pass params from master."""
    return x.astype(mx.bfloat16)

def to_bf16_dict(params: dict) -> dict:
    """Recursively cast all leaf tensors in a parameter dict to bf16.

    NOTE: keeps the same nesting structure as the input. The caller is responsible
    for passing this dict to the model's forward (e.g. via model.update(params)).
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, dict):
            out[k] = to_bf16_dict(v)
        elif isinstance(v, list):
            out[k] = [to_bf16_dict(x) if isinstance(x, dict) else to_bf16_view(x) for x in v]
        elif isinstance(v, mx.array):
            out[k] = to_bf16_view(v)
        else:
            out[k] = v
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_optim.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add optim.py tests/test_optim.py
git commit -m "optim: to_bf16 helpers for fp32 master -> bf16 forward casting"
```

---

## Task 15: Checkpoint save/load (safetensors)

**Files:**
- Create: `checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py
import json
from pathlib import Path
import mlx.core as mx
from checkpoint import save_checkpoint, load_checkpoint, save_run_metadata
from model import Transformer
from config import ModelConfig

def test_save_and_load_roundtrip(tmp_path):
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat_before = _flatten(model.parameters())
    ckpt_path = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=1234, ckpt_path=ckpt_path,
                    optim_state=None, rng_state=None)
    loaded, step = load_checkpoint(ckpt_path)
    assert step == 1234
    flat_after = _flatten(loaded)
    assert set(flat_before.keys()) == set(flat_after.keys())
    for name in flat_before:
        assert mx.array_equal(flat_before[name], flat_after[name]).item(), f"mismatch: {name}"

def test_save_run_metadata_writes_expected_files(tmp_path):
    cfg = ModelConfig.stage0()
    save_run_metadata(
        run_dir=tmp_path,
        model_cfg=cfg,
        train_cfg_dict={"peak_lr": 6e-4},
        git_hash="deadbeef",
        git_dirty=False,
        mlx_version="0.20.5",
        seed=0,
        data_meta={"vocab_size": 100277, "tiktoken_version": "0.8.0"},
    )
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "git.txt").exists()
    assert (tmp_path / "mlx_version.txt").exists()
    assert (tmp_path / "tiktoken.txt").exists()
    assert (tmp_path / "data_meta.json").exists()
    assert (tmp_path / "seed.txt").exists()

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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_checkpoint.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `checkpoint.py`**

```python
# checkpoint.py
"""Checkpoint save/load using MLX safetensors + per-run metadata files."""
from __future__ import annotations
from pathlib import Path
import json
import mlx.core as mx

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{key}.{i}"))
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v
    return out

def _unflatten(flat: dict) -> dict:
    """Inverse of _flatten: turn 'a.b.0.c' keys back into nested dicts/lists."""
    out: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = out
        for i, part in enumerate(parts[:-1]):
            nxt = parts[i + 1]
            nxt_is_int = nxt.isdigit()
            if part.isdigit():
                part = int(part)
                while len(cur) <= part:
                    cur.append({} if not nxt_is_int else [])
                if cur[part] == {} or (isinstance(cur[part], list) and len(cur[part]) == 0):
                    cur[part] = [] if nxt_is_int else {}
                cur = cur[part]
            else:
                if part not in cur:
                    cur[part] = [] if nxt_is_int else {}
                cur = cur[part]
        last = parts[-1]
        if last.isdigit():
            last_i = int(last)
            while len(cur) <= last_i:
                cur.append(None)
            cur[last_i] = val
        else:
            cur[last] = val
    return out

def save_checkpoint(params: dict, step: int, ckpt_path: Path,
                    optim_state=None, rng_state=None) -> None:
    """Save params (and optionally optimizer state / RNG state) to safetensors."""
    flat = _flatten(params)
    metadata = {"step": str(step)}
    mx.save_safetensors(str(ckpt_path), flat, metadata=metadata)
    # optim_state and rng_state will be added in a later iteration (after Stage 0)

def load_checkpoint(ckpt_path: Path) -> tuple[dict, int]:
    """Load a safetensors checkpoint. Returns (params_dict, step)."""
    loaded = mx.load(str(ckpt_path), return_metadata=True)
    tensors, metadata = loaded
    step = int(metadata.get("step", "0"))
    params = _unflatten(dict(tensors))
    return params, step

def save_run_metadata(
    run_dir: Path,
    model_cfg,
    train_cfg_dict: dict,
    git_hash: str,
    git_dirty: bool,
    mlx_version: str,
    seed: int,
    data_meta: dict,
) -> None:
    """Snapshot the run's reproducibility-relevant context."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    full = {"model": _config_to_dict(model_cfg), "train": train_cfg_dict}
    (run_dir / "config.json").write_text(json.dumps(full, indent=2))
    (run_dir / "git.txt").write_text(f"{git_hash}\ndirty={git_dirty}\n")
    (run_dir / "mlx_version.txt").write_text(mlx_version + "\n")
    (run_dir / "tiktoken.txt").write_text(
        f"version={data_meta.get('tiktoken_version', '?')}\n"
        f"encoding={data_meta.get('tokenizer_name', '?')}\n"
        f"vocab_size={data_meta.get('vocab_size', '?')}\n"
    )
    (run_dir / "data_meta.json").write_text(json.dumps(data_meta, indent=2))
    (run_dir / "seed.txt").write_text(str(seed) + "\n")

def _config_to_dict(cfg) -> dict:
    """Convert a frozen dataclass to a plain dict."""
    from dataclasses import is_dataclass, asdict
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg.__dict__)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_checkpoint.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add checkpoint.py tests/test_checkpoint.py
git commit -m "checkpoint: safetensors save/load + per-run metadata snapshot"
```

---

## Task 16: Metrics logger

**Files:**
- Create: `metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import json
from pathlib import Path
from metrics import MetricsLogger

def test_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    logger.log(step=10, train_loss=4.2, lr=1e-4)
    logger.log(step=20, train_loss=4.0, lr=1.5e-4)
    logger.close()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    r1 = json.loads(lines[1])
    assert r0["step"] == 10
    assert r0["train_loss"] == 4.2
    assert r1["lr"] == 1.5e-4

def test_logger_appends_existing_file(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    logger.log(step=1)
    logger.close()
    logger2 = MetricsLogger(log_path)
    logger2.log(step=2)
    logger2.close()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_metrics.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `metrics.py`**

```python
# metrics.py
"""Append-only JSONL metrics logger."""
from __future__ import annotations
from pathlib import Path
import json

class MetricsLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)  # line-buffered

    def log(self, **kwargs) -> None:
        self._f.write(json.dumps(kwargs) + "\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "metrics: append-only JSONL logger"
```

---

## Task 17: Eval function (tier-1 monitoring slice)

**Files:**
- Create: `eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval.py
import numpy as np
import mlx.core as mx
from pathlib import Path
from eval import compute_val_loss
from model import Transformer
from config import ModelConfig
from data.loader import ShardLoader

def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 50_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = (np.arange(n_tokens, dtype=np.uint32)) % 100_277
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:5_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 50000, '
        '"val_token_count": 5000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir

def test_compute_val_loss_returns_finite_float(tmp_path):
    """End-to-end smoke test of eval on a tiny untrained model."""
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=2, qk_head_dim=32,
        vocab_size=100_277, mlp_intermediate=128, block_size=64,
    )
    model = Transformer(cfg)
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader = ShardLoader(shards_dir, split="val")
    loss = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    assert isinstance(loss, float)
    assert loss > 0  # untrained: roughly log(vocab_size) ~ 11.5
    assert np.isfinite(loss)

def test_eval_is_deterministic(tmp_path):
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=2, qk_head_dim=32,
        vocab_size=100_277, mlp_intermediate=128, block_size=64,
    )
    model = Transformer(cfg)
    shards_dir = _make_synthetic_shards(tmp_path)
    val_loader = ShardLoader(shards_dir, split="val")
    loss1 = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    loss2 = compute_val_loss(model, val_loader, block_size=64, micro_batch=4, max_tokens=2000)
    assert loss1 == loss2  # eval walks the val set in a fixed order, no randomness
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_eval.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `eval.py`**

```python
# eval.py
"""Evaluation: token-level NLL over a deterministic prefix of the val set."""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from data.loader import ShardLoader

def compute_val_loss(
    model: nn.Module,
    val_loader: ShardLoader,
    block_size: int,
    micro_batch: int,
    max_tokens: int,
) -> float:
    """Walk the first max_tokens tokens of the val set in non-overlapping windows of
    block_size. Return token-weighted average NLL.

    Deterministic: same loader + same args yields the same loss exactly.
    """
    total_loss = 0.0
    total_tokens = 0
    offset = 0
    while offset + block_size + 1 <= min(val_loader.total_tokens, max_tokens + block_size + 1):
        # batch up micro_batch windows in a row
        windows = []
        for _ in range(micro_batch):
            if offset + block_size + 1 > min(val_loader.total_tokens, max_tokens + block_size + 1):
                break
            windows.append(val_loader.read(offset, block_size + 1))
            offset += block_size
        if not windows:
            break
        import numpy as np
        x = mx.array(np.stack([w[:-1] for w in windows]).astype(np.int32))
        y = mx.array(np.stack([w[1:]  for w in windows]).astype(np.int32))
        logits = model(x).astype(mx.float32)  # fp32 for CE (§9.0)
        # cross-entropy: -log_softmax(logits)[y]
        log_probs = nn.log_softmax(logits, axis=-1)
        # gather log-prob of the target token at each position
        # log_probs: (B, T, V); y: (B, T)
        # gathered: (B, T)
        gathered = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        loss = -gathered.sum().item()
        n = gathered.size
        total_loss += loss
        total_tokens += n
        if offset >= max_tokens:
            break
    return total_loss / max(1, total_tokens)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_eval.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add eval.py tests/test_eval.py
git commit -m "eval: deterministic token-NLL over a val prefix (fp32 logits per §9.0)"
```

---

## Task 18: Training step (single batch, with loss + backward)

**Files:**
- Create: `train_step.py`
- Create: `tests/test_train_step.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_step.py
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from train_step import compute_loss_and_grads, train_step
from model import Transformer
from config import ModelConfig

def test_compute_loss_returns_scalar_fp32():
    cfg = ModelConfig(dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
                       vocab_size=100_277, mlp_intermediate=64, block_size=32)
    model = Transformer(cfg)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 32), dtype=np.int32))
    y = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 32), dtype=np.int32))
    loss = compute_loss_and_grads(model, x, y)[0]
    assert loss.dtype == mx.float32
    assert loss.shape == ()
    assert mx.isfinite(loss).item()

def test_train_step_decreases_loss_on_one_batch_overfit():
    """Smoke: model can overfit a single (x, y) batch — proves backward works."""
    cfg = ModelConfig(dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
                       vocab_size=100_277, mlp_intermediate=64, block_size=16)
    model = Transformer(cfg)
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 16), dtype=np.int32))
    y = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 16), dtype=np.int32))
    losses = []
    for _ in range(50):
        loss = train_step(model, opt, x, y, grad_clip=1.0)
        losses.append(loss)
    assert losses[-1] < losses[0] * 0.5, f"did not overfit: start={losses[0]:.3f} end={losses[-1]:.3f}"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_train_step.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `train_step.py`**

```python
# train_step.py
"""Single training step: forward, CE loss, backward, optimizer step."""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

def _ce_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Mean cross-entropy. Logits cast to fp32 (per design §9.0) for stability."""
    logits = model(x).astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    gathered = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    return -gathered.mean()

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
            for v in g.values(): walk(v)
        elif isinstance(g, list):
            for v in g: walk(v)
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_step.py -v`
Expected: 2 passed (the overfit test may be slow — give it 30-60s)

- [ ] **Step 5: Commit**

```bash
git add train_step.py tests/test_train_step.py
git commit -m "train_step: forward + CE (fp32) + backward + grad clip + optimizer.update"
```

---

## Task 19: git utility (for run metadata)

**Files:**
- Create: `gitinfo.py`
- Create: `tests/test_gitinfo.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gitinfo.py
from gitinfo import current_hash, is_dirty

def test_current_hash_is_40_char_hex():
    h = current_hash()
    assert len(h) == 40
    int(h, 16)  # raises if not hex

def test_is_dirty_returns_bool():
    assert isinstance(is_dirty(), bool)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_gitinfo.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `gitinfo.py`**

```python
# gitinfo.py
"""Capture git state for run reproducibility."""
import subprocess

def current_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def is_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gitinfo.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add gitinfo.py tests/test_gitinfo.py
git commit -m "gitinfo: capture git hash and dirty flag for run metadata"
```

---

## Task 20: Training driver (`train.py`)

**Files:**
- Create: `train.py`
- Create: `tests/test_train_driver.py`

This task assembles the components into a runnable training loop. The smoke test runs ~50 steps on a tiny config and verifies metrics get written.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_driver.py
import json
import numpy as np
from pathlib import Path
import mlx.core as mx
from train import train_run
from config import ModelConfig, TrainConfig

def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 200_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = (np.arange(n_tokens, dtype=np.uint32)) % 100_277
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:10_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 200000, '
        '"val_token_count": 10000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir

def test_train_run_smoke_writes_metrics_and_checkpoint(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    run_dir = tmp_path / "runs" / "smoke"
    # Tiny config: ~5k params, runs in <30s
    model_cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
        vocab_size=100_277, mlp_intermediate=64, block_size=64,
    )
    train_cfg = TrainConfig(
        peak_lr=1e-3, warmup_steps=10, total_tokens=50 * 64 * 2,  # 50 steps
        micro_batch=2, eval_every=20, full_eval_every=50,
        monitoring_tokens=500, full_eval_tokens=2000, save_every=25,
    )
    train_run(model_cfg, train_cfg, shards_dir, run_dir, seed=0, variant="vanilla")
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "latest.safetensors").exists()
    # At least a few train records logged
    lines = (run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 3
    first = json.loads(lines[0])
    assert "step" in first and "train_loss" in first
    # Loss should be at least somewhat finite at the end
    final = json.loads(lines[-1])
    assert np.isfinite(final["train_loss"])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_train_driver.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `train.py`**

```python
# train.py
"""Training driver. Single GPU, single seed. Phase A: vanilla MHA only."""
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mlx.core as mx

from config import ModelConfig, TrainConfig
from model import Transformer
from data.loader import ShardLoader, sample_batch
from train_step import train_step
from eval import compute_val_loss
from schedule import cosine_lr_with_warmup
from optim import make_adamw
from metrics import MetricsLogger
from checkpoint import save_checkpoint, save_run_metadata
from gitinfo import current_hash, is_dirty
import json
import time

def train_run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    shards_dir: Path,
    run_dir: Path,
    seed: int = 0,
    variant: str = "vanilla",
) -> None:
    """Run one training stage to completion. Writes metrics, checkpoints, metadata."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(shards_dir)

    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = ShardLoader(shards_dir, "train")
    val_loader = ShardLoader(shards_dir, "val")
    data_meta = json.loads((shards_dir / "meta.json").read_text())

    # Build model (fp32 master) and optimizer
    model = Transformer(model_cfg)
    optimizer = make_adamw(
        lr=0.0,  # actual LR is set per-step from schedule
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.adam_beta1,
        beta2=train_cfg.adam_beta2,
        eps=train_cfg.adam_eps,
    )

    # Save run metadata up front
    save_run_metadata(
        run_dir=run_dir, model_cfg=model_cfg,
        train_cfg_dict=asdict(train_cfg),
        git_hash=current_hash(), git_dirty=is_dirty(),
        mlx_version=mx.__version__ if hasattr(mx, "__version__") else "unknown",
        seed=seed, data_meta=data_meta,
    )
    (run_dir / "variant.txt").write_text(variant + "\n")

    eff_tokens_per_step = train_cfg.micro_batch * model_cfg.block_size * train_cfg.grad_accum
    total_steps = max(1, train_cfg.total_tokens // eff_tokens_per_step)
    print(f"[train] {total_steps} steps, ~{eff_tokens_per_step} tokens/step, "
          f"{train_cfg.total_tokens / 1e6:.1f}M total tokens")

    logger = MetricsLogger(run_dir / "metrics.jsonl")
    t0 = time.time()
    step = 0
    while step < total_steps:
        # update LR
        lr = cosine_lr_with_warmup(
            step, train_cfg.peak_lr, train_cfg.warmup_steps, total_steps,
            min_lr_frac=0.1,
        )
        optimizer.learning_rate = lr

        x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
        x = mx.array(x_np)
        y = mx.array(y_np)
        loss = train_step(model, optimizer, x, y, grad_clip=train_cfg.grad_clip)

        # Tier-1 monitoring eval
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

def _build_cfgs(stage: str) -> tuple[ModelConfig, TrainConfig]:
    return getattr(ModelConfig, stage)(), getattr(TrainConfig, stage)()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    args = p.parse_args()
    if args.variant != "vanilla":
        raise SystemExit("Phase A: only vanilla supported. Diff lands in Phase B.")
    model_cfg, train_cfg = _build_cfgs(args.stage)
    train_run(model_cfg, train_cfg, args.shards_dir, args.run_dir, args.seed, args.variant)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_driver.py -v`
Expected: 1 passed (takes 30-60s)

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_driver.py
git commit -m "train: training driver with two-tier eval, checkpoints, metrics, metadata"
```

---

## Task 21: Stage 0 vanilla smoke run (real data, ~30M params, ~6,250 steps)

This is the integration test — it's not a pytest but a script you actually run.

**Files:**
- Create: `runs/stage0-vanilla-seed0/` (will be auto-created by train.py)
- Create: `scripts/stage0_vanilla.sh` (one-liner for reproducibility)

- [ ] **Step 1: Create the runner script**

```bash
# scripts/stage0_vanilla.sh
#!/bin/bash
set -euo pipefail
mkdir -p runs
python -m train \
  --stage stage0 \
  --shards_dir data/shards \
  --run_dir runs/stage0-vanilla-seed0 \
  --seed 0 \
  --variant vanilla
```

```bash
mkdir -p scripts
chmod +x scripts/stage0_vanilla.sh
```

- [ ] **Step 2: Confirm shards are ready**

Run:
```bash
ls -lh data/shards/
cat data/shards/meta.json
```

Expected: `train-0000.bin` of ~100-300 MB, `val.bin` of ~10-30 MB, valid `meta.json`.

If shards aren't ready, return to Task 5 and run the tokenizer.

- [ ] **Step 3: Pre-flight check: dry-run with 50 steps**

To avoid wasting 1 day if something is broken, first run a 50-step dry run by overriding `total_tokens` temporarily. Edit `config.py` `TrainConfig.stage0` and change `total_tokens=100_000_000` → `total_tokens=50 * 16 * 1024` for the dry run only.

Run:
```bash
caffeinate -di ./scripts/stage0_vanilla.sh
```

Expected: trains 50 steps in ~1-2 minutes. Loss should descend smoothly from ~11.5 (= log(100277), random init NLL) toward something lower. No NaN.

Verify:
```bash
tail -5 runs/stage0-vanilla-seed0/metrics.jsonl
```

Should show step, train_loss (decreasing), lr (warming up then decaying), tps, wall.

- [ ] **Step 4: Revert the `total_tokens` override and start the full Stage 0 run**

Revert `config.py` Stage 0 `total_tokens` to `100_000_000`. Commit the revert:
```bash
git add config.py
git commit -m "config: restore Stage 0 total_tokens after dry-run"
```

Delete the dry-run output and start the full run:
```bash
rm -rf runs/stage0-vanilla-seed0
caffeinate -di ./scripts/stage0_vanilla.sh 2>&1 | tee runs/stage0-vanilla-seed0.log &
```

Expected wall time: ~1 day on M5 Max. The run produces `runs/stage0-vanilla-seed0/metrics.jsonl` with ~6,250 training records and periodic eval records.

**While it runs:** the design doc §4.1 warnings apply. Don't load Anubis 70B in LM Studio; don't run heavy GPU tasks.

- [ ] **Step 5: After the run completes, verify expected outcomes**

Open `runs/stage0-vanilla-seed0/metrics.jsonl` (it's JSONL — `jq` or Python both work). Check:

1. **Loss decreased.** Final `train_loss` should be roughly 3-5 (perplexity ~20-150). Random init NLL was ~11.5.
2. **No NaN/Inf.** Filter the file for non-finite values:
   ```bash
   python -c "import json; [print(l) for l in open('runs/stage0-vanilla-seed0/metrics.jsonl') if 'NaN' in l or 'Inf' in l]"
   ```
   Should print nothing.
3. **Throughput projection.** At step 200, read `tps` from the metrics. Project Stage 1 wall: `2e9 / tps / 86400` days. Project Stage 2: `4e9 / tps / 86400` days. Both should be feasible (the design doc §5.2 says cut Stage 2 to 3B tokens or drop seeds if Stage 2 projects past 14 days).

- [ ] **Step 6: Save Stage 0 notes + final commit**

```bash
cat > runs/stage0-vanilla-seed0/NOTES.md <<'EOF'
# Stage 0 vanilla seed 0 — notes

## Run summary
- Total steps: 6,250
- Wall time: <fill in from metrics.jsonl>
- Final train loss: <fill in>
- Final val_loss_full: <fill in>
- Tokens/sec at step 200: <fill in>

## Projections (from step-200 throughput)
- Stage 1 (~2B tokens, B=32, T=2048): <fill in> days
- Stage 2 (~4B tokens, B=32, T=2048, grad_accum=4): <fill in> days

## Anything notable
<spikes, instabilities, surprises>
EOF
```

After filling in the numbers:
```bash
git add scripts/stage0_vanilla.sh runs/stage0-vanilla-seed0/NOTES.md
git commit -m "stage0: vanilla seed 0 smoke run complete; throughput calibration notes"
```

**Do not commit** `runs/stage0-vanilla-seed0/metrics.jsonl` or `latest.safetensors` — `runs/` is gitignored. The NOTES.md is the human-readable artifact.

---

## Task 22: Phase A retrospective — write up what works and what's brittle

**Files:**
- Create: `docs/2026-05-20-phase-a-retro.md`

- [ ] **Step 1: Run all tests one more time to make sure nothing regressed**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 2: Write the retro**

```markdown
# Phase A retro

**Date:** <fill in>
**Status:** Complete — vanilla GPT at Stage 0 trained successfully.

## What works
- Project scaffold: ...
- Data pipeline: ...
- Model + training loop: ...
- Stage 0 smoke run: <final loss, wall time, throughput>

## What's brittle (to address in Phase B or later)
- <e.g.: MLX optimizer dtype handling — verify fp32 master assumption>
- <e.g.: tiktoken vocab edge cases hit during tokenization?>
- <e.g.: any RMSNorm precision concerns observed in metrics>

## Throughput projections
| Stage | Projected wall | Decision |
|---|---|---|
| Stage 1 (2B tokens, 4 runs) | <X> days | proceed / cut to ... |
| Stage 2 (4B tokens, 4-6 runs) | <Y> days | proceed / cut to ... |

## Ready for Phase B?
- [ ] All Phase A tests green
- [ ] Stage 0 vanilla loss curve looks sane (descending, no NaN, plausible final value)
- [ ] Throughput projections fit the 2-3 week target (per design §0)

## What Phase B adds
- DiffAttention module (v0 = two SDPA calls + Python subtract)
- Lambda parameters + per-layer lambda computation
- subln (per-head RMSNorm over 2D, post-subtraction)
- Paired-seed init protocol (design §9.7)
- Reference cross-check vs PyTorch (design §7.4)
- Stage 0 paired vanilla/diff smoke run
```

- [ ] **Step 3: Commit the retro**

```bash
git add docs/2026-05-20-phase-a-retro.md
git commit -m "docs: Phase A retrospective + readiness check for Phase B"
```

---

## Task 23: Tag the Phase A milestone

- [ ] **Step 1: Tag the commit**

```bash
git tag -a phase-a-complete -m "Phase A: infrastructure + vanilla Stage 0 baseline complete"
```

- [ ] **Step 2: Verify**

```bash
git log --oneline | head -25
git tag --list
```

Phase A is done. The next plan (`docs/2026-05-20-diffattn-mlx-implementation-plan-phase-b.md`) picks up here with diff-attn implementation.

---

## Self-review against the design doc

Spec coverage checklist (every Phase A scope item from the design doc):

- ✅ Project setup, version pins (`pyproject.toml`, Task 1)
- ✅ Tokenizer wrapper (cl100k_base, Task 3)
- ✅ FineWeb-Edu download + tokenization (uint32 shards, Tasks 4-5)
- ✅ Deterministic data loader (Task 6)
- ✅ Backbone modules: RMSNorm, SwiGLU, VanillaMHA, Block, Transformer (Tasks 7-11)
- ✅ MLX SDPA called with `mask="causal"`, `scale=...` (Task 9)
- ✅ `mx.fast.rope` with `traditional=False` (LLaMA convention, Task 9)
- ✅ Tied embeddings (Task 11)
- ✅ Final RMSNorm before LM head (Task 11)
- ✅ Cosine LR with warmup (Task 12)
- ✅ AdamW + weight-decay exclusions (Task 13)
- ✅ fp32 master / bf16 forward helpers (Task 14)
- ✅ Checkpoint save/load + metadata snapshots (Task 15)
- ✅ Metrics JSONL (Task 16)
- ✅ Two-tier eval cadence (Task 17, used in Task 20)
- ✅ Training step with grad clip (Task 18)
- ✅ Git state capture (Task 19)
- ✅ Training driver, Stage 0 config (Task 20)
- ✅ Stage 0 vanilla smoke run (Task 21)

Phase A out-of-scope (per design + planning decision, deferred to later phases):
- DiffAttention module → Phase B
- Lambda parameter machinery → Phase B
- Paired-seed init protocol → Phase B
- Reference cross-check vs PyTorch → Phase B
- Custom Metal kernels (P1, P2, v1, v2) → Phase C
- Stages 1 and 2 full runs → Phase D
- Eval slices (AR-hit, LAMBADA) → Phase D
- Final writeup → Phase D
