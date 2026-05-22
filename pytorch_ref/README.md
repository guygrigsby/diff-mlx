# pytorch_ref

PyTorch port of the MLX Differential Transformer for **cross-stack validation**. Same algorithm, different framework, different hardware. Used to confirm the MLX result isn't an artifact of MLX-specific kernels or Metal precision quirks.

**Status:** scaffold only. Model port pending. Setup the toolchain first, then we wire it up.

## Why

Project's main implementation is MLX on Apple Silicon (parent dir). This subdir is a controlled cross-check: train the same Stage 0 paired config on an NVIDIA GPU and compare paired δ trajectories. If MLX and PyTorch agree within seed noise, the directional δ replication claim is independent of stack.

## Hardware

Built for RTX 3070 Ti (Ampere, sm_86, 8 GB VRAM). Should work on any sm_75+ GPU. Stage 0 (30M params) fits easily; Stage 1 (162M) is tight at 8 GB and would need micro_batch=2 with grad_accum=16.

## Setup (Windows, with uv)

If `uv` isn't installed, install it first:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then from this directory:

```powershell
cd pytorch_ref
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e .[dev]
```

Verify CUDA torch installed:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Should print `cuda: True` and `device: NVIDIA GeForce RTX 3070 Ti` (or similar).

If `cuda: False`: your CUDA driver doesn't match the `cu124` index in `pyproject.toml`. Switch to `cu121` or `cu118` by editing `pyproject.toml` and rerunning `uv pip install -e .[dev]`.

## Setup (macOS / Linux, with uv)

Same as above but use the unix install:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd pytorch_ref
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]
```

CUDA torch wheels won't have GPU support on macOS / non-NVIDIA Linux; on those machines you'd run CPU-only (slow but functional for the toy cross-check test).

## What gets ported

When the model port lands here, it will mirror the MLX side:

- `model.py`: `VanillaMHA`, `DiffAttention`, `Block`, `Transformer` in PyTorch. Same paper-canonical math (interleaved head split, traditional RoPE, fp32 lambda, subln RMSNorm, fp32 logits).
- `paired_init.py`: same byte-identical shared-weight protocol (design §9.7).
- `train_step.py`, `train.py`: forward + backward + AdamW; auto-resume support.
- `tests/test_diff_reference.py`: same cross-check against `../data/ref_fixtures/diffattn_toy_v1.npz`. The PyTorch port must hit the same `1e-3` fp32 tolerance against the vendored reference.
- `scripts/run_stage0_paired.py`: paired Stage 0 driver.

## Data

Training needs the token shards at `../data/shards/` (uint32 binary, `train-NNN.bin` + `val.bin` + `meta.json`). Copy from the Mac side or regenerate from `cl100k_base` if you don't want to transfer ~3 GB.

## Out of scope

- Stage 1/2 on the 3070 Ti. The 8 GB VRAM is tight at Stage 1; possible at micro_batch=2 with grad_accum=16, but tight enough that we don't promise it. Stage 0 cross-check is the primary deliverable.
- Distributed / multi-GPU. Single device only.
- Production training. Not the goal here.
