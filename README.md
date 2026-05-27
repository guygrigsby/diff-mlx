# diff-mlx

MLX implementation of the Differential Transformer (Ye et al., ICLR 2025; [arXiv 2410.05258](https://arxiv.org/abs/2410.05258)) on Apple Silicon, with custom Metal kernels for the differential-attention forward pass. A small-scale, controlled, paired-init reproduction of the diff-attn mechanism, cross-validated against the vendored Microsoft PyTorch reference and a second run in PyTorch on NVIDIA CUDA.

**Status: complete.** Full writeup: [`docs/2026-05-23-final-writeup.md`](docs/2026-05-23-final-writeup.md).

## Result in one paragraph

At Stage 0 (30M params, 100M tokens) the paired δ replicated the paper's directional claim, diff beating vanilla by 0.020 nats on held-out val. At Stage 1 (162M params, 2.0B tokens) it did **not**: diff ended 0.11 nats ahead on *train* loss but 0.035 nats *behind* vanilla on held-out val, with a clear overfitting signature. A position-binned eval found vanilla uniformly better across the entire 2048-token window with no widening of diff's deficit at later positions, so the architecture's signature long-context advantage did not appear at this scale either. Net: in this small-scale, short-context, single-seed regime, DiffAttention shows no generalization benefit. This is three orders of magnitude below the paper's 3B-param / 1T-token setup, so it refutes nothing about the paper. It is an honest negative for the small-scale regime.

![Stage 1 diff vs vanilla](docs/stage1_diff_vs_vanilla.png)

## What's here

- **MLX implementation** of vanilla MHA + Differential Attention (`model.py`), paper-canonical interleaved head split, RoPE (interleaved), SwiGLU, RMSNorm, tied embeddings.
- **Custom Metal kernels** via `mx.fast.metal_kernel`: P1 row-wise softmax (`kernels/softmax_p1.py`), P2 causal SDPA (`kernels/sdpa_p2.py`), and a v1 diff composition that swaps them in with `mx.custom_function` autograd hooks.
- **Paired-init protocol**: byte-identical shared weights between variants so single-seed δ is meaningful.
- **bf16 mixed precision** via `LinearAMP`; optimizer-state checkpoints, auto-resume, gradient accumulation through a compiled step.
- **PyTorch cross-stack reference** (`pytorch_ref/`) run on an RTX 3070 Ti to rule out MLX/Metal artifacts.
- **116 tests** in `tests/` (plus the PyTorch side in `pytorch_ref/tests/`).

## Correctness

- Cross-check vs vendored Microsoft reference (`tests/test_diff_reference.py`): **3.58e-7** max diff on CPU stream.
- P1 softmax: ~3e-8 (fp32) / ~2e-3 (bf16) vs `mx.softmax`.
- P2 SDPA: within the bf16 ULP-noise band vs `mx.fast.scaled_dot_product_attention`.

## Model checkpoints

Both final Stage 1 checkpoints (162M params, 2.0B tokens, seed 0, safetensors) are published on Hugging Face: **[huggingface.co/guygrigsby/diff-mlx](https://huggingface.co/guygrigsby/diff-mlx)** (`diff/` and `vanilla/` subfolders).

## Reproducing

```bash
# env (Apple Silicon, MLX)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# tests
pytest -q

# Stage 1 paired run (long; needs prepared shards in data/shards/)
python scripts/stage1_paired.py --data_seed 0 --model_seed 0 --out_root runs/stage1-paired

# position-binned held-out eval on the final checkpoints
python scripts/eval_position_binned.py
```

Data shards and training runs are gitignored (see `.gitignore`); only code, docs, and the small reference fixture are tracked.

## Findings worth reading even if you don't care about diff-attn

- **Apple Silicon throughput is dispatch-bound** at these shapes (~14k tok/s, ~5-10% of bf16 peak). macmon GPU-% is utilization, not throughput.
- **Swap cliff:** per-token cost is flat then falls off a cliff at the unified-memory budget. `micro_batch=32` thrashed swap and read 14× slow; `micro_batch=8 grad_accum=4` fixed it. See `docs/2026-05-22-swap-cliff-and-scope-restore.md`.
- **Thermal + power throttling on a laptop chassis:** stock fan curve throttles within ~10 min; aggressive cooling roughly doubles sustained throughput. And a low temperature does *not* rule out throttling: a Thunderbolt dock silently capped charging at 100W (vs the 140W MagSafe), shaving GPU clocks while the chip sat cool at 73°C. See `docs/2026-05-24-thermal-empirical-notes.md`.

## Docs

- `docs/2026-05-23-final-writeup.md` — the full writeup (start here).
- `docs/2026-05-20-diffattn-mlx-reproduction-design.md` — design; kernel specs in §5.1, §5.1b, §7 are authoritative.
- `docs/2026-05-24-thermal-empirical-notes.md` — thermal + power throttling on M5 Max.
- `docs/2026-05-22-swap-cliff-and-scope-restore.md` — the swap-cliff investigation.
- `docs/2026-05-21-bf16-mixed-precision-design.md` — bf16 design.
- `docs/archive/` — superseded plans and phase retros, kept for history.

## License

MIT. See `LICENSE`.
