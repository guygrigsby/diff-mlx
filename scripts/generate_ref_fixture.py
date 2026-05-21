"""Generate a PyTorch reference fixture for the diff-attn cross-check test.

One-time setup. Saves: data/ref_fixtures/diffattn_toy_v1.npz with:
- input_x: (B, T, dim) random tensor
- ref_output: PyTorch reference forward output
- ref_cos / ref_sin: (T, head_dim/2) RoPE tables used in the forward
- weight__*: the PyTorch state_dict (each tensor as fp32 numpy)
- meta: shape + version info

The reference uses GPT-J-style interleaved RoPE (pairs of consecutive even/odd
dims). Task 7 (MLX cross-check test) will need to either pass interleaved-style
RoPE on the MLX side (mx.fast.rope traditional=True) or pre-rotate weights to
compensate for the convention difference. The cos/sin tables are saved so the
test can reuse them verbatim.

Toy shape per design §7.4: B=2, T=16, dim=64, n_heads_diff=2, qk_head_dim=16.
The reference's num_heads argument equals our n_heads_diff (because
head_dim = embed_dim // num_heads // 2 in the reference).
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure both the project root and the vendored ref_fixtures dir are importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "data" / "ref_fixtures"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REF_DIR))

import numpy as np
import torch

import multihead_diffattn_reference as ref  # noqa: E402


# Toy shape per design §7.4
B, T, DIM = 2, 16, 64
NUM_HEADS_REF = 2  # matches our n_heads_diff; gives reference head_dim = 64//2//2 = 16
DEPTH = 0          # 0-indexed: lambda_init_fn(0) = 0.8 - 0.6 = 0.2 (== paper layer 1)
ROPE_BASE = 10000.0


def build_rope_tables(seqlen: int, rotary_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables of shape (seqlen, rotary_dim/2) for interleaved RoPE.

    Matches the convention expected by microsoft/unilm Diff-Transformer's
    apply_rotary_emb: cos/sin are sized as (T, head_dim/2), one frequency per
    even/odd dimension pair. inv_freq[k] = 1 / base^(2k / D), k in [0, D/2).
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rotary_dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.einsum("t,k->tk", t, inv_freq)  # (T, D/2)
    return freqs.cos(), freqs.sin()


def main() -> None:
    torch.manual_seed(42)

    attn = ref.MultiheadDiffAttn(
        embed_dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS_REF,
    )
    attn.eval()

    # Sanity: confirm internal head_dim matches our qk_head_dim assumption.
    assert attn.head_dim == 16, f"expected head_dim=16, got {attn.head_dim}"
    assert abs(attn.lambda_init - 0.2) < 1e-6, (
        f"expected lambda_init=0.2 at depth={DEPTH}, got {attn.lambda_init}"
    )

    head_dim = attn.head_dim
    cos, sin = build_rope_tables(T, head_dim, base=ROPE_BASE)
    rel_pos = (cos, sin)

    x = torch.randn(B, T, DIM, dtype=torch.float32)

    with torch.no_grad():
        output = attn(x, rel_pos)

    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in attn.state_dict().items()}

    out_path = REF_DIR / "diffattn_toy_v1.npz"
    np.savez(
        out_path,
        input_x=x.detach().cpu().numpy().astype(np.float32),
        ref_output=output.detach().cpu().numpy().astype(np.float32),
        ref_cos=cos.detach().cpu().numpy().astype(np.float32),
        ref_sin=sin.detach().cpu().numpy().astype(np.float32),
        B=np.array(B),
        T=np.array(T),
        dim=np.array(DIM),
        num_heads_ref=np.array(NUM_HEADS_REF),
        depth=np.array(DEPTH),
        head_dim=np.array(head_dim),
        lambda_init=np.array(attn.lambda_init, dtype=np.float32),
        rope_base=np.array(ROPE_BASE, dtype=np.float32),
        **{f"weight__{k.replace('.', '__')}": v for k, v in state.items()},
    )

    print(f"Saved fixture: {out_path}")
    print(f"  Input shape:  {tuple(x.shape)}; Output shape: {tuple(output.shape)}")
    print(f"  Output stats: mean={output.mean().item():.4f} std={output.std().item():.4f}")
    print(f"  lambda_init:  {attn.lambda_init:.4f}  (paper layer 1 -> 0.2)")
    print()
    print("PyTorch state_dict keys (stored as weight__* in npz):")
    for k, v in state.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
