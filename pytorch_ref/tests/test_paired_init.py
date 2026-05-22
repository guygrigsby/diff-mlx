"""Paired-seed init must produce byte-identical shared weights."""
import torch

from config import ModelConfig
from paired_init import build_paired_models


def _tiny_cfg():
    return ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
    )


def test_paired_init_shared_weights_byte_identical():
    cfg = _tiny_cfg()
    vanilla, diff = build_paired_models(cfg, seed=0)
    v = vanilla.state_dict()
    d = diff.state_dict()

    # Token embedding must match.
    assert torch.equal(v["tok_embed.weight"], d["tok_embed.weight"])

    # All four attention projections on every block.
    for li in range(cfg.n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"blocks.{li}.attn.{proj}.weight"
            assert torch.equal(v[key], d[key]), f"shared weight mismatch at {key}"

    # Final norm must match.
    assert torch.equal(v["final_norm.scale"], d["final_norm.scale"])


def test_paired_init_diff_lambdas_nonzero():
    """Diff lambda vectors come from a separate RNG stream and should not be zeros."""
    cfg = _tiny_cfg()
    _, diff = build_paired_models(cfg, seed=0)
    d = diff.state_dict()
    for li in range(cfg.n_layers):
        for lk in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            key = f"blocks.{li}.attn.{lk}"
            assert key in d, f"diff is missing {key}"
            assert d[key].abs().max() > 0, f"{key} is all zeros"
