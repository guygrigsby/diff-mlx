"""Save/load round-trip for the PyTorch checkpoint format (model + opt state)."""
import torch

from config import ModelConfig
from model import Transformer
from optim import make_adamw
from checkpoint import save_checkpoint, load_checkpoint


def _tiny_cfg():
    return ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=4, qk_head_dim=8,
        vocab_size=128, mlp_intermediate=64, block_size=16,
    )


def test_save_and_load_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    model = Transformer(cfg)
    opt = make_adamw(model, lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)

    # One backward + step to populate optimizer state.
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    loss = (model(x) ** 2).sum()
    loss.backward()
    opt.step()

    ckpt = tmp_path / "ckpt.safetensors"
    save_checkpoint(model, opt, step=7, ckpt_path=ckpt)

    # Build a fresh model + optimizer, load, verify.
    model2 = Transformer(cfg)
    opt2 = make_adamw(model2, lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    loaded_state, step, opt_state = load_checkpoint(ckpt)
    model2.load_state_dict(loaded_state)
    assert step == 7
    assert opt_state is not None

    # Every model param matches.
    for name, p in model.state_dict().items():
        assert torch.equal(p, model2.state_dict()[name]), f"mismatch at {name}"

    # Optimizer m,v step round-trip: reconstruct optimizer.state_dict() shape.
    opt2.load_state_dict({
        "state": opt_state,
        "param_groups": opt2.state_dict()["param_groups"],
    })

    # Spot-check: state of first param should match.
    pid0 = list(opt.state_dict()["state"].keys())[0]
    src = opt.state_dict()["state"][pid0]
    dst = opt2.state_dict()["state"][pid0]
    assert src.keys() == dst.keys()
    for k in src:
        if isinstance(src[k], torch.Tensor):
            assert torch.equal(src[k], dst[k]), f"opt tensor mismatch at param {pid0} key {k}"
        else:
            assert src[k] == dst[k]
