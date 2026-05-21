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
