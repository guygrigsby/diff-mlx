from schedule import cosine_lr_with_warmup


def test_warmup_zero_at_step_zero():
    lr = cosine_lr_with_warmup(step=0, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 0.0) < 1e-9


def test_warmup_peak_at_warmup_end():
    lr = cosine_lr_with_warmup(step=100, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-3) < 1e-9


def test_cosine_midway_between_peak_and_min():
    step = 100 + (1000 - 100) // 2
    lr = cosine_lr_with_warmup(step=step, peak_lr=1.0, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 0.55) < 1e-6


def test_min_lr_at_total_steps():
    lr = cosine_lr_with_warmup(step=1000, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-4) < 1e-9


def test_holds_min_after_total_steps():
    lr = cosine_lr_with_warmup(step=5000, peak_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_frac=0.1)
    assert abs(lr - 1e-4) < 1e-9
