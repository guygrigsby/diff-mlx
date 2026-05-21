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
        assert b > a, f"non-monotonic: {a} -> {b}"


def test_paper_formula_exact_layer_1():
    expected = 0.8 - 0.6 * math.exp(-0.3 * (1 - 1))
    assert lambda_init_for_layer(1) == expected
