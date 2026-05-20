from gitinfo import current_hash, is_dirty


def test_current_hash_is_40_char_hex():
    h = current_hash()
    assert len(h) == 40
    int(h, 16)  # raises if not hex


def test_is_dirty_returns_bool():
    assert isinstance(is_dirty(), bool)
