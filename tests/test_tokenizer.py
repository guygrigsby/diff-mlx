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
