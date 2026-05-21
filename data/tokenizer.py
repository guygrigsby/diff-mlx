"""cl100k_base tokenizer wrapper. Pinned via tiktoken version in pyproject.toml."""
import tiktoken

_ENCODING_NAME = "cl100k_base"
_enc = tiktoken.get_encoding(_ENCODING_NAME)

VOCAB_SIZE = _enc.n_vocab  # 100_277 for cl100k_base
EOT_ID = _enc.eot_token

def get_tokenizer() -> tiktoken.Encoding:
    return _enc

def tiktoken_version() -> str:
    import tiktoken as _t
    return _t.__version__
