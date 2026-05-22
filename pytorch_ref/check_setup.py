"""Smoke test for the pytorch_ref toolchain.

Run after `uv pip install -e .[dev]` to verify torch + CUDA install.

Usage:
    python check_setup.py
"""
import sys


def main():
    print(f"python: {sys.version}")

    try:
        import torch
    except ImportError as e:
        print(f"FAIL: torch not installed: {e}")
        sys.exit(1)
    print(f"torch: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"  VRAM (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
        # Quick bf16 matmul sanity
        x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        z = x @ y
        print(f"  bf16 matmul: OK (output dtype {z.dtype})")

    try:
        import numpy as np
        print(f"numpy: {np.__version__}")
    except ImportError as e:
        print(f"FAIL: numpy not installed: {e}")
        sys.exit(1)

    try:
        import tiktoken
        print(f"tiktoken: {tiktoken.__version__}")
        enc = tiktoken.get_encoding("cl100k_base")
        print(f"  cl100k_base loaded, vocab_size: {enc.n_vocab}")
    except ImportError as e:
        print(f"FAIL: tiktoken not installed: {e}")
        sys.exit(1)

    try:
        import safetensors
        print(f"safetensors: {safetensors.__version__}")
    except ImportError as e:
        print(f"FAIL: safetensors not installed: {e}")
        sys.exit(1)

    print()
    print("setup OK")


if __name__ == "__main__":
    main()
