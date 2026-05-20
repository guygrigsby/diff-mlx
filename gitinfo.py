"""Capture git state for run reproducibility."""
import subprocess


def current_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def is_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())
