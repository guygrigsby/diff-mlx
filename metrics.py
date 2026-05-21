"""Append-only JSONL metrics logger."""
from __future__ import annotations
from pathlib import Path
import json


class MetricsLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)  # line-buffered

    def log(self, **kwargs) -> None:
        self._f.write(json.dumps(kwargs) + "\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
