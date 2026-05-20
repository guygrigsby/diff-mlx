from pathlib import Path
import inspect
from data.download import target_files_for, download_fineweb_edu_sample


def test_target_files_returns_parquet_paths(tmp_path):
    files = target_files_for(out_dir=tmp_path, n_files=3)
    assert len(files) == 3
    assert all(str(f).endswith(".parquet") for f in files)
    assert all(f.parent == tmp_path for f in files)


def test_download_signature_exists():
    sig = inspect.signature(download_fineweb_edu_sample)
    assert "out_dir" in sig.parameters
    assert "n_files" in sig.parameters
