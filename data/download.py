"""Download FineWeb-Edu sample parquet shards from HuggingFace Hub.

Run once per machine: python -m data.download --out_dir data/raw --n_files 8
"""
from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"


def target_files_for(out_dir: Path, n_files: int) -> list[Path]:
    """Return the local paths we'll download to (does not download)."""
    out_dir = Path(out_dir)
    return [out_dir / f"shard_{i:04d}.parquet" for i in range(n_files)]


def download_fineweb_edu_sample(out_dir: Path, n_files: int) -> list[Path]:
    """Download N parquet shards from the sample-10BT subset.

    Each shard is ~500MB and contains ~500k documents. n_files=4-6 yields
    enough text for Stage 0 (~100M tokens after tokenization).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_files):
        remote = f"{SUBSET}/{i:03d}_00000.parquet"
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote,
            repo_type="dataset",
            local_dir=out_dir,
        )
        paths.append(Path(local))
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--n_files", type=int, default=4)
    args = p.parse_args()
    paths = download_fineweb_edu_sample(args.out_dir, args.n_files)
    print(f"Downloaded {len(paths)} shards to {args.out_dir}")
    for p_ in paths:
        size_mb = p_.stat().st_size / 1e6
        print(f"  {p_.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
