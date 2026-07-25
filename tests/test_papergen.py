"""The paper's numbers are generated from committed evidence, never typed.

gen.py runs over docs/paper/data/ (committed, deterministic), so the test
asserts real output values, not fixture stand-ins.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "docs" / "paper"


def test_gen_produces_all_artifacts(tmp_path):
    out = subprocess.run([sys.executable, str(PAPER / "gen.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    gen = PAPER / "generated"
    for f in ["summary_macros.tex", "stage0_table.tex", "stage1_table.tex",
              "posbin_table.tex", "stage0_band.tex", "stage1_curves.tex",
              "posbin_fig.tex"]:
        assert (gen / f).exists(), f"missing {f}"

    macros = (gen / "summary_macros.tex").read_text()
    # Stage 1 endpoints are fixed committed evidence; assert the known values
    # so a data or aggregation regression cannot pass silently.
    assert "\\newcommand{\\StageOneValDeltaSigned}{+0.035}" in macros
    assert "\\newcommand{\\StageZeroSeedCount}{4}" in macros
    # One seed flipped sign in the band.
    assert "\\newcommand{\\StageZeroFlipCount}{1}" in macros

    band = (gen / "stage0_table.tex").read_text()
    assert band.count("\\\\") >= 6  # header + recorded seed 0 + 4 seeds

    # Position-binned table has 8 bins plus overall.
    pos = (gen / "posbin_table.tex").read_text()
    assert "1792--2047" in pos and "overall" in pos
