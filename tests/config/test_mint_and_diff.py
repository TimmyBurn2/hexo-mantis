"""Mint/diff tool behavior tests (exit-code contracts) + O8 one-key-diff over the grown
(nested) template. The dev template now carries every WP8 field, so mint/diff must handle
dotted nested keys (eval.random_model_sims) and list deltas (radius schedule)."""
import subprocess
import sys
from pathlib import Path

from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
DIFF = REPO_ROOT / "tools" / "config_diff.py"
TEMPLATE = REPO_ROOT / "tools" / "config_templates" / "dev.yaml"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=False)


def _mint(tmp_path: Path, name: str, *deltas: str):
    out = tmp_path / name
    argv = [str(MINT), "--template", "dev", "--out", str(out)]
    for delta in deltas:
        argv += ["--set", delta]
    return out, _run(*argv)


def test_mint_output_validates(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "run_id=mint_check")
    assert proc.returncode == 0, proc.stderr
    cfg = load_config(out)
    assert cfg.run_id == "mint_check"


def test_mint_stamps_template_and_delta_header(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "run_id=mint_check")
    assert proc.returncode == 0, proc.stderr
    head = out.read_text().splitlines()[:3]
    assert head[0] == "# minted-by: tools/mint_config.py"
    assert head[1] == "# template: dev"
    assert head[2] == "# delta: run_id: template_dev -> mint_check"


def test_mint_nested_key_delta(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "run_id=nn", "eval.random_model_sims=64")
    assert proc.returncode == 0, proc.stderr
    cfg = load_config(out)
    assert cfg.eval.random_model_sims == 64


def test_mint_rejects_unknown_delta_key(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "identity.bogus=1")
    assert proc.returncode == 2
    assert "unknown delta key" in proc.stderr
    assert not out.exists()


def test_mint_rejects_an_unknown_INTERMEDIATE_delta_segment(tmp_path):
    """The BEHAVIOUR — an unknown intermediate segment of a dotted key is rc 2 by name — with
    the honest note that it is **not** a flip row for `_resolve_parent`'s `part not in node`.

    WPAX ADJ-13 corrective pass, R72 rule R72-C. That conjunct came out UNCOVERED, and the
    reason is that it is **provably redundant**, not that nobody wrote a test: deleting it lets
    the loop fall into `node[part]`, which raises the same `KeyError` the caller already
    catches, and nothing reads the exception's argument. Measured — `--set nosuchsection.seed=1`,
    `identity.nosuch.leaf=1` and `identity.bogus=1` each produce **byte-identical rc 2 and
    stderr** with the conjunct and with it replaced by `False`. No test can distinguish it, so a
    row claiming to kill it would be a row that cannot fail, which is exactly what R72 exists to
    find. Stated here rather than dressed up (the precedent is R72 row L5's
    `max(learners) >= 1`), and carded as CARD-MINT-RESOLVE-PARENT-CONJUNCT.

    `test_mint_rejects_unknown_delta_key` does not cover this behaviour either: `identity.bogus`
    walks a REAL section and fails on the LEAF check one line further down, so the intermediate
    arm has no producer at all. This row is that producer.
    """
    out, proc = _mint(tmp_path, "minted.yaml", "nosuchsection.seed=1")
    assert proc.returncode == 2, (proc.stdout + proc.stderr)[-2000:]
    assert "unknown delta key" in proc.stderr, proc.stderr[-2000:]
    assert not out.exists()
    deep_out, deep = _mint(tmp_path, "deep.yaml", "identity.nosuch.leaf=1")
    assert deep.returncode == 2 and "unknown delta key" in deep.stderr, deep.stderr[-2000:]
    assert not deep_out.exists()


def test_mint_refuses_to_overwrite_WITHOUT_force_and_obeys_it_WITH_it(tmp_path):
    """R72 (same pass): the `args.force` conjunct of `if out_path.exists() and not args.force`
    was uncovered — replacing it with a constant left the full default tier green in BOTH
    directions, so the flag was decoration.

    Both arms matter and neither is cosmetic. Without `--force`, a mint that silently
    overwrites is a minted config replaced with no record — the R1/LAW-12 provenance failure.
    With `--force`, a mint that refuses anyway makes the documented re-mint path dead, which is
    how an operator ends up hand-editing a config instead (the very thing minting exists to
    stop).
    """
    out, first = _mint(tmp_path, "minted.yaml", "run_id=first")
    assert first.returncode == 0, first.stderr
    body = out.read_text()

    _, refused = _mint(tmp_path, "minted.yaml", "run_id=second")
    assert refused.returncode == 2, (refused.stdout + refused.stderr)[-2000:]
    assert "refusing to overwrite" in refused.stderr
    assert out.read_text() == body, "refused, and the existing file must be untouched"

    argv = [str(MINT), "--template", "dev", "--out", str(out), "--set", "run_id=second",
            "--force"]
    forced = _run(*argv)
    assert forced.returncode == 0, (forced.stdout + forced.stderr)[-2000:]
    assert load_config(out).run_id == "second", (
        "--force must actually overwrite, or the flag names a capability the tool lacks"
    )


def test_diff_exit_0_on_exactly_claimed_one_key_diff(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "MATCH" in res.stdout


def test_diff_exit_0_on_nested_key_diff(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "eval.random_model_sims=64")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "eval.random_model_sims")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "MATCH" in res.stdout


def test_diff_exit_1_on_unclaimed_extra_diff(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check", "seed=999")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id")
    assert res.returncode == 1
    assert "UNCLAIMED diff on seed" in res.stdout


def test_diff_exit_1_when_claimed_key_identical(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id", "--expect", "seed")
    assert res.returncode == 1
    assert "expected diff on seed but values are identical" in res.stdout
