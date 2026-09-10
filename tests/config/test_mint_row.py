"""`--mint-row` — the flag that writes a row every template omits by design.

WHY THIS FILE EXISTS. `identity.arch_kind` and `identity.warm_start` are schema-OPTIONAL and
absent from every committed config and every template: R323(b) rules that they enter production
configs ONLY as a minted row at run6's mint. `--set` refuses a key the template does not carry,
so before `--mint-row` the tool could not write the one class of row a mint exists for, and the
run6 mint act had no mechanism. The tests below pin the capability AND the two refusals that
keep it from becoming "create any key you like".

R347/CONFIG-1 MADE THIS FLAG THE MAIN ROAD rather than a two-key special case. The operational
constants now carry schema defaults and leave the template, so overriding one is a minted row —
which is the point: the override stays a deliberate act with a stamped header line, exactly as
R323(b) wanted for the identity rows. Two of those constants are BLOCKS (`monitor.drain`,
`monitor.disk_guard`), and a block is minted WHOLE, because `--mint-row` adds a leaf to an
existing block and does not build one. That is a property worth a row of its own rather than a
limitation to work around: a half-minted block whose other leaves came silently from the schema
is exactly the "which authority set this?" question R1 exists to make unaskable.
"""
import subprocess
import sys
from pathlib import Path

from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
DIFF = REPO_ROOT / "tools" / "config_diff.py"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=False)


def _mint(out: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(str(MINT), "--template", "dev", "--out", str(out), *args)


def test_mint_row_writes_a_row_the_template_omits(tmp_path: Path) -> None:
    out = tmp_path / "arch.yaml"
    proc = _mint(out, "--mint-row", "identity.arch_kind=GnnArchV2")
    assert proc.returncode == 0, proc.stderr
    assert load_config(out).identity.arch_kind == "GnnArchV2"


def test_the_stamped_old_value_is_the_SCHEMAS_resolution_not_a_guess(tmp_path: Path) -> None:
    """The header must state what validation resolves the template to, or the diff gate lies."""
    out = tmp_path / "arch.yaml"
    assert _mint(out, "--mint-row", "identity.arch_kind=GnnArchV2").returncode == 0
    header = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    assert "# delta: identity.arch_kind: null -> GnnArchV2" in header, header


def test_config_diff_from_header_AGREES_with_a_minted_row(tmp_path: Path) -> None:
    """The real coupling: `config_diff` diffs VALIDATED config against VALIDATED template, so a
    row the header did not claim (or claimed wrongly) reds here."""
    out = tmp_path / "arch.yaml"
    assert _mint(out, "--mint-row", "identity.arch_kind=GnnArchV2").returncode == 0
    proc = _run(str(DIFF), "--from-header", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identity.arch_kind" in proc.stdout


def test_mint_row_REFUSES_a_key_the_template_already_carries(tmp_path: Path) -> None:
    proc = _mint(tmp_path / "x.yaml", "--mint-row", "identity.encoding=gnn_axis_r8")
    assert proc.returncode == 2
    assert "ALREADY in the template" in proc.stderr
    assert "Use --set" in proc.stderr


def test_mint_row_REFUSES_a_key_the_schema_does_not_admit(tmp_path: Path) -> None:
    """extra='forbid' is the backstop; the refusal must name the key, not dump a stack."""
    proc = _mint(tmp_path / "x.yaml", "--mint-row", "identity.arch_kindd=GnnArchV2")
    assert proc.returncode == 2
    assert "arch_kindd" in proc.stderr


def test_mint_row_REFUSES_a_parent_block_that_does_not_exist(tmp_path: Path) -> None:
    proc = _mint(tmp_path / "x.yaml", "--mint-row", "no_such_block.leaf=1")
    assert proc.returncode == 2
    assert "parent path stops at" in proc.stderr


def test_set_STILL_refuses_a_key_the_template_omits(tmp_path: Path) -> None:
    """The guard --mint-row exists beside, not instead of: a typo'd --set must still fail."""
    proc = _mint(tmp_path / "x.yaml", "--set", "identity.arch_kind=GnnArchV2")
    assert proc.returncode == 2
    assert "unknown delta key" in proc.stderr


def test_a_minted_row_and_a_delta_compose_in_one_act(tmp_path: Path) -> None:
    """Run6's mint sets existing keys AND adds rows in the same invocation; prove it here."""
    out = tmp_path / "both.yaml"
    proc = _mint(
        out,
        "--set", "identity.encoding=gnn_axis_r8",
        "--mint-row", "identity.arch_kind=GnnArchV2",
    )
    assert proc.returncode == 0, proc.stderr
    cfg = load_config(out)
    assert (cfg.identity.encoding, cfg.identity.arch_kind) == ("gnn_axis_r8", "GnnArchV2")
    assert _run(str(DIFF), "--from-header", str(out)).returncode == 0


def test_a_defaulted_BLOCK_is_minted_WHOLE_and_its_old_slot_is_the_schemas(tmp_path: Path) -> None:
    """R347/CONFIG-1. `monitor.drain` left the template with a schema default, so a run that
    wants a different cap mints the block — all four leaves, one row — and the header's OLD
    slot is what the schema resolves the template to, not a guess and not `null`.

    The leaf-level attempt is refused by `_resolve_new_parent` and is asserted here beside the
    working form, so the two are read together: the refusal is the reason the whole-block mint
    is the shape, not an obstacle it routes around.
    """
    leaf = _mint(tmp_path / "leaf.yaml", "--mint-row",
                 "monitor.drain.terminal_eval_hard_cap_sec=7200.0")
    assert leaf.returncode == 2, leaf.stdout + leaf.stderr
    assert "does not build the block" in (leaf.stdout + leaf.stderr)

    out = tmp_path / "block.yaml"
    proc = _mint(out, "--mint-row",
                 "monitor.drain={final_eval_drain_timeout_sec: 900.0, "
                 "eval_final_drain_safety_factor: 3.0, "
                 "eval_final_drain_hard_cap_sec: 14400.0, "
                 "terminal_eval_hard_cap_sec: 7200.0}")
    assert proc.returncode == 0, proc.stderr
    caps = load_config(out).monitor.drain
    assert caps.terminal_eval_hard_cap_sec == 7200.0
    assert caps.final_eval_drain_timeout_sec == 900.0
    header = "\n".join(ln for ln in out.read_text(encoding="utf-8").splitlines()
                        if ln.startswith("# delta:"))
    assert "terminal_eval_hard_cap_sec: 14400.0} ->" in header, (
        "the old slot must be the SCHEMA's resolution of the omitted block, or "
        f"`config_diff --from-header` disagrees with the file. got {header}"
    )
