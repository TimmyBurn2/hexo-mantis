"""CI gate 17's producer test (LAW-07): the rule-7 host-content detector must BITE.

rule7-gate: file-ok -- THIS FILE IS THE ORACLE. Every host-shaped literal below is a planted
fixture (RFC 5737 / RFC 2606 reserved), never a real machine; the set of files allowed to say
this is pinned by `test_file_level_hatch_is_confined_to_the_two_pattern_files` below.

The gate exists because rule 7 was enforced by memory alone until the R280(c) scan ran it
against `origin/dev` and found 101 absolute box paths already committed in a fixture. So the
first half of this file proves the detector fires on each registered class.

The SECOND half matters as much, and is why the pattern register carries carve-outs rather
than an exemption list: `gate@test.invalid` and `gate3c@example.invalid` are live in this very
directory, `127.0.0.1` and `0.0.0.0` are ordinary bind addresses, and a version string is four
dot-separated numbers. A gate that fires on any of those gets switched off within a week.

Every case is driven through `rule7_gate.scan_text` — the SAME function the gate's own scan
and its in-process self-test call. An oracle that re-implemented the decision could drift from
the thing it certifies, which is the class this repo keeps finding (gate 16's producer test
says so in its own words).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "rule7_gate.py"


def _load_gate():
    """Load the gate by PATH (R5/LAW-17 ban `sys.path` mutation; `tools/` is not a package)."""
    spec = importlib.util.spec_from_file_location("_rule7_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _classes(text: str) -> set[str]:
    return {name for _rel, _ln, name, _m, _why in GATE.scan_text("probe.txt", text)}


# ── the detector fires ────────────────────────────────────────────────────────────────
#: One planted line per registered class. Hoisted to a constant so the coverage test below can
#: DERIVE what is covered from the same object pytest parametrizes over, rather than from a
#: second transcribed list that would drift from it.
# rule7-gate: ok -- planted fixtures; they name no real machine (RFC 5737 / RFC 2606)
FIRING_CASES: tuple[tuple[str, str], ...] = (
    ("abs-root-path", "repo_path_on_box = /root/hexo-mantis"),
    ("abs-home-path", "outdir = /home/operator/null_probe_out"),
    ("detached-run", "nohup uv run python -m mantis.run &"),
    ("box-outdir", "tail -f shakedown_out/run5/train.log"),
    ("ssh-userhost", "rsync to boxuser@gpu-rig.example-provider.net"),
    ("ssh-invocation", "scp -r checkpoints/ boxhost:/tmp/out"),
    ("ssh-config", "IdentityFile ~/.ssh/id_ed25519"),
    ("provider", "the 5080 was rented from vast.ai"),
    ("ipv4", "the box answered on 203.0.113.7"),
)


@pytest.mark.parametrize(("name", "line"), FIRING_CASES)
def test_each_registered_class_fires(name: str, line: str) -> None:
    assert name in _classes(line), f"{name} did not fire on {line!r}"


def test_every_registered_pattern_has_a_firing_case() -> None:
    """DERIVED, never transcribed: a pattern added to the register without a case here would
    otherwise be shipped uncovered — LAW-07's own failure class, one level up."""
    covered = {name for name, _line in FIRING_CASES}
    assert covered == set(GATE.PATTERNS), (
        f"register/oracle drift — uncovered: {sorted(set(GATE.PATTERNS) - covered)}; "
        f"stale: {sorted(covered - set(GATE.PATTERNS))}"
    )


# ── the detector stays silent on correct content ──────────────────────────────────────
@pytest.mark.parametrize(
    "line",
    [
        "contact gate@test.invalid",             # RFC 2606 reserved: can never resolve
        "id = gate3c@example.invalid",           # live in tests/tools/test_test_count_gate.py
        "server.bind('127.0.0.1', 0)",           # loopback names no machine
        "host: 0.0.0.0",                         # unspecified address
        "torch 2.11.0+cu128, rustc 1.97.1",      # version strings are not IPv4
        "sha256 = '431ccf2846615fa6ac06d073af008dcee6969a4286139aea4cebe316ddf5b740'",
        "see docs/registers/laws.md for LAW-07",
    ],
)
def test_no_false_positive(line: str) -> None:
    assert not GATE.scan_text("probe.txt", line), f"false positive on {line!r}"


def test_reserved_tld_carve_out_does_not_swallow_a_real_host() -> None:
    """The first draft anchored the reserved label with `\\b`, so `example-provider.net` read
    as reserved — `e` -> `-` IS a word boundary. The gate's own self-test caught it. This pins
    the fix: the reserved label must be the FINAL label, not merely word-bounded."""
    assert "ssh-userhost" in _classes("user@example-provider.net")
    assert "ssh-userhost" not in _classes("user@example.com")
    assert "ssh-userhost" not in _classes("user@sub.example.invalid")


# ── the escape hatch ──────────────────────────────────────────────────────────────────
def test_escape_hatch_suppresses_on_the_line() -> None:
    assert not GATE.scan_text("p.txt", "path = /root/x  # rule7-gate: ok -- register entry")


def test_escape_hatch_suppresses_from_the_comment_block_above() -> None:
    text = "# rule7-gate: ok -- fixture\n# second comment line\npath = /root/x\n"
    assert not GATE.scan_text("p.txt", text)


def test_escape_hatch_does_not_leak_past_a_blank_line() -> None:
    """A hatch must cover its own block only. If a blank line did not stop the walk, one hatch
    near the top of a file would silence everything below it."""
    text = "# rule7-gate: ok -- fixture\n\npath = /root/x\n"
    assert GATE.scan_text("p.txt", text)


def test_escape_hatch_requires_the_reason_marker() -> None:
    """`ESCAPE` ends in `--`, so a bare mention of the gate name does not suppress."""
    assert GATE.scan_text("p.txt", "path = /root/x  # rule7-gate")


# ── the gate's own guards ─────────────────────────────────────────────────────────────
def test_self_test_passes_in_process() -> None:
    """The gate runs this on EVERY invocation; if it ever goes red, the gate exits 2 rather
    than reporting a green tree it can no longer certify."""
    assert GATE.self_test() is True


def test_exempt_register_ships_empty() -> None:
    """R98: the gate adopted over a CLEAN tree. The one known blob was sanitized in the commit
    before the gate, not exempted here. If this ever goes non-empty, the entry must carry
    grounds AND a blob sha, and both halves self-expire (see EXEMPT's own comment)."""
    for path, sub, sha, grounds in GATE.EXEMPT:
        assert path and sub and grounds, "an exemption needs a path, a substring and grounds"
        assert len(sha) == 64, f"exemption for {path} must pin the blob sha256"


def test_file_level_hatch_is_confined_to_the_two_pattern_files() -> None:
    """THE GUARD ON THE BIG HAMMER. `rule7-gate: file-ok` silences a WHOLE file, so exactly two
    files may carry it: this gate's pattern register and this oracle. Both are made of planted
    literals by construction. Any third file acquiring one reds HERE — which is the only thing
    standing between a convenience marker and a silently-disabled gate.

    This test is also why the first draft was caught: the gate's own first full-tree run
    reported a clean tree while both of these files were still UNTRACKED, so `git ls-files`
    never handed them to the scan. A green run that scanned nothing is the failure this repo
    keeps re-finding, and the non-vacuity floor below is the other half of the answer.
    """
    allowed = {"tools/ci_gates/rule7_gate.py", "tests/tools/test_rule7_gate.py"}
    carrying = {
        rel for rel in GATE.target_files(None)
        if (t := GATE._read_text(REPO_ROOT / rel)) is not None and GATE.has_file_escape(t)
    }
    assert carrying == allowed, (
        f"file-level hatch set drifted — unexpected: {sorted(carrying - allowed)}; "
        f"missing: {sorted(allowed - carrying)}"
    )


def test_full_tree_floor_is_below_the_live_count() -> None:
    """A gate that scans nothing finds nothing. The floor must sit below the real count with
    headroom, and far enough above zero that a broken `git ls-files` cannot pass silently."""
    live = len([p for p in GATE.target_files(None) if GATE._read_text(REPO_ROOT / p) is not None])
    assert GATE.MIN_FULL_TREE_FILES < live, (
        f"non-vacuity floor {GATE.MIN_FULL_TREE_FILES} is at or above the live text-file "
        f"count {live} — the gate would refuse itself"
    )
    assert GATE.MIN_FULL_TREE_FILES >= 100
