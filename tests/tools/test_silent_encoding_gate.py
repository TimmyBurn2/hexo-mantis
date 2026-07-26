"""Producer test + mutation self-test for CI gate 11 (LAW-07, R4, R45).

R4: no gate input without a producer test. A grep gate that cannot be shown to BITE is
decoration — and gate 11 exists precisely because a guard that reported green while the
thing it guarded was absent is how the gate-1 rot survived the whole migration.

The mutation probes run the gate's own matching logic over synthetic lines rather than
mutating the real tree, so the suite is order-independent and leaves nothing behind.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "silent_encoding_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("silent_encoding_gate", GATE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


GATE = _load_gate()


def _fires(line: str, prev: str = "", suffix: str = ".py") -> bool:
    """True if the gate would flag `line` (with `prev` as the line above it).

    Routes through `_logical_lines` rather than matching the raw text, so this mirrors the
    real scan path — comment stripping and continuation joining included. An earlier
    version matched raw lines and so could not see the `return "v6"  # comment` evasion
    that the gate itself handles correctly.
    """
    lines = [prev, line] if prev else [line]
    for idx, logical in GATE._logical_lines(lines, suffix):
        for rx, _why in GATE._COMPILED:
            if rx.search(logical):
                return not GATE._is_justified(lines, idx, suffix)
    return False


def test_gate_is_green_on_the_current_tree():
    """The whole point of landing it in this commit: the arms are actually closed."""
    assert GATE.find_violations() == []


def test_gate_actually_scanned_something():
    """A gate that scans nothing finds nothing.

    `find_violations() == []` is satisfied just as well by a broken file walk, which is
    precisely the rot this gate family exists to prevent — so the floor is asserted, not
    assumed.
    """
    _violations, _debt, files_scanned, _matched = GATE.scan()
    assert files_scanned >= GATE.MIN_SCANNED_FILES


def test_known_debt_register_is_tamper_evident():
    """Every KNOWN_DEBT entry must still match real source.

    The register is not an escape hatch: it asserts "this IS an arm, tracked and owned".
    If the code moves out from under an entry, the gate must FAIL so the exemption is
    re-adjudicated rather than silently inherited by whatever replaced it.
    """
    _v, debt_hits, _f, matched = GATE.scan()
    assert len(matched) == len(GATE.KNOWN_DEBT), "a KNOWN_DEBT entry matched nothing"
    assert debt_hits, "registered-open arms must be reported on every run, not silently"
    for _path, _text, reason in GATE.KNOWN_DEBT:
        assert "owner" in reason.lower() or "WP" in reason, "debt needs a named owner"


def test_main_returns_nonzero_when_an_arm_is_present(monkeypatch, capsys):
    """The EXIT CODE is what CI reads — assert it, not just the violations list."""
    monkeypatch.setattr(GATE, "find_violations", lambda: ["synthetic:1: arm"])
    monkeypatch.setattr(GATE, "scan", lambda: (["synthetic:1: arm"], [], 999, {0}))
    assert GATE.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_zero_on_the_real_tree(capsys):
    assert GATE.main() == 0
    assert "no unregistered silent encoding-fallback arms" in capsys.readouterr().out


@pytest.mark.parametrize(
    "line",
    [
        'return str(enc.get("version", "v6"))',          # arm 5, validate.py
        '    return "v6"',                                # arms 5 and 6, terminal default
        'encoding = cfg.get("encoding", "v6w25")',
        'name = getattr(spec, "name", "gnn_axis_v1")',
        'enc = declared or "v6"',
        'let name = declared.unwrap_or("v6");',
        'let name = declared.unwrap_or_else(|| "v6_live2_ls");',
        # ── shapes REVIEW-impl used to defeat the first draft (27 of 31 evaded) ──
        '    return "v6"  # a trailing comment used to walk straight through',
        'spec = declared if declared is not None else lookup("v6")',   # arm 8's shape
        'def make_collate(augment: bool, encoding: str = "v6"):',      # arm 7's shape
        'parser.add_argument("--encoding", default="v6")',             # how arm 6 regresses
        'DEFAULT_ENCODING = "v6"',                                     # variable indirection
        'enc = cfg.setdefault("encoding", "v6")',
        'enc = cfg.pop("encoding", "v6")',
        'name = spec.name if spec else "v6w25"',
        'let name = declared.map_or("v6", |d| d);',
        '    #[pyo3(signature = (capacity, encoding = "v6"))]',        # arms 9/10's shape
    ],
)
def test_gate_bites_on_every_known_fallback_shape(line):
    assert _fires(line), f"gate 11 failed to flag a silent fallback: {line!r}"


def test_gate_sees_rust_attributes_as_code_not_comments():
    """`#[pyo3(...)]` is an attribute, not a comment.

    An earlier draft blanked every line starting with `#`, which hid arms 9 and 10 (the
    pyo3 signature defaults on ReplayBuffer/HexgBuffer) and reported green over both.
    """
    lines = ['    #[pyo3(signature = (capacity, encoding = "v6"))]']
    assert GATE._logical_lines(lines, ".rs")[0][1] != ""
    assert _fires(lines[0], suffix=".rs")
    # ...and a real Rust comment IS still stripped (yielding no logical line at all)
    assert GATE._logical_lines(['    // encoding = "v6" in prose'], ".rs") == []


def test_multiline_call_is_joined_before_matching():
    """A `.get(` split across lines was never matched by the first draft."""
    lines = ['enc = cfg.get(', '    "encoding",', '    "v6",', ')']
    joined = GATE._logical_lines(lines, ".py")
    assert len(joined) == 1
    assert any(rx.search(joined[0][1]) for rx, _ in GATE._COMPILED)


def test_skip_dirs_are_matched_on_the_relative_path_only():
    """A checkout living under a directory named `target`/`tests` must still be scanned.

    Matching skip parts against the ABSOLUTE path made the gate vacuously green depending
    on where the repo happened to sit — and this project uses git worktrees.
    """
    assert GATE.REPO_ROOT.is_absolute()
    scanned = list(GATE._iter_files())
    assert len(scanned) >= GATE.MIN_SCANNED_FILES


@pytest.mark.parametrize(
    "line",
    [
        'spec = lookup("v6")',                            # explicit dispatch, not a default
        'return lookup("gnn_axis_v1")',                   # affirmative marker dispatch
        'ReplayBuffer::new(cap, "v6")',                   # a named argument, not a fallback
        'assert spec.name == "v6w25"',                    # an assertion
        'ENCODINGS = ("v6", "v6w25")',                    # a census tuple
    ],
)
def test_gate_does_not_fire_on_legitimate_encoding_literals(line):
    """A gate that fires on correct code trains reviewers to ignore it."""
    assert not _fires(line), f"gate 11 false-positived on: {line!r}"


def test_escape_hatch_silences_a_justified_site():
    just = "# silent-encoding-gate: ok -- diagnostic-only guess, never resolves for real work"
    assert not _fires('    return "v6"', prev=just)


def test_escape_hatch_requires_an_actual_reason():
    """`ok --` with nothing after it must not silence anything."""
    assert _fires('    return "v6"', prev="# silent-encoding-gate: ok --")
    assert _fires('    return "v6"', prev="# silent-encoding-gate: ok")


def test_escape_hatch_reads_the_whole_comment_block_not_just_one_line():
    """A justification worth writing usually needs a sentence.

    The first draft only looked one line back, so a two-line justification did not
    silence the site it justified — caught when the real audit_sections.py escape failed.
    """
    lines = [
        "# silent-encoding-gate: ok -- diagnostic-only guess, compared against the",
        "# declared sidecar value to raise a warning; never resolves for real work.",
        '    return "v6"',
    ]
    assert GATE._is_justified(lines, 2)


def test_v6_alternation_does_not_shadow_longer_names():
    """`v6` must not match inside `v6w25` and mislabel the finding."""
    assert GATE.ENCODINGS.index("v6") == len(GATE.ENCODINGS) - 1
    assert _fires('return "v6w25"')
