# >300 justify (R8). The file grew at WP12-R with
# the KNOWN_DEBT register's three-way split, O-7a/b/c, which gave the gate's
# stale and reporting branches the producers they never had. It is ONE gate's producer
# suite over ONE loaded module object (`GATE`) plus ONE corpus fixture, and R5 bars
# cross-test imports — a split forks the loader and the `_fires` helper into two copies that
# drift apart while both stay green, which is this gate's own defect class.
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
        if GATE.line_hit(logical) is not None:
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


_SYNTHETIC_FALLBACK = 'spec = declared if declared is not None else lookup("v6")'


def _synthetic_tree(tmp_path: Path, *, body: str) -> Path:
    """A one-file `src/` tree for the register tests to scan instead of the real repo.

    The gate's own `test_skip_dirs_are_matched_on_the_relative_path_only` already relies on
    `REPO_ROOT` being redirectable. Redirecting it here is what makes these two oracles
    INDEPENDENT of the real tree: the whole repo contains exactly one line that is both
    pattern-hit and unjustified (arm 8's ternary), and WP12-R Phase C deletes it, so a
    register test written against real source could not be green both before and after.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "probe_module.py").write_text(
        f"def f(declared):\n    {body}\n    return spec\n"
    )
    return tmp_path


def test_known_debt_register_is_empty():
    """⊕ O-7a. Arm 8 was the last registered-open arm, and it is CLOSED.

    HEAD: RED — one entry (`inference_local.py`'s ternary), which WP12-R Phase C deletes.

    The two assertions carried over from the tamper-evidence test this replaces are VACUOUS
    against an empty register (`0 == 0`; a loop over nothing) and are NOT counted as
    coverage — they are future-proofing for the next owned arm. The gate's two debt
    branches are held by O-7b and O-7c below, which is where `assert debt_hits`'s real
    coverage went rather than being dropped.
    """
    _v, _debt_hits, _f, matched = GATE.scan()
    assert GATE.KNOWN_DEBT == (), (
        f"KNOWN_DEBT is not empty: {[entry[0] for entry in GATE.KNOWN_DEBT]}. A registered-"
        f"open arm is owned debt, not an exemption — re-adjudicate it, do not inherit it."
    )
    assert len(matched) == len(GATE.KNOWN_DEBT), "a KNOWN_DEBT entry matched nothing"
    for _path, _text, reason in GATE.KNOWN_DEBT:
        assert "owner" in reason.lower() or "WP" in reason, "debt needs a named owner"


def test_a_stale_debt_entry_fails_the_gate(tmp_path, monkeypatch, capsys):
    """⊕ O-7b. An entry that matches nothing must FAIL the gate.

    Producer for `silent_encoding_gate.py:338-344`, which had NO producer test at HEAD. The
    register asserts "this IS a real arm, here, now"; if the code moves out from under an
    entry the gate must fail so the exemption is re-adjudicated rather than silently
    inherited by whatever replaced that line.

    The synthetic tree is CLEAN (no fallback shape at all), so rc 1 is attributable to the
    stale branch alone. Over the real tree the same rc is produced by any unrelated
    violation, and the assertion would not be an oracle for this branch.
    """
    monkeypatch.setattr(
        GATE, "REPO_ROOT", _synthetic_tree(tmp_path, body="spec = lookup(declared)")
    )
    monkeypatch.setattr(GATE, "MIN_SCANNED_FILES", 1)
    monkeypatch.setattr(
        GATE,
        "KNOWN_DEBT",
        (("src/probe_module.py", 'else lookup("v6w25")', "OWNER: WP-ORACLE. ADJ-TEST."),),
    )

    assert GATE.main() == 1
    assert "matched nothing" in capsys.readouterr().out


def test_a_matching_debt_entry_is_reported_and_passes(tmp_path, monkeypatch, capsys):
    """⊕ O-7c. A matching entry is REPORTED on every run, and does not fail the gate.

    Producer for BOTH halves of what `assert debt_hits` used to hold: `scan()`'s debt
    matching (`:304-314`) and `main()`'s reporting branch (`:346-351`). Deleting the
    reporting branch loses `REGISTERED-OPEN`; deleting the matching turns the line into a
    hard violation and rc 1. Registered debt that stops being visible stops being debt and
    starts being the status quo.

    `files_scanned == 1` is the load-bearing number: the real `src/`/`crates/` trees are
    never walked, so this oracle's verdict cannot depend on the arm-8 line Phase C deletes.
    """
    monkeypatch.setattr(
        GATE, "REPO_ROOT", _synthetic_tree(tmp_path, body=_SYNTHETIC_FALLBACK)
    )
    monkeypatch.setattr(GATE, "MIN_SCANNED_FILES", 1)
    monkeypatch.setattr(
        GATE,
        "KNOWN_DEBT",
        (
            (
                "src/probe_module.py",
                'declared if declared is not None else lookup("v6")',
                "OWNER: WP-ORACLE. ADJ-TEST.",
            ),
        ),
    )

    violations, debt_hits, files_scanned, matched = GATE.scan()
    assert files_scanned == 1
    assert len(debt_hits) == 1
    assert matched == {0}
    assert violations == []

    assert GATE.main() == 0
    assert "REGISTERED-OPEN" in capsys.readouterr().out


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


def _load_corpus() -> list[dict]:
    import tomllib

    path = REPO_ROOT / "tests" / "fixtures" / "silent_encoding_evasions.toml"
    return tomllib.loads(path.read_text())["case"]


CORPUS = _load_corpus()


def test_the_evasion_corpus_is_the_one_review_impl_built():
    """Guards the fixture itself against quiet shrinkage.

    The corpus is only worth committing if it cannot be trimmed when a case becomes
    inconvenient — the failure mode that let the first draft claim coverage it never had.
    """
    import tomllib

    path = REPO_ROOT / "tests" / "fixtures" / "silent_encoding_evasions.toml"
    meta = tomllib.loads(path.read_text())["meta"]
    assert meta["candidates_tried"] == 31
    assert len(CORPUS) >= 31, f"corpus shrank to {len(CORPUS)} cases"
    assert {c["expect"] for c in CORPUS} <= {"fires", "quiet", "gap"}


@pytest.mark.parametrize(
    "case", [c for c in CORPUS if c["expect"] == "fires"], ids=lambda c: c["id"]
)
def test_corpus_shapes_the_gate_must_flag(case):
    assert _fires(case["code"], suffix=f".{case['lang']}"), (
        f"{case['id']} evaded gate 11: {case['code']!r}"
    )


@pytest.mark.parametrize(
    "case", [c for c in CORPUS if c["expect"] == "quiet"], ids=lambda c: c["id"]
)
def test_corpus_shapes_the_gate_must_ignore(case):
    assert not _fires(case["code"], suffix=f".{case['lang']}"), (
        f"{case['id']} false-positived: {case['code']!r}"
    )


@pytest.mark.parametrize(
    "case", [c for c in CORPUS if c["expect"] == "gap"], ids=lambda c: c["id"]
)
def test_corpus_accepted_gaps_are_still_gaps(case):
    """Pins the ADMITTED limits, so the gate's real power stays honestly stated.

    If one of these starts firing, that is good news — but the fixture is then lying about
    this gate's coverage, and the claim must be re-derived rather than left stale.
    """
    assert not _fires(case["code"], suffix=f".{case['lang']}"), (
        f"{case['id']} now fires — the accepted-residue claim in the fixture is stale"
    )


def test_v6_alternation_does_not_shadow_longer_names():
    """`v6` must not match inside `v6w25` and mislabel the finding."""
    assert GATE.ENCODINGS.index("v6") == len(GATE.ENCODINGS) - 1
    assert _fires('return "v6w25"')
