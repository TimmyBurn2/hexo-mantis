"""CI gate 16's producer test (LAW-07): the encoding-less text-I/O detector must BITE.

A gate is only worth its CI minute if it fires on the defect and stays silent on correct code.
The second half matters as much as the first here: `"rb"` correctly takes no `encoding`, and a
gate that flags binary reads gets switched off within a week.

Every case below is driven through `encoding_io_gate.is_unsafe` - the SAME function the gate's
scan calls. An oracle that re-implemented the decision could drift from the thing it certifies,
which is the defect class this repo keeps finding (gate 11's docstring says so in its own words).
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "encoding_io_gate.py"


def _load_gate():
    """Load the gate by PATH.

    R5/LAW-17 ban `sys.path` mutation, and `tools/` is not an importable package, so the gate is
    spec-loaded from its file exactly as the other gate oracles do it.
    """
    spec = importlib.util.spec_from_file_location("_encoding_io_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _first_call(source: str) -> ast.Call:
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call))
    return node


# --- the five that MUST fire -------------------------------------------------------------

MUST_FIRE = [
    pytest.param('open("f.txt")', id="builtin-open-default-mode"),
    pytest.param('open("f.txt", "w")', id="builtin-open-explicit-text-mode"),
    pytest.param("p.read_text()", id="path-read_text"),
    pytest.param("p.write_text(data)", id="path-write_text"),
    pytest.param('p.open("r")', id="path-open-text-mode"),
]


@pytest.mark.parametrize("source", MUST_FIRE)
def test_detector_fires_on_encoding_less_text_io(source: str) -> None:
    assert GATE.is_unsafe(_first_call(source)) is True, (
        f"gate 16 MISSED an encoding-less text call: {source!r}. "
        "This is the S-19 defect that took down pytest collection on Windows."
    )


# --- the five that MUST NOT fire ---------------------------------------------------------

MUST_NOT_FIRE = [
    pytest.param('open("f.txt", encoding="utf-8")', id="encoding-as-keyword"),
    # Binary mode takes no `encoding` AT ALL. Flagging it would be wrong, and the WP that
    # commissioned this gate named it as the false positive that gets a gate disabled.
    pytest.param('open("f.bin", "rb")', id="builtin-open-binary-positional"),
    pytest.param('p.open("wb")', id="path-open-binary-positional"),
    # Positional encoding. The builtin and the Path method differ by one index; an earlier
    # draft of this gate shared one table and got BOTH of these wrong.
    pytest.param('open("f.txt", "r", -1, "utf-8")', id="builtin-open-encoding-positional"),
    pytest.param("p.read_text(enc)", id="path-read_text-encoding-positional-variable"),
]


@pytest.mark.parametrize("source", MUST_NOT_FIRE)
def test_detector_is_silent_on_correct_code(source: str) -> None:
    assert GATE.is_unsafe(_first_call(source)) is False, (
        f"gate 16 FALSE POSITIVE on correct code: {source!r}. "
        "A gate that fires on correct code trains reviewers to ignore it."
    )


# --- shapes that are easy to get wrong ---------------------------------------------------


def test_path_open_binary_is_not_confused_with_builtin_open() -> None:
    """`Path.open` has no `file` parameter, so mode sits at index 0, not 1.

    Sharing one positional table between the builtin and the method makes `p.open("rb")` look
    like text mode. This is the regression guard for that exact bug.
    """
    assert GATE.is_unsafe(_first_call('p.open("rb")')) is False
    assert GATE.is_unsafe(_first_call('open("f", "rb")')) is False


def test_kwargs_forwarding_is_not_claimed_as_a_violation() -> None:
    """`**kwargs` may carry `encoding`; absence is not statically provable, so do not claim it."""
    assert GATE.is_unsafe(_first_call("p.read_text(**kwargs)")) is False


def test_unrelated_calls_are_ignored() -> None:
    assert GATE.is_unsafe(_first_call("json.loads(text)")) is False
    assert GATE.is_unsafe(_first_call("shutil.copyfile(a, b)")) is False


def test_known_limitation_any_dot_open_is_flagged_regardless_of_receiver() -> None:
    """The gate cannot tell `Path.open` from `os.open` / `ZipFile.open` / a mock's `.open`.

    This is a DELIBERATE over-approximation, recorded as a test so it is a known property rather
    than a surprise. The reasoning, measured at P0-05:

    * Across `tools/`, `tests/` and `src/` there are 13 `.open()` attribute calls. In the GATED
      scope (`tools/`, and `tests/` at module scope) every receiver is Path-like, so the
      over-approximation costs zero false positives today.
    * The one genuine counter-example is `os.open(path, flags)` in `src/` - an fd-level call that
      takes no `encoding`. `src/` is deliberately out of scope, so it is not reached; if `src/`
      is ever added, that site needs the escape hatch.
    * Narrowing this (proving the receiver is a `Path`) is not reliably decidable statically, and
      would buy precision with false NEGATIVES - the wrong trade for a gate whose entire purpose
      is catching a class that fails silently.
    """
    assert GATE.is_unsafe(_first_call("os.open(path, flags)")) is True
    assert GATE.is_unsafe(_first_call("zipfile.ZipFile(z).open(name)")) is True


# --- the gate as a whole -----------------------------------------------------------------


def test_gate_is_green_on_the_committed_tree() -> None:
    """The baseline R98 requires: a gate may only be adopted over a clean baseline."""
    violations, scanned, _matched = GATE.scan()
    assert not violations, "gate 16 baseline is dirty:\n" + "\n".join(violations)
    assert scanned["tools"] >= GATE.MIN_FILES["tools"]
    assert scanned["tests"] >= GATE.MIN_FILES["tests"]


def test_every_registered_exemption_still_matches() -> None:
    """A stale exemption FAILS the gate, so it can never be inherited by a rewritten line."""
    _violations, _scanned, matched = GATE.scan()
    unmatched = [GATE.EXEMPT[k][0] for k in range(len(GATE.EXEMPT)) if k not in matched]
    assert not unmatched, (
        f"registered exemption(s) matched nothing and must be re-adjudicated: {unmatched}"
    )


def test_every_exemption_carries_grounds() -> None:
    """An exemption without a stated reason is an escape hatch wearing a register's clothes."""
    for path, snippet, reason in GATE.EXEMPT:
        assert (REPO_ROOT / path).is_file(), f"exempted path does not exist: {path}"
        assert snippet.strip(), f"exemption for {path} has an empty match snippet"
        assert len(reason) > 40, f"exemption for {path} needs real grounds, got: {reason!r}"


def test_escape_hatch_requires_a_reason() -> None:
    """`# encoding-gate: ok` with nothing after it must not silence anything."""
    assert GATE.ESCAPE.endswith("--"), (
        "the escape marker must end with `--` so a bare marker cannot match; "
        f"got {GATE.ESCAPE!r}"
    )
