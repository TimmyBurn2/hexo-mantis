"""CI gate 16's producer test: the encoding-less text-I/O detector must BITE.

Both halves matter: `"rb"` correctly takes no `encoding`, and a gate that flags binary reads
gets switched off within a week. Every case is driven through `encoding_io_gate.is_unsafe`,
the same function the gate's own scan calls, so the oracle cannot drift from what it certifies.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
from _toolpath import load_module_by_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "encoding_io_gate.py"


def _load_gate():
    """Load the gate module by path, since `tools/` is not importable and `sys.path` may not move."""
    return load_module_by_path("_encoding_io_gate", GATE_PATH)


GATE = _load_gate()


def _first_call(source: str) -> ast.Call:
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call))
    return node


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


MUST_NOT_FIRE = [
    pytest.param('open("f.txt", encoding="utf-8")', id="encoding-as-keyword"),
    # Binary mode takes no `encoding` at all; flagging it is the false positive that gets a
    # gate disabled.
    pytest.param('open("f.bin", "rb")', id="builtin-open-binary-positional"),
    pytest.param('p.open("wb")', id="path-open-binary-positional"),
    # Positional encoding: the builtin and the Path method differ by one index.
    pytest.param('open("f.txt", "r", -1, "utf-8")', id="builtin-open-encoding-positional"),
    pytest.param("p.read_text(enc)", id="path-read_text-encoding-positional-variable"),
]


@pytest.mark.parametrize("source", MUST_NOT_FIRE)
def test_detector_is_silent_on_correct_code(source: str) -> None:
    assert GATE.is_unsafe(_first_call(source)) is False, (
        f"gate 16 FALSE POSITIVE on correct code: {source!r}. "
        "A gate that fires on correct code trains reviewers to ignore it."
    )


def test_path_open_binary_is_not_confused_with_builtin_open() -> None:
    """Prove `Path.open`'s mode index (0, not 1) is not shared with the builtin's."""
    assert GATE.is_unsafe(_first_call('p.open("rb")')) is False
    assert GATE.is_unsafe(_first_call('open("f", "rb")')) is False


def test_kwargs_forwarding_is_not_claimed_as_a_violation() -> None:
    """Prove `**kwargs` forwarding is not claimed: its absence is not statically provable."""
    assert GATE.is_unsafe(_first_call("p.read_text(**kwargs)")) is False


def test_unrelated_calls_are_ignored() -> None:
    assert GATE.is_unsafe(_first_call("json.loads(text)")) is False
    assert GATE.is_unsafe(_first_call("shutil.copyfile(a, b)")) is False


def test_known_limitation_any_dot_open_is_flagged_regardless_of_receiver() -> None:
    """Record the deliberate over-approximation: any `.open` is flagged whatever the receiver.

    Measured: in the gated scope every receiver is Path-like, so this costs zero false
    positives. `os.open` is the one exemption and is skipped at scan level by `_is_os_open` —
    an fd call whose second argument is flags — while `is_unsafe` keeps flagging it, so a
    future receiver named like it stays visible.
    """
    assert GATE.is_unsafe(_first_call("os.open(path, flags)")) is True
    assert GATE.is_unsafe(_first_call("zipfile.ZipFile(z).open(name)")) is True
    assert GATE._is_os_open(_first_call("os.open(path, flags)")) is True
    assert GATE._is_os_open(_first_call("p.open(path)")) is False


def test_gate_is_green_on_the_committed_tree() -> None:
    """Prove the gate is green on the committed tree: a gate is adopted only over a clean baseline."""
    violations, scanned, _matched = GATE.scan()
    assert not violations, "gate 16 baseline is dirty:\n" + "\n".join(violations)
    assert scanned["tree"] >= GATE.MIN_FILES["tree"]


def test_every_registered_exemption_still_matches() -> None:
    """Prove every registered exemption still matches, so none can be inherited by a rewritten line."""
    _violations, _scanned, matched = GATE.scan()
    unmatched = [GATE.EXEMPT[k][0] for k in range(len(GATE.EXEMPT)) if k not in matched]
    assert not unmatched, (
        f"registered exemption(s) matched nothing and must be re-adjudicated: {unmatched}"
    )


def test_every_exemption_carries_grounds() -> None:
    """Prove every exemption carries real grounds, not just a path."""
    for path, snippet, reason in GATE.EXEMPT:
        assert (REPO_ROOT / path).is_file(), f"exempted path does not exist: {path}"
        assert snippet.strip(), f"exemption for {path} has an empty match snippet"
        assert len(reason) > 40, f"exemption for {path} needs real grounds, got: {reason!r}"


def test_escape_hatch_requires_a_reason() -> None:
    """Prove the escape marker ends with `--`, so a bare marker cannot match."""
    assert GATE.ESCAPE_TOKEN.endswith("--"), (
        "the escape marker must end with `--` so a bare marker cannot match; "
        f"got {GATE.ESCAPE_TOKEN!r}"
    )


@pytest.mark.parametrize(
    "comment",
    ["# encoding-gate: ok --", "# encoding-gate: ok --   ", "# encoding-gate: ok"],
    ids=["bare-dashes", "dashes-and-space", "no-dashes"],
)
def test_a_hatch_with_NO_reason_text_suppresses_nothing(comment: str) -> None:
    """Prove a hatch with no reason text suppresses nothing, driven through the deciding function."""
    assert not GATE._justified([comment, "open('x')"], 2), comment


def test_a_hatch_WITH_a_reason_still_suppresses() -> None:
    """Control: a hatch with a reason still suppresses, so the repair is not a wall."""
    assert GATE._justified(["open('x')  # encoding-gate: ok -- a zipfile member, not text"], 1)
    assert GATE._justified(
        ["# encoding-gate: ok -- the receiver is a mock, not a path", "open('x')"], 2)
