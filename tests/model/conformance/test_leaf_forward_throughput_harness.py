# >300 justify (R8): the instrument and the self-tests that show it measures what its docstring
# names are one unit — a timer whose exclusion of graph construction is asserted in another file
# can be widened silently, and the self-tests are the only thing making this tier non-vacuous.
"""The leaf-forward throughput INSTRUMENT. It measures; it never judges.

Wall-clock microseconds per leaf of the MODEL FORWARD ONLY, at a ladder of candidate counts,
batched as at MCTS leaves, with `torch.cuda.synchronize()` around the timed region on CUDA —
without which the number is queue-submission time, not compute. MEDIAN with IQR after discarded
warm-ups. THERE IS NO FLOOR, NO THRESHOLD AND NO PASS/FAIL, and no number lands in a tracked path.

The ladder is DERIVED from `Board.legal_move_count()`; points above the reachable ceiling are a
SYNTHETIC block, LABELLED, never mixed with the reachable one.

TIER PLACEMENT NEARLY DISARMED THIS TIER, so only the MEASUREMENT carries `slow`: a `slow`-marked
control is green-by-never-executing under CI's default `-m "not integration and not slow"`. Both
self-tests would be unfalsifiable stated structurally, and any magnitude assertion here is RED, so
one asserts a RELATION BETWEEN TWO MEASUREMENTS IN THE SAME PROCESS and the other COUNTS CALLS.
"""
from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board, HexgBuffer

from _corpus import ConformanceRefusal, build_board, roster

REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_REACHABLE = "reachable"
LADDER_SYNTHETIC = "synthetic"
#: Instrument parameter for the within-process differential, not a threshold on any subject.
_DIFFERENTIAL_SLEEP_S = 0.005


class UnlabelledLadderPoint(ConformanceRefusal):
    """A ladder point carries no block label, so a synthetic figure could pass for reachable."""


class HardwareGateClosed(ConformanceRefusal):
    """The harness was asked to run on a device it cannot reach. It FAILS; it never skips."""


class MagnitudeWouldLand(ConformanceRefusal):
    """A measured magnitude was about to be written into a tracked path."""


class LadderNotDerived(ConformanceRefusal):
    """The reachable ladder did not follow the engine's own legal-move count."""


@dataclass(frozen=True)
class LadderPoint:
    """A candidate count and WHICH BLOCK it belongs to. `label` has no default, deliberately."""

    candidates: int
    label: str


@dataclass(frozen=True)
class Measurement:
    """Median and IQR in nanoseconds, plus the sync-call count. No verdict, no comparison."""

    median_ns: float
    iqr_ns: float
    sync_calls: int
    repeats: int


def require_labelled_ladder(points: list[LadderPoint]) -> int:
    for point in points:
        if point.label not in (LADDER_REACHABLE, LADDER_SYNTHETIC):
            raise UnlabelledLadderPoint(
                f"ladder point with {point.candidates} candidates carries label "
                f"{point.label!r}; the block label is a required field, not a convention — an "
                "unlabelled synthetic point reads as a system measurement."
            )
    return len(points)


def reachable_ladder(enc: str, spans: tuple[int, ...], counter=None) -> list[LadderPoint]:
    """Candidate counts READ from the engine, one per constructed position. `counter` is the seam
    the derivation control stubs: pass a callable taking a Board and the ladder must follow it."""
    read = counter if counter is not None else (lambda board: board.legal_move_count())
    points: list[LadderPoint] = []
    for span in spans:
        board = build_board(enc, [(i, 0) for i in range(span + 1)])
        points.append(LadderPoint(int(read(board)), LADDER_REACHABLE))
    return points


def synthetic_ladder(ceiling: int, multipliers: tuple[int, ...]) -> list[LadderPoint]:
    """Points ABOVE the reachable ceiling, labelled, reported in their own block."""
    return [LadderPoint(ceiling * m, LADDER_SYNTHETIC) for m in multipliers]


def require_device_available(requested: str, available: bool) -> str:
    if not available:
        raise HardwareGateClosed(
            f"the harness was asked to measure on {requested!r} and the device is not "
            "reachable. This FAILS loudly with a named reason; it does not skip, because a "
            "silent skip is the vacuous pass this suite exists to prevent."
        )
    return requested


def require_no_magnitude_lands(path: Path) -> Path:
    if REPO_ROOT in path.resolve().parents or path.resolve() == REPO_ROOT:
        raise MagnitudeWouldLand(
            f"refusing to write a measured magnitude into {path} — it is inside the tracked "
            "tree. A µs/leaf figure is host-attested or it is mechanism evidence; the "
            "verdict-bearing number is operator-forwarded, never committed."
        )
    return path


def _timed_once(
    build_input: Callable[[], Any],
    forward: Callable[[Any], Any],
    device_type: str,
    sync: Callable[[], None] | None,
) -> tuple[int, int]:
    """One `(elapsed_ns, sync_calls)` sample. THE one timing primitive, so "the timed region
    excludes input construction" is a property of a single function, not a coincidence."""
    payload = build_input()
    syncs = 0
    if device_type == "cuda" and sync is not None:
        sync()
        syncs += 1
    start = time.perf_counter_ns()
    forward(payload)
    if device_type == "cuda" and sync is not None:
        sync()
        syncs += 1
    return time.perf_counter_ns() - start, syncs


def _summarise(samples: list[int], syncs: int) -> Measurement:
    """Median and IQR over one arm's samples. No verdict, no comparison."""
    ordered = sorted(samples)
    mid = len(ordered) // 2
    median = float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)
    lower = ordered[: len(ordered) // 2]
    upper = ordered[(len(ordered) + 1) // 2 :]
    iqr = float((upper[len(upper) // 2] if upper else 0) - (lower[len(lower) // 2] if lower else 0))
    return Measurement(median_ns=median, iqr_ns=iqr, sync_calls=syncs, repeats=len(ordered))


def measure_forward(
    build_input: Callable[[], Any],
    forward: Callable[[Any], Any],
    *,
    repeats: int,
    warmup: int,
    device_type: str,
    sync: Callable[[], None] | None = None,
) -> Measurement:
    """Time `forward` only, `build_input` OUTSIDE the timed region every repeat. The sync call
    count is returned so the CUDA branch is observable without a GPU."""
    samples: list[int] = []
    syncs = 0
    for index in range(warmup + repeats):
        elapsed, sync_calls = _timed_once(build_input, forward, device_type, sync)
        syncs += sync_calls
        if index >= warmup:
            samples.append(elapsed)
    return _summarise(samples, syncs)


def measure_forward_paired(
    first_build: Callable[[], Any],
    first_forward: Callable[[Any], Any],
    second_build: Callable[[], Any],
    second_forward: Callable[[Any], Any],
    *,
    repeats: int,
    warmup: int,
    device_type: str,
    sync: Callable[[], None] | None = None,
) -> tuple[Measurement, Measurement]:
    """The same timer, over TWO arms ALTERNATELY, so both meet the same machine state.

    Two `measure_forward` calls hand each arm its own window of host noise: measured at load ~19 on
    16 cores, the same pair read 163.588 ms and 0.309 ms for the SAME arm, and reversing the call
    order moved rows in and out of inversion. Each arm keeps its own samples, never a sum.
    """
    first: list[int] = []
    second: list[int] = []
    first_syncs = 0
    second_syncs = 0
    for index in range(warmup + repeats):
        first_elapsed, first_calls = _timed_once(first_build, first_forward, device_type, sync)
        second_elapsed, second_calls = _timed_once(second_build, second_forward, device_type, sync)
        first_syncs += first_calls
        second_syncs += second_calls
        if index >= warmup:
            first.append(first_elapsed)
            second.append(second_elapsed)
    return _summarise(first, first_syncs), _summarise(second, second_syncs)


def test_the_timer_EXCLUDES_input_construction_and_INCLUDES_the_forward(derived):
    """The within-process differential, two-sided. Neither half commits a number: both are
    relations between measurements taken in this same process."""
    sleep_ns = int(_DIFFERENTIAL_SLEEP_S * 1e9)

    outside = measure_forward(
        lambda: time.sleep(_DIFFERENTIAL_SLEEP_S),
        lambda payload: None,
        repeats=3, warmup=1, device_type="cpu",
    )
    inside = measure_forward(
        lambda: None,
        lambda payload: time.sleep(_DIFFERENTIAL_SLEEP_S),
        repeats=3, warmup=1, device_type="cpu",
    )
    derived("t6.differential.outside_median_ns", outside.median_ns)
    derived("t6.differential.inside_median_ns", inside.median_ns)
    assert inside.median_ns - outside.median_ns >= sleep_ns, (
        "moving the sleep INTO the timed region did not grow the reported median by the sleep. "
        "Either the timer includes input construction — in which case both arms carry it and "
        "the difference collapses — or it excludes the forward. This is the RELATION the "
        "docstring promises: both terms are measured in this same process, so no host-dependent "
        "bound is committed. Comparing `outside` against the sleep constant alone, which is what "
        "stood here, is a wall-clock ceiling on an empty call and flakes on a loaded host."
    )
    assert inside.median_ns >= sleep_ns, (
        "a sleep moved INSIDE the timed region did NOT move the reported median — the timer is "
        "not measuring the forward at all"
    )


def test_the_PAIRED_timer_reads_its_two_arms_SEPARATELY_and_never_as_a_sum(derived):
    """A paired timer returning one reading twice, or the SUM of the two arms in both slots, would
    satisfy every caller's type and silently make `second / first` a constant. The sleep is in the
    second arm alone, so the relation commits no number."""
    sleep_ns = int(_DIFFERENTIAL_SLEEP_S * 1e9)

    first, second = measure_forward_paired(
        lambda: None, lambda payload: None,
        lambda: None, lambda payload: time.sleep(_DIFFERENTIAL_SLEEP_S),
        repeats=3, warmup=1, device_type="cpu",
    )
    derived("t6.paired.first_median_ns", first.median_ns)
    derived("t6.paired.second_median_ns", second.median_ns)
    assert second.median_ns - first.median_ns >= sleep_ns, (
        "the sleep in the SECOND arm alone did not separate the two readings — the paired "
        "timer is returning one measurement for both arms, or the sum of them"
    )
    assert first.median_ns < sleep_ns, (
        "the first arm's reading carries the second arm's sleep, so the paired timer is "
        "timing the pair rather than each arm"
    )


def test_the_PAIRED_timer_ALTERNATES_rather_than_running_one_arm_to_COMPLETION(derived):
    """The interleave itself, COUNTED: it is not observable in a duration, and a timer that ran
    one arm to completion then the other would return two readings of two different moments."""
    order: list[str] = []
    repeats, warmup = 3, 1
    measure_forward_paired(
        lambda: None, lambda payload: order.append("first"),
        lambda: None, lambda payload: order.append("second"),
        repeats=repeats, warmup=warmup, device_type="cpu",
    )
    derived("t6.paired.call_order", order)
    assert order == ["first", "second"] * (repeats + warmup), (
        f"the paired timer did not alternate its arms: {order}. One arm run to completion "
        "before the other is the sequential measurement this timer exists to replace"
    )


def test_the_CUDA_sync_branch_fires_on_cuda_and_NOT_on_cpu(derived):
    """The counting stub with its required negative control, constructible with no GPU — the CUDA
    branch otherwise never executes where CI runs."""
    calls = {"n": 0}

    def counting_sync() -> None:
        calls["n"] += 1

    on_cuda = measure_forward(
        lambda: None, lambda p: None, repeats=2, warmup=1, device_type="cuda",
        sync=counting_sync,
    )
    derived("t6.sync_calls.cuda", on_cuda.sync_calls)
    assert on_cuda.sync_calls > 0

    calls["n"] = 0
    on_cpu = measure_forward(
        lambda: None, lambda p: None, repeats=2, warmup=1, device_type="cpu",
        sync=counting_sync,
    )
    derived("t6.sync_calls.cpu", on_cpu.sync_calls)
    assert on_cpu.sync_calls == 0, (
        "the sync counter is non-zero on a CPU device, so this self-test fires on both branches "
        "and is measuring nothing"
    )


def test_the_reachable_ladder_FOLLOWS_a_stubbed_legal_move_count(derived):
    """A ladder that does not move when the engine's count moves is a transcribed literal wearing
    a derivation."""
    spec = roster()[0]
    spans = (1, 3, 5)
    real = reachable_ladder(spec.name, spans)
    stubbed = reachable_ladder(spec.name, spans, counter=lambda board: board.legal_move_count() + 1)
    derived("t6.ladder.reachable", [p.candidates for p in real])
    assert all(p.candidates > 0 for p in real)
    if [p.candidates for p in stubbed] != [p.candidates + 1 for p in real]:
        raise LadderNotDerived(
            f"the reachable ladder did not follow the stubbed count: {stubbed} vs {real}"
        )


def test_an_UNLABELLED_synthetic_point_is_refused():
    """The label is a required field, so the refusal is what makes it one."""
    good = synthetic_ladder(100, (2, 4))
    assert require_labelled_ladder(good) == 2
    with pytest.raises(UnlabelledLadderPoint, match="required field"):
        require_labelled_ladder([LadderPoint(400, "")])


def test_a_CLOSED_hardware_gate_FAILS_rather_than_skips():
    """The only admissible conditional in this suite is a hardware gate, and it must fail loudly
    with a named reason when it is asked to run."""
    assert require_device_available("cpu", True) == "cpu"
    with pytest.raises(HardwareGateClosed, match="does not skip"):
        require_device_available("cuda", False)


#: THE MARKER FAMILY THAT REMOVES A TEST FROM THE RUN, enumerated and NON-EXHAUSTIVE, matched only
#: as an attribute of `pytest.mark` / `mark` so a field named `skip` is not a hit. `xfail` is in
#: the family because a swallowed failure has stopped asserting.
_DISARMING_MARKS = frozenset({"skip", "skipif", "xfail"})
#: The call spellings. `importorskip` is the one that arrives disguised as an import.
_DISARMING_CALLS = frozenset({"skip", "importorskip", "xfail"})
#: The unittest spelling, which pytest honours and which no `pytest.` prefix announces.
_DISARMING_EXCEPTIONS = frozenset({"SkipTest"})


def _is_mark_root(node: ast.expr) -> bool:
    """`pytest.mark` or a bare `mark` bound by `from pytest import mark`."""
    if isinstance(node, ast.Attribute):
        return node.attr == "mark" and isinstance(node.value, ast.Name) and node.value.id == "pytest"
    return isinstance(node, ast.Name) and node.id == "mark"


def _pytest_callee(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.attr if func.value.id == "pytest" else ""
    return ""


def _empty_parametrize(node: ast.Call) -> bool:
    """`parametrize(..., [])`. Under pytest's default `empty_parameter_set_mark` an empty
    argvalues collects one SKIPPED item — a disarm containing no skip word at all."""
    func = node.func
    named = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if named != "parametrize" or len(node.args) < 2:
        return False
    argvalues = node.args[1]
    return isinstance(argvalues, ast.List | ast.Tuple | ast.Set) and not argvalues.elts


def disarming_forms(tree: ast.AST) -> list[tuple[int, str]]:
    """`(line, form)` for every construct in one module that removes a test from the run.

    MECHANISM, NOT SCOPE, WAS THE DEFECT: matching only `pytest.skip` calls let one
    `@pytest.mark.skipif(True, ...)` produce a green run, rc 0, past a collected-test floor gate —
    a skipped test is still COLLECTED. Matched on the AST, so this docstring is not itself a hit.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _DISARMING_MARKS:
            if _is_mark_root(node.value):
                found.append((node.lineno, f"mark.{node.attr}"))
        elif isinstance(node, ast.Call):
            callee = _pytest_callee(node.func)
            if callee in _DISARMING_CALLS:
                found.append((node.lineno, f"pytest.{callee}()"))
            elif _empty_parametrize(node):
                found.append((node.lineno, "parametrize with EMPTY argvalues"))
        elif isinstance(node, ast.Name) and node.id in _DISARMING_EXCEPTIONS:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in _DISARMING_EXCEPTIONS:
            found.append((node.lineno, node.attr))
    return sorted(set(found))


def suite_modules() -> list[Path]:
    return sorted(Path(__file__).parent.glob("*.py"))


def test_NO_MODULE_of_the_conformance_suite_DISARMS_a_TEST(derived):
    """No module of this suite disarms a test, in any form that removes one from the run.

    SCOPE was the first finding and MECHANISM the second: a `skipif` decorator on a cross-arm gate
    produced `104 passed, 1 skipped`, rc 0, with every gate in the repo green. The census
    cardinality is a derived output, so an empty glob cannot pass for a clean census.
    """
    modules = suite_modules()
    offenders = [
        f"{path.name}:{line} ({form})"
        for path in modules
        for line, form in disarming_forms(ast.parse(path.read_text(encoding="utf-8")))
    ]
    derived("t6.no_disarm.modules_walked", [p.name for p in modules])
    derived("t6.no_disarm.forms_matched", sorted(_DISARMING_MARKS | _DISARMING_CALLS))
    assert len(modules) > 1, (
        "the no-disarm census walked fewer than two modules of this suite — an empty or "
        "single-file walk reports a clean census over nothing"
    )
    assert not offenders, f"a test in this suite is disarmed at {offenders}"


#: One planted line per disarming spelling. A census that names a family and is only ever shown to
#: catch one member is a family claim resting on one measurement.
_DISARM_PLANTS: tuple[tuple[str, str], ...] = (
    ("decorator-skipif", '@pytest.mark.skipif(True, reason="flaky")\ndef test_x(): pass\n'),
    ("decorator-skip", "@pytest.mark.skip\ndef test_x(): pass\n"),
    ("decorator-xfail", '@pytest.mark.xfail(reason="known")\ndef test_x(): pass\n'),
    ("inline-call", 'def test_x():\n    pytest.skip("later")\n'),
    ("import-or-skip", 'torch = pytest.importorskip("torch")\n'),
    ("param-marks", '@pytest.mark.parametrize("a", [pytest.param(1, marks=pytest.mark.skip)])\n'
                    "def test_x(a): pass\n"),
    ("unittest-exception", "def test_x():\n    raise SkipTest\n"),
    ("module-level", "pytestmark = pytest.mark.skipif(True, reason='off')\n"),
    ("empty-parametrize", '@pytest.mark.parametrize("a", [])\ndef test_x(a): pass\n'),
)


@pytest.mark.parametrize("label,body", _DISARM_PLANTS, ids=[p[0] for p in _DISARM_PLANTS])
def test_EVERY_disarming_spelling_is_seen_by_the_census(label, body, tmp_path):
    """Each row is a line that produces a green run with one fewer assertion."""
    module = tmp_path / f"test_{label.replace('-', '_')}.py"
    module.write_text(body, encoding="utf-8")
    found = disarming_forms(ast.parse(module.read_text(encoding="utf-8")))
    assert found, f"{label}: the census did not see {body!r}"


def test_the_census_does_NOT_fire_on_an_ORDINARY_test_module(tmp_path):
    """Negative control, as binding as the positive rows: a census that fired on ordinary code
    would be widened until its green stopped meaning its name."""
    module = tmp_path / "test_ordinary.py"
    module.write_text(
        "import pytest\n\n"
        "@pytest.fixture\ndef thing(): return 1\n\n"
        '@pytest.mark.parametrize("a", [1, 2])\ndef test_x(a, thing): assert a\n\n'
        "@pytest.mark.slow\ndef test_y(): assert True\n\n"
        "class Cfg:\n    skip = 3\n\n"
        "def test_z(): assert Cfg.skip == 3\n",
        encoding="utf-8",
    )
    assert disarming_forms(ast.parse(module.read_text(encoding="utf-8"))) == []


#: TIER PLACEMENT, DECLARED: `(module, test)` for every test here that may carry `slow`. Not a
#: threshold and not a number, so the only way to change it is to say in a diff which gate stopped
#: running in CI. Deriving it from the marker source would be a one-source comparison.
_SLOW_TIER_MEMBERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("test_leaf_forward_throughput_harness.py", "test_leaf_forward_throughput_ladder"),
        # The dense-arm measurements went with the K-cluster window.
        (
            "test_window_frame_midpoint_translation_boundary.py",
            "test_report_the_graph_node_feature_translation_residual",
        ),
        (
            "test_arch_states_its_perf_floor.py",
            "test_report_the_per_arch_floor_and_serving_overhead",
        ),
        (
            "test_arch_states_its_memory_envelope.py",
            "test_report_the_per_arch_memory_envelope",
        ),
    }
)

#: Markers a test here may carry AT ALL. CI's default gate selects on markers, so an unknown one
#: is refused outright: its effect on tier membership cannot be predicted.
_ALLOWED_MARKERS = frozenset({"parametrize", "slow"})

#: `pytestmark = pytest.mark.slow` deselects every test in a file at once and appears in no
#: decorator list.
_MODULE_LEVEL = "<module-level pytestmark>"


class MarkerOutsideTheDeclaredSet(ConformanceRefusal):
    """A test in this suite carries a marker the tier-placement census does not know."""


class TierPlacementChanged(ConformanceRefusal):
    """A test moved into or out of the CI default tier without the declaration moving with it."""


def marker_census(path: Path) -> list[tuple[str, str, str]]:
    """`(module, test, marker)` for every `pytest.mark.X` on a test — both the decorator and the
    module-level `pytestmark`, which takes a whole file out of the tier in one line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            for decorator in node.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(func, ast.Attribute) and _is_mark_root(func.value):
                    rows.append((path.name, node.name, func.attr))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Attribute) and _is_mark_root(inner.value):
                    rows.append((path.name, _MODULE_LEVEL, inner.attr))
    return sorted(set(rows))


def require_declared_tier_placement(
    census: list[tuple[str, str, str]], declared: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]]:
    """Refuse an unknown marker, and refuse tier placement drifting from the declaration.

    BOTH DIRECTIONS. One `slow` line removed a cross-crate claim from CI while the suite reported
    `96 passed, 5 deselected` and every gate stayed rc 0, because a DESELECTED test is still
    COLLECTED; a measurement moved IN makes a magnitude a verdict by accident.
    """
    unknown = [row for row in census if row[2] not in _ALLOWED_MARKERS]
    if unknown:
        raise MarkerOutsideTheDeclaredSet(
            f"markers outside the declared set on {sorted(unknown)}. CI's default gate selects "
            f"on markers, so a marker this census does not know is one whose effect on tier "
            f"membership it cannot predict. Declared: {sorted(_ALLOWED_MARKERS)}."
        )
    observed = frozenset((module, test) for module, test, mark in census if mark == "slow")
    if observed != declared:
        raise TierPlacementChanged(
            f"tier placement changed. Moved OUT of the CI default tier (now `slow`, not "
            f"declared): {sorted(observed - declared)}; moved IN (declared `slow`, no longer "
            f"marked): {sorted(declared - observed)}. A deselected test is still COLLECTED, so "
            "neither the suite's own green nor the collected-test floor gate reports this."
        )
    return observed


def test_the_TIER_PLACEMENT_of_every_test_in_this_suite_matches_the_declaration(derived):
    """Every test's tier placement matches the declaration — the guard this module's own docstring
    called "the thing that nearly disarmed this tier" before anything asserted it."""
    modules = suite_modules()
    census = [row for path in modules for row in marker_census(path)]
    derived("t6.tier.marker_census", census)
    assert census, "the marker census is EMPTY — it is reporting a clean tier over nothing"
    derived(
        "t6.tier.slow_members",
        sorted(require_declared_tier_placement(census, _SLOW_TIER_MEMBERS)),
    )


def test_a_GATE_moved_OUT_of_the_CI_default_tier_is_refused():
    """The red-team plant, driven through the gate's own helper: one `slow` line above a cross-arm
    gate, which no other instrument in the repo would report."""
    census = [row for path in suite_modules() for row in marker_census(path)]
    planted = [
        *census,
        (
            "test_legal_move_coverage_boundary.py",
            "test_the_graph_wire_carries_exactly_mantis_cores_legal_set",
            "slow",
        ),
    ]
    with pytest.raises(TierPlacementChanged, match="Moved OUT of the CI default tier"):
        require_declared_tier_placement(planted, _SLOW_TIER_MEMBERS)


def test_a_MEASUREMENT_moved_INTO_the_CI_default_tier_is_refused():
    """The other direction, and not symmetry for its own sake: the ladder in the default tier puts
    a host-dependent wall-clock number on every CI run."""
    census = [row for path in suite_modules() for row in marker_census(path)]
    without_ladder = [row for row in census if row[1] != "test_leaf_forward_throughput_ladder"]
    with pytest.raises(TierPlacementChanged, match="moved IN"):
        require_declared_tier_placement(without_ladder, _SLOW_TIER_MEMBERS)


def test_a_MODULE_LEVEL_pytestmark_is_seen_by_the_tier_census(tmp_path):
    """`pytestmark = pytest.mark.slow` takes a whole file out of the tier in one line and appears
    in no decorator list, so a decorator-only census cannot see it."""
    module = tmp_path / "test_planted.py"
    module.write_text(
        "import pytest\n\npytestmark = pytest.mark.slow\n\ndef test_x(): assert True\n",
        encoding="utf-8",
    )
    assert marker_census(module) == [("test_planted.py", _MODULE_LEVEL, "slow")]
    with pytest.raises(TierPlacementChanged, match="Moved OUT"):
        require_declared_tier_placement(marker_census(module), _SLOW_TIER_MEMBERS)


def test_an_UNKNOWN_marker_is_refused_rather_than_ignored(tmp_path):
    """A marker the census does not know is one whose tier effect it cannot predict, and ignoring
    it is how a new `-m` expression silently deselects a gate."""
    module = tmp_path / "test_planted.py"
    # rule7-gate: ok -- the `\n@pytest.mark.flaky` in this planted source reads as user@host to
    # the ssh-userhost pattern: the "user" is the n of an escaped newline and the "domain" is a
    # pytest marker path. No account, no machine, and the string is a fixture this test writes.
    planted = 'import pytest\n\n@pytest.mark.flaky\ndef test_x(): assert True\n'
    module.write_text(planted, encoding="utf-8")
    with pytest.raises(MarkerOutsideTheDeclaredSet, match="flaky"):
        require_declared_tier_placement(marker_census(module), _SLOW_TIER_MEMBERS)


def test_a_MAGNITUDE_cannot_be_written_into_a_tracked_path(tmp_path):
    """Keeps "what does NOT land" enforceable rather than aspirational, by attempting it."""
    assert require_no_magnitude_lands(tmp_path / "block.md") == tmp_path / "block.md"
    with pytest.raises(MagnitudeWouldLand, match="tracked tree"):
        require_no_magnitude_lands(REPO_ROOT / "tools" / "bench_floors.toml")


def _tiny_graph_forward(spec):
    """The production surfaces, at the smallest net the repo's own precedent uses."""
    import torch

    from mantis.model.arch import GnnArch
    from mantis.model.build import build_net
    from mantis.selfplay.graph_collate import collate_graph_batch

    net = build_net(
        GnnArch(
            in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
            hidden=8, num_layers=1, policy_hidden=8, value_hidden=8,
        )
    ).eval()

    def build(span: int):
        board = build_board(spec.name, [(i, 0) for i in range(span + 1)])
        legal = board.legal_moves()
        buffer = HexgBuffer(2, spec.name, 8)
        buffer.push_graph_position(
            board.get_stones(), [(legal[0][0], legal[0][1], 1.0)],
            board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8,
        )
        wire, _targets = buffer.sample_graph_batch(1, augment=False)
        batch = collate_graph_batch(
            wire, trunk_size=spec.trunk_size, win_length=spec.win_length,
            node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
        )
        n_nodes = batch.x.shape[0]
        stone_mask = torch.zeros(n_nodes, dtype=torch.bool)
        stone_mask[: int(batch.n_stones.sum())] = True
        return batch, stone_mask, board.legal_move_count()

    def forward(payload) -> None:
        batch, stone_mask, _count = payload
        with torch.no_grad():
            net.forward_batch(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.legal_node_gather, stone_mask, batch.node_offsets,
            )

    return build, forward


@pytest.mark.slow
def test_leaf_forward_throughput_ladder(derived):
    """The measurement: a TABLE, asserting nothing about any magnitude in it.

    COVERAGE, STATED: the reachable block of the ONE registered graph encoding, on CPU. NOT the
    grid encodings (a dense arm is a different net) and NOT the synthetic block, which is a
    projection point with no board that produces it.
    """
    spec = next(s for s in roster() if s.is_graph)
    build, forward = _tiny_graph_forward(spec)
    spans = (1, 4, 8)
    points = reachable_ladder(spec.name, spans)
    ceiling = max(p.candidates for p in points)
    points += synthetic_ladder(ceiling, (2,))
    require_labelled_ladder(points)
    device_type = require_device_available("cpu", True)
    rows: list[dict] = []
    for span, point in zip(spans, points[: len(spans)], strict=True):
        payload_builder = lambda span=span: build(span)  # noqa: E731
        measurement = measure_forward(
            payload_builder, forward, repeats=5, warmup=2, device_type=device_type,
        )
        rows.append(
            {
                "encoding": spec.name,
                "candidates": point.candidates,
                "block": point.label,
                "repeats": measurement.repeats,
                "median_us_per_leaf": measurement.median_ns / 1000.0 / max(point.candidates, 1),
                "iqr_ns": measurement.iqr_ns,
                "device": device_type,
            }
        )
    derived("t6.measurement.rows", rows)
    derived(
        "t6.measurement.synthetic_block_UNMEASURED",
        [p.candidates for p in points[len(spans):]],
    )
    derived("t6.measurement.encodings_measured", sorted({row["encoding"] for row in rows}))
    assert rows, "the ladder produced no row"
    assert {row["block"] for row in rows} == {LADDER_REACHABLE}, (
        "a synthetic projection point carries a timing in the measured table; the synthetic "
        "block is a candidate-count projection and every measured row is reachable"
    )
