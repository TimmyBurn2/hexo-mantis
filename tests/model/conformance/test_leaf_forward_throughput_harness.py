# >300 justify (R8): the instrument and the self-tests that show it measures what its docstring
# names are one unit — a timer whose exclusion of graph construction is asserted in a different
# file can be widened silently, and the whole point of this tier is that its self-tests are the
# only thing making it non-vacuous.
"""T6 — the leaf-forward throughput INSTRUMENT. It measures; it never judges.

WHAT IT MEASURES: wall-clock microseconds per leaf of the MODEL FORWARD ONLY, at a ladder of
candidate counts, batched as at MCTS leaves. Not FLOPs, not steps/sec, not end-to-end. The timer
is `time.perf_counter_ns` around the forward, with `torch.cuda.synchronize()` before and after
when the device is CUDA — without which the number is queue-submission time, not compute.
Warm-up repeats are discarded and the statistic is the MEDIAN with IQR, matching what the
repo's IQR-gated bench discipline expects. THERE IS NO FLOOR, NO THRESHOLD AND NO PASS/FAIL.

WHAT DOES NOT LAND: any number, from any host. A µs/leaf figure is host-attested or it is
mechanism evidence, never a verdict, and the verdict-bearing number is operator-forwarded only.
Nothing here is written into `tools/bench_floors.toml` — that file carries criterion/Rust floors
attested against a pinned rustc, and a Python/torch figure would break the attestation it
exists to carry.

THE LADDER IS DERIVED, NOT TRANSCRIBED. Its reachable block comes from
`Board.legal_move_count()` on constructed positions; a `10^4` point does not exist on any real
position, so points above the reachable ceiling are built as a SYNTHETIC block, LABELLED, and
never mixed with the reachable one — a number from a position no game can produce is mechanism
evidence about the kernel, not about the system.

TIER PLACEMENT IS THE THING THAT NEARLY DISARMED THIS TIER, so it is stated: only the
MEASUREMENT carries `slow`. Every self-test and every planted break below is DEFAULT tier,
because CI's default gate runs `-m "not integration and not slow"` and a `slow`-marked control
is green-by-never-executing — the same vacuous pass one layer out.

BOTH SELF-TESTS WERE UNFALSIFIABLE IN THE ORIGINAL SPECIFICATION, and the remedy commits no
number. "Excludes graph construction" and "syncs on CUDA" are not structurally observable — you
can only see them in a duration or a call — while any magnitude assertion here is RED. So the
first self-test asserts a RELATION BETWEEN TWO MEASUREMENTS TAKEN IN THE SAME PROCESS (which is
host-independent and is not a µs/leaf figure) and the second COUNTS CALLS. The fixed sleep the
differential uses is an instrument parameter, not a threshold on any subject.

THE GRAVES THIS TIER IS NOT RE-DIGGING (read at HEAD, `docs/registers/falsified.md`): F-21, a
sibling-project CUDA-kernel borrow, falsified and red-teamed, whose stated fallback order is
torch.compile → smaller net → quantized eval; F-17/F-18/F-19, bench-falsified legal-move-set
perf ideas, with F-19's build-once-per-leaf corollary as standing doctrine; F-01, static probes
cannot validate dynamic equivariance. **T6 proposes NO optimization.** It changes no hot path,
adds no kernel, touches no builder. The rows are cited because the moment a T6 number tempts
someone toward a fix, they are the fence that fix has to clear first.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from mantis._engine import Board, HexgBuffer

from _corpus import ConformanceRefusal, build_board, roster

REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_REACHABLE = "reachable"
LADDER_SYNTHETIC = "synthetic"
#: Instrument parameter for the within-process differential. NOT a threshold on any subject:
#: it exists only to make a relation between two measurements observable.
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
    """Candidate counts READ from the engine, one per constructed position.

    `counter` is the seam the derivation control below stubs: pass a callable taking a Board
    and returning its legal-move count, and the ladder must follow it.
    """
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


def measure_forward(
    build_input,
    forward,
    *,
    repeats: int,
    warmup: int,
    device_type: str,
    sync=None,
) -> Measurement:
    """Time `forward` only. `build_input` runs OUTSIDE the timed region, every repeat.

    The sync callable is invoked before and after the timed region when the device reports
    CUDA, and the call count is returned so the branch is observable without a GPU.
    """
    samples: list[int] = []
    syncs = 0
    for index in range(warmup + repeats):
        payload = build_input()
        if device_type == "cuda" and sync is not None:
            sync()
            syncs += 1
        start = time.perf_counter_ns()
        forward(payload)
        if device_type == "cuda" and sync is not None:
            sync()
            syncs += 1
        elapsed = time.perf_counter_ns() - start
        if index >= warmup:
            samples.append(elapsed)
    samples.sort()
    mid = len(samples) // 2
    median = float(samples[mid] if len(samples) % 2 else (samples[mid - 1] + samples[mid]) / 2)
    lower = samples[: len(samples) // 2]
    upper = samples[(len(samples) + 1) // 2 :]
    iqr = float((upper[len(upper) // 2] if upper else 0) - (lower[len(lower) // 2] if lower else 0))
    return Measurement(median_ns=median, iqr_ns=iqr, sync_calls=syncs, repeats=len(samples))


# --------------------------------------------------------------------------------------- #
# Self-tests and planted breaks — ALL DEFAULT TIER
# --------------------------------------------------------------------------------------- #
def test_the_timer_EXCLUDES_input_construction_and_INCLUDES_the_forward(derived):
    """Self-test 1 — the within-process differential, two-sided. Neither half commits a
    number: both are relations between measurements taken in this same process."""
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


def test_the_CUDA_sync_branch_fires_on_cuda_and_NOT_on_cpu(derived):
    """Self-test 2 — the counting stub with its required negative control. Constructible with
    no GPU, which matters: the CUDA branch otherwise never executes where CI actually runs, so
    "sync is applied when CUDA" would be vacuous everywhere it is evaluated."""
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
    """Self-test 3 / PB-44. A ladder that does not move when the engine's count moves is a
    transcribed literal wearing a derivation."""
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
    """PB-45. The label is a required field, so the refusal is what makes it one."""
    good = synthetic_ladder(100, (2, 4))
    assert require_labelled_ladder(good) == 2
    with pytest.raises(UnlabelledLadderPoint, match="required field"):
        require_labelled_ladder([LadderPoint(400, "")])


def test_a_CLOSED_hardware_gate_FAILS_rather_than_skips():
    """PB-46. The only admissible conditional in this suite is a hardware gate, and it must
    fail loudly with a named reason when it is asked to run."""
    assert require_device_available("cpu", True) == "cpu"
    with pytest.raises(HardwareGateClosed, match="does not skip"):
        require_device_available("cuda", False)


#: THE MARKER FAMILY THAT REMOVES A TEST FROM THE RUN, enumerated and NON-EXHAUSTIVE, and
#: matched only as an attribute of `pytest.mark` / `mark` so an unrelated field named `skip`
#: is not a hit. `xfail` is in the family because a test whose failure is swallowed has stopped
#: asserting as surely as one that never runs.
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
    argvalues collects one SKIPPED item — a disarm that contains no skip word at all."""
    func = node.func
    named = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if named != "parametrize" or len(node.args) < 2:
        return False
    argvalues = node.args[1]
    return isinstance(argvalues, ast.List | ast.Tuple | ast.Set) and not argvalues.elts


def disarming_forms(tree: ast.AST) -> list[tuple[int, str]]:
    """`(line, form)` for every construct in one module that removes a test from the run.

    MECHANISM, NOT SCOPE, IS WHAT WAS WRONG. The control this replaces matched `ast.Call` nodes
    whose func was `pytest.skip`, so one `@pytest.mark.skipif(True, reason=...)` line above any
    gate in this suite produced a green run with one silent skip, `rc 0`, and a collected-test
    floor gate that also returned `rc 0` — because a skipped test is still COLLECTED. The prior
    fix widened this census from one module to every module of the suite and left the single
    spelling in place; nine modules times one spelling of six is still one spelling of six.

    Matched on the AST, so this very docstring — which necessarily contains the words — is not
    a hit, and so is the family declaration above.
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
    """PB-46's other half, over EVERY module of this suite and every form that disarms one.

    SCOPE WAS THE FIRST FINDING HERE and MECHANISM was the second. Parsing `Path(__file__)`
    asserted the discipline over one module of eight; matching only `pytest.skip(...)` asserted
    it over one spelling of six, and a `skipif` DECORATOR on T1's cross-arm gate — the half of
    T1 no other instrument makes — produced `104 passed, 1 skipped`, `rc 0`, with every gate in
    the repo green. A silent skip is this suite's headline failure mode.

    It stays in this module because PB-46's two halves — a hardware gate that FAILS rather than
    skips, and no module that disarms a test at all — are one claim, and a control that lives
    apart from its check can be deleted without the check going red. The census cardinality is
    a derived output, so an empty glob cannot pass for a clean census.
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


#: One planted line per disarming spelling, each written the way a hurried commit writes it.
#: A census that names a family and is only ever shown to catch one member of it is a family
#: claim resting on one measurement.
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
    """PB-46b. Each row is a line that produces a green run with one fewer assertion."""
    module = tmp_path / f"test_{label.replace('-', '_')}.py"
    module.write_text(body, encoding="utf-8")
    found = disarming_forms(ast.parse(module.read_text(encoding="utf-8")))
    assert found, f"{label}: the census did not see {body!r}"


def test_the_census_does_NOT_fire_on_an_ORDINARY_test_module(tmp_path):
    """Negative control, and as binding as the positive rows. A census that flags `parametrize`
    with real argvalues, a fixture, or a field that happens to be named `skip` would be widened
    until it fired on ordinary code, and its green would stop meaning its name."""
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


def test_a_MAGNITUDE_cannot_be_written_into_a_tracked_path(tmp_path):
    """PB-47. Keeps "what does NOT land" enforceable rather than aspirational, by attempting it."""
    assert require_no_magnitude_lands(tmp_path / "block.md") == tmp_path / "block.md"
    with pytest.raises(MagnitudeWouldLand, match="tracked tree"):
        require_no_magnitude_lands(REPO_ROOT / "tools" / "bench_floors.toml")


# --------------------------------------------------------------------------------------- #
# The MEASUREMENT (`slow`) — tables only, no magnitude asserted anywhere
# --------------------------------------------------------------------------------------- #
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
    """The measurement. Returns a TABLE — encoding, candidate count, block label, repeats,
    median, IQR, device — and asserts nothing about any magnitude in it.

    COVERAGE, STATED, because the table is narrower than the design asks and a table is read as
    its own scope. MEASURED: the reachable block of the ONE registered graph encoding, on CPU.
    NOT MEASURED, and named rather than implied: the registered GRID encodings, because this
    harness builds a GNN forward and a dense arm is a different net rather than a different
    parameter of this one; and the SYNTHETIC block, which is a projection point with no board
    that produces it — it is reported as a candidate count and carries no timing. The row set is
    asserted to contain no synthetic label below, so a projection cannot later drift into the
    measured table wearing a reachable figure.
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
