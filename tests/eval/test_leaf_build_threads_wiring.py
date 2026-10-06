"""The eval leaf build's WIDTH is threaded end to end, or it is nothing.

`leaf_build_threads` defaults to `1` at three layers, and `1` is the SERIAL path that shipped,
so a break anywhere in the chain produces byte-identical results and no error: correct graphs,
correct promotions, and most of the eval path back in a serial loop nobody notices. A default
of `1` is only defensible with this file beside it.

    run.py  --resolve_leaf_build_threads-->  build_eval_pipeline
            --self._leaf_build_threads-->    RoundSpec (crosses the process seam)
            --spec.leaf_build_threads-->     LocalInferenceEngine (both eval sites)
            --self._leaf_build_threads-->    submit_graphs_and_wait_ls(positions, n)

STRUCTURE, NEVER TEXT: every check reads an AST or drives a real object, because a grep would
pass on a commented-out line, a docstring, or a keyword computed and then discarded.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _call_kwargs(source: str, func_name: str) -> dict[str, ast.expr]:
    """The keyword arguments of the FIRST `func_name(...)` call in `source`, as AST nodes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == func_name:
            return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no call to {func_name}(...) found")


def test_run_py_threads_a_DERIVED_width_and_not_a_literal() -> None:
    """The top of the chain. A literal here would be a host reservation `run.py` invented,
    which is precisely what `resolve_leaf_build_threads` exists to stop."""
    source = (_REPO / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    kwargs = _call_kwargs(source, "build_eval_pipeline")
    assert "leaf_build_threads" in kwargs, (
        "run.py builds the eval pipeline without threading a leaf-build width, so every "
        "eval round runs the SERIAL build and the lever is silently off"
    )
    value = kwargs["leaf_build_threads"]
    assert isinstance(value, ast.Call) and getattr(
        value.func, "id", getattr(value.func, "attr", None)) == "resolve_leaf_build_threads", (
        f"run.py's leaf_build_threads is not the resolver call itself: {ast.dump(value)[:200]}"
    )


def test_the_pipeline_puts_its_width_on_every_round_spec() -> None:
    """The seam. `RoundSpec` is the ONLY route from the parent to the eval child."""
    source = inspect.getsource(__import__("mantis.eval.pipeline", fromlist=["x"]))
    kwargs = _call_kwargs(source, "RoundSpec")
    assert "leaf_build_threads" in kwargs, "RoundSpec is built without the width"
    value = kwargs["leaf_build_threads"]
    assert isinstance(value, ast.Attribute) and value.attr == "_leaf_build_threads", (
        "the spec's width must be the pipeline's OWN resolved value, not a fresh literal"
    )


def test_both_eval_engine_constructions_thread_the_specs_width() -> None:
    """BOTH sites, counted rather than spot-checked: the gate block builds a second engine for
    the anchor, and a width threaded to only one of them is a half-on lever."""
    source = (_REPO / "src" / "mantis" / "eval" / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "LocalInferenceEngine"
    ]
    assert len(sites) == 2, f"expected two eval engine constructions, found {len(sites)}"
    for site in sites:
        kw = {k.arg: k.value for k in site.keywords if k.arg is not None}
        value = kw.get("leaf_build_threads")
        assert value is not None, (
            f"the LocalInferenceEngine at line {site.lineno} threads no leaf-build width"
        )
        assert isinstance(value, ast.Attribute) and value.attr == "leaf_build_threads", (
            f"line {site.lineno}: the width must come from the ROUND SPEC, never from a "
            f"literal or a re-resolution in the child (which has no RunConfig)"
        )


def test_the_engine_passes_its_width_to_the_rust_call() -> None:
    """The last link: the engine may hold the width and still call the Rust entry point with its
    serial default."""
    from mantis.selfplay import inference_local

    source = inspect.getsource(inference_local)
    call = _call_kwargs  # noqa: F841 — the positional form below is what production uses
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == \
                "submit_graphs_and_wait_ls":
            found.append(node)
    assert found, "no submit_graphs_and_wait_ls call in inference_local"
    for node in found:
        args = list(node.args) + [k.value for k in node.keywords]
        assert any(
            isinstance(a, ast.Attribute) and a.attr == "_leaf_build_threads" for a in args
        ), (
            f"line {node.lineno}: submit_graphs_and_wait_ls is called without the engine's "
            f"own width, so the Rust default (serial) applies and the lever is off"
        )


@pytest.mark.parametrize(("cores", "n_workers", "want"), [
    (24, 12, 11), (24, 1, 22), (4, 8, 1), (2, 0, 1),
])
def test_the_derivation_reserves_and_never_returns_zero(
    cores: int, n_workers: int, want: int,
) -> None:
    """The arithmetic as cases rather than an assertion about this machine. The floor of 1
    matters: a budget of "no threads at all" is not a state the build loop can be in."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads

    got = resolve_leaf_build_threads({"selfplay": {"n_workers": n_workers}}, cpu_count=cores)
    assert got == want


def test_the_derivation_is_the_RING_S_arithmetic_and_not_a_second_copy() -> None:
    """ONE authority for the reservation: if these two disagree, changing the ring's reservation
    has silently changed an eval path too, or failed to."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
    from mantis.config.resolve.sample_threads import resolve_sample_threads

    for cores in (2, 4, 8, 24, 64):
        for n_workers in (0, 1, 12, 100):
            cfg = {"selfplay": {"n_workers": n_workers}}
            assert (resolve_leaf_build_threads(cfg, cpu_count=cores)
                    == resolve_sample_threads(cfg, cpu_count=cores))


def test_a_missing_reservation_input_RAISES_rather_than_defaulting() -> None:
    """Absent is an error: a silent fallback would hand the eval child every core the self-play
    workers are using."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
    from mantis.config.resolve.sample_threads import MissingSampleThreadsInputError

    with pytest.raises(MissingSampleThreadsInputError):
        resolve_leaf_build_threads({"selfplay": {}}, cpu_count=8)
    with pytest.raises(MissingSampleThreadsInputError):
        resolve_leaf_build_threads({}, cpu_count=8)


@pytest.mark.parametrize(("cores", "leaf_batch", "concurrency", "want"), [
    (16, 8, 1, 8), (16, 8, 8, 1), (4, 8, 1, 3), (1, 8, 1, 1), (16, 4, 1, 4), (32, 8, 4, 7), (2, 8, 8, 1),
])
def test_the_standalone_width_shares_the_host_and_never_exceeds_a_leaf_batch(
    cores: int, leaf_batch: int, concurrency: int, want: int,
) -> None:
    """A host running no self-play: the cores less the serving thread, shared by the games in flight,
    at most one leaf batch (a wider pool has no leaf to build)."""
    from mantis.config.resolve.leaf_build_threads import resolve_standalone_leaf_build_threads

    cfg = {"selfplay": {"n_workers": 32, "leaf_batch_size": leaf_batch}}
    assert resolve_standalone_leaf_build_threads(cfg, concurrency=concurrency, cpu_count=cores) == want


def test_a_missing_standalone_input_RAISES_rather_than_defaulting() -> None:
    from mantis.config.resolve.leaf_build_threads import resolve_standalone_leaf_build_threads
    from mantis.config.resolve.sample_threads import MissingSampleThreadsInputError

    with pytest.raises(MissingSampleThreadsInputError):
        resolve_standalone_leaf_build_threads({"selfplay": {}}, concurrency=1, cpu_count=8)
    with pytest.raises(ValueError):
        resolve_standalone_leaf_build_threads({"selfplay": {"leaf_batch_size": 8}}, concurrency=0, cpu_count=8)


def _calls(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "id", getattr(node.func, "attr", None)) == name)


def test_the_in_run_eval_beside_self_play_keeps_the_reservation() -> None:
    """The pin: widening the build beside self-play takes cores from its workers, so the run's eval seam
    never reaches the standalone width."""
    for rel in ("src/mantis/run.py", "src/mantis/eval/pipeline.py", "src/mantis/eval/worker.py"):
        assert _calls(_REPO / rel, "resolve_standalone_leaf_build_threads") == 0, rel
    assert _calls(_REPO / "src/mantis/run.py", "resolve_leaf_build_threads") == 1


def test_nothing_widens_unasked_and_a_cell_or_the_ladder_widens_only_on_the_hosts_word() -> None:
    """A cell, the ladder or the dash may sit beside a run; only the host's own word widens the build."""
    for rel in ("tools/ladder/backends.py", "tools/strength_frontier.py"):
        assert _calls(_REPO / rel, "resolve_standalone_leaf_build_threads") == 1, rel
        assert _calls(_REPO / rel, "resolve_leaf_build_threads") >= 1, rel
    assert _calls(_REPO / "tools/dash/engine/engines.py", "resolve_standalone_leaf_build_threads") == 0
    assert _calls(_REPO / "tools/strix_follower.py", "resolve_standalone_leaf_build_threads") == 0


@pytest.mark.parametrize("standalone", [False, True])
def test_a_cell_keeps_the_reservation_unless_its_host_is_standalone(standalone: bool) -> None:
    """Eight games in flight on a standalone host share it; beside a run a cell takes what the run has not promised."""
    from mantis.config.census import production_configs
    from mantis.config.loader import load_config
    from mantis.config.resolve.leaf_build_threads import (
        resolve_leaf_build_threads,
        resolve_standalone_leaf_build_threads,
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("strength_frontier_wiring_t", _REPO / "tools/strength_frontier.py")
    assert spec is not None and spec.loader is not None
    sf = importlib.util.module_from_spec(spec)
    sys.modules["strength_frontier_wiring_t"] = sf
    try:
        spec.loader.exec_module(sf)
        config = load_config(production_configs(_REPO)[0])
        base = sf.base_round_spec(config, work_dir=Path("/nonexistent"))
        cell = {"label": "c", "search_kind": "puct", "sims": 128, "games": 2, "opponent": "six",
                "six_net": "gen0030", "six_nodes": 16, "concurrency": 8, **({"standalone_host": True} if standalone else {})}
        got = sf.cell_spec(cell, base, cell_dir=Path("/nonexistent"), config=config)
        dump = config.model_dump()
        want = (resolve_standalone_leaf_build_threads(dump, concurrency=8) if standalone
                else resolve_leaf_build_threads(dump))
        assert got.leaf_build_threads == want
    finally:
        sys.modules.pop("strength_frontier_wiring_t", None)


def test_every_round_result_carries_its_engines_serving_rows() -> None:
    """Each engine's wake and bucket rows fire per pop, so every exit of the round reports them."""
    tree = ast.parse((_REPO / "src" / "mantis" / "eval" / "worker.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_round_result"]
    assert len(calls) == 2 and all(any(k.arg == "serving" for k in c.keywords) for c in calls)
    from mantis.eval import worker

    class _Engine:
        def batch_timing_snapshot(self) -> dict:
            return {"wake": {"submitters": 8, "all_submitted": 3}, "served_graphs": {"bucket_parts": {"1025": 2}},
                    "unverified_graph_builds": 5, "queue_wait": None}
    assert worker.serving_rows(_Engine()) == {"wake": {"submitters": 8, "all_submitted": 3},
                                              "served_graphs": {"bucket_parts": {"1025": 2}},
                                              "unverified_graph_builds": 5}
