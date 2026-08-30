"""NIGHTRUN-1 E1 — the eval leaf build's WIDTH is threaded end to end, or it is nothing.

WHY THIS FILE IS THE LEVER'S ONLY WITNESS. `leaf_build_threads` defaults to `1` at three
layers, and `1` is the SERIAL path — the behaviour that shipped. So a break anywhere in the
chain produces **byte-identical results and no error**: correct graphs, correct policies,
correct promotions, and 95 % of the eval path back in a serial loop nobody notices for a run.
That is the silently-disabled-knob class R1 and LAW-08 exist for, and a default of `1` is only
defensible with this file beside it.

THE CHAIN, and every link is asserted here:

    run.py  --resolve_leaf_build_threads-->  build_eval_pipeline
            --self._leaf_build_threads-->    RoundSpec (crosses the process seam)
            --spec.leaf_build_threads-->     LocalInferenceEngine (both eval sites)
            --self._leaf_build_threads-->    submit_graphs_and_wait_ls(positions, n)

STRUCTURE, NEVER TEXT (R296(f)). Every check below reads an AST or drives a real object. A
grep for `leaf_build_threads` would pass on a commented-out line, on a docstring, and on a
keyword that is computed and then discarded.

THE GRID ARM IS A ROW, not an omission. A grid round builds no leaf graphs, so its width must
be the serial `1` and must NOT call the resolver — reading a graph-only host reservation on a
grid run would make it a grid dependency, the same reason `fused_graph_caps` is graph-only.
"""
from __future__ import annotations

import ast
import inspect
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
    # An `IfExp` is the graph/grid split; either branch may be the resolver call.
    calls = [n for n in ast.walk(value) if isinstance(n, ast.Call)]
    names = {getattr(c.func, "id", getattr(c.func, "attr", None)) for c in calls}
    assert "resolve_leaf_build_threads" in names, (
        f"run.py's leaf_build_threads is not a resolver call: {ast.dump(value)[:200]}"
    )
    assert isinstance(value, ast.IfExp), (
        "the width must be graph-only: a grid round builds no leaf graphs, and resolving a "
        "graph-path host reservation on a grid run makes it a grid dependency"
    )


def test_the_grid_arm_is_the_serial_width() -> None:
    source = (_REPO / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    value = _call_kwargs(source, "build_eval_pipeline")["leaf_build_threads"]
    assert isinstance(value, ast.IfExp)
    assert isinstance(value.orelse, ast.Constant) and value.orelse.value == 1, (
        "the grid arm must be the serial width 1, not a resolver call and not another literal"
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
    """BOTH sites, counted rather than spot-checked: the gate block builds a second engine
    for the anchor, and a width threaded to only one of them is a half-on lever."""
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
    """The last link, and the one a reviewer is most likely to miss: the engine may hold the
    width and still call the Rust entry point with its serial default."""
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
    """The arithmetic, stated as cases rather than asserted about this machine. The floor of
    1 matters: 1 is the serial path, and a budget of "no threads at all" is not a state the
    build loop can be in."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads

    got = resolve_leaf_build_threads({"selfplay": {"n_workers": n_workers}}, cpu_count=cores)
    assert got == want


def test_the_derivation_is_the_RING_S_arithmetic_and_not_a_second_copy() -> None:
    """ONE authority for the reservation. If these two ever disagree, the next person to
    change the ring's reservation has silently changed an eval path too — or failed to."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
    from mantis.config.resolve.sample_threads import resolve_sample_threads

    for cores in (2, 4, 8, 24, 64):
        for n_workers in (0, 1, 12, 100):
            cfg = {"selfplay": {"n_workers": n_workers}}
            assert (resolve_leaf_build_threads(cfg, cpu_count=cores)
                    == resolve_sample_threads(cfg, cpu_count=cores))


def test_a_missing_reservation_input_RAISES_rather_than_defaulting() -> None:
    """LAW-11: absent is an error. A silent fallback here would hand the eval child every
    core the self-play workers are using."""
    from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
    from mantis.config.resolve.sample_threads import MissingSampleThreadsInputError

    with pytest.raises(MissingSampleThreadsInputError):
        resolve_leaf_build_threads({"selfplay": {}}, cpu_count=8)
    with pytest.raises(MissingSampleThreadsInputError):
        resolve_leaf_build_threads({}, cpu_count=8)
