# >300 justify (R8): ONE oracle over ONE landing. The reroute, the ring's geometry contract,
# the refused dense-arm flags and the buried corpus-mix loader are four faces of a single
# decision — BC on the graph arch goes THROUGH the shared seam and arms nothing — and a split
# would let one half go green while the property they jointly assert was false.
"""The BC-pretrain graph REROUTE and the buried corpus-mix loader.

Three groups, each for a defect the others cannot see: the graph arm reaches the SAME declared
train-step seam the self-play loop takes; the ring's geometry is READ from provenance rather
than guessed; and nothing selects the route, the dense-only loader being refused BY NAME.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mantis.encoding import resolvers
from mantis.train.pretrain import graph_route
from mantis.train.pretrain.graph_route import (
    BC_RECENCY_WEIGHT,
    GraphPretrainError,
    load_ring,
    read_ring_provenance,
    DENSE_ARM_FLAGS,
    refuse_dense_arm_flags,
    resolve_step_budget,
    run_graph_pretrain,
)

_REPO = Path(__file__).resolve().parents[2]


def _write_ring(tmp_path: Path, **overrides: Any) -> Path:
    """A `.hexg` stand-in plus the provenance sidecar the route reads its geometry from."""
    ring = tmp_path / "corpus.hexg"
    ring.write_bytes(b"not-a-real-ring")
    prov: dict[str, Any] = {
        "encoding": "gnn_axis_v1", "ring_capacity": 64, "ring_visit_capacity": 8, "plies": 40,
        "games": 4,
    }
    prov.update(overrides)
    ring.with_name(ring.name + ".provenance.json").write_text(
        json.dumps(prov), encoding="utf-8"
    )
    return ring


def test_an_absent_sidecar_REFUSES_rather_than_guessing_a_capacity(tmp_path: Path) -> None:
    ring = tmp_path / "corpus.hexg"
    ring.write_bytes(b"x")
    with pytest.raises(GraphPretrainError, match="no provenance sidecar"):
        read_ring_provenance(ring, encoding="gnn_axis_v1")


def test_a_sidecar_for_a_DIFFERENT_encoding_is_refused(tmp_path: Path) -> None:
    ring = _write_ring(tmp_path, encoding="v6w25")
    with pytest.raises(GraphPretrainError, match="different training sets"):
        read_ring_provenance(ring, encoding="gnn_axis_v1")


@pytest.mark.parametrize("missing", ["ring_capacity", "ring_visit_capacity", "plies", "encoding"])
def test_every_geometry_key_is_required(tmp_path: Path, missing: str) -> None:
    ring = _write_ring(tmp_path)
    sidecar = ring.with_name(ring.name + ".provenance.json")
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    del prov[missing]
    sidecar.write_text(json.dumps(prov), encoding="utf-8")
    with pytest.raises(GraphPretrainError):
        read_ring_provenance(ring, encoding="gnn_axis_v1")


def test_an_unparseable_sidecar_is_refused(tmp_path: Path) -> None:
    ring = tmp_path / "corpus.hexg"
    ring.write_bytes(b"x")
    ring.with_name(ring.name + ".provenance.json").write_text("{ nope", encoding="utf-8")
    with pytest.raises(GraphPretrainError, match="unparseable"):
        read_ring_provenance(ring, encoding="gnn_axis_v1")


def test_an_absent_ring_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GraphPretrainError, match="corpus ring not found"):
        load_ring(tmp_path / "nope.hexg", encoding="gnn_axis_v1")


def test_a_ring_that_loads_zero_records_is_refused(tmp_path: Path, monkeypatch) -> None:
    """An empty corpus is not a run — the refusal is about the LOADED COUNT, not the bytes."""
    ring = _write_ring(tmp_path)

    class _Buf:
        def __init__(self, *_a, **_k) -> None: ...
        def load_from_path(self, _p: str) -> int:
            return 0

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)
    with pytest.raises(GraphPretrainError, match="loaded 0 records"):
        load_ring(ring, encoding="gnn_axis_v1")


def test_the_buffer_is_built_at_the_SIDECARS_geometry(tmp_path: Path, monkeypatch) -> None:
    """Capacity and visit capacity come from the artifact's own record, not a plausible constant
    smaller than the corpus."""
    ring = _write_ring(tmp_path, ring_capacity=4096, ring_visit_capacity=13)
    seen: dict[str, Any] = {}

    class _Buf:
        def __init__(self, capacity, encoding, visit_capacity) -> None:
            seen.update(capacity=capacity, encoding=encoding, visit_capacity=visit_capacity)
        def load_from_path(self, _p: str) -> int:
            return 40

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)
    load_ring(ring, encoding="gnn_axis_v1")
    assert seen == {"capacity": 4096, "encoding": "gnn_axis_v1", "visit_capacity": 13}


def _stub_buffer(monkeypatch, records: int = 40) -> None:
    class _Buf:
        def __init__(self, *_a, **_k) -> None: ...
        def load_from_path(self, _p: str) -> int:
            return records

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)


def test_an_encoding_with_NO_launch_pin_is_not_enforced(tmp_path: Path, monkeypatch) -> None:
    """A `None` pin means NOT ENFORCED — the arm that runs at HEAD, where no graph encoding
    registers a pin, so a green BC run proves nothing without the arm below."""
    ring = _write_ring(tmp_path)
    _stub_buffer(monkeypatch)
    monkeypatch.setattr(resolvers, "_CORPUS_SHA_PINS", {}, raising=True)
    load_ring(ring, encoding="gnn_axis_v1")


def test_a_ring_that_is_not_the_PINNED_corpus_is_refused(tmp_path: Path, monkeypatch) -> None:
    """THE PRODUCER: a launch pin says two hosts train on byte-identical bytes. Registered
    synthetically because no graph encoding is pinned at HEAD."""
    ring = _write_ring(tmp_path)
    _stub_buffer(monkeypatch)
    monkeypatch.setattr(resolvers, "_CORPUS_SHA_PINS", {"gnn_axis_v1": "f" * 64}, raising=True)
    with pytest.raises(GraphPretrainError, match="not the launch-pinned corpus"):
        load_ring(ring, encoding="gnn_axis_v1")


def test_the_pin_is_taken_over_the_RING_BYTES_not_the_sidecar(tmp_path: Path, monkeypatch) -> None:
    """A pin read off provenance would certify a swapped ring, so it is taken over the BYTES;
    rewriting the sidecar does not move the verdict."""
    ring = _write_ring(tmp_path)
    true_sha = hashlib.sha256(ring.read_bytes()).hexdigest()
    _stub_buffer(monkeypatch)
    monkeypatch.setattr(resolvers, "_CORPUS_SHA_PINS", {"gnn_axis_v1": true_sha}, raising=True)
    load_ring(ring, encoding="gnn_axis_v1")

    sidecar = ring.with_name(ring.name + ".provenance.json")
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    prov["source_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(prov), encoding="utf-8")
    load_ring(ring, encoding="gnn_axis_v1")

    ring.write_bytes(b"a-different-ring-entirely")
    with pytest.raises(GraphPretrainError, match="not the launch-pinned corpus"):
        load_ring(ring, encoding="gnn_axis_v1")


def test_the_pin_check_runs_INSIDE_load_ring_and_not_only_in_a_helper() -> None:
    """The seam, structurally: a guard reachable only by calling it directly is uncalled."""
    src = (_REPO / "src" / "mantis" / "train" / "pretrain" / "graph_route.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_ring")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_assert_launch_pin" in called, (
        "load_ring no longer calls the launch-pin guard — the corpus it loads is unchecked"
    )


def test_explicit_steps_wins_over_epochs() -> None:
    assert resolve_step_budget(1000, batch_size=32, steps=7, epochs=99) == 7


def test_epochs_derive_the_dense_arms_way() -> None:
    """`epochs * ceil(ring / batch)` — the dense arm's `epochs * len(loader)`."""
    assert resolve_step_budget(100, batch_size=32, steps=None, epochs=3) == 3 * 4


def test_a_zero_budget_is_refused() -> None:
    with pytest.raises(GraphPretrainError, match="nothing would be trained"):
        resolve_step_budget(100, batch_size=32, steps=0, epochs=1)


def test_the_graph_arm_trains_THROUGH_the_declared_seam(tmp_path: Path, monkeypatch) -> None:
    """The gradient step is `run_declared_train_step`, not a second graph training path,
    asserted on the CALL because a parallel path would still produce a checkpoint."""
    ring = _write_ring(tmp_path, plies=64)
    calls: list[dict[str, Any]] = []

    class _Buf:
        def __init__(self, *_a, **_k) -> None: ...
        def load_from_path(self, _p: str) -> int:
            return 64

    class _Spec:
        name = "gnn_axis_v1"
        representation = "graph"

    class _Trainer:
        def __init__(self, *_a, **_k) -> None:
            self.saved: Any = None
        def save_checkpoint(self, loss_info):
            self.saved = loss_info
            return tmp_path / "ckpt.pt"

    class _Knobs:
        batch_size = 8
        augment = True

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)
    monkeypatch.setattr(graph_route, "Trainer", _Trainer)
    monkeypatch.setattr(graph_route, "build_net", lambda _arch: object())
    monkeypatch.setattr(graph_route, "arch_from_spec_and_config", lambda _s, _c: object())
    monkeypatch.setattr(graph_route, "resolve_coordinator_knobs", lambda _t: _Knobs())
    monkeypatch.setattr(graph_route, "resolve_microbatch_caps", lambda _c: "CAPS")
    monkeypatch.setattr(graph_route, "resolve_sample_threads", lambda _c: 3)

    def _spy(trainer, buffer, spec, **kw):
        calls.append({"spec": spec, **kw})
        return {"loss": 1.0}

    monkeypatch.setattr(graph_route, "run_declared_train_step", _spy)

    out = run_graph_pretrain(
        spec=_Spec(), full_config={"train": {}}, train_section=object(), ring_path=ring,
        checkpoint_dir=tmp_path, device=None, steps=3, epochs=1, dense_arm_flags={},
    )
    assert out == tmp_path / "ckpt.pt"
    assert len(calls) == 3, "one seam call per step"
    first = calls[0]
    assert first["spec"].representation == "graph"
    assert first["batch_size"] == 8 and first["augment"] is True
    assert first["recent_buffer"] is None, "the graph route takes no dense recent buffer"
    # The providers are CALLABLES, not resolved values — the dispatcher's own contract.
    assert callable(first["caps_provider"]) and first["caps_provider"]() == "CAPS"
    assert callable(first["sample_threads_provider"]) and first["sample_threads_provider"]() == 3


def test_the_BC_route_passes_zero_recency_and_it_is_STRUCTURAL(tmp_path: Path, monkeypatch) -> None:
    """A corpus ring has no time ordering, so `recent_frac` has no subject over it."""
    assert BC_RECENCY_WEIGHT == 0.0
    ring = _write_ring(tmp_path)
    seen: list[float] = []

    class _Buf:
        def __init__(self, *_a, **_k) -> None: ...
        def load_from_path(self, _p: str) -> int:
            return 40

    class _Spec:
        name = "gnn_axis_v1"
        representation = "graph"

    class _Knobs:
        batch_size = 4
        augment = False

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)
    monkeypatch.setattr(graph_route, "Trainer",
                        lambda *_a, **_k: type("T", (), {"save_checkpoint": lambda s, li: tmp_path / "c.pt"})())
    monkeypatch.setattr(graph_route, "build_net", lambda _arch: object())
    monkeypatch.setattr(graph_route, "arch_from_spec_and_config", lambda _s, _c: object())
    monkeypatch.setattr(graph_route, "resolve_coordinator_knobs", lambda _t: _Knobs())
    monkeypatch.setattr(graph_route, "resolve_microbatch_caps", lambda _c: None)
    monkeypatch.setattr(graph_route, "resolve_sample_threads", lambda _c: 1)
    monkeypatch.setattr(graph_route, "run_declared_train_step",
                        lambda _t, _b, _s, **kw: seen.append(kw["recency_weight"]) or {"loss": 0.0})
    run_graph_pretrain(
        spec=_Spec(), full_config={}, train_section=object(), ring_path=ring,
        checkpoint_dir=tmp_path, device=None, steps=2, epochs=1, dense_arm_flags={},
    )
    assert seen == [0.0, 0.0]


def test_the_cli_routes_on_the_declared_representation() -> None:
    """The route is chosen by the declaration, never by an `isinstance` on the built model."""
    src = (_REPO / "src/mantis/train/pretrain/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "pretrain")
    branch_src = ast.get_source_segment(src, fn) or ""
    assert 'representation", None) == "graph"' in branch_src
    graph_at = branch_src.index('== "graph"')
    sniff_at = branch_src.find("isinstance(model")
    assert sniff_at == -1 or graph_at < sniff_at, (
        "the graph route must be decided by the declaration BEFORE any model sniff"
    )


def test_the_cli_exposes_a_hexg_corpus_override() -> None:
    src = (_REPO / "src/mantis/train/pretrain/cli.py").read_text(encoding="utf-8")
    assert "--corpus-hexg" in src


#: The buried symbol, named ONCE so a rename cannot leave the census watching a dead string.
_BURIED = "load_pretrained_buffer"

#: Where a resurrection could hide. `tests/` is included deliberately: a test that re-defines or
#: re-imports the symbol would make the census green while the thing was back.
_CENSUS_ROOTS = ("src", "tools", "tests")


def _census(predicate: Callable[[ast.AST], bool]) -> list[str]:
    """`path:lineno` for every AST node under the census roots that `predicate` accepts."""
    hits: list[str] = []
    for root in _CENSUS_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if predicate(node):
                    hits.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    return hits


def test_the_corpus_mix_loader_IS_BURIED_and_nothing_defines_it() -> None:
    """The corpus-mix loader is deleted, censused over DEFINITIONS so the grave comment naming
    the symbol cannot read as a resurrection."""
    hits = _census(lambda n: isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == _BURIED)
    assert not hits, (
        f"{_BURIED} is DEFINED at {hits}. R326(d) deleted it: posture (A) excludes corpus-mix "
        "by definition, so a redefinition needs a ruling that un-buries it, not an edit"
    )


def test_nothing_IMPORTS_the_buried_loader() -> None:
    """The import half: a definition census alone passes while a stale import sits in a module,
    and that is an `ImportError` at collection time, which takes a whole tier down."""
    def _imports_it(node: ast.AST) -> bool:
        if isinstance(node, ast.ImportFrom):
            return any(alias.name == _BURIED for alias in node.names)
        if isinstance(node, ast.Import):
            return any(alias.name.rsplit(".", 1)[-1] == _BURIED for alias in node.names)
        return False

    hits = _census(_imports_it)
    assert not hits, f"{_BURIED} is imported at {hits} — the symbol does not exist"


def test_nothing_CALLS_the_buried_loader_by_either_spelling() -> None:
    """The call census, WIDENED to both spellings — it watched bare `Name` calls only, which
    did not matter while the function existed and does now."""
    def _calls_it(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return ((isinstance(func, ast.Name) and func.id == _BURIED)
                or (isinstance(func, ast.Attribute) and func.attr == _BURIED))

    hits = _census(_calls_it)
    assert not hits, f"{_BURIED} is called at {hits} — the symbol does not exist"


def test_only_the_pretrain_cli_reaches_the_graph_route() -> None:
    """LANDING IS NOT ARMING: reachable from the manual CLI entry point and nothing else."""
    hits: list[str] = []
    for path in (_REPO / "src").rglob("*.py"):
        if path.name in {"graph_route.py", "cli.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("graph_route" in n for n in names):
                hits.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert not hits, f"the BC graph route is reachable from production code at {hits}"


def test_no_shipped_config_names_the_pretrain_route() -> None:
    """Asserted on the MODULE PATHS a config must name, not the substring "pretrain", which
    legitimately appears in `mixing.pretrained_buffer_path`."""
    for cfg in sorted((_REPO / "configs").glob("*.yaml")):
        text = cfg.read_text(encoding="utf-8")
        for token in ("graph_route", "mantis.train.pretrain", "train/pretrain"):
            assert token not in text, f"{cfg.name} names {token}"


@pytest.mark.parametrize("flag", sorted(DENSE_ARM_FLAGS))
def test_EVERY_dense_arm_flag_is_refused_not_ignored(flag: str) -> None:
    """Driven over the SET: a flag added without a refusal, or without a reason, fails here."""
    with pytest.raises(GraphPretrainError, match="does not read"):
        refuse_dense_arm_flags({flag: 1.0})
    assert DENSE_ARM_FLAGS[flag].strip(), f"{flag} carries no reason"


def test_store_true_flags_read_False_as_NOT_supplied() -> None:
    """`store_true` flags read `False` as NOT supplied — otherwise every graph pretrain is
    refused."""
    refuse_dense_arm_flags({"--freeze-trunk-entry": False, "--filters": None})


def test_a_flag_outside_the_declared_set_is_itself_an_error() -> None:
    """The set is the authority — an undeclared flag would otherwise get a silent pass."""
    with pytest.raises(GraphPretrainError, match="not in DENSE_ARM_FLAGS"):
        refuse_dense_arm_flags({"--not-a-flag": 1})


def test_the_cli_hands_over_every_declared_dense_arm_flag() -> None:
    """The other half: a declared flag the CLI never passes is refused in theory only."""
    src = (_REPO / "src/mantis/train/pretrain/cli.py").read_text(encoding="utf-8")
    for flag in DENSE_ARM_FLAGS:
        assert f'"{flag}": args.' in src, f"the CLI never hands {flag} to the refusal"


def test_no_dense_arm_flags_is_the_clean_case() -> None:
    refuse_dense_arm_flags({f: None for f in DENSE_ARM_FLAGS})


def test_the_graph_arch_is_built_from_the_NESTED_config(tmp_path: Path, monkeypatch) -> None:
    """Production resolves the arch from the NESTED config, so the CLI's flat term dict would
    resolve a different arch the day a `gnn_*` width key is minted. Asserted on the ARGUMENT."""
    ring = _write_ring(tmp_path)
    seen: list[Any] = []
    nested = {"train": {"batch_size": 4}, "identity": {"encoding": "gnn_axis_v1"}}

    class _Buf:
        def __init__(self, *_a, **_k) -> None: ...
        def load_from_path(self, _p: str) -> int:
            return 40

    class _Spec:
        name = "gnn_axis_v1"
        representation = "graph"

    class _Knobs:
        batch_size = 4
        augment = False

    import mantis._engine as _engine
    monkeypatch.setattr(_engine, "HexgBuffer", _Buf, raising=False)
    monkeypatch.setattr(graph_route, "Trainer",
                        lambda *_a, **_k: type("T", (), {"save_checkpoint": lambda s, li: tmp_path / "c.pt"})())
    monkeypatch.setattr(graph_route, "build_net", lambda _arch: object())
    monkeypatch.setattr(graph_route, "arch_from_spec_and_config",
                        lambda _s, cfg: seen.append(cfg) or object())
    monkeypatch.setattr(graph_route, "resolve_coordinator_knobs", lambda _t: _Knobs())
    monkeypatch.setattr(graph_route, "resolve_microbatch_caps", lambda _c: None)
    monkeypatch.setattr(graph_route, "resolve_sample_threads", lambda _c: 1)
    monkeypatch.setattr(graph_route, "run_declared_train_step",
                        lambda *_a, **_k: {"loss": 0.0})
    run_graph_pretrain(
        spec=_Spec(), full_config=nested, train_section=object(), ring_path=ring,
        checkpoint_dir=tmp_path, device=None, steps=1, epochs=1, dense_arm_flags={},
    )
    assert seen == [nested]


def test_the_flag_refusal_is_REQUIRED_and_undefaulted() -> None:
    """A caller that could omit `dense_arm_flags` would silently skip the refusal."""
    import inspect
    sig = inspect.signature(run_graph_pretrain)
    assert sig.parameters["dense_arm_flags"].default is inspect.Parameter.empty
