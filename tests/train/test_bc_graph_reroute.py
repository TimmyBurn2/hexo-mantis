# >300 justify (R8): ONE oracle over ONE landing. The reroute, the ring's geometry contract,
# the refused dense-arm flags and the gap-4 disposition are four faces of a single
# decision — that BC on the graph arch goes THROUGH the shared seam and arms nothing — and a
# split would let one half go green while the property they jointly assert was false.
"""The BC-pretrain graph REROUTE and the gap-4 disposition (R325(c)).

Three groups, each for a defect the others cannot see:

  * THE REROUTE IS A REROUTE — the graph arm reaches `run_declared_train_step`, the SAME
    declared seam the self-play loop takes, with the providers passed as callables. A test
    that only checked "a checkpoint appeared" would pass on a second, parallel training path,
    which is the exact thing this design refuses.
  * THE RING'S GEOMETRY IS READ, NOT GUESSED — every provenance refusal bites. A guessed
    capacity silently drops the head of the corpus and every downstream check still passes.
  * UNARMEDNESS AND THE GAP-4 CONTRACT — nothing selects the route, and the dense-only
    corpus-mix loader now refuses a non-grid representation BY NAME rather than inside numpy.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

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


# ── the ring's geometry is READ, not guessed ─────────────────────────────────────────────
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
    """An empty corpus is not a run. Driven through `load_ring` with a stub buffer, because
    the refusal is about the LOADED COUNT and not about the file's bytes."""
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
    """The capacity and visit capacity come from the artifact's own record — the defect this
    catches is a plausible constant that is smaller than the corpus."""
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


# ── the step budget, on the dense arm's own convention ───────────────────────────────────
def test_explicit_steps_wins_over_epochs() -> None:
    assert resolve_step_budget(1000, batch_size=32, steps=7, epochs=99) == 7


def test_epochs_derive_the_dense_arms_way() -> None:
    """`epochs * ceil(ring / batch)` — the dense arm's `epochs * len(loader)`."""
    assert resolve_step_budget(100, batch_size=32, steps=None, epochs=3) == 3 * 4


def test_a_zero_budget_is_refused() -> None:
    with pytest.raises(GraphPretrainError, match="nothing would be trained"):
        resolve_step_budget(100, batch_size=32, steps=0, epochs=1)


# ── THE REROUTE IS A REROUTE ─────────────────────────────────────────────────────────────
def test_the_graph_arm_trains_THROUGH_the_declared_seam(tmp_path: Path, monkeypatch) -> None:
    """The property this module exists for: the gradient step is
    `run_declared_train_step` — the SAME seam the self-play loop takes — and not a second
    graph training path. Asserted on the CALL, with its spec and its providers, because a
    parallel path would still produce a checkpoint."""
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
    """A corpus ring has no time ordering, so `recent_frac` has no subject over it. Pinned
    because reading `train.recency_weight` here would silently treat the newest ring rows as
    a recent window they are not."""
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


# ── the CLI branches on the DECLARED representation, not on a model sniff ────────────────
def test_the_cli_routes_on_the_declared_representation() -> None:
    """R102's ban: the route is chosen by the declaration, never by an `isinstance` on the
    built model. Asserted structurally on the CLI's own source."""
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


# ── GAP 4: the corpus-mix loader states its contract instead of assuming it ──────────────
def test_the_corpus_mix_loader_REFUSES_a_graph_representation_by_name() -> None:
    """The gap-4 disposition. Before this, a graph run reaching the loader died inside
    `np.load` on a `.hexg` file; now it dies at the contract with the reason named."""
    from mantis.train.batch_assembly import load_pretrained_buffer
    from mantis.train.coordinator.dispatch import RepresentationRouteError

    with pytest.raises(RepresentationRouteError, match="no corpus-mix route"):
        load_pretrained_buffer(
            {"pretrained_buffer_path": "data/gnn_corpus_v1.hexg"},
            {"encoding": "gnn_axis_v1"},
            lambda _e: None, 0, 0,
        )


def test_the_corpus_mix_loader_is_STILL_uncalled_and_that_is_deliberate() -> None:
    """R289(q) holds this path RESERVED: it is one arm of an operator-owed decision, so it is
    not deleted — and it is not wired either. An `ast` call census, so a docstring mention
    does not read as a call site."""
    hits: list[str] = []
    for path in (_REPO / "src").rglob("*.py"):
        if path.name == "batch_assembly.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "load_pretrained_buffer":
                hits.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert not hits, f"corpus-mix is wired at {hits} — that is a posture the operator has not chosen"


# ── unarmedness of the reroute itself ────────────────────────────────────────────────────
def test_only_the_pretrain_cli_reaches_the_graph_route() -> None:
    """LANDING IS NOT ARMING. The reroute is reachable from the manual CLI entry point and
    from nothing else under `src/`."""
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
    """The other half of unarmedness. Asserted on the MODULE PATHS a config would have to
    name to select this route — not on the substring "pretrain", which legitimately appears in
    `mixing.pretrained_buffer_path` and `mixing.pretrain_max_samples` and would make this row
    either vacuous or wrong."""
    for cfg in sorted((_REPO / "configs").glob("*.yaml")):
        text = cfg.read_text(encoding="utf-8")
        for token in ("graph_route", "mantis.train.pretrain", "train/pretrain"):
            assert token not in text, f"{cfg.name} names {token}"


# ── grid-only flags are refused, not ignored ─────────────────────────────────────────────
@pytest.mark.parametrize("flag", sorted(DENSE_ARM_FLAGS))
def test_EVERY_dense_arm_flag_is_refused_not_ignored(flag: str) -> None:
    """Driven over the SET, not over a hand-listed copy of it: a flag added to
    `DENSE_ARM_FLAGS` without a refusal, or refused without a stated reason, fails here."""
    with pytest.raises(GraphPretrainError, match="does not read"):
        refuse_dense_arm_flags({flag: 1.0})
    assert DENSE_ARM_FLAGS[flag].strip(), f"{flag} carries no reason"


def test_store_true_flags_read_False_as_NOT_supplied() -> None:
    """`--freeze-trunk-entry` is `store_true`, so its unsupplied value is `False`, not None.
    Treating False as supplied would refuse every graph pretrain."""
    refuse_dense_arm_flags({"--freeze-trunk-entry": False, "--filters": None})


def test_a_flag_outside_the_declared_set_is_itself_an_error() -> None:
    """The set is the authority. A caller passing an undeclared flag would otherwise get a
    silent pass — the phantom-gate shape."""
    with pytest.raises(GraphPretrainError, match="not in DENSE_ARM_FLAGS"):
        refuse_dense_arm_flags({"--not-a-flag": 1})


def test_the_cli_hands_over_every_declared_dense_arm_flag() -> None:
    """The other half: a flag declared in the set but never passed by the CLI would be
    refused in theory and ignored in practice."""
    src = (_REPO / "src/mantis/train/pretrain/cli.py").read_text(encoding="utf-8")
    for flag in DENSE_ARM_FLAGS:
        assert f'"{flag}": args.' in src, f"the CLI never hands {flag} to the refusal"


def test_no_dense_arm_flags_is_the_clean_case() -> None:
    refuse_dense_arm_flags({f: None for f in DENSE_ARM_FLAGS})


def test_the_graph_arch_is_built_from_the_NESTED_config(tmp_path: Path, monkeypatch) -> None:
    """Production (`Trainer._derive_arch`, `train.orchestrator`) resolves the arch from
    `RunConfig.model_dump()`. The CLI's flat term dict is the DENSE arm's shape and would
    resolve a different arch the day a `gnn_*` width key is minted, so this route passes the
    nested mapping — asserted on the ARGUMENT, not on the resulting object."""
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
    """A caller that could omit `dense_arm_flags` would silently skip the refusal — the same
    shape as the ignored flags it exists to catch."""
    import inspect
    sig = inspect.signature(run_graph_pretrain)
    assert sig.parameters["dense_arm_flags"].default is inspect.Parameter.empty
