"""⊕ F-816-10 F6 — no construction site can omit the caps (dispatcher ruling D-1).

Written by ORACLE-WRITE **before** the feature exists.

THE BLOCKER THIS SUITE EXISTS FOR. The design's §2.4 claimed the bound "covers the eval child
automatically, per-process". That is FALSE as designed and REVIEW-design caught it:
`LocalInferenceEngine.__init__` (`src/mantis/selfplay/inference_local.py`) builds its
graph-route `InferenceServer` from a HAND-BUILT DICT LITERAL with no `RunConfig` at all — its
own comment says "this standalone caller has no RunConfig to draw from" — so under an eager
resolve it raises on every graph-representation eval run, permanently, and R119's "the
operator mints the value" story never reaches it.

D-1 OVERRULED the reviewer's own recommended fix (hardcode a value into that same literal) on
three grounds, and this suite pins all three:
  (a) a hardcoded cap is a SECOND AUTHORITY over one byte budget — the exact defect
      `MicrobatchCapsConfig`'s docstring refuses ("two independent keys would give two
      authorities over one byte budget"). FG6-07 is the census that makes it unwritable.
  (b) the eval child runs on `eval.worker_device: cuda` with its OWN allocator, so a wrong
      hardcoded value there is unbounded in practice on the very arm design §2.2 shows OOM'd.
  (c) an EXACT in-repo precedent exists at this seam: `RoundSpec` already carries
      resolver-produced frozen dataclasses across the eval process boundary
      (`ply_cap_adjudication`, `strength_floor`, documented there as "resolved ONCE in the
      parent by `mantis.config.resolve.*` and carried across the process seam as plain
      dataclasses"). FG6-08 pins that `fused_graph_caps` takes that shape.

Test sites pass an EXPLICIT value; grid sites pass `None` explicitly, never an omitted
argument (D-1's closing sentence).

The defect each row is the ONLY witness to:

- **FG6-01** — a graph server constructed with no bound at all, which is the state HEAD is in
  and the state this packet exists to make unreachable.
- **FG6-02** — the grid route acquiring a requirement it has no use for. The dense batch is a
  fixed-shape tensor already bounded by `inference_batch_size`; a caps requirement there would
  break four frozen grid coordinators for nothing.
- **FG6-03/04** — a DEFAULTED parameter on the standalone engine. A default is a value nobody
  minted, sitting on the one construction path that has no config to be minted from — R1's
  duplicated-default-authority class, and the eight inert `InferenceHParams` defaults already
  in that literal are the flag design §12.1 raises about it.
- **FG6-05** — a grid caller forced to invent a cap, or an omitted argument silently taking a
  default on the grid arm.
- **FG6-06** — a threaded value accepted and then ignored, with the site quietly reading a
  config that is not there.
- **FG6-07** — the hardcode D-1 refused, added later by someone who reads only the design.
- **FG6-08** — the caps stopping at the process boundary. The eval worker is a SECOND
  allocator on the same card that no in-process bound can see; if the spec does not survive
  `asdict -> JSON -> from_dict`, the child runs unbounded and the parent's oracle cannot tell.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

import _fused_graph_harness as H
from mantis.config.resolve.fused_graph_caps import (
    FusedGraphCapsSpec,
    MissingFusedGraphCapsError,
)
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.rounds import GateSpec, RoundSpec
from mantis.model import GnnArch, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.inference_server import InferenceServer

_CPU = torch.device("cpu")
_REPO = Path(__file__).resolve().parents[2]
_INFERENCE_LOCAL = _REPO / "src" / "mantis" / "selfplay" / "inference_local.py"


class _DummyBatcher:
    def close(self) -> None:
        return None


# ═══ FG6-01/02 — the server ══════════════════════════════════════════════════════════════
def test_fg6_01_a_graph_server_cannot_be_built_without_a_caps_value() -> None:
    """FG6-01 — a graph-route `InferenceServer` whose config carries no block and whose
    caller passes no explicit value RAISES, by name, at construction.

    This is the whole mechanism in one assertion: after this row holds, an unbounded fused
    graph forward is UNCONSTRUCTIBLE rather than merely discouraged."""
    with pytest.raises(MissingFusedGraphCapsError) as exc:
        InferenceServer(
            H.SentinelGraphNet(), _CPU, H.graph_cfg(omit_block=True),
            batcher=_DummyBatcher(), encoding_spec=H.GRAPH_SPEC,
        )
    assert "inference.fused_graph_caps" in str(exc.value)


def test_fg6_01_an_explicit_caps_argument_is_honoured_over_an_absent_block() -> None:
    """FG6-01 second limb — the THREADED arm (D-1). A caller with no `RunConfig` supplies the
    resolver-produced spec directly and the server uses it, without ever reading a config
    section that does not exist on that path."""
    caps = FusedGraphCapsSpec(max_fused_edges=4_500_000, max_fused_nodes=170_000)
    server = InferenceServer(
        H.SentinelGraphNet(), _CPU, H.graph_cfg(omit_block=True),
        batcher=_DummyBatcher(), encoding_spec=H.GRAPH_SPEC, fused_graph_caps=caps,
    )
    assert server.batch_timing_snapshot()["fusion"]["caps"] == {
        "max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}


def test_fg6_02_a_grid_server_is_unaffected_by_the_new_requirement() -> None:
    """FG6-02 — the grid route never reads the block and never needs one.

    The dense batch is a fixed-shape tensor already bounded by `inference_batch_size`, so
    there is no unbounded quantity there for a cap to bound — the same scoping argument
    `MicrobatchCapsConfig` makes for the train side. A grid construction that acquired this
    requirement would break the four frozen grid coordinators for nothing."""
    server = InferenceServer(
        torch.nn.Linear(1, 1), _CPU, H.grid_cfg(omit_block=True),
        batcher=_DummyBatcher(), encoding_spec=H.GRID_SPEC,
    )
    assert server.batch_timing_snapshot()["fusion"] is None


# ═══ FG6-03/04/05/06 — the standalone engine ═════════════════════════════════════════════
def test_fg6_03_the_local_engine_takes_a_required_keyword_only_caps_parameter() -> None:
    """FG6-03 — REQUIRED and KEYWORD-ONLY, with NO default.

    A default here is the R1 defect in its purest form: a value nobody minted, on the ONE
    construction path that has no config to mint it from. `encoding_spec` on this same class
    is the precedent and the reasoning is copied verbatim from its docstring — "a required
    parameter makes absent UNCONSTRUCTIBLE, which pyright catches before a worker ever
    spawns"."""
    params = inspect.signature(LocalInferenceEngine.__init__).parameters
    assert "fused_graph_caps" in params, (
        "`LocalInferenceEngine` takes no `fused_graph_caps` parameter — the eval-side graph "
        "server is still built from a hand-made dict with no bound (D-1 BLOCKER)")
    caps = params["fused_graph_caps"]
    assert caps.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"`fused_graph_caps` must be keyword-only; it is {caps.kind}")
    assert caps.default is inspect.Parameter.empty, (
        f"`fused_graph_caps` carries a default ({caps.default!r}). A default is a SECOND "
        "authority over one byte budget, on the path that has no config to be the first")


def test_fg6_04_a_graph_engine_cannot_be_built_without_the_caps() -> None:
    """FG6-04 — omitting the argument is a TypeError at the call, not a runtime surprise
    three hours into an eval round in a subprocess whose stderr nobody is reading."""
    net = build_net(GnnArch(in_dim=H.GRAPH_SPEC.node_feat_dim,
                            edge_dim=H.GRAPH_SPEC.edge_feat_dim, hidden=16, num_layers=1,
                            policy_hidden=16, value_hidden=16)).to(_CPU)
    net.eval()
    with pytest.raises(TypeError):
        LocalInferenceEngine(net, _CPU, encoding_spec=H.GRAPH_SPEC)  # type: ignore[call-arg]


def test_fg6_05_a_grid_engine_passes_none_explicitly_and_constructs() -> None:
    """FG6-05 — D-1's closing sentence, pinned: grid sites pass `None` EXPLICITLY, never an
    omitted argument. `None` here means "this route has no fused graph forward to bound", and
    it is written at the call site so a reader sees the decision rather than a silence."""
    engine = LocalInferenceEngine(torch.nn.Linear(1, 1), _CPU,
                                  encoding_spec=H.GRID_SPEC, fused_graph_caps=None,
 inference_batching=None, max_in_flight=0,
                                  amp_dtype="bf16")
    assert engine._graph_server is None, "a grid engine constructs no graph server"


def test_fg6_06_the_threaded_caps_reach_the_engines_own_server() -> None:
    """FG6-06 — accepted AND used. A parameter that is stored and never consulted satisfies
    FG6-03/04 completely and bounds nothing; this row is the difference."""
    net = build_net(GnnArch(in_dim=H.GRAPH_SPEC.node_feat_dim,
                            edge_dim=H.GRAPH_SPEC.edge_feat_dim, hidden=16, num_layers=1,
                            policy_hidden=16, value_hidden=16)).to(_CPU)
    net.eval()
    caps = FusedGraphCapsSpec(max_fused_edges=1_234_567, max_fused_nodes=89_012)
    engine = LocalInferenceEngine(net, _CPU, encoding_spec=H.GRAPH_SPEC,
                                  fused_graph_caps=caps,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8,
                                  amp_dtype="bf16")
    try:
        assert engine._graph_server is not None
        assert engine._graph_server.batch_timing_snapshot()["fusion"]["caps"] == {
            "max_fused_edges": 1_234_567, "max_fused_nodes": 89_012}, (
            "the threaded caps did not reach the engine's own server — the eval arm is still "
            "unbounded, which is the arm design §2.2 shows OOM'd")
    finally:
        engine.close()


def test_fg6_07_no_cap_value_is_hardcoded_at_the_standalone_construction_site() -> None:
    """FG6-07 — D-1's refusal, made unwritable. No INTEGER LITERAL may be assigned to either
    member anywhere in `inference_local.py`.

    An `ast` census and not a grep: the reviewer's proposed fix was a literal inside the same
    eight-key dict this class already carries, and the eight inert `InferenceHParams` defaults
    sitting beside it are exactly why that shape looks acceptable at a glance. A hardcoded cap
    is a SECOND AUTHORITY over one byte budget and it is invisible to every behavioural row —
    the run would be bounded, at a number nobody measured, on the arm with its own allocator.
    """
    tree = ast.parse(_INFERENCE_LOCAL.read_text(encoding="utf-8"))
    members = {"max_fused_edges", "max_fused_nodes"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (isinstance(key, ast.Constant) and key.value in members
                    and isinstance(value, ast.Constant)):
                raise AssertionError(
                    f"{key.value} is hardcoded to {value.value!r} in "
                    f"{_INFERENCE_LOCAL.relative_to(_REPO)}. D-1 refused this: the caps are "
                    "THREADED from the parent's resolver, never written at this site.")
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "fused_graph_caps":
                raise AssertionError(
                    "`fused_graph_caps` is written into the hand-built config dict at "
                    f"{_INFERENCE_LOCAL.relative_to(_REPO)}. The spec is threaded as a "
                    "resolver-produced dataclass (the `RoundSpec` precedent), not smuggled "
                    "back through a config-shaped literal.")


# ═══ FG6-08 — the process seam ═══════════════════════════════════════════════════════════
def _round_spec_base() -> dict:
    """The `RoundSpec` field set minus the two posture members and the new caps member —
    lifted from `tests/eval/test_eval_posture_inert.py`'s round-trip row so the two stay one
    shape."""
    return dict(
        round_id="r1", step=1, candidate_snapshot="c.pt", best_snapshot=None, best_step=None,
        encoding="gnn_axis_v1", worker_device="cpu",
        gate=GateSpec(stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=1, opening_book="b",
                      bootstrap_resamples=1, min_distinct_per_pair=1, seed_base=1,
                      run_gate=False),
        rung_jobs=[], random_floor_games=0, random_model_sims=1, sealbot_model_sims=1,
        kraken_model_sims=1, strix_model_sims=1, seed_base=1, round_timeout_sec=1.0,
        result_path="r.json", progress_path="p.txt", ladder_bootstrap_resamples=1,
        ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1,
        ply_cap_adjudication=None, strength_floor=None,
    )


def test_fg6_08_the_round_spec_carries_the_caps_across_the_process_seam() -> None:
    """FG6-08 — `asdict -> JSON -> from_dict` REHYDRATES the spec to its dataclass, not to a
    raw mapping.

    The distinction is the whole point of the precedent: `from_dict` explicitly rehydrates
    `ply_cap_adjudication` and `strength_floor` because a raw mapping crossing the seam gives
    the child attribute errors at the first read — which, on this member, would be at the
    moment it tries to bound a forward."""
    assert "fused_graph_caps" in RoundSpec.__dataclass_fields__, (
        "`RoundSpec` carries no `fused_graph_caps` field — the resolved caps stop at the "
        "process boundary and the eval child (its OWN allocator, `eval.worker_device: cuda`) "
        "runs unbounded")
    caps = FusedGraphCapsSpec(max_fused_edges=4_500_000, max_fused_nodes=170_000)
    spec = RoundSpec(**_round_spec_base(), fused_graph_caps=caps, leaf_batch_size=1, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
                     inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10))
    back = RoundSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert isinstance(back.fused_graph_caps, FusedGraphCapsSpec), (
        f"the caps came back as {type(back.fused_graph_caps).__name__}, not the dataclass — "
        "the child would raise on the first attribute read")
    assert back.fused_graph_caps == caps
    assert back == spec


def test_fg6_08_a_grid_round_carries_none_across_the_same_seam() -> None:
    """FG6-08 second limb — the `None` arm survives unchanged, exactly as the two posture
    members' disarmed arm does. A grid eval round has no fused graph forward to bound, and
    `None` must round-trip as `None` rather than as a rehydration failure."""
    spec = RoundSpec(**_round_spec_base(), fused_graph_caps=None, leaf_batch_size=1, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
                     inference_batching=None)
    back = RoundSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert back.fused_graph_caps is None
    assert back == spec
