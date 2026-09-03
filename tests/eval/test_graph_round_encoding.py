"""⊕ WP12-R Phases B+C — an eval round must decode the encoding the round DECLARED.

Oracle-first (PREREG WP12-R §1), byte-frozen through IMPL. At HEAD `mantis.eval.worker`
constructs `LocalInferenceEngine` with no spec at both sites (`:78`, `:193`), so
`inference_local.py:70-71` binds `lookup("v6")` for EVERY declared encoding: board geometry
comes from `RoundSpec.encoding` while the inference decode comes from a constant, and the
two are never shown to agree. All four registered encodings were driven end to end through
the real `run_round` at HEAD (PREREG §3): `v6` correct; `v6w25` COMPLETES a round while
decoding a 362-wide policy for a 626-action board (silently wrong — every flat index >= 361
is dropped at `inference_local.py:200-201`); `v6_live2_ls` dies in the conv channel check;
`gnn_axis_v1` dies because the dense arm calls `GnnNet.forward`, which R138 forbids adding.

Pre-registered HEAD verdicts (PREREG §1). These are RED at RUN, not at collection: every
module they import exists at HEAD — this is a behaviour defect, not a missing port.

    RED   test_graph_eval_round_runs_end_to_end             NotImplementedError ... forward
    RED   test_both_engines_bind_the_declared_graph_spec    same raise; bindings are "v6"
    GREEN test_dense_v6_round_is_byte_stable_and_deterministic     R20-protected grid arm
    GREEN test_declared_grid_encoding_is_bound_and_decodes[v6]     fix is a no-op here
    RED   test_declared_grid_encoding_is_bound_and_decodes[v6w25]  completes, binds "v6"
    RED   test_no_drop_pooling_encoding_is_refused_with_a_named_error   conv 4-vs-8 channels
    RED   test_the_decode_capability_set_is_closed_over_the_registry    guard absent

The rounds here run IN-PROCESS. Production spawns a child (`eval/pipeline.py:364`); that
seam is already covered by the integration-tier `tests/eval/test_round_end_to_end.py`, and
the graph `InferenceServer` is a daemon thread (`inference_server.py:66`), so an in-process
round cannot hang the suite at exit. What is covered NOWHERE is the decode — that is what
this file adds, and it belongs in the tier CI runs first because it guards a mint blocker.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import all_specs, lookup
from mantis.eval import worker
from mantis.eval.rounds import GateSpec, RoundSpec, RungJob
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import CnnArch, GnnArch, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine

# The ONE opening book in the repo, and the probe's parameter set (PREREG §5) verbatim, so
# every sha recorded there is re-derivable from these fixtures: candidate seed 1 / best
# seed 2, deploy_sims=2, seed_base=20260625, minimal width and depth.
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(enc_name: str, *, seed: int) -> torch.nn.Module:
    """A registry-TRUE net for `enc_name`: its dims come from the spec, never a literal.

    A net sized from a hard-coded board_size/in_channels would agree with the wrong
    encoding by coincidence — which is precisely the confusion these oracles exist to
    detect — so the arch is derived from the same registry row the round declares.
    """
    spec = lookup(enc_name)
    torch.manual_seed(seed)
    arch: CnnArch | GnnArch
    if spec.representation == "graph":
        arch = GnnArch(
            in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
            hidden=16, num_layers=1, policy_hidden=16, value_hidden=16,
        )
    else:
        arch = CnnArch(
            board_size=spec.board_size, in_channels=spec.n_planes, filters=8, res_blocks=1,
        )
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _caps_for(enc_name: str):
    """The fused-forward memory bound this encoding's route needs (F-816-10 D-1).

    Derived from the encoding, not chosen per call site: the graph route resolves the bound
    EAGERLY when its `InferenceServer` is constructed, and the grid route never reads it. The
    value is the template's NON-BINDING-BY-CONSTRUCTION pair, so no round here splits.
    """
    from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
    from mantis.encoding import lookup

    if lookup(enc_name).representation != "graph":
        return None
    return FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)


def _round_spec(
    tmp_path: Path, enc_name: str, *, rung_games: int = 0, floor_games: int = 0
) -> RoundSpec:
    """A real `RoundSpec` for `enc_name` with a gate block of 2 screen + 2 confirm games."""
    candidate = tmp_path / f"candidate_{enc_name}.pt"
    best = tmp_path / f"best_{enc_name}.pt"
    write_model_snapshot(_net(enc_name, seed=1), candidate)
    write_model_snapshot(_net(enc_name, seed=2), best)

    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    rung_jobs = [
        RungJob(
            name="random_rung", bot="random", variant="raw", depth=None, opponent_sims=None,
            opening_book=_BOOK, deploy_matched=True, games=rung_games,
        )
    ]
    return RoundSpec(
        leaf_batch_size=1, amp_dtype="bf16", leaf_build_threads=1,
        round_id=f"oracle_{enc_name}", step=1, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=enc_name, worker_device="cpu",
        gate=gate, rung_jobs=rung_jobs, random_floor_games=floor_games,
        random_model_sims=2, sealbot_model_sims=2, kraken_model_sims=2, strix_model_sims=2,
        seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"), progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=_caps_for(enc_name),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )


def _recorded_bindings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Record `(spec.name, is_graph)` for every engine the round ACTUALLY constructs.

    A real subclass that delegates to the real `__init__` — never a stub — so what is
    recorded is what the production engine bound, including the representation dispatch it
    derived from that spec. The returned list is filled in construction order.
    """
    bound: list[tuple[str, bool]] = []

    class _RecordingEngine(LocalInferenceEngine):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            bound.append((self.encoding_spec.name, self._is_graph))

    monkeypatch.setattr(worker, "LocalInferenceEngine", _RecordingEngine)
    return bound


# ── ⊕ O-1 ─────────────────────────────────────────────────────────────────────────────
def test_graph_eval_round_runs_end_to_end(tmp_path: Path) -> None:
    """A `gnn_axis_v1` round completes — the run5 encoding, on the real worker path.

    HEAD: RED with `NotImplementedError: Module [GnnNet] is missing the required "forward"
    function` — the dense arm, reached because the engine bound the dense default, calls
    `model(...)` on a graph net. Exercises BOTH construction sites: `:78` (the best anchor,
    via the gate block) and `:193` (the candidate).
    """
    result = worker.run_round(_round_spec(tmp_path, "gnn_axis_v1"))

    assert result["gate"] is not None, "run_gate=True with a best snapshot must play a gate"
    assert result["gate"]["n_screen"] == 2


# ── ⊕ O-2 ─────────────────────────────────────────────────────────────────────────────
def test_both_engines_bind_the_declared_graph_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every engine the round builds binds the DECLARED spec, not a constant.

    O-1 proves the round runs; this proves it runs on the right decode. At HEAD the
    recorded bindings are `[('v6', False), ('v6', False)]` for a `gnn_axis_v1` round
    (measured), which is the whole defect: the declared encoding reaches `board_factory`
    and the `RegimeKey` stamps but never the inference decode.
    """
    bound = _recorded_bindings(monkeypatch)

    worker.run_round(_round_spec(tmp_path, "gnn_axis_v1"))

    assert len(bound) == 2, f"expected the candidate and the best anchor, got {bound}"
    assert [name for name, _is_graph in bound] == ["gnn_axis_v1", "gnn_axis_v1"]
    assert all(is_graph for _name, is_graph in bound), f"graph dispatch not taken: {bound}"


# ── ⊕ᶜ O-3 (R20-protected grid arm) ───────────────────────────────────────────────────
def test_dense_v6_round_is_byte_stable_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`v6` is an R20-protected grid encoding: this card must not move it.

    NAMING CORRECTED (ADJ-WP12R-18, under R148). This docstring previously called `v6`
    "the operator-locked dense control arm". R148 rules that the dense control arm IS
    `v6_live2_ls`, consistent with R117 — so the TITLE was wrong here, at four sites in
    this file, which were the only four occurrences of the phrase in the repo. What the
    test actually pins is unchanged and remains correct: `v6` round determinism and
    byte-stability under R20, which protects `representation="grid"` as a CLASS and names
    no encoding. A naming defect (R73 name-truth), never a behaviour defect — the
    assertions below are byte-identical to the shipped ones.

    The committed assertion is the platform-independent property — two runs of the SAME
    spec in the same process return equal result dicts, and both rounds bind `v6` down the
    dense arm. A hard-coded golden sha would pin the repo to one BLAS/torch build; the
    this-box sha (`4d8d6321…`, identical at HEAD and under the fix) is corroborating
    evidence in PREREG §3, reproducible from the preserved probe, not the gate.
    """
    bound = _recorded_bindings(monkeypatch)
    spec = _round_spec(tmp_path, "v6", rung_games=2, floor_games=2)

    first = worker.run_round(spec)
    second = worker.run_round(spec)

    # KeyError, not a defaulted pop: the worker's own contract says the key is always there.
    first.pop("worker_pid")
    second.pop("worker_pid")
    assert first == second, "the R20-protected v6 grid round is not deterministic in-process"
    assert bound == [("v6", False)] * 4, f"the dense round did not bind v6 dense: {bound}"


# ── ⊕ O-9 ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("enc_name", ["v6", "v6w25"])
def test_declared_grid_encoding_is_bound_and_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enc_name: str
) -> None:
    """Both grid encodings this decode supports must bind THEMSELVES, not `v6`.

    The asymmetry is the point. `[v6]` is GREEN at HEAD (the fix is a no-op there by
    construction); `[v6w25]` is RED at HEAD *while completing the round* — it binds `v6`,
    so a 362-wide policy is decoded for a 626-action board and every action index >= 361 is
    discarded. A regression that re-broke non-`v6` grid decoding while leaving `v6` and the
    graph arm green would pass every other oracle in this file.
    """
    bound = _recorded_bindings(monkeypatch)

    result = worker.run_round(_round_spec(tmp_path, enc_name))

    assert result["gate"]["n_screen"] == 2
    assert {name for name, _is_graph in bound} == {enc_name}
    assert not any(is_graph for _name, is_graph in bound), f"grid spec took graph arm: {bound}"


# ── ⊕ O-8 (the §c.7 guard) ────────────────────────────────────────────────────────────
def test_no_drop_pooling_encoding_is_refused_with_a_named_error(tmp_path: Path) -> None:
    """A declared no-drop pooling this decode cannot honour is REFUSED, by name.

    `v6_live2_ls` declares `policy_pool="legal_set_scatter_max"` — the no-drop legal-set
    pool — while the eval decode entrance (`DeployHeadPlayer` -> `engine.infer` ->
    `infer_batch`) scatter-MAXes and drops off-window cells. Threading the spec without a
    guard would turn HEAD's loud crash into a silent, plausible, wrongly-pooled eval
    result; the guard makes it a named refusal instead.

    The catch is on `RuntimeError` — `EvalDecodeUnsupportedError`'s own base — and the
    exact type is asserted afterwards, deliberately: at HEAD the error class does not exist
    yet, and naming it in the `raises` line would make this oracle fail on an ImportError
    instead of on the pre-registered mechanism. This form fails at HEAD *showing* the
    pre-registered `RuntimeError: Given groups=1, weight of size [8, 4, 3, 3], expected
    input[2, 8, 19, 19] to have 4 channels, but got 8 channels`, and `type(...) is` is
    strictly tighter than `pytest.raises` on the class (no subclass may satisfy it).
    """
    spec = _round_spec(tmp_path, "v6_live2_ls")

    with pytest.raises(RuntimeError) as excinfo:
        worker.run_round(spec)

    message = str(excinfo.value)
    assert "v6_live2_ls" in message, message
    assert "legal_set_scatter_max" in message, message

    from mantis.eval.errors import EvalDecodeUnsupportedError

    assert type(excinfo.value) is EvalDecodeUnsupportedError


# ── ⊕ O-8b ────────────────────────────────────────────────────────────────────────────
def test_the_decode_capability_set_is_closed_over_the_registry() -> None:
    """The guard's REACH over the live registry, pinned to a literal written here.

    The expected set is frozen in this test and is never derived from the constant under
    test: an assertion of the form "the helper fires iff the pool is outside the helper's
    own constant" is not an oracle — widening the constant flips both sides together and it
    stays green (measured, PREREG §5b). A future registry row declaring an unimplemented
    pool reds this, and so does any widening of the capability set.
    """
    from mantis.eval.errors import EvalDecodeUnsupportedError

    def _guard_fires(spec) -> bool:
        try:
            worker._assert_decode_implements_declared_pooling(spec)
        except EvalDecodeUnsupportedError:
            return True
        return False

    assert {spec.name for spec in all_specs() if _guard_fires(spec)} == {"v6_live2_ls"}
    assert worker._DECODE_IMPLEMENTED_POLICY_POOLS == frozenset({"none", "scatter_max"})
