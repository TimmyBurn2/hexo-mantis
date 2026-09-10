# >300 justify (R8): one question — which encoding an eval round binds and decodes — asked
# once per arm over ONE shared round-spec builder. The grid arms, the graph arm and the
# refusal arm are only comparable because they are constructed identically; split across
# files, a builder edit could move one arm's geometry while every file stayed green.
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
from mantis.model import GnnArch, build_net
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
    arch = GnnArch(
        in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
        hidden=16, num_layers=1, policy_hidden=16, value_hidden=16,
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
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1, concurrency=1,
        round_index=0, round_id=f"oracle_{enc_name}", step=1, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=enc_name, worker_device="cpu",
        gate=gate, rung_jobs=rung_jobs, random_floor_games=floor_games,
        random_model_sims=2, sealbot_model_sims=2, seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"), progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        game_record=None,
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=_caps_for(enc_name),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )


def _openings_at(enc_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the round from openings DERIVED at `enc_name`'s own geometry.

    `book_v1_s20260625_p4` is the repo's only book and is minted against `gnn_axis_v1`
    (`tools/mint_opening_book.py`, radius 6). Measured over its 512 openings, 292 of them
    (57.03%) require `legal_move_radius >= 6` to replay — so under a radius-5 grid encoding
    like `v6` the round starts from positions the rules cannot reach. Nothing detected that
    until R345(b)(2) put a legality boundary in the match loop; before it, those openings
    were simply played.

    This suite's subject is which SPEC the round binds and whether the round is
    deterministic, not which openings it draws, so the openings are derived here rather than
    drawn from a book whose geometry does not match. Deriving them also removes a silent
    dependency on WHICH single opening `seed_base` happened to select. Whether the repo
    should ALSO ship a radius-5 book is an artifact decision on the architect's ledger, not
    this suite's to make.
    """
    from mantis._engine import Board
    from mantis.arena.books import Opening

    def _derived(book_id: str, *, n_pairs: int, seed_base: int, round_index: int,
                 **_kw) -> list[Opening]:
        openings: list[Opening] = []
        for i in range(max(int(n_pairs), 1)):
            board = Board.with_encoding_name(enc_name)
            moves: list[tuple[int, int]] = []
            for ply in range(4):
                legal = sorted(board.legal_moves())
                move = legal[(seed_base + round_index + i * 7 + ply * 3) % len(legal)]
                board.apply_move(*move)
                moves.append(move)
            openings.append(Opening(opening_id=f"{book_id}-derived-{i}", moves=moves))
        return openings

    monkeypatch.setattr(worker, "round_openings", _derived)


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

    # EMPTY since R346(f): the three grid rows carried the unimplemented pools, and every
    # registered row now declares `policy_pool="none"`. Frozen as a literal all the same —
    # a future row declaring an unimplemented pool reds this, and so does any widening of
    # the capability set below.
    assert {spec.name for spec in all_specs() if _guard_fires(spec)} == set()
    assert worker._DECODE_IMPLEMENTED_POLICY_POOLS == frozenset({"none", "scatter_max"})
