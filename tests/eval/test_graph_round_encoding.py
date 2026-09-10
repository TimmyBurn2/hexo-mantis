# >300 justify (R8): one question — which encoding an eval round binds and decodes — asked
# once per arm over ONE shared round-spec builder, so the arms are only comparable because they
# are constructed identically.
"""Prove an eval round decodes the encoding the round DECLARED, not a constant.

Board geometry comes from `RoundSpec.encoding` while the inference decode came from a hardcoded
lookup: a mismatched pair can COMPLETE a round while silently dropping every flat policy index
past the narrower board's width.
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

# The repo's one opening book and the probe's parameter set verbatim, so every recorded sha is
# re-derivable: candidate seed 1, best seed 2, deploy_sims=2, seed_base=20260625.
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(enc_name: str, *, seed: int) -> torch.nn.Module:
    """Build a net whose dims come from `enc_name`'s registry row: a hard-coded size would agree
    with the wrong encoding by coincidence, the exact confusion these oracles detect."""
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
    """Derive the fused-forward memory bound this encoding's route needs; the graph route resolves
    it eagerly at construction, at a non-binding pair so no round here splits."""
    from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
    from mantis.encoding import lookup

    if lookup(enc_name).representation != "graph":
        return None
    return FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)


def _round_spec(
    tmp_path: Path, enc_name: str, *, rung_games: int = 0, floor_games: int = 0
) -> RoundSpec:
    """Build a real `RoundSpec` for `enc_name` with a 2-screen, 2-confirm gate block."""
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
    """Drive the round from openings derived at `enc_name`'s own geometry.

    The repo's only book is minted at radius 6 and 292 of its 512 openings (57.03%, measured) need
    `legal_move_radius >= 6` to replay, so a narrower encoding would start from unreachable
    positions; deriving them also drops a dependency on which one `seed_base` selected.
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
    """Record `(spec.name, is_graph)` in construction order for every engine the round builds,
    through a real subclass so what is recorded is what production bound."""
    bound: list[tuple[str, bool]] = []

    class _RecordingEngine(LocalInferenceEngine):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            bound.append((self.encoding_spec.name, self._is_graph))

    monkeypatch.setattr(worker, "LocalInferenceEngine", _RecordingEngine)
    return bound


def test_graph_eval_round_runs_end_to_end(tmp_path: Path) -> None:
    """Prove a graph round completes on the real worker path, exercising both engine construction
    sites: the best anchor via the gate block, and the candidate."""
    result = worker.run_round(_round_spec(tmp_path, "gnn_axis_v1"))

    assert result["gate"] is not None, "run_gate=True with a best snapshot must play a gate"
    assert result["gate"]["n_screen"] == 2


def test_both_engines_bind_the_declared_graph_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove every engine the round builds binds the declared spec: the row above proves the
    round runs, this proves it runs on the right decode."""
    bound = _recorded_bindings(monkeypatch)

    worker.run_round(_round_spec(tmp_path, "gnn_axis_v1"))

    assert len(bound) == 2, f"expected the candidate and the best anchor, got {bound}"
    assert [name for name, _is_graph in bound] == ["gnn_axis_v1", "gnn_axis_v1"]
    assert all(is_graph for _name, is_graph in bound), f"graph dispatch not taken: {bound}"


def test_the_decode_capability_set_is_closed_over_the_registry() -> None:
    """Pin the guard's reach over the live registry to a literal written here: deriving it from
    the constant under test would let a widening flip both sides together and stay green."""
    from mantis.eval.errors import EvalDecodeUnsupportedError

    def _guard_fires(spec) -> bool:
        try:
            worker._assert_decode_implements_declared_pooling(spec)
        except EvalDecodeUnsupportedError:
            return True
        return False

    # Empty because every registered row now declares `policy_pool="none"`; frozen as a
    # literal all the same, so a future unimplemented pool or a widened set reds this.
    assert {spec.name for spec in all_specs() if _guard_fires(spec)} == set()
    assert worker._DECODE_IMPLEMENTED_POLICY_POOLS == frozenset({"none", "scatter_max"})
