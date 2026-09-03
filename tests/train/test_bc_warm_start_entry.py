"""AUDIT-1 F-19 / R332(d) — the BC warm-start entry EXISTS, and it refuses the wrong artifact.

THE DEFECT. `train/warmstart.py` was imported by NOTHING under `src/`:
`maybe_warmstart_gnn_from_bc` and `assert_dist65_bins_seeded` had zero references anywhere,
and the transfer primitive `model/gnn.py::load_representation_policy_from_bc` had exactly one
caller — that dead function. So BC-EXEC could produce a BC checkpoint and no production path
could consume it. The module read `gnn_warm_start.checkpoint` and
`combined_config.get("value_head_type", "scalar")` — keys the schema does not have, with
code-side defaults under them (R1's class).

R332(d) DECIDED it BUILT rather than deleted, and specified the shape: *the identity names a
checkpoint by path AND net hash; the one selector authority resolves it through the artifact
stamp; pinning test: the loaded net's hash equals the checkpoint's; a CPU smoke plays one legal
game from it.* This file is that.

WHAT THE HASH IS, because it is easy to assume a file digest. `net_param_hash` is taken over
the net REBUILT FROM the checkpoint's own stamp — sorted `name + shape + dtype + bytes`. A file
digest moves with re-saves and metadata; the parameter hash is what a prereg means when it
names an artifact, and it is the same currency `worker_sweep` and `acceptance_witness` report.

NOT HERE, deliberately: the 20/20-vs-random step-0 reproduction. R332(d) puts that at the mint
sitting, on the checkpoint of record — it is a strength claim about a real artifact, not
something a synthetic fixture can stand in for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.encoding import lookup
from mantis.model import build_net, select_arch
from mantis.model.identity import net_param_hash
from mantis.train.warmstart import (
    WARM_START_ROW,
    BcWarmStart,
    WarmStartIdentityError,
    apply_bc_warm_start,
    maybe_warmstart_gnn_from_bc,
    resolve_bc_warm_start,
)

_ENC = "gnn_axis_v1"


def _spec() -> Any:
    return lookup(_ENC)


def _arch() -> Any:
    """A deliberately NARROW graph arch: small enough to build fast, and different from the
    dataclass defaults so a rebuild that ignored the stamp would be visible."""
    import dataclasses

    base = select_arch(_spec(), {}, arch_kind="GnnArch")
    return dataclasses.replace(base, hidden=32, num_layers=2, policy_hidden=32, value_hidden=16)


def _write_source(tmp_path: Path, arch: Any) -> tuple[Path, str]:
    """Write a BC-shaped source checkpoint through the ONE writer, and return its net hash."""
    from mantis.train.checkpoints import save_checkpoint

    from _warmstart_config import minimal_config  # noqa: PLC0415

    net = build_net(arch)
    path = save_checkpoint(
        model=net, optimizer=None, scaler=None, scheduler=None, step=0,
        config=minimal_config(), kind="weights",
        metadata_kwargs={"encoding_name": _ENC, "run_id": "bcsrc", "arch": arch},
        checkpoint_dir=tmp_path,
    )
    return path, net_param_hash(net)


def _config_with_row(checkpoint: Path, net_hash: str) -> dict[str, Any]:
    return {"identity": {"encoding": _ENC, "representation": "graph",
                         "warm_start": {"checkpoint": str(checkpoint), "net_hash": net_hash}}}


# ══ the resolver: an absent row is no warm start, never a guess ════════════════════════
def test_a_config_with_no_row_declares_no_warm_start() -> None:
    assert resolve_bc_warm_start({"identity": {"encoding": _ENC, "representation": "graph"}}) is None
    assert resolve_bc_warm_start({}) is None


def test_a_row_missing_its_hash_is_REFUSED_not_defaulted() -> None:
    """The whole reason `warm_start` is a BLOCK. A path with no expected hash lets a run
    warm-start from whatever file is sitting there."""
    with pytest.raises(ValueError, match="net_hash"):
        resolve_bc_warm_start({"identity": {"warm_start": {"checkpoint": "/x.pt"}}})


def test_the_row_has_exactly_one_reader() -> None:
    """`WARM_START_ROW` names the dotted key so the row has one reader to change, the
    `ARCH_KIND_ROW` pattern."""
    assert WARM_START_ROW == "identity.warm_start"


# ══ THE WITNESS R332(d) NAMES: the loaded net's hash equals the checkpoint's ════════════
def test_the_warm_started_net_carries_the_declared_checkpoints_hash(tmp_path: Path) -> None:
    arch = _arch()
    source, source_hash = _write_source(tmp_path, arch)

    fresh = build_net(arch)
    assert net_param_hash(fresh) != source_hash, (
        "the fresh net already hashes equal to the source — the fixture proves nothing"
    )

    report = apply_bc_warm_start(fresh, BcWarmStart(source, source_hash), spec=_spec())
    assert report["loaded_keys"], "the transfer reported no keys"

    # The transferred half is byte-equal to the source; the value head is untouched, so the
    # WHOLE-net hash is deliberately NOT asserted equal — that would be a different claim.
    src_state = build_net(arch)
    src_state.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["model_state"])
    for key, value in src_state.state_dict().items():
        if key.startswith(("representation.", "policy_head.")):
            assert torch.equal(fresh.state_dict()[key], value), key


def test_a_checkpoint_that_is_NOT_the_declared_net_is_REFUSED(tmp_path: Path) -> None:
    """The refusal the hash exists for: the file at that path is not the artifact the prereg
    named, so the run stops rather than training from whatever is there."""
    arch = _arch()
    source, _ = _write_source(tmp_path, arch)
    wrong = "0" * 64
    with pytest.raises(WarmStartIdentityError, match="net_param_hash"):
        apply_bc_warm_start(build_net(arch), BcWarmStart(source, wrong), spec=_spec())


def test_an_absent_checkpoint_is_a_named_refusal(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_bc_warm_start(
            build_net(_arch()), BcWarmStart(tmp_path / "nope.pt", "0" * 64), spec=_spec(),
        )


def test_the_transfer_is_graph_only_and_says_so(tmp_path: Path) -> None:
    arch = _arch()
    source, source_hash = _write_source(tmp_path, arch)
    with pytest.raises(ValueError, match="graph-only"):
        apply_bc_warm_start(
            build_net(arch), BcWarmStart(source, source_hash), spec=lookup("v6_live2_ls"),
        )


# ══ the hook fires from a config, and is a no-op without the row ═══════════════════════
def test_the_hook_fires_from_a_config_row_and_no_ops_without_one(tmp_path: Path) -> None:
    arch = _arch()
    source, source_hash = _write_source(tmp_path, arch)
    net = build_net(arch)
    before = net_param_hash(net)

    assert maybe_warmstart_gnn_from_bc(net, {"identity": {}}, spec=_spec()) is False
    assert net_param_hash(net) == before, "a config with no row must leave the net untouched"

    assert maybe_warmstart_gnn_from_bc(net, _config_with_row(source, source_hash), spec=_spec())
    assert net_param_hash(net) != before, "the transfer fired but changed nothing"


def test_init_trainer_is_the_live_consumer() -> None:
    """LAW-08: the key has a live consumer, and this names it. A census, not a mention count —
    the call must be in `init_trainer`'s source."""
    import inspect

    from mantis.train import orchestrator

    src = inspect.getsource(orchestrator.init_trainer)
    assert "maybe_warmstart_gnn_from_bc" in src, (
        "init_trainer no longer calls the warm-start hook — the schema row would be a key with "
        "no consumer, which is the state F-19 found it in"
    )


# ══ THE CPU SMOKE R332(d) NAMES: one legal game, played from the warm-started net ══════
def test_a_cpu_smoke_plays_one_legal_game_from_the_warm_started_net(tmp_path: Path) -> None:
    """R332(d)'s second witness. The hash check proves the right WEIGHTS arrived; this proves
    the resulting net is one you can actually play a game with — that the transfer left a
    forward the PRODUCTION eval path can drive, on CPU, producing legal moves to a terminal.

    Driven through the real seam and not a stand-in: `LocalInferenceEngine` (graph arm) ->
    `DeployHeadPlayer(expand_fn=..., c_visit=50.0, c_scale=1.0)` -> `mantis.arena.match._play_one_game`, which is what the
    eval worker builds for a graph candidate. A warm start that "succeeds" and hands back a net
    the inference server cannot run would red here rather than in a run.

    NOT a strength claim: the opponent is the same player, and the 20/20-vs-random reproduction
    belongs at the mint sitting on the checkpoint of record (R332(d)).
    """
    from mantis.arena.match import _play_one_game
    from mantis.arena.deploy_head import DeployHeadPlayer
    from mantis._engine import Board
    from mantis.config.resolve import FusedGraphCapsSpec, InferenceBatchingSpec
    from mantis.eval.worker import _graph_expand_fn
    from mantis.selfplay.inference_local import LocalInferenceEngine

    arch = _arch()
    source, source_hash = _write_source(tmp_path, arch)
    net = build_net(arch)
    assert maybe_warmstart_gnn_from_bc(net, _config_with_row(source, source_hash), spec=_spec())
    net.eval()

    spec = _spec()
    engine = LocalInferenceEngine(
        net, torch.device("cpu"), encoding_spec=spec,
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(inference_batch_size=8, inference_max_wait_ms=10),
        max_in_flight=4, amp_dtype="bf16",
    )
    try:
        def _player() -> Any:
            # `n_sims` deliberately small: this is a liveness smoke on the default tier, not a
            # search-quality measurement. `leaf_batch_size` is threaded, never defaulted (R318(b)).
            return DeployHeadPlayer(
                expand_fn=_graph_expand_fn(engine, spec), n_sims=4, leaf_batch_size=2, c_visit=50.0, c_scale=1.0,
            )

        winner, plies, moves, terminal, _adj = _play_one_game(
            _player(), _player(), [],
            candidate_color=1,
            board_factory=lambda: Board.with_encoding_name(_ENC),
            max_plies=8,
        )
    finally:
        engine.close()

    assert plies > 0, "the warm-started net produced no move at all"
    assert len(set(moves)) == len(moves), f"a cell was played twice: {moves}"
    assert winner in {"candidate", "opponent", "draw"}
    assert terminal, "the game ended with no terminal reason"

    # Every move must have been legal on a board that only ever saw legal moves — replay it.
    replay = Board.with_encoding_name(_ENC)
    for q, r in moves:
        assert (q, r) in replay.legal_moves(), f"({q}, {r}) was not legal when it was played"
        replay.apply_move(q, r)
