# >300 justify (R8): one row, and its arming question cannot be split. The fixtures that
# build a match, the G=1-versus-factory byte-equality arm and the which-block-carries-it arm
# all read the SAME constructed round; separating them would let two files disagree about
# what an unarmed default does, which is the only thing this row asserts.
"""R339(b) — `eval.concurrency`: the row exists, reaches ONE block, and is inert at its default.

WHAT THE ROW IS FOR. Run6's first real eval round ran 50 min 22 s to 93 games without
finishing, against its own `round_timeout_sec 3600`. R339(b) rules that the gate GEOMETRY does
not move and the timeout is not lengthened — it is the stall watchdog's limit — so the lever is
the concurrency capability tranche-3 landed UNARMED, raised to a config key. The VALUE is picked
on the box by measurement; nothing here proposes one.

WHY THE SCOPE IS THE CLAIM AND NOT AN IMPLEMENTATION DETAIL. The gate block is ~93 % of the
round's wall, so it is the only block worth arming; the other three are ~200 s of a ~7 000 s
round and each pays a real price for threading. The floor probe is a LAW-07 gate input, and the
rung battery is LAW-04's Elo channel whose per-game trajectory identity is unprovable under
threading on CUDA (`index_add_`). "Only the gate block is armed" is therefore a property to
PIN, not a coincidence of where the kwarg was typed — hence the census below counts every
`play_paired_match` call in a real round and asserts what each one carries.

THE DEFAULT IS THE BEHAVIOUR, NOT A PLACEHOLDER. `concurrency=1` takes the identical serial
branch on the identical objects, so an absent row and a minted `1` are the SAME round rather
than merely both legal — which is what makes the row safe to land before its value is known.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError

from mantis.arena.match import DEFAULT_MAX_PLIES, play_paired_match
from mantis.arena.regime import RegimeKey
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.rounds import EVAL_CONCURRENCY_ROW, GateSpec, RoundSpec
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import GnnArch, build_net

#: `book_v1_s20260625_p4` is minted against `gnn_axis_v1` and 292 of its 512 openings need
#: radius >= 6 to replay, so the round's encoding has to cover that.
_ENC = "gnn_axis_v1"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625
_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "run6.yaml"
#: run6 MINTS the row now — R339(b) said the value is picked on the box, and it was — so the
#: absent-row parity arm needs a file that still omits it. The smoke profile is that file.
_UNMINTED_CONFIG = _CONFIG.with_name("smoke_preflight_armed.yaml")


# ── 1. the schema row ──────────────────────────────────────────────────────────────────
def _raw(source: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((source or _CONFIG).read_text(encoding="utf-8"))


def test_the_row_name_is_the_key_path_the_schema_actually_carries() -> None:
    """`EVAL_CONCURRENCY_ROW` is what the exemption list and the contract doc cite, so a rename
    that moved the field without moving the constant would leave both citing a dead path."""
    section, _, field = EVAL_CONCURRENCY_ROW.partition(".")
    assert section == "eval"
    assert field in RunConfig.model_fields["eval"].annotation.model_fields


def test_an_absent_row_and_a_minted_one_are_the_SAME_config() -> None:
    """The parity witness at the config layer: byte-identical dumps, not merely both valid.

    Every config that does not mint the row inherits the default, and a default that produced
    a DIFFERENT config from the minted `1` would make the row's arrival a silent behaviour
    change on every file that never mentions it. run6 has since minted its own value, so the
    subject here is the config that has not — the arm is about the DEFAULT, and a file
    carrying an armed 8 cannot witness it.
    """
    raw = _raw(_UNMINTED_CONFIG)
    assert "concurrency" not in raw["eval"], (
        f"{_UNMINTED_CONFIG.name} has grown the key — this row's premise is that some "
        "committed config still omits it, so the default has a live subject"
    )
    minted = copy.deepcopy(raw)
    minted["eval"]["concurrency"] = 1

    absent_cfg = RunConfig.model_validate(raw)
    minted_cfg = RunConfig.model_validate(minted)
    assert absent_cfg.eval.concurrency == 1
    assert absent_cfg.model_dump() == minted_cfg.model_dump(), (
        "an absent row and a minted `1` produced different configs — the default is then a "
        "value with its own meaning, not the serial behaviour that was already there"
    )


@pytest.mark.parametrize("bad", [0, -1, -8])
def test_the_row_is_refused_below_one(bad: int) -> None:
    """`ge=1`. Zero games in flight is not a slower round, it is a round that plays nothing."""
    raw = _raw()
    raw["eval"]["concurrency"] = bad
    with pytest.raises(ValidationError, match="concurrency"):
        RunConfig.model_validate(raw)


def test_a_planted_neighbour_key_is_still_refused() -> None:
    """R1's `extra="forbid"` survives the addition (the packet's named plant).

    `games_in_flight` is the name the capability was nearly given and the one a reader is most
    likely to reach for; under `extra="forbid"` it must be a boot error rather than a knob that
    looks armed and reaches nothing.
    """
    raw = _raw()
    raw["eval"]["games_in_flight"] = 4
    with pytest.raises(ValidationError, match="games_in_flight"):
        RunConfig.model_validate(raw)


# ── 2. the serial arm is byte-exact with the factory present ───────────────────────────
class _CountingBot:
    """Stateful by design: a shared instance across concurrent games would interleave two
    games' counters, which is what `player_factory` exists to prevent."""

    def __init__(self, stride: int) -> None:
        self._stride, self._i = stride, 0

    def new_game(self) -> None:
        self._i = 0

    def select_move(self, board):
        legal = board.legal_moves()
        mv = legal[(self._i * self._stride) % len(legal)]
        self._i += 1
        return mv


class _Opening:
    """One four-ply opening DERIVED from the engine's legal set, not hand-written.

    The coordinates this replaces ran off the legal set at `op3`: an empty board's legal
    region is the 5x5 block around the origin whatever the radius, so `(3, 0)` was never a
    playable first move. R345(b)(2)'s legality boundary is what surfaced it.
    """

    def __init__(self, i: int) -> None:
        from mantis._engine import Board

        self.opening_id = f"op{i}"
        board = Board.with_encoding_name(_ENC)
        moves: list[tuple[int, int]] = []
        for ply in range(4):
            legal = sorted(board.legal_moves())
            move = legal[(i * 7 + ply * 3) % len(legal)]
            board.apply_move(*move)
            moves.append(move)
        self.moves = moves


def _play(*, with_factory: bool):
    from mantis._engine import Board

    def _pair():
        return (_CountingBot(3), _CountingBot(5))

    cand, opp = _pair()
    return play_paired_match(
        cand, opp, [_Opening(i) for i in range(4)],
        regime_key=RegimeKey(
            bot="candidate", variant="test", model_sims=1, opponent_spec="fixed",
            opening_book="test_book", deploy_matched=False, encoding=_ENC,
        ),
        board_factory=lambda: Board.with_encoding_name(_ENC),
        max_plies=DEFAULT_MAX_PLIES, record_sink=None, adjudicator=None,
        **({"player_factory": _pair, "concurrency": 1} if with_factory else {}),
    )


def test_passing_the_factory_at_G1_changes_nothing() -> None:
    """The gate block now ALWAYS hands `play_paired_match` a factory, armed or not — so "the
    serial path is byte-exact" has to hold with the factory PRESENT, which is a different
    statement from the one tranche-3 pinned (serial vs concurrent).

    Compared as whole records: `trajectory_hash` is LAW-04's dedupe input and it is in there.
    """
    assert _play(with_factory=False) == _play(with_factory=True), (
        "handing the serial arm a `player_factory` moved a record — at G=1 the factory must "
        "never be called and the two passed-in players must be the ones that play"
    )


# ── 3. the round: which blocks are armed ───────────────────────────────────────────────
def _net(seed: int):
    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _round_spec(tmp_path: Path, concurrency: int) -> RoundSpec:
    candidate, best = tmp_path / "candidate.pt", tmp_path / "best.pt"
    write_model_snapshot(_net(seed=1), candidate)
    write_model_snapshot(_net(seed=2), best)
    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    return RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=32,
        leaf_build_threads=1, concurrency=concurrency,
        round_index=0, round_id="concurrency_wiring", step=1, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=_ENC, worker_device="cpu",
        gate=gate, rung_jobs=[], random_floor_games=2,
        random_model_sims=2, sealbot_model_sims=2, seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        game_record=None,
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441,
                                            max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(
            inference_batch_size=64, inference_max_wait_ms=10
        ),
    )


def _census(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None, bool]]:
    """One `(phase, concurrency-kwarg, factory-present)` row per `play_paired_match` call.

    The phase is read off the record sink's own name rather than off the regime key, because
    the gate's screen and confirm share a regime key and the two calls must be distinguishable.
    """
    rows: list[tuple[str, int | None, bool]] = []
    real = worker.play_paired_match

    def _spy(candidate, opponent, openings, **kwargs):
        rows.append((
            _phase_of(kwargs),
            kwargs.get("concurrency"),
            kwargs.get("player_factory") is not None,
        ))
        return real(candidate, opponent, openings, **kwargs)

    monkeypatch.setattr(worker, "play_paired_match", _spy)
    return rows


def _phase_of(kwargs: dict[str, Any]) -> str:
    """`_RoundProgress.sink` closes over its phase name; read it back off the closure.

    Reading the phase from the object the production code already built beats threading a
    label through the spy: a label the test invents could agree with the wrong call.

    R344(b) put a FAN-OUT between the two — `record_sink` is now `_both(progress.sink(...),
    games.sink(...))`, whose own closure holds a tuple of sinks rather than a phase — so the
    walk descends through nested closures and tuples to the first string it finds. Reading
    cell 0 of the outermost closure was never a contract, only the shape that happened to
    hold; this states what is actually being asked for."""
    return _first_str_in_closure(kwargs["record_sink"])


def _first_str_in_closure(value: Any, depth: int = 0) -> str:
    if depth > 4:                                    # a wrapper stack this deep is a defect
        return "?"
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        for item in value:
            found = _first_str_in_closure(item, depth + 1)
            if found != "?":
                return found
        return "?"
    for cell in getattr(value, "__closure__", None) or ():
        found = _first_str_in_closure(cell.cell_contents, depth + 1)
        if found != "?":
            return found
    return "?"


@pytest.mark.parametrize("armed", [1, 3])
def test_only_the_gate_block_carries_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, armed: int
) -> None:
    """THE SCOPE WITNESS, and its own mutation control.

    Parametrised over an UNARMED and an ARMED value so this is not a test that would pass
    against a hardcoded `concurrency=1`: at `armed=3` the two gate calls must carry 3, and the
    floor/rung/random calls must still carry nothing. A single-value version of this test would
    be green against a wire that reached nothing.
    """
    rows = _census(monkeypatch)
    worker.run_round(_round_spec(tmp_path, armed))

    gate_rows = [r for r in rows if r[0].startswith("gate_")]
    other_rows = [r for r in rows if not r[0].startswith("gate_")]
    assert gate_rows, f"the gate block never played; phases seen: {[r[0] for r in rows]}"
    assert other_rows, "no non-gate block played, so the negative half proves nothing"

    for phase, conc, has_factory in gate_rows:
        assert conc == armed, f"{phase} played at concurrency={conc}, config said {armed}"
        assert has_factory, f"{phase} was armed without a player_factory — G>1 would raise"
    for phase, conc, has_factory in other_rows:
        assert conc is None and not has_factory, (
            f"{phase} carries the concurrency row; only the gate block may. The floor probe is "
            "a LAW-07 gate input and the rung block is LAW-04's Elo channel — both stay serial "
            "and deterministic by ruling, not by omission."
        )
