"""The strength floor's ROUND WIRING — a refusing probe stops the round before the gate block.

WHAT WAS ALREADY PROVEN AND WHAT WAS NOT. `test_strength_floor_gate.py` proves the DECISION
RULE as a pure function, and `test_eval_posture_inert.py` proves the result payload's key set
and the event channel. Neither drives `run_round`, so the claim the mechanism actually exists
for — **a failing probe refuses the round CHEAPLY, before the expensive phase runs** — had no
witness anywhere in the tree. That claim is about `run_round`'s branching, not about
arithmetic, and it is what this file adds.

CHEAPNESS IS ASSERTED STRUCTURALLY, NOT BY WALL TIME. The round's own phase series
(`device_memory["phases"]`, written by `DeviceMemoryProbe.mark` at every phase boundary and
attached UNCONDITIONALLY on both exit paths) shows whether `gate_block` was ever entered. A
timing assertion would measure this machine; the phase series measures the branch. The second
half is a game census over the ONE function every phase plays through, so "no gate games were
played" is counted rather than inferred from the absence of a result key.

WHY THE PROBE'S OUTCOMES ARE SYNTHETIC AND NOTHING ELSE IS. The rule reads
`GameRecord.terminal` and `GameRecord.winner`, and no checkpoint that exists off-box produces
a controlled decisive rate — EVAL-CHANNEL-1 measured 0/40 against a near-self opponent and had
no arm at all against `random`, which is who the probe plays. So the probe's RECORDS are
planted, per test, and everything downstream of them is production: `evaluate_strength_floor`
decides unpatched, `run_round` branches unpatched, and every OTHER phase plays real games
through the real arena. The seam is one function and it is the same one production calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.arena.adjudicate import TERMINAL_PLY_CAP, TERMINAL_WIN
from mantis.arena.match import GameRecord
from mantis.arena.regime import RegimeKey
from mantis.config.resolve.eval_posture import StrengthFloorSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.floor_gate import FLOOR_PROBE_VARIANT
from mantis.eval.rounds import GateSpec, RoundSpec
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import CnnArch, build_net

#: The grid encoding and the one shipped book — a real registry row and a real book, so the
#: round this file drives is the production round shape and not a fixture-only one.
_ENC = "v6"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(*, seed: int) -> torch.nn.Module:
    """A registry-TRUE net for `_ENC`, minimal width and depth.

    Dims come from the spec rather than from literals: a net sized by hand would agree with
    the wrong encoding by coincidence, which is the confusion `test_graph_round_encoding.py`
    exists to detect and which this file has no reason to re-introduce.
    """
    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = CnnArch(
        board_size=spec.board_size, in_channels=spec.n_planes, filters=8, res_blocks=1
    )
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _round_spec(tmp_path: Path, floor: StrengthFloorSpec | None) -> RoundSpec:
    """A real `RoundSpec` with a REAL anchor, so the gate block is reachable.

    `best_snapshot` is not None and `run_gate` is True on purpose: a round with no anchor
    skips the gate block for an unrelated reason, and a refusal test that could not tell those
    two apart would pass against a mechanism that does nothing.
    """
    candidate = tmp_path / "candidate.pt"
    best = tmp_path / "best.pt"
    write_model_snapshot(_net(seed=1), candidate)
    write_model_snapshot(_net(seed=2), best)

    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    return RoundSpec(
        leaf_batch_size=1, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
        round_id="floor_wiring", step=1, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=_ENC, worker_device="cpu",
        gate=gate, rung_jobs=[], random_floor_games=0,
        random_model_sims=2, sealbot_model_sims=2, kraken_model_sims=2, strix_model_sims=2,
        seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        ply_cap_adjudication=None, strength_floor=floor,
        fused_graph_caps=None,
        inference_batching=InferenceBatchingSpec(
            inference_batch_size=64, inference_max_wait_ms=10
        ),
    )


def _record(regime_key: RegimeKey, *, winner: str, terminal: str, idx: int) -> GameRecord:
    return GameRecord(
        regime_key=regime_key, opening_id=f"synthetic_{idx}",
        colors={"candidate": 1, "opponent": -1},
        trajectory_hash=f"{idx:064x}", winner=winner, plies=128, moves=(),
        terminal=terminal, adjudication=None,
    )


def _plant_probe_outcomes(
    monkeypatch: pytest.MonkeyPatch, *, winner: str, terminal: str
) -> list[tuple[str, str, int]]:
    """Plant the FLOOR PROBE's outcomes only; every other phase plays for real.

    Returns the live game census — one `(bot, variant, n_records)` row per
    `play_paired_match` call, in call order — which is what makes "the gate block played
    nothing" a COUNT rather than an inference from a missing result key.
    """
    census: list[tuple[str, str, int]] = []
    real = worker.play_paired_match

    def _spy(candidate, opponent, openings, *, regime_key, **kwargs):
        if regime_key.variant == FLOOR_PROBE_VARIANT:
            planted = [
                _record(regime_key, winner=winner, terminal=terminal, idx=i)
                for i in range(len(list(openings)) * 2)
            ]
            census.append((regime_key.bot, regime_key.variant, len(planted)))
            return planted
        records = real(candidate, opponent, openings, regime_key=regime_key, **kwargs)
        census.append((regime_key.bot, regime_key.variant, len(records)))
        return records

    monkeypatch.setattr(worker, "play_paired_match", _spy)
    return census


def _phases(result: dict[str, Any]) -> list[str]:
    return [p["phase"] for p in result["device_memory"]["phases"]]


# ── the two rounds ─────────────────────────────────────────────────────────────────────

def test_a_no_signal_round_REFUSES_before_the_gate_block_ever_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EVAL-CHANNEL-1's regime, driven through the real round: every probe game a ply-cap
    non-result, so `decisive_rate` is 0.0 and any non-zero bar refuses.

    The three assertions are deliberately different KINDS of evidence — the verdict, the phase
    series and the game census — because a round could satisfy any one of them for the wrong
    reason: a `passed: False` verdict says nothing about whether the round then stopped, and a
    missing `gate` key is also what a round with no anchor produces.
    """
    census = _plant_probe_outcomes(monkeypatch, winner="draw", terminal=TERMINAL_PLY_CAP)
    floor = StrengthFloorSpec(probe_games=4, min_decisive_rate=0.25, min_winrate=0.0)

    result = worker.run_round(_round_spec(tmp_path, floor))

    assert result["strength_floor"]["passed"] is False
    assert result["strength_floor"]["decisive_rate"] == 0.0
    assert result["strength_floor"]["failed_bars"] == ["decisive_rate"]

    # The refusal is CHEAP, and this is the sentence that says so structurally.
    assert _phases(result) == ["round_start", "floor_probe", "round_end"], _phases(result)
    assert census == [("random", FLOOR_PROBE_VARIANT, 4)], census

    # And the round is a healthy REFUSAL, not a broken one: it returns a normal result whose
    # gate is absent, which is the same route a no-anchor round takes to not promoting.
    assert result["gate"] is None
    assert result["rungs"] == {}
    assert result["random"] == {"games": 0, "wr": None}


def test_a_decisive_round_PASSES_and_the_gate_block_then_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same round, the same bar, the same net — only the probe's outcomes differ.

    Holding everything else fixed is what makes the pair a CONTROL rather than two unrelated
    rounds: the only thing that can explain the different phase series is the verdict.
    """
    census = _plant_probe_outcomes(monkeypatch, winner="candidate", terminal=TERMINAL_WIN)
    floor = StrengthFloorSpec(probe_games=4, min_decisive_rate=0.25, min_winrate=0.0)

    result = worker.run_round(_round_spec(tmp_path, floor))

    assert result["strength_floor"]["passed"] is True
    assert result["strength_floor"]["decisive_rate"] == 1.0
    assert result["strength_floor"]["failed_bars"] == []

    phases = _phases(result)
    assert phases[:3] == ["round_start", "floor_probe", "gate_block"], phases
    assert phases[-1] == "round_end", phases
    assert result["gate"] is not None

    # The probe played its 4 games AND the gate block played its own — the pass arm pays the
    # probe's cost on top, which is the trade §5.2 of EVAL_POSTURE_OPTIONS prices.
    assert census[0] == ("random", FLOOR_PROBE_VARIANT, 4), census
    assert len(census) > 1 and sum(n for _, _, n in census[1:]) > 0, census


def test_the_disarmed_posture_plays_no_probe_and_reaches_the_gate_block_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`strength_floor: null` — every committed config — must leave the phase order alone.

    This is the identity arm the two above are measured against. Without it, a mechanism that
    reordered the round unconditionally would satisfy both of them.
    """
    census = _plant_probe_outcomes(monkeypatch, winner="draw", terminal=TERMINAL_PLY_CAP)

    result = worker.run_round(_round_spec(tmp_path, None))

    assert "strength_floor" not in result
    phases = _phases(result)
    assert "floor_probe" not in phases, phases
    assert phases[:2] == ["round_start", "gate_block"], phases
    assert all(variant != FLOOR_PROBE_VARIANT for _, variant, _ in census), census
