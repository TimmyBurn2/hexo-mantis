"""LadderState — opponent-ladder scheduling + CI-hysteresis graduation (design §a.3
ladder.py; STATE §5 verbatim). Every threshold/cadence/n is a schema field
(`mantis.config.schema.LadderConfig`) — never a code literal (rule 4).

Transitions:
  * activation — a DORMANT rung activates when its IMMEDIATE PREDECESSOR's most recent
    MEASURED round's lower-CI >= `activation_wr_lower_ci` (sticky); a rung that is itself
    handed a measured round while still dormant self-activates on that same evidence (it
    is plainly being played).
  * graduation — an ACTIVE rung SATURATES when the pair-bootstrap lower CI of WR >=
    `graduation_wr_lower_ci` for `graduation_consec_rounds` CONSECUTIVE MEASURED
    qualifying rounds (STATE §5 verbatim, no "full-n" qualifier). Counter semantics
    (pre-registered, MUST-FIX 5): a round where the rung recorded ZERO games neither
    advances nor resets the streak (transparent); a MEASURED sub-threshold round resets it
    to 0. Saturated is terminal — calibration cadence forever, never retired.

Scheduling: `allocate_games` splits `round_games` across ACTIVE rungs proportional to
p*(1-p) (numpy, largest-remainder rounding), floors each at `min_games_per_active_rung`,
then CLAMPS each at `games_max` (no redistribution of the clamped excess — deterministic,
total may undershoot). SATURATED rungs get `calibration_games` every
`calibration_every_k_rounds`-th round, forever.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mantis.config.schema import LadderConfig, LadderRung
from mantis.eval.errors import LadderStateError

_DORMANT = "dormant"
_ACTIVE = "active"
_SATURATED = "saturated"

#: RED-TEAM F1 (BLOCKER) fix, layer 1 (root cause). `allocate_games` must be TOTAL over
#: every ACTIVE rung, not just the rungs present in this round's freshly-fit `bt_probs`
#: (an active rung that just activated this same round, or that failed to resolve while a
#: sibling played, is absent from `bt_probs` and was previously an unconditional
#: `bt_probs[name]` KeyError — see mantis-migration/wp/WP11A/RED_TEAM.md Finding F1).
#:
#: This is NOT a tunable knob (hence no schema field, no `Field(...)` bound, no mint
#: value) — it is the exact mathematical boundary of the STATE §5 information-weighting
#: formula: scheduling weight is `p*(1-p)` ("play whoever yields the most information",
#: plan/STATE_2026-07-24.md:104-105, the KataGo/BayesElo variance-sampling citation). The
#: no-information point is `p=0.5`; the derivative `d/dp[p(1-p)] = 1-2p` is zero exactly at
#: `p=0.5`, i.e. `p(1-p)` is MAXIMIZED there (`0.25`, the largest value the weight can take).
#: A rung the fit has no data for this round is, by definition, the point of maximum
#: uncertainty — so it is CORRECT (not merely a safe default) for it to receive the
#: highest-information scheduling weight, exactly the STATE §5 intent, until a real p̂
#: measurement narrows it away from 0.5.
UNINFORMATIVE_P_HAT = 0.5


@dataclass
class _RungState:
    name: str
    status: str
    consec: int = 0
    history: list = field(default_factory=list)  # [{"round_idx","games","wr","ci_lo"}]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "consec": self.consec,
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> _RungState:
        return cls(name=payload["name"], status=payload["status"], consec=payload["consec"],
                   history=list(payload.get("history", [])))


class LadderState:
    """The ladder's per-rung status/streak/history + the round game-allocation policy."""

    def __init__(self, cfg: LadderConfig, rung_states: dict[str, _RungState]) -> None:
        self._cfg = cfg
        self._rungs = rung_states

    @classmethod
    def initial(cls, ladder_cfg: LadderConfig) -> LadderState:
        rung_states: dict[str, _RungState] = {}
        for index, rung in enumerate(ladder_cfg.rungs):
            status = _ACTIVE if index == 0 else _DORMANT
            rung_states[rung.name] = _RungState(name=rung.name, status=status)
        return cls(ladder_cfg, rung_states)

    # ── accessors ─────────────────────────────────────────────────────────────────────
    def status(self, rung: str) -> str:
        return self._rungs[rung].status

    def consec(self, rung: str) -> int:
        return self._rungs[rung].consec

    def rungs(self) -> list[LadderRung]:
        return list(self._cfg.rungs)

    # ── round recording (activation + graduation transitions) ──────────────────────────
    def record_round(
        self, round_idx: int, results: Mapping[str, Mapping[str, Any]], *, sink: Any = None
    ) -> None:
        """`results` maps rung name -> `{"games", "wr", "ci_lo"}` for every rung that
        recorded >=1 game this round. A rung ABSENT from `results` played zero games and
        is untouched (streak HELD — the transparent case)."""
        rung_names = [r.name for r in self._cfg.rungs]
        for name in rung_names:
            entry = results.get(name)
            if entry is None:
                continue  # absent -> zero games this round; streak HELD, no history entry
            self._record_one_rung(name, round_idx, entry, sink=sink)
        self._apply_activation_pass(round_idx, sink=sink)

    def _record_one_rung(
        self, name: str, round_idx: int, entry: Mapping[str, Any], *, sink: Any
    ) -> None:
        state = self._rungs[name]
        games = int(entry.get("games", 0))
        wr = entry.get("wr")
        ci_lo = entry.get("ci_lo")
        state.history.append({"round_idx": round_idx, "games": games, "wr": wr, "ci_lo": ci_lo})

        if games == 0:
            # A rung PRESENT with games=0 is an explicit zero-game round: HELD, loud —
            # distinguished from "absent" only for bookkeeping/history completeness.
            if sink is not None:
                sink.emit({
                    "event": "eval_ladder_zero_game_round", "rung": name, "round_id": str(round_idx),
                })
            return

        if state.status == _DORMANT:
            # Direct evidence of play while dormant is itself an activation signal (the
            # scheduler only ever hands games to a rung it means to have running).
            state.status = _ACTIVE

        if state.status == _SATURATED:
            return  # terminal: calibration WR still recorded above, never a streak change

        qualifies = ci_lo is not None and ci_lo >= self._cfg.graduation_wr_lower_ci
        if qualifies:
            state.consec += 1
            if state.consec >= self._cfg.graduation_consec_rounds:
                state.status = _SATURATED
                if sink is not None:
                    sink.emit({
                        "event": "eval_rung_graduated", "rung": name,
                        "round_id": str(round_idx), "consec": state.consec,
                    })
        else:
            state.consec = 0

    def _apply_activation_pass(self, round_idx: int, *, sink: Any) -> None:
        rung_names = [r.name for r in self._cfg.rungs]
        for index in range(len(rung_names) - 1):
            predecessor = self._rungs[rung_names[index]]
            successor = self._rungs[rung_names[index + 1]]
            if successor.status != _DORMANT:
                continue
            if not predecessor.history:
                continue
            last = predecessor.history[-1]
            if last["round_idx"] != round_idx or last["games"] == 0:
                continue
            ci_lo = last["ci_lo"]
            if ci_lo is not None and ci_lo >= self._cfg.activation_wr_lower_ci:
                successor.status = _ACTIVE
                if sink is not None:
                    sink.emit({
                        "event": "eval_rung_activated", "rung": successor.name,
                        "round_id": str(round_idx), "trigger_ci": ci_lo,
                    })

    # ── scheduling ───────────────────────────────────────────────────────────────────
    def allocate_games(
        self, round_idx: int, bt_probs: Mapping[str, float]
    ) -> dict[str, int]:
        """`round_games` split across ACTIVE rungs ∝ p(1-p) (largest-remainder rounding),
        floored at `min_games_per_active_rung`, CLAMPED at each rung's `games_max` (excess
        not redistributed). SATURATED rungs get `calibration_games` every
        `calibration_every_k_rounds`-th round, else 0 — forever."""
        alloc: dict[str, int] = {}
        games_max = {rung.name: rung.games_max for rung in self._cfg.rungs}
        active_names = [
            rung.name for rung in self._cfg.rungs if self._rungs[rung.name].status == _ACTIVE
        ]
        if active_names:
            # F1 fix (layer 1): `.get(name, UNINFORMATIVE_P_HAT)`, never a bare `[name]` —
            # total over ALL active rungs, not just the ones `bt_probs` happens to cover.
            weights = np.array(
                [
                    bt_probs.get(name, UNINFORMATIVE_P_HAT)
                    * (1.0 - bt_probs.get(name, UNINFORMATIVE_P_HAT))
                    for name in active_names
                ],
                dtype=np.float64,
            )
            total_weight = float(weights.sum())
            if total_weight > 0:
                shares = self._cfg.round_games * weights / total_weight
            else:
                shares = np.full(len(active_names), self._cfg.round_games / len(active_names))
            floor_shares = np.floor(shares).astype(np.int64)
            remainder = int(self._cfg.round_games - int(floor_shares.sum()))
            frac = shares - floor_shares
            order = np.argsort(-frac, kind="stable")
            alloc_arr = floor_shares.copy()
            for i in range(max(remainder, 0)):
                alloc_arr[order[i % len(order)]] += 1
            alloc_arr = np.maximum(alloc_arr, self._cfg.min_games_per_active_rung)
            for name, n in zip(active_names, alloc_arr):
                alloc[name] = min(int(n), games_max[name])

        for rung in self._cfg.rungs:
            if self._rungs[rung.name].status != _SATURATED:
                continue
            if round_idx % self._cfg.calibration_every_k_rounds == 0:
                alloc[rung.name] = min(self._cfg.calibration_games, rung.games_max)
            else:
                alloc[rung.name] = 0
        return alloc

    # ── persistence (LAW-14: no silent except) ─────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        target = Path(path)
        payload = {name: state.to_dict() for name, state in self._rungs.items()}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(target)
        except OSError as exc:
            raise LadderStateError(f"LadderState.save failed for {target}: {exc!r}") from exc

    @classmethod
    def load(cls, path: str | Path, *, ladder_cfg: LadderConfig | None = None) -> LadderState:
        target = Path(path)
        try:
            raw = json.loads(target.read_text())
        except (OSError, ValueError) as exc:
            raise LadderStateError(f"LadderState.load failed for {target}: {exc!r}") from exc
        rung_states = {name: _RungState.from_dict(payload) for name, payload in raw.items()}
        cfg = ladder_cfg if ladder_cfg is not None else _synthetic_cfg_from_rungs(rung_states)
        return cls(cfg, rung_states)


def _synthetic_cfg_from_rungs(rung_states: Mapping[str, _RungState]) -> LadderConfig:
    """Round-trip convenience: rebuild a MINIMAL valid `LadderConfig` naming exactly the
    persisted rungs, for callers that only need `status`/`consec` back (no live
    scheduling config available at this call site)."""
    rungs = [
        LadderRung(
            name=name, bot="random", variant="raw", depth=None, opponent_sims=None,
            opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=1_000_000,
        )
        for name in rung_states
    ]
    # NOTE: these numbers are placeholder scheduling knobs for a config-less round-trip
    # ONLY (this call site never has the ORIGINAL LadderConfig — the caller passing one
    # explicitly always wins). They deliberately avoid the STATE §5 threshold VALUES
    # (never as bare code literals, rule 4 — see test_minted_configs_carry_the_ladder_
    # verbatim's source-grep oracle).
    return LadderConfig(
        rungs=rungs, round_games=1, min_games_per_active_rung=0,
        graduation_wr_lower_ci=0.99, graduation_consec_rounds=3, activation_wr_lower_ci=0.5,
        calibration_every_k_rounds=4, calibration_games=1, bootstrap_resamples=100,
        bootstrap_ci_level=0.9, bt_prior_games=1.0, bootstrap_seed=1,
    )


__all__ = ["LadderState", "UNINFORMATIVE_P_HAT"]
