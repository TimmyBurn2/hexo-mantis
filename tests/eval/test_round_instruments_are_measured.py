"""AUDIT-1 F-28 rows B02, B03, B04, B05 — the eval round's constants become measurements.

FOUR ROWS, ONE CLASS. Each publishes a value that reads as a measurement and is not one:

* **B02** — the worker child stamped `"status": "active"` on EVERY rung result. The child has
  no `LadderState`, so a SATURATED rung playing its off-cadence calibration games was labelled
  active every round. The parent now stamps the real status, read BEFORE `record_round` so it
  is the status the rung was PLAYED under. Beside it, the random floor's `RegimeKey` claimed
  `deploy_matched=True` while playing at `random_model_sims` against a uniform bot — M-3's
  mislabel one block over. Only the GATE block is deploy-matched (both sides at
  `spec.gate.deploy_sims`), and that `True` is derived from the construction, so it stays.
* **B03** — the progress writer defaulted `plies` to `0`, publishing a game that ended at ply
  zero for a record shape carrying no ply count. `event_manifest.md`'s own row for this writer
  says an unrecognised shape writes NULLS.
* **B04** — `eval_round_complete.promoted: false` covered three different rounds: the gate ran
  and refused, the gate was not scheduled, there was no anchor. `None` is now "no promotion
  decision was taken".
* **B05** — the terminal round emitted no `eval_round_started`, while the `eval_round_wall`
  manifest row names the PAIR as its producer. The one round whose wall time the drain budget
  is judged on had no start timestamp.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")


# ── B03: the progress writer ──────────────────────────────────────────────────────────

def _progress_row(tmp_path: Path, record: Any) -> dict[str, Any]:
    """One row through the PRODUCTION writer."""
    from mantis.eval.worker import _RoundProgress

    path = tmp_path / "progress.jsonl"
    _RoundProgress(path).sink("gate_screen")(record)
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_a_record_with_no_ply_count_writes_null_not_a_game_that_ended_at_ply_zero(
    tmp_path: Path,
) -> None:
    row = _progress_row(tmp_path, SimpleNamespace(terminal="win", winner="candidate"))
    assert row["plies"] is None, row


def test_a_record_that_HAS_a_ply_count_still_carries_it(tmp_path: Path) -> None:
    """The control, including the genuinely-zero case the convention must not eat."""
    row = _progress_row(tmp_path, SimpleNamespace(plies=37, terminal="win", winner="candidate"))
    assert row["plies"] == 37
    zero = _progress_row(tmp_path, SimpleNamespace(plies=0, terminal="win", winner="candidate"))
    assert zero["plies"] == 0 and zero["plies"] is not None


# ── B02: the deploy-matched label ─────────────────────────────────────────────────────

def test_only_the_GATE_block_claims_to_be_deploy_matched() -> None:
    """Structural, over the worker's own source: `deploy_matched=True` is legitimate exactly
    where both sides play at `spec.gate.deploy_sims`. The random floor plays at
    `random_model_sims` against a uniform bot and used to claim it too."""
    import ast
    import inspect

    import mantis.eval.worker as worker

    source = inspect.getsource(worker)
    tree = ast.parse(source)
    claims: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RegimeKey"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        bot = kw.get("bot")
        name = bot.value if isinstance(bot, ast.Constant) else "<rung>"
        flag = kw.get("deploy_matched")
        claims[name] = flag.value if isinstance(flag, ast.Constant) else "<derived>"
    assert claims.get("best_anchor") is True, claims
    assert claims.get("random") is False, (
        f"the random floor still claims the deploy-matched label: {claims}"
    )


# ── B02: the rung status the parent stamps ────────────────────────────────────────────

class _FakePipeline:
    """`EvalPipeline._success_result` lifted off the class, ladder collaborator REAL enough to
    carry statuses — the same stand-in shape `test_strength_floor_refusal_reaches_the_gate.py`
    uses, so the code exercised is production."""

    class _Ladder:
        rungs: tuple = ()
        bt_prior_games = 1.0

    def __init__(self, sink: Any, statuses: dict[str, str]) -> None:
        self._sink = sink
        self._statuses = statuses
        self._eval_cfg = SimpleNamespace(ladder=self._Ladder())
        self._ladder_state_path = Path("/nonexistent/ladder.json")
        self._last_p_hat: dict = {}
        self._floor_checked_total = 0
        self._floor_skipped_total = 0
        outer = self

        class _State:
            def status(self, rung: str) -> str:
                return outer._statuses[rung]

            def record_round(self, *a: Any, **k: Any) -> None:
                # recording is what MOVES a status; anything read after this is the wrong fact
                outer._statuses = {n: "saturated" for n in outer._statuses}

            def save(self, *a: Any, **k: Any) -> None: ...

            def allocate_games(self, *a: Any, **k: Any) -> dict:
                return {}

        self._state = _State()

    def _ensure_ladder_state(self) -> Any:
        return self._state

    def _current_p_hat(self) -> dict:
        return self._last_p_hat

    def _name_the_sealbot_rung(self, rungs_raw, reported_wr, *, round_id):
        """The PRODUCTION method, bound through the class — not a stub (AUDIT-1 F-14). It
        walks `self._eval_cfg.ladder.rungs`, which this stand-in supplies."""
        from mantis.eval.pipeline import EvalPipeline

        return EvalPipeline._name_the_sealbot_rung(
            self, rungs_raw, reported_wr, round_id=round_id)

    def _emit_posture_events(self, inflight: Any, raw: Any) -> None: ...


def _drive(raw: dict[str, Any], statuses: dict[str, str]) -> tuple[dict, list[dict]]:
    from mantis.eval.pipeline import EvalPipeline

    events: list[dict[str, Any]] = []

    class _Sink:
        def emit(self, payload: dict) -> None:
            events.append(dict(payload))

    fake = _FakePipeline(_Sink(), dict(statuses))
    result = EvalPipeline._success_result(
        fake, {"round_id": "r000001_5000", "step": 5000, "round_idx": 1}, raw, wall_sec=9.0,
    )
    return result, events


def test_a_SATURATED_rungs_calibration_games_are_not_labelled_active() -> None:
    """THE PIN (B02). The child stamped `active` unconditionally; the parent reads the truth."""
    raw = {"rungs": {"sealbot_d5": {"games": 8, "wr": 0.5, "wr_ci_lower": 0.2}},
           "gate": None, "random": {"games": 0, "wr": None}, "skipped_rungs": []}
    result, _events = _drive(raw, {"sealbot_d5": "saturated"})
    assert result["rungs"]["sealbot_d5"]["status"] == "saturated", result["rungs"]


def test_the_status_is_the_one_the_rung_was_PLAYED_under() -> None:
    """Read BEFORE `record_round`. Recording this round's result is what MOVES a status, so a
    read after it reports the rung's next state as though the games were played in it."""
    raw = {"rungs": {"sealbot_d5": {"games": 8, "wr": 1.0, "wr_ci_lower": 0.9}},
           "gate": None, "random": {"games": 0, "wr": None}, "skipped_rungs": []}
    result, _events = _drive(raw, {"sealbot_d5": "active"})
    assert result["rungs"]["sealbot_d5"]["status"] == "active", (
        "the status was read AFTER record_round, so it describes the round that follows"
    )


def test_a_rung_the_ladder_does_not_know_is_absent_not_active() -> None:
    raw = {"rungs": {"mystery": {"games": 4, "wr": 0.5, "wr_ci_lower": 0.1}},
           "gate": None, "random": {"games": 0, "wr": None}, "skipped_rungs": []}
    result, _events = _drive(raw, {})
    assert result["rungs"]["mystery"]["status"] is None


# ── B04: the three rounds that used to be one observable ──────────────────────────────

def _complete(events: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [e for e in events if e.get("event") == "eval_round_complete"]
    assert len(matches) == 1, matches
    return matches[0]


def test_a_round_with_no_gate_scheduled_reports_NO_promotion_decision() -> None:
    """THE PIN (B04). Before: `promoted: false`, identical to a gate that ran and refused."""
    raw = {"rungs": {}, "gate": None, "random": {"games": 4, "wr": 0.5}, "skipped_rungs": []}
    _result, events = _drive(raw, {})
    assert _complete(events)["promoted"] is None


@pytest.mark.parametrize("promoted", [True, False], ids=["promoted", "refused"])
def test_a_gate_that_RAN_reports_its_decision_either_way(promoted: bool) -> None:
    """The control. `None` must mean "no decision", never "the gate said no"."""
    raw = {"rungs": {}, "random": {"games": 0, "wr": None}, "skipped_rungs": [],
           "gate": {"wr_screen": 0.6, "wr_confirm": 0.6, "n_screen": 80, "n_confirm": 0,
                    "n_pooled": 80, "escalated": False, "elo_ci_lower_boot": 1.0,
                    "low_power": False, "eff_n": 80, "reason": "", "deploy_matched": True,
                    "promoted": promoted}}
    _result, events = _drive(raw, {})
    assert _complete(events)["promoted"] is promoted
    assert _complete(events)["promoted"] is not None


# ── B05: the terminal round's start event ─────────────────────────────────────────────

def test_the_terminal_round_emits_the_START_half_of_the_manifest_pair() -> None:
    """B05, structural over the production source: `_run_terminal_sync` must call
    `emit_round_started`. The `eval_round_wall` manifest row names the PAIR as its producer,
    and the terminal round is the one whose wall time the drain budget is judged on."""
    import ast
    import inspect

    from mantis.eval.pipeline import EvalPipeline

    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(EvalPipeline._run_terminal_sync)))
    called = {getattr(node.func, "id", None) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    assert "emit_round_started" in called, sorted(n for n in called if n)
