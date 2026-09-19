"""The eval round's constants become measurements.

Three rows, one class — each published a value that read as a measurement and was not one:

* the random floor's `RegimeKey` claimed `deploy_matched=True` while playing at
  `random_model_sims` against a uniform bot; only the GATE block is deploy-matched. (The rung
  STATUS row this file also carried — the child's constant `"active"` against the ladder's real
  status — left with the sealbot rung, R362(c).)
* the progress writer defaulted `plies` to `0`, publishing a game that ended at ply zero for a
  record shape carrying no ply count. An unrecognised shape writes NULLS.
* `eval_round_complete.promoted: false` covered three different rounds — the gate ran and
  refused, the gate was not scheduled, there was no anchor. `None` is now "no decision taken".
* the terminal round emitted no `eval_round_started`, while the `eval_round_wall` manifest row
  names the PAIR as its producer.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")



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



def test_only_the_GATE_block_claims_to_be_deploy_matched() -> None:
    """Structural, over the worker's own source: `deploy_matched=True` is legitimate exactly
    where both sides play at `spec.gate.deploy_sims`."""
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



class _FakePipeline:
    """`EvalPipeline._success_result` lifted off the class, so the code exercised is production."""

    def __init__(self, sink: Any) -> None:
        self._sink = sink
        self._round_counter = 0
        self._floor_checked_total = 0
        self._floor_skipped_total = 0

    def _emit_posture_events(self, inflight: Any, raw: Any) -> None: ...


def _drive(raw: dict[str, Any]) -> tuple[dict, list[dict]]:
    from mantis.eval.pipeline import EvalPipeline

    events: list[dict[str, Any]] = []

    class _Sink:
        def emit(self, payload: dict) -> None:
            events.append(dict(payload))

    fake = _FakePipeline(_Sink())
    result = EvalPipeline._success_result(
        fake, {"round_id": "r000001_5000", "step": 5000, "round_idx": 1}, raw, wall_sec=9.0,
    )
    return result, events



def _complete(events: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [e for e in events if e.get("event") == "eval_round_complete"]
    assert len(matches) == 1, matches
    return matches[0]


def test_a_round_with_no_gate_scheduled_reports_NO_promotion_decision() -> None:
    """Before this, `promoted: false` was identical to a gate that ran and refused."""
    raw = {"rungs": {}, "gate": None, "random": {"games": 4, "wr": 0.5}, "skipped_rungs": []}
    _result, events = _drive(raw)
    assert _complete(events)["promoted"] is None


@pytest.mark.parametrize("promoted", [True, False], ids=["promoted", "refused"])
def test_a_gate_that_RAN_reports_its_decision_either_way(promoted: bool) -> None:
    """The control. `None` must mean "no decision", never "the gate said no"."""
    raw = {"rungs": {}, "random": {"games": 0, "wr": None}, "skipped_rungs": [],
           "gate": {"wr_screen": 0.6, "wr_confirm": 0.6, "n_screen": 80, "n_confirm": 0,
                    "n_pooled": 80, "escalated": False, "elo_ci_lower_boot": 1.0,
                    "low_power": False, "eff_n": 80, "reason": "", "deploy_matched": True,
                    "promoted": promoted}}
    _result, events = _drive(raw)
    assert _complete(events)["promoted"] is promoted
    assert _complete(events)["promoted"] is not None



def test_the_terminal_round_emits_the_START_half_of_the_manifest_pair() -> None:
    """Structural over the production source: `_run_terminal_sync` must call
    `emit_round_started`. The `eval_round_wall` manifest row names the PAIR as its producer, and
    the terminal round is the one whose wall time the drain budget is judged on."""
    import ast
    import inspect

    from mantis.eval.pipeline import EvalPipeline

    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(EvalPipeline._run_terminal_sync)))
    called = {getattr(node.func, "id", None) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    assert "emit_round_started" in called, sorted(n for n in called if n)
