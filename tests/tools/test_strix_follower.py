"""`tools/strix_follower.py` (R356(a)): triggers off the event stream, one cell per receipt, the stamp untouched."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RUN = "runx"


@pytest.fixture(scope="module")
def follower_mod():
    path = _REPO / "tools" / "strix_follower.py"
    spec = importlib.util.spec_from_file_location("strix_follower_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / _RUN
    (run / "logs").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    return run


def _plant(run: Path, rows: list[dict[str, Any]], seg: int = 1) -> Path:
    path = run / "logs" / f"events_{_RUN}_seg{seg:04d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _checkpoint(run: Path, step: int, sha8: str = "0badf00d") -> Path:
    path = run / "checkpoints" / f"{_RUN}_{step:08d}_{sha8}.ckpt"
    path.write_bytes(f"stamped weights at {step}".encode())
    return path


def _heartbeat(run: Path, wall_ts: float) -> None:
    (run / "logs" / f"heartbeat_{_RUN}.json").write_text(json.dumps({"wall_ts": wall_ts}),
                                                          encoding="utf-8")


class _FakeCells:
    """Stands in for `strength_frontier.run_cell`: counts calls, returns the record shape it returns."""

    def __init__(self, rc: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.rc = rc

    def __call__(self, cell: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(cell))
        record: dict[str, Any] = {
            "label": cell["label"], "cell": dict(cell), "rc": self.rc, "wall_sec": 12.5,
            "provenance": {"candidate": {"net_hash": "net" + "0" * 61, "step": cell["step"]}},
            "strix_findings": {"count": 0, "first": []},
        }
        if self.rc == 0:
            record["readout"] = {"games": cell["games"], "eff_n": cell["games"], "pairs": cell["games"] // 2,
                                 "wins": 40, "losses": 248, "draws": 0, "wr": 40 / 288,
                                 "wr_ci_lower": 0.10, "wr_ci_upper": 0.18, "median_plies": 43.0,
                                 "sec_per_game": 0.04}
        else:
            record["error"] = "planted failure"
        return record


def _follower(mod, run: Path, cells: _FakeCells, **kw):
    return mod.Follower(run_dir=run, run_id=_RUN, run_cell=cells, strix_pin={"commit": "abc",
                        "checkpoint": "ck.pt", "checkpoint_sha256": "f" * 64},
                        clock=lambda: 1_000.0, log=lambda _s: None, **kw)


# triggers

def test_triggers_are_cadence_multiples_and_promotions_only(follower_mod) -> None:
    rows = [
        {"event": "periodic_checkpoint_save", "step": 15000, "path": "/x/a.ckpt"},
        {"event": "periodic_checkpoint_save", "step": 18000, "path": "/x/b.ckpt"},
        {"event": "eval_round_complete", "step": 3000, "promoted": True},
        {"event": "eval_round_complete", "step": 6000, "promoted": False},
        {"event": "eval_round_complete", "step": 9000, "promoted": None},
        {"event": "trainer_step", "step": 15000},
        {"event": "periodic_checkpoint_save", "step": "30000"},
    ]
    got = follower_mod.triggers_from_rows(iter(rows), 15_000)
    assert [(t.step, t.kind, t.path) for t in got] == [(15000, "cadence", "/x/a.ckpt"),
                                                        (3000, "promotion", None)]


def test_promotions_off_reads_the_cadence_points_only(follower_mod, tmp_path: Path) -> None:
    """R361(a): with the promotion trigger OFF a promoted round fires nothing; the cadence save still does."""
    rows = [{"event": "eval_round_complete", "step": 3000, "promoted": True},
            {"event": "periodic_checkpoint_save", "step": 15000, "path": "/x/a.ckpt"}]
    got = follower_mod.triggers_from_rows(iter(rows), 15_000, promotions=False)
    assert [(t.step, t.kind) for t in got] == [(15000, "cadence")]
    run = _run_dir(tmp_path)
    ckpt = _checkpoint(run, 3000)
    _plant(run, [{"event": "eval_round_complete", "step": 3000, "promoted": True}])
    cells = _FakeCells()
    f = _follower(follower_mod, run, cells, promotions=False)
    assert f.poll() == [] and f.pending == {} and cells.calls == []
    assert not ckpt.with_name(ckpt.name + ".strix256.json").exists()
    on = _follower(follower_mod, run, cells)
    assert len(on.poll()) == 1, "the default is today's behaviour: the promotion fires"


def test_the_cli_default_is_promotions_on_and_no_promotions_switches_it_off(follower_mod, monkeypatch, tmp_path: Path) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(follower_mod, "_real_run_cell", lambda _config, _work: _FakeCells())
    monkeypatch.setattr(follower_mod, "_strix_pin", lambda: {})
    monkeypatch.setattr(follower_mod.Follower, "follow", lambda self, _poll: seen.append(self.promotions))
    base = ["--config", "c.yaml", "--run-dir", str(tmp_path), "--run-id", _RUN, "--work-dir", str(tmp_path / "w")]
    assert follower_mod.main([*base, "--follow"]) == 0
    assert follower_mod.main([*base, "--follow", "--no-promotions"]) == 0
    assert follower_mod.main([*base, "--follow", "--promotions"]) == 0
    assert seen == [True, False, True]


def test_the_tail_reads_only_new_lines_and_follows_a_new_segment(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    _plant(run, [{"event": "periodic_checkpoint_save", "step": 15000, "path": "p"},
                 {"event": "trainer_step", "step": 1}])
    tail = follower_mod.EventTail(run, _RUN)
    assert [r["step"] for r in tail.read_new()] == [15000]
    assert tail.read_new() == []
    _plant(run, [{"event": "eval_round_complete", "step": 3000, "promoted": True}], seg=2)
    path = _plant(run, [{"event": "periodic_checkpoint_save", "step": 30000, "path": "q"}])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "periodic_checkpoint_save", "step": 45000')  # mid-append, no newline
    assert sorted(r["step"] for r in tail.read_new()) == [3000, 30000]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(', "path": "r"}\n')
    assert [r["step"] for r in tail.read_new()] == [45000]


# the producer test (LAW-07): planted event -> fires once; planted duplicate -> not twice

def test_a_planted_cadence_event_fires_one_cell_and_writes_the_sidecar(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    ckpt = _checkpoint(run, 15000)
    before = ckpt.read_bytes()
    _heartbeat(run, wall_ts=1_000.0 - 30.0)
    _plant(run, [{"event": "periodic_checkpoint_save", "step": 15000, "path": str(ckpt)}])
    cells = _FakeCells()
    f = _follower(follower_mod, run, cells)
    written = f.poll()
    assert len(cells.calls) == 1
    cell = cells.calls[0]
    assert (cell["search_kind"], cell["sims"], cell["opponent"], cell["strix_sims"], cell["games"]) == (
        "puct", 256, "strix", 256, 288), "the equal-work unit, 288 paired games"
    assert cell["candidate"] == str(ckpt) and cell["step"] == 15000
    assert written == [ckpt.with_name(ckpt.name + ".strix256.json")]
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert body["unit"] == "equal_work" and body["ours"]["sims"] == 256 and body["strix"]["sims"] == 256
    assert body["trigger"] == "cadence" and body["regime"] == "CONTENDED"
    assert body["regime_evidence"]["heartbeat_age_sec_self"] == 30.0
    assert body["regime_evidence"]["live"] == [f"{_RUN}/logs/heartbeat_{_RUN}.json"]
    assert body["net_hash"] == "net" + "0" * 61
    assert body["checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert body["strix"]["checkpoint_sha256"] == "f" * 64
    assert (body["games"], body["eff_n"], body["wr"], body["wr_ci_lower"], body["wr_ci_upper"]) == (
        288, 288, 40 / 288, 0.10, 0.18)
    assert ckpt.read_bytes() == before, "the stamp is never touched (LAW-12)"


def test_a_planted_duplicate_does_not_fire_twice(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    ckpt = _checkpoint(run, 15000)
    row = {"event": "periodic_checkpoint_save", "step": 15000, "path": str(ckpt)}
    _plant(run, [row])
    cells = _FakeCells()
    f = _follower(follower_mod, run, cells)
    assert len(f.poll()) == 1
    _plant(run, [row, {"event": "eval_round_complete", "step": 15000, "promoted": True}])
    assert f.poll() == [] and len(cells.calls) == 1, "the sidecar is the receipt"
    # a fresh follower (a restart) over the same stream reads the receipt, not the cell
    again = _follower(follower_mod, run, cells)
    assert again.poll() == [] and len(cells.calls) == 1


def test_a_promotion_fires_on_the_step_s_checkpoint_once_it_exists(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    _plant(run, [{"event": "eval_round_complete", "step": 3000, "promoted": True},
                 {"event": "eval_round_complete", "step": 6000, "promoted": False}])
    cells = _FakeCells()
    f = _follower(follower_mod, run, cells)
    assert f.poll() == [] and 3000 in f.pending, "no checkpoint on disk yet: pending, not dropped"
    ckpt = _checkpoint(run, 3000)
    written = f.poll()
    assert [c["candidate"] for c in cells.calls] == [str(ckpt)]
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert body["trigger"] == "promotion" and body["step"] == 3000
    assert body["regime"] == "IDLE" and body["regime_evidence"]["heartbeat_age_sec_self"] is None


def test_a_failed_cell_leaves_no_receipt(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    ckpt = _checkpoint(run, 30000)
    _plant(run, [{"event": "periodic_checkpoint_save", "step": 30000, "path": str(ckpt)}])
    cells = _FakeCells(rc=1)
    f = _follower(follower_mod, run, cells)
    assert f.poll() == []
    assert not ckpt.with_name(ckpt.name + ".strix256.json").exists()
    failed = ckpt.with_name(ckpt.name + ".strix256.failed.json")
    assert json.loads(failed.read_text(encoding="utf-8"))["error"] == "planted failure"


def test_once_reads_the_named_checkpoint_in_the_named_unit(follower_mod, tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    ckpt = _checkpoint(run, 42000)
    cells = _FakeCells()
    f = _follower(follower_mod, run, cells, unit="as_shipped")
    status, out = f.read_one(ckpt, trigger="once")
    assert status == "written" and out.name == ckpt.name + ".strix512.json"
    assert (cells.calls[0]["sims"], cells.calls[0]["strix_sims"]) == (512, 128)
    assert f.read_one(ckpt, trigger="once") == ("receipted", out) and len(cells.calls) == 1


def test_the_equal_work_cell_composes_through_the_frontier_as_the_256_256_rung(follower_mod, tmp_path: Path) -> None:
    """The unit the sidecar names is the RoundSpec the child plays: ours 256, strix 256, 288 games, 8 in flight."""
    from mantis.config.loader import load_config

    path = _REPO / "tools" / "strength_frontier.py"
    spec = importlib.util.spec_from_file_location("strength_frontier_for_follower", path)
    assert spec is not None and spec.loader is not None
    frontier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frontier
    spec.loader.exec_module(frontier)
    config = load_config(str(_REPO / "configs" / "run8.yaml"))
    base = frontier.base_round_spec(config, work_dir=tmp_path / "w")
    cell = follower_mod.compose_cell(Path("/x/run8_00015000_deadbeef.ckpt"), unit="equal_work",
                                     step=15000, games=288, concurrency=8, label="equal_work_run8_15000")
    round_spec = frontier.cell_spec(cell, base, cell_dir=tmp_path / "c", config=config)
    job = round_spec.rung_jobs[0]
    assert (round_spec.search_kind, round_spec.strix_model_sims, job.bot, job.opponent_sims, job.games) == (
        "puct", 256, "strix", 256, 288)
    assert round_spec.rung_concurrency == 8 and round_spec.step == 15000
    assert job.opening_book == config.eval.gate.opening_book == "book_v1_s20260625_p4"
    assert frontier.cell_channel(cell) == "external"


def test_a_sibling_runs_live_heartbeat_makes_the_cell_contended(follower_mod, tmp_path: Path) -> None:
    """The parent run or a shakedown twin shares the card as much as this run does."""
    run = _run_dir(tmp_path)
    twin = tmp_path / "runs" / "runx-shakedown" / "logs"
    twin.mkdir(parents=True)
    (twin / "heartbeat_runx-shakedown.json").write_text(json.dumps({"wall_ts": 1_000.0 - 5.0}),
                                                        encoding="utf-8")
    ckpt = _checkpoint(run, 15000)
    _plant(run, [{"event": "periodic_checkpoint_save", "step": 15000, "path": str(ckpt)}])
    f = _follower(follower_mod, run, _FakeCells())
    body = json.loads(f.poll()[0].read_text(encoding="utf-8"))
    assert body["regime"] == "CONTENDED"
    assert body["regime_evidence"]["live"] == ["runx-shakedown/logs/heartbeat_runx-shakedown.json"]
    assert body["regime_evidence"]["heartbeat_age_sec_self"] is None, "this run's own heartbeat is absent"


def test_a_stale_preflight_heartbeat_with_the_same_filename_cannot_hide_the_live_run(follower_mod, tmp_path: Path) -> None:
    """Box fact 2026-09-18: a stale `<run>-preflight/logs/heartbeat_<run>.json` shares its NAME with the live one and, keyed by name, hid it (run8's cell read IDLE)."""
    run = _run_dir(tmp_path)
    _heartbeat(run, 1_000.0 - 2.0)
    stale = tmp_path / "runs" / "runx-preflight" / "logs"
    stale.mkdir(parents=True)
    (stale / "heartbeat_runx.json").write_text(json.dumps({"wall_ts": 1_000.0 - 18_000.0}), encoding="utf-8")
    name, evidence = follower_mod.regime(run, _RUN, 1_000.0)
    assert name == "CONTENDED", evidence
    assert evidence["heartbeat_age_sec_self"] == 2.0
    assert "runx/logs/heartbeat_runx.json" in evidence["live"] and len(evidence["live"]) == 1
    assert evidence["heartbeat_age_sec"]["runx-preflight/logs/heartbeat_runx.json"] == 18_000.0
