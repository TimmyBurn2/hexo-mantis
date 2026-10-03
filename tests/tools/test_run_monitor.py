"""The run monitor: the games after a save, the halting rows and their planted break, the idle-window rates, one halt."""
from __future__ import annotations

import importlib
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import load_ring
from mantis.monitor.event_tail import EventTail
from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
mon = importlib.import_module("run_monitor.monitor")
rings = importlib.import_module("run_monitor.rings")
cli = importlib.import_module("run_monitor.cli")

_RUN = "arm"
_ENC = "gnn_axis_v1"


def _ring(path: Path, games: range) -> Path:
    buf = HexgBuffer(64, _ENC, 128)
    for g in games:
        for ply in (2, 3):
            buf.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, ply, True,
                                    1.0 if g % 2 else -1.0, True, 10 + g, game_id=g)
    buf.save_to_path(str(path))
    return path


def test_the_unseen_ring_holds_only_the_games_after_the_save(tmp_path: Path) -> None:
    own, later = _ring(tmp_path / "s.bin", range(0, 4)), _ring(tmp_path / "t.bin", range(2, 8))
    meta = rings.unseen_ring(later, rings.max_game_id(own), tmp_path / "u.bin")
    kept = load_ring(tmp_path / "u.bin")
    assert sorted(set(kept.game_id.tolist())) == [4, 5, 6, 7] and meta["games"] == 4 and meta["rows"] == 8
    with pytest.raises(ValueError, match="no game after"):
        rings.unseen_ring(own, rings.max_game_id(later), tmp_path / "none.bin")


def _exam(mean: float, floor: float) -> dict[str, Any]:
    return {"calibrated_mean": mean, "floor": floor, "holds": mean >= floor}


def _assert_the_rows_halt(halting_rows: Any) -> None:
    """Below a floor halts, a band miss halts, the known-good passes."""
    good = {"T4_V": _exam(0.29, 0.154), "DEF_V_att": _exam(0.21, 0.100)}
    assert halting_rows(good, {"misses": []}) == []
    low = {**good, "DEF_V_att": _exam(0.0995, 0.100)}
    assert halting_rows(low, {"misses": []}) == ["DEF_V_att calibrated 0.0995 below the floor 0.1"]
    assert halting_rows(good, {"misses": ["cap_rate: 0.07 not lt 0.05"]}) == ["ring band cap_rate: 0.07 not lt 0.05"]


def test_a_reading_below_its_floor_or_a_band_outside_halts_and_the_planted_break_reds() -> None:
    """PLANTED BREAK: a checker that reads the raw floor test as `>` instead of the instrument's `holds` misses the low DEF."""
    _assert_the_rows_halt(mon.halting_rows)
    with pytest.raises(AssertionError):
        _assert_the_rows_halt(lambda exams, bands: [f"ring band {m}" for m in bands["misses"]])


def test_the_rates_read_the_idle_windows_apart_from_the_monitors_busy_ones() -> None:
    rows = [{"ts": 0.0, "games_total": 0, "positions_produced_total": 0, "step": 0},
            {"ts": 1800.0, "games_total": 100, "positions_produced_total": 9000, "step": 240},
            {"ts": 3600.0, "games_total": 120, "positions_produced_total": 10800, "step": 288}]
    out = mon.rates(rows, 0.0, 3600.0, busy=[(1900.0, 3500.0)])
    assert out["games_per_h"] == pytest.approx(120.0) and out["idle_games_per_h"] == pytest.approx(200.0)
    assert out["idle_h"] == pytest.approx(0.5) and out["steps_per_h"] == pytest.approx(288.0)
    assert "NOT MEASURED" in mon.rates(rows[:1], 0.0, 3600.0, busy=[])["note"]


class _Readers:
    """Reads that write nothing a net computed: a fixed GEN body, exams that fail at the chosen step, bands clean."""

    def __init__(self, fail_at: int | None) -> None:
        self.fail_at, self.values, self.step = fail_at, [], 0

    def value(self, ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        self.values.append((ckpt.name, ring.name, None if train is None else train.name))
        band = {"n": 10, "cf_ce": 0.6}
        return {"heldout": {"overall": {"cf_ce": 0.6, "temperature": 2.0, "auc": 0.7}, "policy_ce": 2.3,
                            "plies_0_10": band, "plies_11_40": band, "plies_41_up": band}}

    def exams(self, ckpt: Path, gen_read: Path, floors: dict[str, float]) -> dict[str, Any]:
        low = self.fail_at is not None and f"_{self.fail_at:08d}_" in ckpt.name
        return {"T4_V": _exam(0.10 if low else 0.29, floors["T4_V"]), "DEF_V_att": _exam(0.21, floors["DEF_V_att"])}

    def bands(self, ring: Path, events: Path | None, bands: Path) -> dict[str, Any]:
        return {"rows": {}, "misses": []}


def _run_dir(tmp_path: Path, pid: int) -> Path:
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "logs" / f"heartbeat_{_RUN}.json").write_text(json.dumps({"pid": pid, "seq": 1}), encoding="utf-8")
    return run


def _save(run: Path, step: int, games: range) -> Path:
    ckpt = run / "checkpoints" / f"{_RUN}_{step:08d}_0000abcd.ckpt"
    ckpt.write_bytes(f"stand-in {step}".encode())
    _ring(Path(f"{ckpt}.ring.bin"), games)
    return ckpt


def _monitor(tmp_path: Path, run: Path, readers: _Readers, *, halt: bool) -> Any:
    setup = mon.Setup(run_dir=run, run_id=_RUN, out=tmp_path / "records", gen_ring=tmp_path / "gen.bin", gen_sha256="",
                      exams=tmp_path / "exams.jsonl", exams_sha256="", bands=tmp_path / "bands.md",
                      floors={"T4_V": 0.154, "DEF_V_att": 0.100}, parent=run / "parent.ckpt", batches=1,
                      device="cpu", threads=1, halt=halt)
    return mon.Monitor(setup, mon.Readers(value=readers.value, exams=readers.exams, bands=readers.bands))


def _events(run: Path, rows: list[dict[str, Any]]) -> None:
    with (run / "logs" / f"events_{_RUN}_seg0001.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def test_each_save_is_read_once_paired_with_its_lagged_net_and_a_halting_row_sends_one_sigterm(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save 3k pairs with the parent on the games after it, 6k with 3k; 6k's low exam halts: ONE SIGTERM, then no read."""
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run = _run_dir(tmp_path, victim.pid)
        readers = _Readers(fail_at=6000)
        monitor = _monitor(tmp_path, run, readers, halt=True)
        paired: list[tuple[str, str]] = []
        monkeypatch.setattr(mon._vi, "lagged", lambda a: paired.append((a.current.name, a.lagged.name)) or {
            "diff": -0.01, "ci": [-0.02, 0.0], "se_game": 0.005, "games": 4, "detection": False, "worse": False,
            "bands": {}})
        tail = EventTail(run, _RUN, mon.EVENTS)
        c1, c2, c3 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8)), _save(run, 9000, range(6, 12))
        _events(run, [{"event": "iteration_complete", "ts": 10.0, "games_total": 4, "positions_produced_total": 8,
                       "step": 3000},
                      {"event": "periodic_checkpoint_save", "step": 3000, "path": str(c1), "ts": 20.0},
                      {"event": "trainer_step", "step": 3001},
                      {"event": "periodic_checkpoint_save", "step": 6000, "path": str(c2), "ts": 30.0},
                      {"event": "periodic_checkpoint_save", "step": 9000, "path": str(c3), "ts": 40.0}])
        assert cli.follow(monitor, tail, poll_s=0.0, final_timeout_s=0.0) == 3
        assert victim.wait(timeout=10) == -signal.SIGTERM
        saves = sorted(p.name for p in (tmp_path / "records" / "saves").iterdir())
        assert saves == ["00003000.json", "00006000.json"], "a halted monitor reads no further save"
        assert ("unseen_00003000_current.json", "unseen_00003000_lagged.json") in paired
        assert (c1.name, "unseen_00003000.bin", "ring_00003000.bin") in readers.values, "the gap's train side is its own ring"
        assert ("parent.ckpt", "unseen_00003000.bin", None) in readers.values
        record = json.loads((tmp_path / "records" / "saves" / "00006000.json").read_text(encoding="utf-8"))
        assert record["lagged_of"]["step"] == 3000 and record["lagged_of"]["unseen"]["games"] == 4
        assert record["halting_rows"] == ["T4_V calibrated 0.1000 below the floor 0.154"]
        halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
        assert halt["armed"] and halt["sent"]["pid"] == victim.pid
    finally:
        victim.kill()


def test_an_unarmed_monitor_records_the_halt_and_signals_nothing(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, pid=2**22 + 7)
    monitor = _monitor(tmp_path, run, _Readers(fail_at=3000), halt=False)
    record = monitor.read_save(3000, _save(run, 3000, range(0, 4)), 20.0)
    assert record["halting_rows"] and json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))["armed"] is False


def test_a_stop_reads_its_final_save_once_its_ring_lands(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, pid=2**22 + 7)
    monitor = _monitor(tmp_path, run, _Readers(fail_at=None), halt=False)
    _events(run, [{"event": "shutdown_save", "step": 4321}])
    tail = EventTail(run, _RUN, mon.EVENTS)
    _save(run, 4321, range(0, 4))
    assert cli.follow(monitor, tail, poll_s=0.0, final_timeout_s=5.0) == 0
    assert (tmp_path / "records" / "saves" / "00004321.json").is_file()


def test_a_gen_ring_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    (tmp_path / "gen.bin").write_bytes(b"x")
    (tmp_path / "exams.jsonl").write_text('{"exam": "T4_V", "id": "a", "stones": [], "sign": 1}\n', encoding="utf-8")
    argv = ["once", "--run-dir", str(tmp_path), "--run-id", _RUN, "--out", str(tmp_path / "o"), "--gen-ring",
            str(tmp_path / "gen.bin"), "--gen-sha256", "0" * 64, "--exams", str(tmp_path / "exams.jsonl"),
            "--exams-sha256", "0" * 64, "--bands", "b.md", "--floors", "T4_V=0.154", "--parent", "p.ckpt",
            "--batches", "1", "--device", "cpu", "--threads", "1"]
    assert cli.main(argv) == 2
