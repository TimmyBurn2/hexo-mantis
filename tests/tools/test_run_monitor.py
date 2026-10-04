"""The run monitor: the games after a save, the halting rows and their planted breaks, the rates, one signal, a restart."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import load_ring
from mantis.monitor.event_tail import EventTail
from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
mon = importlib.import_module("run_monitor.monitor")
rules = importlib.import_module("run_monitor.rules")
rings = importlib.import_module("run_monitor.rings")
cli = importlib.import_module("run_monitor.cli")

_RUN = "monitoredrun7"
_ENC = "gnn_axis_v1"
#: A stand-in run: counts every SIGTERM in a file and keeps running, so a second signal is seen, not survived.
_VICTIM = ("import signal, sys, time\n"
           "def h(*_):\n    open(sys.argv[1], 'a', encoding='utf-8').write('TERM\\n')\n"
           "signal.signal(signal.SIGTERM, h)\ntime.sleep(60)\n")


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


_GOOD = {"T4_V": _exam(0.29, 0.154), "DEF_V_att": _exam(0.21, 0.100)}


def _assert_the_rows_halt(halting_rows: Any) -> None:
    """Below a floor halts, a band miss halts, the known-good and an unmeasured ring pass."""
    assert halting_rows(_GOOD, {"misses": []}) == []
    assert halting_rows(_GOOD, {"misses": [], "not_measured": "the ring was gone"}) == []
    low = {**_GOOD, "DEF_V_att": _exam(0.0995, 0.100)}
    assert halting_rows(low, {"misses": []}) == ["DEF_V_att calibrated 0.0995 below the floor 0.1"]
    assert halting_rows(_GOOD, {"misses": ["cap_rate: 0.07 not lt 0.05"]}) == ["ring band cap_rate: 0.07 not lt 0.05"]


@pytest.mark.parametrize("planted", ["exams ignored", "bands ignored"])
def test_a_reading_below_its_floor_or_a_band_outside_halts_and_each_planted_break_reds(planted: str) -> None:
    """PLANTED BREAKS: a checker that drops the exam rule, or the band rule, misses its halting row."""
    _assert_the_rows_halt(rules.halting_rows)
    broken = {"exams ignored": lambda exams, bands: [f"ring band {m}" for m in bands["misses"]],
              "bands ignored": lambda exams, bands: rules.halting_rows(exams, {"misses": []})}[planted]
    with pytest.raises(AssertionError):
        _assert_the_rows_halt(broken)


def test_the_rates_read_the_idle_windows_apart_and_a_missing_counter_is_no_zero() -> None:
    rows = [{"ts": 0.0, "games_total": 0, "positions_produced_total": 0, "step": 0},
            {"ts": 1800.0, "games_total": 100, "positions_produced_total": 9000, "step": 240},
            {"ts": 3600.0, "games_total": 120, "positions_produced_total": 10800, "step": 288}]
    out = rules.rates(rows, 0.0, 3600.0, busy=[(1900.0, 3500.0)])
    assert out["games_per_h"] == pytest.approx(120.0) and out["idle_games_per_h"] == pytest.approx(200.0)
    assert out["idle_h"] == pytest.approx(0.5) and out["steps_per_h"] == pytest.approx(288.0)
    assert "NOT MEASURED" in rules.rates(rows[:1], 0.0, 3600.0, busy=[])["note"]
    assert rules.counter_row({"event": "iteration_complete", "ts": 1.0, "games_total": 3, "step": 9}) is None


class _Readers:
    """Reads with no net: a fixed GEN body, exams that fail at the chosen step, clean bands, an optional raise."""

    def __init__(self, fail_at: int | None, raise_at: int | None = None) -> None:
        self.fail_at, self.raise_at, self.values = fail_at, raise_at, []

    def value(self, ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        if self.raise_at is not None and f"_{self.raise_at:08d}_" in ckpt.name:
            self.raise_at = None  # transient, as an out-of-memory is
            raise RuntimeError("CUDA out of memory (planted)")
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
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "logs" / f"heartbeat_{_RUN}.json").write_text(json.dumps({"pid": pid, "seq": 1}), encoding="utf-8")
    return run


def _save(run: Path, step: int, games: range) -> Path:
    ckpt = run / "checkpoints" / f"{_RUN}_{step:08d}_0000abcd.ckpt"
    ckpt.write_bytes(f"stand-in {step}".encode())
    _ring(Path(f"{ckpt}.ring.bin"), games)
    return ckpt


def _monitor(tmp_path: Path, run: Path, readers: _Readers, *, halt: bool) -> Any:
    setup = mon.Setup(run_dir=run, run_id=_RUN, out=tmp_path / "records", gen_ring=tmp_path / "gen.bin", gen_sha256="",
                      exams=tmp_path / "exams.jsonl", exams_sha256="", bands=tmp_path / "bands.md", bands_sha256="",
                      floors={"T4_V": 0.154, "DEF_V_att": 0.100}, line=0.012, parent=run / "parent.ckpt", batches=1,
                      device="cpu", threads=1, halt=halt)
    return mon.Monitor(setup, mon.Readers(value=readers.value, exams=readers.exams, bands=readers.bands))


def _events(run: Path, rows: list[dict[str, Any]]) -> None:
    with (run / "logs" / f"events_{_RUN}_seg0001.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def _saved(step: int, ckpt: Path) -> dict[str, Any]:
    return {"event": "periodic_checkpoint_save", "step": step, "path": str(ckpt), "ts": float(step)}


@pytest.fixture
def victim(tmp_path: Path) -> Any:
    count = tmp_path / "sigterms.txt"
    proc = subprocess.Popen([sys.executable, "-c", _VICTIM, str(count), _RUN])
    time.sleep(0.3)
    yield proc, count
    proc.kill()


def _sigterms(count: Path) -> int:
    return len(count.read_text(encoding="utf-8").splitlines()) if count.is_file() else 0


def _no_lagged(monkeypatch: pytest.MonkeyPatch, paired: list[tuple[str, str]]) -> None:
    monkeypatch.setattr(mon._vi, "lagged", lambda a: paired.append((a.current.name, a.lagged.name)) or {
        "diff": -0.01, "ci": [-0.02, 0.0], "se_game": 0.005, "games": 4, "detection": False, "worse": False,
        "bands": {}})


def test_one_halt_sends_one_sigterm_and_a_restarted_monitor_neither_rereads_nor_resends(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    paired: list[tuple[str, str]] = []
    _no_lagged(monkeypatch, paired)
    c1, c2, c3 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8)), _save(run, 9000, range(6, 12))
    _events(run, [{"event": "iteration_complete", "ts": 10.0, "games_total": 4, "positions_produced_total": 8,
                   "step": 3000}, _saved(3000, c1), _saved(6000, c2), _saved(9000, c3)])
    readers = _Readers(fail_at=6000)
    assert cli.follow(_monitor(tmp_path, run, readers, halt=True), EventTail(run, _RUN, mon.EVENTS), 0.0, 0.0) == 3
    time.sleep(0.3)
    assert _sigterms(count) == 1 and proc.poll() is None, "one SIGTERM, and the stand-in run kept running"
    saves = sorted(p.name for p in (tmp_path / "records" / "saves").iterdir())
    assert saves == ["00003000.json", "00006000.json"], "a halted monitor reads no further save"
    assert ("unseen_00003000_current.json", "unseen_00003000_lagged.json") in paired
    assert (c1.name, "unseen_00003000.bin", "ring_00003000.bin") in readers.values, "the gap's train side is its own ring"
    assert ("parent.ckpt", "unseen_00003000.bin", None) in readers.values
    record = json.loads((tmp_path / "records" / "saves" / "00006000.json").read_text(encoding="utf-8"))
    assert record["lagged_of"]["step"] == 3000 and record["lagged_of"]["unseen"]["games"] == 4
    assert record["halting_rows"] == ["T4_V calibrated 0.1000 below the floor 0.154"]
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert halt["armed"] and halt["signal"]["sent"] is True and halt["signal"]["pid"] == proc.pid

    again = _Readers(fail_at=6000)
    assert cli.follow(_monitor(tmp_path, run, again, halt=True), EventTail(run, _RUN, mon.EVENTS), 0.0, 0.0) == 3
    time.sleep(0.3)
    assert _sigterms(count) == 1 and again.values == [], "a restart re-read a save or re-sent the halt"


def test_a_restart_resumes_its_pairing_and_reads_only_the_new_saves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, _count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1, c2 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8))
    _events(run, [_saved(3000, c1), _saved(6000, c2)])
    _monitor(tmp_path, run, _Readers(fail_at=None), halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    Path(f"{c1}.ring.bin").unlink()  # the run prunes old bundles' rings
    c3 = _save(run, 9000, range(6, 12))
    _events(run, [_saved(9000, c3)])
    readers = _Readers(fail_at=None)
    _monitor(tmp_path, run, readers, halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    assert {v[0] for v in readers.values} == {c3.name, c2.name, c1.name}, "only 9000 read, 6000 paired with 3000"
    record = json.loads((tmp_path / "records" / "saves" / "00009000.json").read_text(encoding="utf-8"))
    assert record["lagged_of"]["step"] == 6000 and record["lagged_of"]["lagged_net"] == str(c1)
    assert not (tmp_path / "records" / "HALT.json").exists()


def test_a_save_whose_ring_is_gone_is_unmeasured_not_halted(tmp_path: Path, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    ckpt = _save(run, 3000, range(0, 4))
    Path(f"{ckpt}.ring.bin").unlink()
    record = _monitor(tmp_path, run, _Readers(fail_at=None), halt=True).read_save(3000, ckpt, 1.0, stopping=False)
    assert record["halting_rows"] == [] and "not_measured" in record["ring_bands"] and _sigterms(count) == 0


def test_a_halting_row_at_the_final_save_is_recorded_and_never_signalled(tmp_path: Path, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    _save(run, 4321, range(0, 4))
    _events(run, [{"event": "shutdown_save", "step": 4321}])
    monitor = _monitor(tmp_path, run, _Readers(fail_at=4321), halt=True)
    assert cli.follow(monitor, EventTail(run, _RUN, mon.EVENTS), 0.0, 5.0) == 3
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert halt["final_save"] and "signal" not in halt and _sigterms(count) == 0


def test_a_failed_read_is_recorded_and_the_next_save_is_still_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, _count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1, c2 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8))
    _events(run, [_saved(3000, c1), _saved(6000, c2)])
    _monitor(tmp_path, run, _Readers(fail_at=None, raise_at=3000), halt=True).on_events(
        EventTail(run, _RUN, mon.EVENTS).read_new())
    log = [json.loads(x) for x in (tmp_path / "records" / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in log] == ["save_read_failed", "save_read"]
    assert (tmp_path / "records" / "saves" / "00006000.json").is_file()


def test_a_run_that_died_without_a_stop_event_has_its_newest_save_read_and_the_follow_ends(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, pid=2**22 + 7)  # no such process: the run is gone
    _save(run, 5100, range(0, 4))
    monitor = _monitor(tmp_path, run, _Readers(fail_at=None), halt=True)
    assert not monitor.run_alive()
    assert cli.follow(monitor, EventTail(run, _RUN, mon.EVENTS), 0.0, 5.0) == 0
    assert (tmp_path / "records" / "saves" / "00005100.json").is_file()
    assert json.loads((tmp_path / "records" / "monitor_alive.json").read_text(encoding="utf-8"))["last_step"] == 5100


def test_a_pid_that_is_not_the_run_is_never_signalled(tmp_path: Path) -> None:
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run = _run_dir(tmp_path, other.pid)
        assert rules.halt_run(run, _RUN)["sent"] is False and other.poll() is None
    finally:
        other.kill()


def test_an_input_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    (tmp_path / "gen.bin").write_bytes(b"x")
    (tmp_path / "exams.jsonl").write_text('{"exam": "T4_V", "id": "a", "stones": [], "sign": 1}\n', encoding="utf-8")
    (tmp_path / "b.md").write_text("bands\n", encoding="utf-8")
    argv = ["once", "--run-dir", str(tmp_path), "--run-id", _RUN, "--out", str(tmp_path / "o"), "--gen-ring",
            str(tmp_path / "gen.bin"), "--gen-sha256", "0" * 64, "--exams", str(tmp_path / "exams.jsonl"),
            "--exams-sha256", "0" * 64, "--bands", str(tmp_path / "b.md"), "--bands-sha256", "0" * 64,
            "--floors", "T4_V=0.154", "--line", "0.012", "--parent", "p.ckpt", "--batches", "1", "--device", "cpu",
            "--gpu-mem-fraction", "0.25", "--threads", "1"]
    assert cli.main(argv) == 2
