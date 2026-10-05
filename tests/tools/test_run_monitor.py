"""The run monitor: the games after a save, one signal, a restart, a resumed run, the floors' first pass and the gap rule's record."""
# >300 justify (R8): every scenario drives one harness, a stand-in run that counts its signals over real rings
# and event segments; split, each half would rebuild that run and its fixtures.
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
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


class _Readers:
    """Reads with no net: a GEN body (no temperature where chosen), exams failing where chosen, a gap, bands, a raise."""

    def __init__(self, fail_at: tuple[int, ...] = (), raise_at: int | None = None, gap: float = 0.0,
                 no_skill_at: tuple[int, ...] = (), band_miss_at: tuple[int, ...] = (), nan_rows: bool = False) -> None:
        self.fail_at, self.raise_at, self.gap, self.values = fail_at, raise_at, gap, []
        self.no_skill_at, self.band_miss_at, self.nan_rows = no_skill_at, band_miss_at, nan_rows

    def value(self, ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        if self.raise_at is not None and f"_{self.raise_at:08d}_" in ckpt.name:
            self.raise_at = None  # transient, as an out-of-memory is
            raise RuntimeError("CUDA out of memory (planted)")
        self.values.append((ckpt.name, ring.name, None if train is None else train.name))
        band = {"n": 10, "cf_ce": 0.6}
        t = None if train is None and any(f"_{x:08d}_" in ckpt.name for x in self.no_skill_at) else 2.0
        body = {"heldout": {"overall": {"cf_ce": 0.6, "temperature": t, "auc": 0.7}, "policy_ce": 2.3,
                            "plies_0_10": band, "plies_11_40": band, "plies_41_up": band}}
        return body if train is None else {**body, "gap": {"cf_ce": self.gap}}

    def exams(self, ckpt: Path, gen_read: Path, floors: dict[str, float]) -> dict[str, Any]:
        low = any(f"_{step:08d}_" in ckpt.name for step in self.fail_at)
        return {"T4_V": _exam(0.10 if low else 0.29, floors["T4_V"]), "DEF_V_att": _exam(0.21, floors["DEF_V_att"])}

    def bands(self, ring: Path, events: Path | None, bands: Path) -> dict[str, Any]:
        miss = any(ring.name == f"ring_{x:08d}.bin" for x in self.band_miss_at)
        rows = {"unwritable": float("nan")} if miss and self.nan_rows else {}
        return {"rows": rows, "misses": ["cap_rate: 0.07 not lt 0.05"] if miss else []}


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


def _monitor(tmp_path: Path, run: Path, readers: _Readers, *, halt: bool, first_pass: bool = False) -> Any:
    setup = mon.Setup(run_dir=run, run_id=_RUN, out=tmp_path / "records", gen_ring=tmp_path / "gen.bin", gen_sha256="",
                      exams=tmp_path / "exams.jsonl", exams_sha256="", bands=tmp_path / "bands.md", bands_sha256="",
                      floors={"T4_V": 0.154, "DEF_V_att": 0.100}, line=0.012, parent=run / "parent.ckpt", batches=1,
                      device="cpu", threads=1, halt=halt, gap_line=0.05, floors_from_first_pass=first_pass,
                      bands_from_step=0)
    return mon.Monitor(setup, mon.Readers(value=readers.value, exams=readers.exams, bands=readers.bands))


def _events(run: Path, rows: list[dict[str, Any]], segment: int = 1) -> None:
    with (run / "logs" / f"events_{_RUN}_seg{segment:04d}.jsonl").open("a", encoding="utf-8") as fh:
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
    steps = (3000, 6000, 9000, 12000, 15000, 18000)
    ckpts = [_save(run, step, range(2 * i, 2 * i + 4)) for i, step in enumerate(steps)]
    c1 = ckpts[0]
    _events(run, [{"event": "iteration_complete", "ts": 10.0, "games_total": 4, "positions_produced_total": 8,
                   "step": 3000}, *(_saved(step, c) for step, c in zip(steps, ckpts, strict=True))])
    readers = _Readers(fail_at=(6000, 12000, 15000))
    assert cli.follow(_monitor(tmp_path, run, readers, halt=True), EventTail(run, _RUN, mon.EVENTS), 0.0, 0.0) == 3
    time.sleep(0.3)
    assert _sigterms(count) == 1 and proc.poll() is None, "one SIGTERM, and the stand-in run kept running"
    saves = sorted(p.name for p in (tmp_path / "records" / "saves").iterdir())
    assert saves == [f"{step:08d}.json" for step in steps[:5]], "the miss at 6000 armed, 9000 disarmed, 15000 fired"
    assert ("unseen_00003000_current.json", "unseen_00003000_lagged.json") in paired
    assert (c1.name, "unseen_00003000.bin", "ring_00003000.bin") in readers.values, "the gap's train side is its own ring"
    assert ("parent.ckpt", "unseen_00003000.bin", None) in readers.values
    record = json.loads((tmp_path / "records" / "saves" / "00006000.json").read_text(encoding="utf-8"))
    assert record["lagged_of"]["step"] == 3000 and record["lagged_of"]["unseen"]["games"] == 2
    assert record["halting_rows"] == [] and record["armed_floors"] == ["T4_V"]
    fired = json.loads((tmp_path / "records" / "saves" / "00015000.json").read_text(encoding="utf-8"))
    assert fired["halting_rows"] == ["T4_V calibrated 0.1000 below the floor 0.154, its second miss in a row"]
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert halt["step"] == 15000 and halt["armed"] and halt["signal"]["sent"] is True and halt["signal"]["pid"] == proc.pid

    again = _Readers(fail_at=(6000, 12000, 15000))
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
    _monitor(tmp_path, run, _Readers(), halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    Path(f"{c1}.ring.bin").unlink()  # the run prunes old bundles' rings
    c3 = _save(run, 9000, range(6, 12))
    _events(run, [_saved(9000, c3)])
    readers = _Readers()
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
    record = _monitor(tmp_path, run, _Readers(), halt=True).read_save(3000, ckpt, 1.0, stopping=False)
    assert record["halting_rows"] == [] and "not_measured" in record["ring_bands"] and _sigterms(count) == 0


def test_a_halting_row_at_the_final_save_is_recorded_and_never_signalled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1 = _save(run, 3000, range(0, 4))
    _save(run, 4321, range(2, 6))
    _events(run, [_saved(3000, c1), {"event": "shutdown_save", "step": 4321}])
    monitor = _monitor(tmp_path, run, _Readers(fail_at=(3000, 4321)), halt=True)
    assert cli.follow(monitor, EventTail(run, _RUN, mon.EVENTS), 0.0, 5.0) == 3
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert halt["step"] == 4321 and halt["final_save"] and "signal" not in halt and _sigterms(count) == 0


def test_a_failed_read_is_recorded_and_the_next_save_is_still_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, _count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1, c2 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8))
    c1.unlink()  # the checkpoint's own bytes are gone: the read itself fails, not one reader
    _events(run, [_saved(3000, c1), _saved(6000, c2)])
    _monitor(tmp_path, run, _Readers(), halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    log = [json.loads(x) for x in (tmp_path / "records" / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in log] == ["save_read_failed", "save_read"]
    assert (tmp_path / "records" / "saves" / "00006000.json").is_file()


def test_a_run_that_died_without_a_stop_event_has_its_newest_save_read_and_the_follow_ends(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, pid=2**22 + 7)  # no such process: the run is gone
    _save(run, 5100, range(0, 4))
    monitor = _monitor(tmp_path, run, _Readers(), halt=True)
    assert not monitor.run_alive()
    assert cli.follow(monitor, EventTail(run, _RUN, mon.EVENTS), 0.1, 1.0) == 0
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
            "--gpu-mem-fraction", "0.25", "--threads", "1", "--gap-line", "0.05", "--floors-from", "start",
            "--bands-from-step", "0"]
    assert cli.main(argv) == 2


def test_a_resumed_run_is_followed_past_its_earlier_stop_and_its_first_save_pairs_with_the_stop_save(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, _count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1, stop = _save(run, 3000, range(0, 4)), _save(run, 4321, range(2, 6))
    _events(run, [_saved(3000, c1), {"event": "shutdown_save", "step": 4321}])
    assert cli.follow(_monitor(tmp_path, run, _Readers(), halt=True), EventTail(run, _RUN, mon.EVENTS), 0.0, 5.0) == 0

    def second_life() -> None:  # the resumed run's segment lands after the restarted monitor's first poll
        c3, _c4 = _save(run, 6000, range(4, 9)), _save(run, 7000, range(6, 11))
        _events(run, [{"event": "run_segment_started", "segment": 2}, _saved(6000, c3),
                      {"event": "shutdown_save", "step": 7000}], segment=2)
    later = threading.Timer(1.0, second_life)
    later.start()
    readers = _Readers()
    assert cli.follow(_monitor(tmp_path, run, readers, halt=True), EventTail(run, _RUN, mon.EVENTS), 0.1, 0.5) == 0
    later.join()
    saves = sorted(p.name for p in (tmp_path / "records" / "saves").iterdir())
    assert saves == ["00003000.json", "00004321.json", "00006000.json", "00007000.json"], "the old stop ended the follow"
    record = json.loads((tmp_path / "records" / "saves" / "00006000.json").read_text(encoding="utf-8"))
    assert record["lagged_of"]["step"] == 4321 and record["lagged_of"]["lagged_net"] == str(c1)
    assert stop.name in {v[0] for v in readers.values}


def test_floors_report_until_their_first_pass_then_halt_on_two_reads_across_a_restart_and_an_unread_save(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    """3000 misses (reported), 6000 passes (live), 9000 misses (armed); restart; 12000's GEN read fails; 15000 fires."""
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    steps = (3000, 6000, 9000, 12000, 15000)
    ckpts = {step: _save(run, step, range(2 * i, 2 * i + 4)) for i, step in enumerate(steps)}
    readers = _Readers(fail_at=(3000, 9000, 12000, 15000), raise_at=12000)
    _events(run, [_saved(step, ckpts[step]) for step in steps[:3]])
    _monitor(tmp_path, run, readers, halt=True, first_pass=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    early = json.loads((tmp_path / "records" / "saves" / "00003000.json").read_text(encoding="utf-8"))
    assert early["halting_rows"] == [] and early["reported_rows"] and early["floors_live"] is False
    armed = json.loads((tmp_path / "records" / "saves" / "00009000.json").read_text(encoding="utf-8"))
    assert armed["armed_floors"] == ["T4_V"] and not (tmp_path / "records" / "HALT.json").exists()
    _events(run, [_saved(step, ckpts[step]) for step in steps[3:]])
    _monitor(tmp_path, run, readers, halt=True, first_pass=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    time.sleep(0.3)
    unread = json.loads((tmp_path / "records" / "saves" / "00012000.json").read_text(encoding="utf-8"))
    assert unread["exams"]["T4_V"]["holds"] is None and unread["armed_floors"] == ["T4_V"], "an unread save keeps the arm"
    assert json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))["step"] == 15000
    assert _sigterms(count) == 1


def test_a_gen_read_with_no_skill_leaves_the_floors_unread_and_a_band_miss_still_halts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    c1, c2 = _save(run, 3000, range(0, 4)), _save(run, 6000, range(2, 8))
    _events(run, [_saved(3000, c1), _saved(6000, c2)])
    readers = _Readers(fail_at=(3000,), no_skill_at=(3000, 6000), band_miss_at=(6000,))
    _monitor(tmp_path, run, readers, halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    time.sleep(0.3)
    first = json.loads((tmp_path / "records" / "saves" / "00003000.json").read_text(encoding="utf-8"))
    assert first["gen"]["temperature"] is None and first["exams"]["T4_V"]["holds"] is None
    assert first["halting_rows"] == [] and first["armed_floors"] == []
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert halt["step"] == 6000 and halt["halting_rows"] == ["ring band cap_rate: 0.07 not lt 0.05"]
    assert _sigterms(count) == 1


def test_a_resume_under_a_stale_heartbeat_is_followed_by_its_new_segments_pid(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, _count = victim
    run = _run_dir(tmp_path, 2**22 + 7)  # the earlier life's heartbeat: its process is gone
    _no_lagged(monkeypatch, [])
    c1 = _save(run, 3000, range(0, 4))
    _events(run, [_saved(3000, c1), {"event": "shutdown_save", "step": 3000}])
    _events(run, [{"event": "run_segment_started", "segment": 2, "pid": proc.pid}], segment=2)
    monitor = _monitor(tmp_path, run, _Readers(), halt=True)
    follower = threading.Thread(target=cli.follow, args=(monitor, EventTail(run, _RUN, mon.EVENTS), 0.1, 1.0))
    follower.start()
    time.sleep(2.5)
    assert follower.is_alive(), "the follow ended on the earlier life's heartbeat"
    c2, _c3 = _save(run, 6000, range(2, 8)), _save(run, 7000, range(4, 10))
    _events(run, [_saved(6000, c2), {"event": "shutdown_save", "step": 7000}], segment=2)
    follower.join(timeout=10)
    assert not follower.is_alive()
    saves = sorted(p.name for p in (tmp_path / "records" / "saves").iterdir())
    assert saves == ["00003000.json", "00006000.json", "00007000.json"]


def test_the_gap_rule_records_its_verdict_and_never_signals_the_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    _no_lagged(monkeypatch, [])
    ckpts = [_save(run, step, range(2 * i, 2 * i + 4)) for i, step in enumerate((3000, 6000, 9000))]
    _events(run, [_saved(step, c) for step, c in zip((3000, 6000, 9000), ckpts, strict=True)])
    _monitor(tmp_path, run, _Readers(gap=0.08), halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    rule = json.loads((tmp_path / "records" / "GAP_RULE.json").read_text(encoding="utf-8"))
    assert rule["over"] == [3000, 6000] and rule["fired"] and rule["step"] == 9000
    assert _sigterms(count) == 0 and not (tmp_path / "records" / "HALT.json").exists()


def test_a_halt_is_sent_even_when_its_save_record_cannot_be_written(tmp_path: Path, victim: Any) -> None:
    proc, count = victim
    run = _run_dir(tmp_path, proc.pid)
    c1 = _save(run, 3000, range(0, 4))
    _events(run, [_saved(3000, c1)])
    readers = _Readers(band_miss_at=(3000,), nan_rows=True)  # the band rows carry a NaN the record refuses
    _monitor(tmp_path, run, readers, halt=True).on_events(EventTail(run, _RUN, mon.EVENTS).read_new())
    time.sleep(0.3)
    assert not (tmp_path / "records" / "saves" / "00003000.json").exists()
    assert json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))["step"] == 3000
    assert _sigterms(count) == 1
