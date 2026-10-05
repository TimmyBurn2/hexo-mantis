"""The dash reader of the run monitor's records, pinned against the monitor's real output; null reads stay unmeasured."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import HexgBuffer
from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
mon = importlib.import_module("run_monitor.monitor")

_RUN = "producerrun"


@pytest.fixture(scope="module")
def saves(dash):
    return importlib.import_module("dash.readers.saves")


def _ring(path: Path, games: range) -> None:
    buf = HexgBuffer(64, "gnn_axis_v1", 128)
    for g in games:
        for ply in (2, 3):
            buf.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, ply, True,
                                    1.0 if g % 2 else -1.0, True, 10 + g, game_id=g)
    buf.save_to_path(str(path))


class _Readers:
    """The monitor's three reads with no net: a GEN body (no temperature when asked), passing exams, clean bands."""

    def __init__(self, temperature: float | None = 1.2) -> None:
        self.temperature = temperature

    def value(self, ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        band = {"n": 10, "cf_ce": 0.61}
        body = {"heldout": {"overall": {"cf_ce": 0.6, "temperature": self.temperature, "auc": 0.7}, "policy_ce": 2.3,
                            "plies_0_10": band, "plies_11_40": band, "plies_41_up": band}}
        return body if train is None else {**body, "gap": {"cf_ce": 0.01}}

    def exams(self, ckpt: Path, gen_read: Path, floors: dict[str, float]) -> dict[str, Any]:
        return {k: {"calibrated_mean": 0.3, "floor": f, "holds": True} for k, f in floors.items()}

    def bands(self, ring: Path, events: Path | None, bands: Path) -> dict[str, Any]:
        return {"rows": {"cap_rate": 0.01}, "misses": []}


def _produce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temperature: float | None = 1.2) -> Path:
    """Two saves read by the real `run_monitor.Monitor`, so the second record carries its lagged read; returns the records dir."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    monkeypatch.setattr(mon._vi, "lagged", lambda a: {"diff": -0.01, "ci": [-0.02, -0.001], "se_game": 0.005,
                                                       "games": 4, "detection": True, "worse": False, "bands": {}})
    setup = mon.Setup(run_dir=run, run_id=_RUN, out=tmp_path / "records", gen_ring=tmp_path / "gen.bin", gen_sha256="",
                      exams=tmp_path / "e.jsonl", exams_sha256="", bands=tmp_path / "b.md", bands_sha256="",
                      floors={"T4_V": 0.154, "DEF_V_att": 0.1}, line=0.012, parent=run / "parent.ckpt", batches=1,
                      device="cpu", threads=1, halt=False, gap_line=0.05, floors_from_first_pass=False,
                      bands_from_step=0)
    readers = _Readers(temperature)
    monitor = mon.Monitor(setup, mon.Readers(value=readers.value, exams=readers.exams, bands=readers.bands))
    for step, games in ((3000, range(0, 4)), (6000, range(2, 8))):
        ckpt = run / "checkpoints" / f"{_RUN}_{step:08d}_0000abcd.ckpt"
        ckpt.write_bytes(f"stand-in {step}".encode())
        _ring(Path(f"{ckpt}.ring.bin"), games)
        monitor.read_save(step, ckpt, float(step), stopping=False)
    return tmp_path / "records"


def test_the_monitors_real_save_record_carries_every_pinned_key(saves, tmp_path, monkeypatch):
    records = _produce(tmp_path, monkeypatch)
    raw = json.loads((records / "saves" / "00006000.json").read_text(encoding="utf-8"))
    assert saves.missing_keys(raw) == []
    read = saves.load(records)
    second = read.saves[-1]
    assert [s.step for s in read.saves] == [3000, 6000]
    assert (second.cf_ce, second.temperature, second.auc, second.policy_ce) == (0.6, 1.2, 0.7, 2.3)
    assert second.bands == {b: 0.61 for b in saves.PLY_BANDS}
    assert second.exams["T4_V"].holds is True and second.exams["T4_V"].floor == 0.154
    assert second.lagged.step == 3000 and second.lagged.diff == -0.01 and second.lagged.gap_cf_ce == 0.01
    assert second.gap == 0.01 and second.gap_line == 0.05 and second.gap_fired is False
    assert second.floors_live is True and second.halting_rows == ()


@pytest.mark.parametrize("key", ["gen.cf_ce", "exams.T4_V.holds", "lagged_of.current.gap.cf_ce", "gap_rule.line"])
def test_the_pin_checker_bites_on_a_record_missing_a_key(saves, tmp_path, monkeypatch, key):
    records = _produce(tmp_path, monkeypatch)
    raw = json.loads((records / "saves" / "00006000.json").read_text(encoding="utf-8"))
    *parents, leaf = key.split(".")
    node = raw
    for part in parents:
        node = node[part]
    del node[leaf]
    assert saves.missing_keys(raw) == [key]


def test_a_gen_read_with_no_temperature_leaves_the_exams_unmeasured_not_zero(saves, tmp_path, monkeypatch):
    records = _produce(tmp_path, monkeypatch, temperature=None)
    first = saves.load(records).saves[0]
    assert first.temperature is None
    assert all(e.holds is None and e.calibrated_mean is None and e.not_measured for e in first.exams.values())


def test_an_unparseable_save_is_skipped_by_name_and_the_halt_and_gap_rule_are_read(saves, tmp_path):
    root = tmp_path / "records"
    (root / "saves").mkdir(parents=True)
    (root / "saves" / "00003000.json").write_text("{not json", encoding="utf-8")
    (root / "saves" / "00006000.json").write_text(json.dumps({"step": 6000}), encoding="utf-8")
    (root / "HALT.json").write_text(json.dumps({"step": 6000, "halting_rows": ["T4_V"]}), encoding="utf-8")
    (root / "GAP_RULE.json").write_text(json.dumps({"step": 6000, "over": [3000, 6000]}), encoding="utf-8")
    read = saves.load(root)
    assert read.skipped == ("00003000.json",) and [s.step for s in read.saves] == [6000]
    assert read.saves[0].cf_ce is None and read.saves[0].lagged is None
    assert read.halt["halting_rows"] == ["T4_V"] and read.gap_rule["over"] == [3000, 6000]


def test_a_records_dir_without_saves_reads_empty(saves, tmp_path):
    read = saves.load(tmp_path)
    assert read.saves == () and read.halt is None and read.gap_rule is None


class _Halting(_Readers):
    """Clean reads but a ring-band miss at every save, which halts at once; and a gap above the line at every lagged read."""

    def value(self, ckpt, ring, out, train):
        body = super().value(ckpt, ring, out, train)
        return body if train is None else {**body, "gap": {"cf_ce": 0.2}}

    def bands(self, ring, events, bands):
        return {"rows": {"cap_rate": 0.9}, "misses": ["cap_rate: 0.9 not lt 0.05"]}


def test_the_monitors_real_halt_and_gap_rule_files_carry_their_pinned_keys(saves, tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    monkeypatch.setattr(mon._vi, "lagged", lambda a: {"diff": 0.01, "ci": [0.001, 0.02], "se_game": 0.005, "games": 4,
                                                       "detection": True, "worse": True, "bands": {}})
    setup = mon.Setup(run_dir=run, run_id=_RUN, out=tmp_path / "records", gen_ring=tmp_path / "gen.bin", gen_sha256="",
                      exams=tmp_path / "e.jsonl", exams_sha256="", bands=tmp_path / "b.md", bands_sha256="",
                      floors={"T4_V": 0.154}, line=0.012, parent=run / "parent.ckpt", batches=1, device="cpu", threads=1,
                      halt=False, gap_line=0.05, floors_from_first_pass=False, bands_from_step=10**9)
    readers = _Halting()
    monitor = mon.Monitor(setup, mon.Readers(value=readers.value, exams=readers.exams, bands=readers.bands))
    for step, games in ((3000, range(0, 4)), (6000, range(2, 8)), (9000, range(6, 12))):
        ckpt = run / "checkpoints" / f"{_RUN}_{step:08d}_0000abcd.ckpt"
        ckpt.write_bytes(f"stand-in {step}".encode())
        _ring(Path(f"{ckpt}.ring.bin"), games)
        monitor.read_save(step, ckpt, float(step), stopping=False)
    gap = json.loads((tmp_path / "records" / "GAP_RULE.json").read_text(encoding="utf-8"))
    assert saves.missing_keys(gap, saves.PINNED_GAP_RULE) == [] and gap["fired"] is True
    monitor.setup = mon.Setup(**{**monitor.setup.__dict__, "bands_from_step": 0})
    ckpt = run / "checkpoints" / f"{_RUN}_00012000_0000abcd.ckpt"
    ckpt.write_bytes(b"stand-in 12000")
    _ring(Path(f"{ckpt}.ring.bin"), range(10, 16))
    monitor.read_save(12000, ckpt, 12000.0, stopping=False)
    halt = json.loads((tmp_path / "records" / "HALT.json").read_text(encoding="utf-8"))
    assert saves.missing_keys(halt, saves.PINNED_HALT) == [] and halt["armed"] is False
    read = saves.load(tmp_path / "records")
    assert read.halt["step"] == 12000 and read.gap_rule["over"] == [3000, 6000]
    del halt["armed"]
    assert saves.missing_keys(halt, saves.PINNED_HALT) == ["armed"]
