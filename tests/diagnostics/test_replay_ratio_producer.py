"""`ring_audit.replay_ratio`: samples consumed ÷ positions produced over the ring's span, off `--events`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mantis.diagnostics import ring_audit as A

_BINS = [10] * 12


def _row(ts: float, samples: int | None, positions: int | None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"event": "iteration_complete", "ts": ts, "step": 1, "sym_draws": {"bins": _BINS, "empty_skipped": 0}}
    if samples is not None:
        row["samples_consumed_total"] = samples
    if positions is not None:
        row["positions_produced_total"] = positions
    return {**row, **extra}


def _events(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _replay(events: Path | None, ring_size: int) -> A.Row:
    return next(r for r in A.event_rows(events, ring_size=ring_size) if r.key == "replay_ratio")


def test_the_ratio_is_the_diff_of_the_pair_over_the_rings_span(tmp_path: Path) -> None:
    """Four rows; the ring holds 1 000 positions, so the span starts at the first row within 1 000 of the last."""
    rows = [_row(0.0, 0, 0), _row(1800.0, 900, 300), _row(3600.0, 2700, 800), _row(7200.0, 6300, 1800)]
    row = _replay(_events(tmp_path / "e.jsonl", rows), ring_size=1000)
    # rows 2 and 3 are within 1 000 positions of the last (800 ≥ 1 800 − 1 000): Δsamples 3 600 ÷ Δpositions 1 000.
    assert row.value == pytest.approx(3.6)
    assert row.n == 2
    assert "span 1.00 h" in row.note and "positions 800 → 1800" in row.note and "samples 2700 → 6300" in row.note


def test_a_stream_shorter_than_the_ring_spans_every_row(tmp_path: Path) -> None:
    rows = [_row(0.0, 0, 0), _row(3600.0, 512, 256)]
    row = _replay(_events(tmp_path / "e.jsonl", rows), ring_size=100_000)
    assert row.value == pytest.approx(2.0) and row.n == 2 and "span 1.00 h" in row.note


def test_rows_without_the_pair_are_not_measured_and_say_which_key_is_missing(tmp_path: Path) -> None:
    rows = [_row(0.0, None, 0), _row(3600.0, None, 300)]
    row = _replay(_events(tmp_path / "e.jsonl", rows), ring_size=1000)
    assert row.value is None and "samples_consumed_total" in row.note


def test_one_row_is_no_span(tmp_path: Path) -> None:
    row = _replay(_events(tmp_path / "e.jsonl", [_row(0.0, 512, 256)]), ring_size=1000)
    assert row.value is None and "one row" in row.note


def test_a_zero_position_delta_is_not_measured_never_a_division(tmp_path: Path) -> None:
    rows = [_row(0.0, 0, 300), _row(3600.0, 900, 300)]
    row = _replay(_events(tmp_path / "e.jsonl", rows), ring_size=1000)
    assert row.value is None and "no position" in row.note


def test_the_cli_takes_events_and_prints_the_two_rows(tmp_path: Path, capsys) -> None:
    from mantis import _engine

    buf = _engine.HexgBuffer(8, "gnn_axis_r8", 16)
    for i in range(8):
        buf.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 1.0)], 1, 2, 2, True, 1.0, True, 20, i, 0.0)
    ring = tmp_path / "r.ring.bin"
    buf.save_to_path(str(ring))
    # The ring holds 8 rows, so the span is every events row within 8 positions of the last.
    events = _events(tmp_path / "e.jsonl", [_row(0.0, 0, 0), _row(3600.0, 18, 5)])
    assert A.main([str(ring), "--events", str(events)]) == 0
    out = capsys.readouterr().out
    assert "replay_ratio" in out and "3.6" in out and "sym_bin0_over_mean" in out and "NOT MEASURED" not in out.split("replay_ratio")[1].splitlines()[0]
