"""Planted breaks for R342(b)(iv)'s rate bar — each condemnation path must fire on its own.

R342(b)(iv) condemns with no further ruling, so the cost of this check being wrong is either a
run that should have stopped or an instance destroyed for nothing. Every branch is planted here,
including the two ways it could wrongly report CLEAR.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.diagnostics.f816_37_rate_bar import MAX_IN_WINDOW, WINDOW_SEC, evaluate, worst_window


def _dump(record: Path, when_ms: int, path_name: str = "selfplay") -> None:
    d = record / "collate_dumps"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"collate_dump_r1_{when_ms}.json").write_text(
        json.dumps({"finding": "F-816-37", "path": path_name,
                    "error": "edge axis one-hot is not a clean one-hot"}),
        encoding="utf-8",
    )


def test_clean_record_clears(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "run.log").write_text("step 1 ok\n", encoding="utf-8")
    assert evaluate(tmp_path) == 0
    assert "BAR CLEAR" in capsys.readouterr().out


def test_three_in_window_clears_and_four_condemns(tmp_path: Path) -> None:
    """The bar is *more than* 3, so 3 clears and 4 condemns. Off-by-one is the whole risk."""
    (tmp_path / "run.log").write_text("ok\n", encoding="utf-8")
    base = 1_800_000_000_000
    for i in range(MAX_IN_WINDOW):
        _dump(tmp_path, base + i * 1000)
    assert evaluate(tmp_path) == 0, "exactly 3 in a window is AT the limit, not over it"
    _dump(tmp_path, base + MAX_IN_WINDOW * 1000)
    assert evaluate(tmp_path) == 1, "4 in one window must CONDEMN"


def test_the_window_rolls_rather_than_bucketing(tmp_path: Path) -> None:
    """Four firings spanning under 12 h condemn even when spread across the record.

    A fixed-bucket implementation would clear this, which is the defect this pins.
    """
    (tmp_path / "run.log").write_text("ok\n", encoding="utf-8")
    base = 1_800_000_000_000
    span_ms = (WINDOW_SEC - 60) * 1000
    for i in range(4):
        _dump(tmp_path, base + (span_ms // 3) * i)
    assert evaluate(tmp_path) == 1


def test_four_firings_spread_over_more_than_twelve_hours_clear(tmp_path: Path) -> None:
    """The converse: the bar is a RATE, so the same four spread wide must not condemn."""
    (tmp_path / "run.log").write_text("ok\n", encoding="utf-8")
    base = 1_800_000_000_000
    for i in range(4):
        _dump(tmp_path, base + (WINDOW_SEC + 600) * 1000 * i)
    assert evaluate(tmp_path) == 0


@pytest.mark.parametrize(
    "line, where",
    [
        ("CheckpointStampError: content hash abc12345 disagrees with the filename sha8 def67890",
         "weights"),
        ("anchor sha256 mismatch (fresh-init seed)", "weights"),
        ("GraphContractError: ring header checksum failed", "ring headers"),
        ("IndexError: index out of range in legal_offsets", "indices"),
    ],
)
def test_any_out_of_wire_firing_condemns_alone(tmp_path: Path, line: str, where: str) -> None:
    """One out-of-wire signature condemns with ZERO in-wire firings and no rate to consult."""
    (tmp_path / "run.log").write_text(f"step 4 ok\n{line}\n", encoding="utf-8")
    assert evaluate(tmp_path) == 1, f"{where} must condemn on location alone"


def test_out_of_wire_condemns_even_when_the_rate_is_clear(tmp_path: Path) -> None:
    """The two channels are independent — a clear rate must not mask a weights firing."""
    (tmp_path / "run.log").write_text(
        "anchor sha256 mismatch (fresh-init seed)\n", encoding="utf-8")
    _dump(tmp_path, 1_800_000_000_000)
    assert evaluate(tmp_path) == 1


# the two ways it could wrongly report CLEAR
def test_missing_record_refuses(tmp_path: Path) -> None:
    assert evaluate(tmp_path / "nope") == 2


def test_empty_record_refuses(tmp_path: Path) -> None:
    """An empty run record is not a clean one — a scan of nothing must not read as a pass."""
    (tmp_path / "empty").mkdir()
    assert evaluate(tmp_path / "empty") == 2


def test_unparseable_sidecar_raises_rather_than_scoring_zero(tmp_path: Path) -> None:
    """A dump that cannot be read is evidence, not an absence."""
    d = tmp_path / "collate_dumps"
    d.mkdir(parents=True)
    (d / "collate_dump_r1_1800000000000.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        evaluate(tmp_path)


def test_worst_window_is_empty_safe() -> None:
    assert worst_window([]) == (0, 0.0)
