"""`train.ring_preload`: a fresh launch loads its pinned ring before the pool exists; a checkpoint launch never does; bad rings are refused."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run
from mantis._engine import HexgBuffer
from mantis.config.schema import RingPreloadConfig
from mantis.run import _preload_ring
from mantis.train.pinned_ring import PinnedRingError
from mantis.util.hashing import sha256_file

ENC = "gnn_axis_r8"
VISITS = 8


def _ring(path: Path, rows: int) -> Path:
    """A real HEXG ring of `rows` positions, saved through the engine's own writer."""
    buf = HexgBuffer(max(rows, 8), ENC, VISITS)
    for i in range(rows):
        stones = [(q, 0, 1 if q % 2 == 0 else -1) for q in range(4 + i % 3)]
        buf.push_graph_position(stones, [(-1, 0, 0.6), (9, 0, 0.4)], 1, 2, i, True, 0.0, True, 30, 100 + i)
    buf.save_to_path(str(path))
    return path


def _spec(path: Path, sha: str | None = None) -> RingPreloadConfig:
    return RingPreloadConfig(ring=str(path), ring_sha256=sha or sha256_file(path))


def test_a_pinned_ring_loads_whole_into_the_empty_buffer(tmp_path: Path) -> None:
    ring = _ring(tmp_path / "r.bin", 24)
    buf = HexgBuffer(100, ENC, VISITS)
    assert _preload_ring(buf, _spec(ring)) == 24
    assert buf.size == 24


def test_null_loads_nothing(tmp_path: Path) -> None:
    buf = HexgBuffer(100, ENC, VISITS)
    assert _preload_ring(buf, None) == 0
    assert buf.size == 0


def test_a_ring_that_hashes_otherwise_is_refused_before_any_row_loads(tmp_path: Path) -> None:
    """A wrong digest refuses before any row loads (the mutation that reds it: drop the digest comparison)."""
    ring = _ring(tmp_path / "r.bin", 24)
    buf = HexgBuffer(100, ENC, VISITS)
    with pytest.raises(PinnedRingError, match="hashes sha256"):
        _preload_ring(buf, _spec(ring, sha="0" * 64))
    assert buf.size == 0


def test_a_missing_ring_is_refused_not_read_as_empty(tmp_path: Path) -> None:
    """The engine's loader returns 0 for a missing file; the pinned loader must refuse it first."""
    buf = HexgBuffer(100, ENC, VISITS)
    with pytest.raises(FileNotFoundError, match="not found"):
        _preload_ring(buf, RingPreloadConfig(ring=str(tmp_path / "absent.bin"), ring_sha256="0" * 64))


def test_a_ring_with_no_rows_is_refused(tmp_path: Path) -> None:
    ring = _ring(tmp_path / "empty.bin", 0)
    with pytest.raises(PinnedRingError, match="no row"):
        _preload_ring(HexgBuffer(100, ENC, VISITS), _spec(ring))


def test_the_engine_still_refuses_more_rows_than_the_capacity(tmp_path: Path) -> None:
    ring = _ring(tmp_path / "r.bin", 24)
    with pytest.raises(ValueError, match="capacity"):
        _preload_ring(HexgBuffer(16, ENC, VISITS), _spec(ring))


def _drive(monkeypatch: pytest.MonkeyPatch, smoke_run_config: Any, tmp_path: Path, checkpoint_path: str | None,
           restored: Any) -> list[str]:
    """Run `build_run_collaborators` with its heavy collaborators stubbed; returns the ring steps and the pool, in order."""
    seen: list[str] = []
    config = smoke_run_config("smoke_preflight_armed.yaml",
                              train={"ring_preload": {"ring": "x.bin", "ring_sha256": "0" * 64}})
    monkeypatch.setattr(mantis.run, "init_trainer", lambda **_kw: SimpleNamespace(model=None, arch=None))
    monkeypatch.setattr(mantis.run, "_preload_ring", lambda buf, spec: seen.append(f"preload {spec.ring}") or 0)
    monkeypatch.setattr(mantis.run, "_restore_resume_state", lambda buf, ck: seen.append(f"restore {ck}") or restored)
    monkeypatch.setattr(mantis.run, "GameRecorder", lambda **_kw: object())
    monkeypatch.setattr(mantis.run, "WorkerPool", lambda **_kw: seen.append("pool") or object())
    mantis.run.build_run_collaborators(config=config, out_dir=tmp_path / "out", checkpoint_path=checkpoint_path)
    return seen


def test_a_fresh_launch_preloads_before_the_pool_exists(monkeypatch: pytest.MonkeyPatch, smoke_run_config: Any,
                                                       tmp_path: Path) -> None:
    assert _drive(monkeypatch, smoke_run_config, tmp_path, None, None) == ["preload x.bin", "pool"]


@pytest.mark.parametrize("restored", [SimpleNamespace(step=3000), None], ids=["sidecar", "no-sidecar"])
def test_a_checkpoint_launch_never_preloads(monkeypatch: pytest.MonkeyPatch, smoke_run_config: Any, tmp_path: Path,
                                            restored: Any) -> None:
    """A resume keeps its own ring, and a sidecar-less one stays announced-empty (the mutation that reds it: preload on both branches)."""
    assert _drive(monkeypatch, smoke_run_config, tmp_path, "stop.ckpt", restored) == ["restore stop.ckpt", "pool"]
