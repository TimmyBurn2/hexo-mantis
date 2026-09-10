"""Segment rotation on resume: no two process starts ever share a JSONL file.

The contract is `path = log_dir / f"events_{run_id}_seg{seg:04d}.jsonl"` with
`seg = max(existing segs for run_id) + 1`, so a start never appends to a prior segment.

The law is absolute, so it is driven under a race and under a hostile `run_id` too: both were
once defeated (3 of 9 files carried two headers from two pids under 12 concurrent
constructions, and `""` / `"a/b"` / `"../x"` each produced a spanning file). The scan-then-open
TOCTOU is now an `O_CREAT|O_EXCL` claim and `run_id` is validated at the sink boundary.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from mantis.monitor.sink import JsonlEventSink, RunIdError

_SEG_RE = re.compile(r"_seg(\d+)\.jsonl$")


def _seg_index(path: Path) -> int:
    m = _SEG_RE.search(path.name)
    assert m is not None, f"segment filename {path.name!r} must carry a _segNNNN suffix"
    return int(m.group(1))


def _first_event(path: Path) -> str:
    for ln in path.read_text().splitlines():
        if ln.strip():
            return json.loads(ln)["event"]
    raise AssertionError(f"{path} has no lines")


def test_resume_opens_the_next_segment_and_never_appends(tmp_path: Path) -> None:
    """Prove a resume rotates to seg+1, leaves the first file intact, and headers both segments."""
    sink1 = JsonlEventSink(log_dir=tmp_path, run_id="run5")
    sink1.emit({"event": "training_step", "step": 1})
    sink1.close()
    seg1 = _seg_index(sink1.path)
    size1 = sink1.path.stat().st_size

    # Resume: a fresh sink for the SAME (log_dir, run_id).
    sink2 = JsonlEventSink(log_dir=tmp_path, run_id="run5")
    seg2 = _seg_index(sink2.path)
    assert sink2.path != sink1.path, "resume must open a NEW file, never the prior segment"
    assert seg2 == seg1 + 1, f"resume segment {seg2} must be prior max {seg1} + 1"

    sink2.emit({"event": "training_step", "step": 2})
    sink2.close()

    assert sink1.path.stat().st_size == size1, (
        "the prior segment file must be untouched after a resume (never spans two segments)"
    )
    assert _first_event(sink1.path) == "run_segment_started"
    assert _first_event(sink2.path) == "run_segment_started"


def test_third_open_continues_the_monotonic_segment_sequence(tmp_path: Path) -> None:
    """Prove segment indices strictly increase across repeated resumes and never reset."""
    segs: list[int] = []
    for _ in range(3):
        s = JsonlEventSink(log_dir=tmp_path, run_id="run5")
        s.emit({"event": "e"})
        s.close()
        segs.append(_seg_index(s.path))
    assert segs[0] < segs[1] < segs[2], f"segments must strictly increase, got {segs}"
    assert len(set(segs)) == 3, "each resume must claim a distinct segment file"


def test_distinct_run_ids_do_not_share_segments(tmp_path: Path) -> None:
    """Prove segments are per run_id: one run's resumes never bump another's counter."""
    a = JsonlEventSink(log_dir=tmp_path, run_id="runA")
    a.close()
    b = JsonlEventSink(log_dir=tmp_path, run_id="runB")
    b.close()
    assert "runA" in a.path.name and "runB" in b.path.name
    # Re-opening runA rotates ONLY runA's segments.
    a2 = JsonlEventSink(log_dir=tmp_path, run_id="runA")
    a2.close()
    assert _seg_index(a2.path) == _seg_index(a.path) + 1


def _headers(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip() and json.loads(ln).get("event") == "run_segment_started"]


def test_concurrent_constructions_never_share_a_segment_file(tmp_path: Path) -> None:
    """Prove N simultaneous constructions claim N distinct files, each with exactly one header.

    Bites the scan-then-`open("a")` TOCTOU, where two racers computing the same next index
    both opened it for append. The claim is `O_CREAT|O_EXCL` with a bounded re-scan.
    """
    n = 12
    sinks: list[JsonlEventSink] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def _construct() -> None:
        try:
            barrier.wait(timeout=5.0)                 # Maximise the collision window.
            sinks.append(JsonlEventSink(log_dir=tmp_path, run_id="racerun"))
        except BaseException as exc:                  # noqa: BLE001 — reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=_construct) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"no construction may fail: {errors}"
    assert len(sinks) == n
    paths = [s.path for s in sinks]
    for s in sinks:
        s.close()

    assert len(set(paths)) == n, f"{n} constructions must claim {n} DISTINCT files: {paths}"
    assert len({_seg_index(p) for p in paths}) == n, "segment indices must be distinct"
    for path in paths:
        heads = _headers(path)
        assert len(heads) == 1, (
            f"{path.name} carries {len(heads)} run_segment_started headers — a JSONL file "
            f"spanning two run segments is forbidden by §11"
        )


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "..", ".", "a\\b", "x\x00y", "a\ty",
                                 " lead", "trail "])
def test_hostile_run_id_is_rejected_at_the_sink_boundary(tmp_path: Path, bad: str) -> None:
    """Prove a run_id that cannot be a safe filename component is rejected at construction.

    `""` makes the segment regex unmatchable so the index never advances; `a/b` and `../x` put
    the file outside the scanned directory. The schema's own pattern is no defence here,
    because this sink is reachable without it.
    """
    with pytest.raises(RunIdError):
        JsonlEventSink(log_dir=tmp_path, run_id=bad)
    assert list(tmp_path.glob("**/*.jsonl")) == [], "a rejected run_id must create no file"


def test_valid_exotic_run_ids_still_rotate(tmp_path: Path) -> None:
    """Prove unicode, regex-special and seg-lookalike run ids are safe and still rotate."""
    for run_id in ("рун-ид", "a.*b|c[0-9]+", "abc_seg0009", "run.5"):
        first = JsonlEventSink(log_dir=tmp_path, run_id=run_id)
        first.close()
        second = JsonlEventSink(log_dir=tmp_path, run_id=run_id)
        second.close()
        assert _seg_index(second.path) == _seg_index(first.path) + 1
        assert len(_headers(first.path)) == 1 and len(_headers(second.path)) == 1
