"""P6/P9 — the marker channel fails closed, and no figure is printed without its limits.

TWO PROPERTIES, both earned rather than invented.

MARKERS FAIL CLOSED. The 2026-08-22 sitting's own `peaks.py` guessed at a file's shape,
collapsed a whole run into one poll, and reported **1 392 GiB of high-water on a 16 GiB card**.
A plausible-looking wrong number would have been minted against. `mantis.eval.child_memory`
answers that by refusing a file with no markers; this reader inherits the refusal AND its own
token, because a sweep log carrying `MANTIS_EVAL_MEM` lines would be read by the eval-child
reader as an eval drive, which it is not.

EVERY FIGURE CARRIES ITS SAMPLING LIMIT AND ITS PRODUCING RUN. R287(a), and the box block's own
convention: rounds observed and the wall seconds they cover print beside the peaks. A number
lifted out of this report into a sitting record still names what produced it and over how long.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from mantis.diagnostics import worker_sweep as ws

_PLAN = Path(__file__).resolve().parents[2] / "tools" / "worker_sweep_plan.toml"


@pytest.fixture()
def plan() -> ws.SweepPlan:
    return ws.load_plan(_PLAN)


class _Stats:
    def __init__(self, games: int, moves: int) -> None:
        self.games_completed = games
        self.positions_generated = moves


class _Pool:
    _producer_exc = None
    # R317(c)(i): drive_rung hashes `pool.model` right after the build; a mock pool needs one.
    model = type("_NoParams", (), {"state_dict": lambda self: {}})()

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...


@pytest.fixture()
def driven(plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch) -> tuple[ws.RungResult, str]:
    counter = {"n": 0}

    def stats(_pool: object) -> _Stats:
        counter["n"] += 1
        return _Stats(games=counter["n"], moves=counter["n"] * 500)

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _Pool())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    monkeypatch.setattr(ws, "runner_stats", stats)
    sink = io.StringIO()
    result = ws.drive_rung(object(), plan, n_workers=2, device=torch.device("cpu"),
                           label="run5@abc1234", out=sink, sleep=lambda _s: None)
    return result, sink.getvalue()


# ══ the channel ══════════════════════════════════════════════════════════════════════════
def test_a_rung_emits_its_own_markers_and_never_the_eval_child_s(driven) -> None:
    _result, log = driven
    records = ws.parse_sweep_markers(log)
    phases = [r["phase"] for r in records]
    assert phases[0] == "rung_start" and phases[-1] == "rung_end"
    assert any(p.startswith("round_start:") for p in phases)
    assert any(p.startswith("round_end:") for p in phases)
    assert all(r.get("produced_by") == "run5@abc1234" for r in records), (
        "every marker names the run that produced it (R287(a))"
    )
    assert "MANTIS_EVAL_MEM" not in log, (
        "the eval probe's own channel is DISCARDED, not re-emitted: a sweep log carrying eval "
        "markers would be read by mantis.diagnostics.eval_child_memory as an eval drive"
    )


def test_the_rung_end_marker_is_emitted_even_when_the_rung_ooms(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_a: object, **_k: object) -> object:
        raise torch.OutOfMemoryError("synthetic")

    monkeypatch.setattr(ws, "build_sweep_pool", explode)
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    sink = io.StringIO()
    result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                           label="t", out=sink)
    assert result.verdict == ws.OOM
    phases = [r["phase"] for r in ws.parse_sweep_markers(sink.getvalue())]
    assert "rung_oom" in phases and "rung_end" in phases, (
        "a rung that died must still close its own marker span, or an external sampler cannot "
        "tell where the OOM'd rung ended"
    )


# ══ P6 — fail closed ═════════════════════════════════════════════════════════════════════
def test_a_capture_with_no_markers_raises_rather_than_reporting_nothing() -> None:
    with pytest.raises(ValueError, match="no MANTIS_WORKER_SWEEP"):
        ws.parse_sweep_markers("worker_pool_started: n_workers=2\nsome other log line\n")


def test_a_marker_with_an_unreadable_payload_raises_rather_than_being_skipped() -> None:
    with pytest.raises(ValueError, match="no readable json"):
        ws.parse_sweep_markers(f'{ws.MARKER} {{"phase": "rung_start"\n')


def test_a_marker_whose_payload_is_not_an_object_raises() -> None:
    with pytest.raises(ValueError, match="json object"):
        ws.parse_sweep_markers(f"{ws.MARKER} [1, 2, 3]\n")


def test_non_marker_lines_are_skipped_but_do_not_make_the_reader_pass_vacuously() -> None:
    """A real capture carries lines that are not markers; refusing on one would make the reader
    unusable. The refusal comes from finding no MARKERS, which is the thing the reader needs."""
    text = f'noise\n{ws.MARKER} {{"phase": "rung_start"}}\nmore noise\n'
    assert [r["phase"] for r in ws.parse_sweep_markers(text)] == ["rung_start"]


# ══ P9 — the census over the human screen ════════════════════════════════════════════════
def _report(plan: ws.SweepPlan, result: ws.RungResult, *, counters: bool) -> dict:
    prov = {
        "tool": ws.TOOL, "produced_by": "run5@abc1234", "config_name": "run5.yaml",
        "config_sha256": "deadbeef", "git_commit": "abc1234", "git_dirty": False,
        "run_id": "run5", "encoding": "gnn_axis_v1", "representation": "graph",
        "device": "cpu", "torch_version": "2.11.0+cpu", "torch_cuda_version": None,
        "cuda_available": counters, "cuda_counters_available": counters, "gpu_name": None,
        "thread_bound": 16, "thread_bound_source": "os.sched_getaffinity(0)",
        "declared_allocator_posture": None, "allocator_posture_governs_device": False,
        "live_allocator_conf": "", "live_allocator_conf_source_var": None,
    }
    return ws.build_report(plan=plan, prov=prov, results=[result], stopped="test",
                           noise_floor_rel_std=0.0)


def test_every_rung_row_names_its_producing_run_and_its_sampling_limit(
    driven, plan: ws.SweepPlan, capsys,
) -> None:
    result, _log = driven
    report = _report(plan, result, counters=True)
    ws.render(report, __import__("sys").stdout)
    text = capsys.readouterr().out
    for needle in ("produced_by=run5@abc1234", "rounds_measured=", "rounds_unmeasured=",
                   "wall_sec=", "card_samples=", "not a bound"):
        assert needle in text, f"the rung screen omits {needle!r}"


def test_a_host_with_no_counters_says_so_before_it_says_anything_else(
    driven, plan: ws.SweepPlan, capsys,
) -> None:
    """Item 3's requirement, in the tool rather than in a covering note: no figure from a
    counter-less host may be quoted as a floor, and the banner is what stops it."""
    result, _log = driven
    ws.render(_report(plan, result, counters=False), __import__("sys").stdout)
    text = capsys.readouterr().out
    assert "MECHANISM EVIDENCE ONLY" in text
    assert "No figure below is a" in text
    head = text.splitlines()[:4]
    assert any("MECHANISM EVIDENCE ONLY" in line for line in head), (
        "the banner must precede the numbers; a disclaimer under a table is read after the "
        "number has already been copied out"
    )


def test_the_report_carries_the_prereg_and_the_whole_plan_it_ran_under(
    driven, plan: ws.SweepPlan,
) -> None:
    result, _log = driven
    report = _report(plan, result, counters=True)
    assert report["prereg"]["prereg_ruling"] == "R309(f)"
    assert report["plan"]["rungs"] == list(plan.rungs)
    assert report["plan"]["knee_pct"] == plan.knee_pct
    assert report["plan"]["metric"] == plan.metric
    assert report["rungs"][0]["produced_by"] == "run5@abc1234"


def test_the_report_carries_no_host_identifier_fields(driven, plan: ws.SweepPlan) -> None:
    """R112 / CI gate 17: GPU model and thread count are regime facts; a hostname, a home path
    or a provider name is not, and none is read here."""
    result, _log = driven
    prov = _report(plan, result, counters=True)["provenance"]
    banned = {"hostname", "host", "user", "home", "ssh", "ip", "container", "provider"}
    assert not (banned & set(prov)), f"provenance carries host-identifying keys: {sorted(prov)}"
