"""AUDIT-1 F-28 rows A03, A05, A06, A07, A08, A09, A10, A11 — the diagnostics tools' defaults.

Eight rows, one class: a tool publishing a number nothing measured, in an artifact an operator
reads at a sitting or a mint. Each row's mechanism is named at its own test.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pytest

import mantis.diagnostics.worker_sweep as ws
from mantis.diagnostics import eval_child_memory as ecm
from mantis.diagnostics.fusion_calibrate import _recommend

_REPO = Path(__file__).resolve().parents[2]


# ── A03: a plan whose sampler cannot produce a series is refused at LOAD ──────────────

def _plan_text(*, round_sec: float, interval: float) -> str:
    base = (_REPO / "tools" / "worker_sweep_plan.toml").read_text(encoding="utf-8")
    out = []
    for line in base.splitlines():
        if line.startswith("round_sec"):
            out.append(f"round_sec = {round_sec}")
        elif line.startswith("sampler_interval_sec"):
            out.append(f"sampler_interval_sec = {interval}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def test_the_shipped_plan_still_loads(tmp_path: Path) -> None:
    """The control, first: the plan the box actually runs must be unaffected."""
    plan = ws.load_plan(_REPO / "tools" / "worker_sweep_plan.toml")
    assert plan.sampler_interval_sec < plan.round_sec


@pytest.mark.parametrize("interval", [120.0, 240.0], ids=["equal", "longer"])
def test_a_sampler_that_cannot_produce_a_series_is_refused(
    tmp_path: Path, interval: float
) -> None:
    """THE PIN (A03). At an interval at or above `round_sec` a round collects at most ONE
    card sample, so the card high-water is not a series — and the rung PASSED anyway on the
    allocator series alone while its card column read `card_samples<=1`."""
    path = tmp_path / "plan.toml"
    path.write_text(_plan_text(round_sec=120.0, interval=interval), encoding="utf-8")
    with pytest.raises(ValueError, match="sampler_interval_sec"):
        ws.load_plan(path)


# ── A05: "seen in this sweep" includes the points that OOM'd ──────────────────────────

_FIT = {"a_bytes": 1_000_000.0, "b_bytes_per_edge": 100.0, "c_bytes_per_node": 200.0,
        "operating_edges_per_node": 4.0}


def test_a_graph_the_sweep_OOMd_on_counts_as_SEEN(tmp_path: Path) -> None:
    """THE PIN (A05). The flag was computed over MEASURED points only, so the biggest shapes
    — the ones a cap has to admit or refuse, and the ones that OOM — were not "seen", and the
    flag read False for a pair that cannot hold a graph the sweep actually built."""
    measured = [{"largest_graph_nodes": 10, "largest_graph_edges": 40}]
    huge = [{"largest_graph_nodes": 10_000_000, "largest_graph_edges": 40_000_000,
             "not_measured": "cuda_out_of_memory"}]

    without = _recommend(_FIT, 1_000_000_000, 0.85, measured)
    assert without["refuses_a_graph_seen_in_this_sweep"] is False

    with_oom = _recommend(_FIT, 1_000_000_000, 0.85, measured, unmeasured=huge)
    assert with_oom["refuses_a_graph_seen_in_this_sweep"] is True, (
        "a pair that cannot hold the biggest graph the sweep built reported that it could"
    )
    assert with_oom["seen_points"] == 2 and with_oom["seen_points_unmeasured"] == 1, (
        "the denominator must be published, or the flag is a claim with no stated scope"
    )


# ── A06 / A07: the markers reader and the printed wall ────────────────────────────────

def test_a_round_with_no_timestamps_reports_no_wall_instead_of_crashing() -> None:
    """THE PIN (A06). `max(...)` over a sequence containing `None` raises `TypeError`, which
    escaped `--markers` and exited with GROWING's code — an unreadable capture presenting as
    a measured growth verdict."""
    assert ecm._span([None, None]) is None
    assert ecm._span([]) is None
    assert ecm._span([5.0]) is None, "one stamp is an instant, not a span"
    assert ecm._span([None, 1.0, 4.5, None]) == pytest.approx(3.5)


def test_the_printed_wall_total_states_how_many_rounds_it_covers() -> None:
    """THE PIN (A07). `sum(r.wall_sec or 0.0 ...)` counted an unmeasured round as ZERO
    SECONDS, so the printed total was an understatement wearing a total's name."""
    rounds = [
        ecm.RoundReading(round_id="r1", step=None, available=True, peak_bytes=10,
                         reserved_peak_bytes=12, wall_sec=100.0, phases=()),
        ecm.RoundReading(round_id="r2", step=None, available=False, peak_bytes=None,
                         reserved_peak_bytes=None, wall_sec=None, phases=()),
    ]
    out = io.StringIO()
    ecm._render(rounds, plateau_rounds=1, band_pct=1.0, verdict=None, refusal=None, out=out)
    text = out.getvalue()
    assert "wall_sec=100.0 over 1/2 timed round(s)" in text, text


def test_a_series_with_no_timed_round_at_all_says_unmeasured() -> None:
    rounds = [ecm.RoundReading(round_id="r1", step=None, available=False, peak_bytes=None,
                               reserved_peak_bytes=None, wall_sec=None, phases=())]
    out = io.StringIO()
    ecm._render(rounds, plateau_rounds=1, band_pct=1.0, verdict=None, refusal=None, out=out)
    assert "wall_sec=unmeasured (0/1 rounds carry a wall)" in out.getvalue()


# ── A08: rates over zero games ────────────────────────────────────────────────────────

def test_the_witness_reports_no_rate_over_zero_games() -> None:
    """THE PIN (A08). This tool's whole subject is a decisive-rate bar, so `decisive_rate
    0.0` over no games read as the strongest possible refusal evidence when nothing played."""
    from mantis.diagnostics.acceptance_witness import measure_arm

    arm = measure_arm([], encoding_name="gnn_axis_v1", floor=None)
    assert arm["games"] == 0
    for key in ("decisive_rate", "winrate", "mean_plies", "longest_run_max"):
        assert arm[key] is None, f"{key} = {arm[key]!r} over zero games"


def test_a_witness_over_zero_games_is_refused_at_the_CLI() -> None:
    """The other half: the tool should never have been asked for it."""
    from mantis.diagnostics.acceptance_witness import _positive_games

    assert _positive_games("20") == 20
    for bad in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
            _positive_games(bad)


# ── A09: a verdict over zero games ────────────────────────────────────────────────────

def test_the_convention_audit_does_not_certify_over_zero_games() -> None:
    """THE PIN (A09). With nothing in the check both counters are 0, `lacks` is falsy, and the
    tool reported "CONSISTENT ... on every game replayed" — a clean bill issued over nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_abc_probe", _REPO / "tools" / "audit_bootstrap_corpus.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    out = tool.audit_conventions([])
    assert "NOT CHECKED" in out["verdict"], out["verdict"]
    assert "CONSISTENT" not in out["verdict"]
    assert out["games_checked"] == 0


# ── A10: git_dirty ────────────────────────────────────────────────────────────────────

def _provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                *, porcelain: str | None) -> dict[str, Any]:
    """The PRODUCTION `provenance`, over a REAL minted config, with only `_git` stubbed."""
    from mantis.config.loader import load_config

    config = load_config(_REPO / "configs" / "dev_example.yaml")
    monkeypatch.setattr(
        ws, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else porcelain)
    return ws.provenance(config, _REPO / "configs" / "dev_example.yaml",
                         device="cpu", label="t")


def test_git_dirty_keys_on_the_command_that_produced_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE PIN (A10). The guard discriminated on `rev-parse`, not on `status --porcelain`. A
    tree where the commit resolves and the status call FAILS — an index lock, a permissions
    fault — took the `else` arm and published `bool(None)` = CLEAN."""
    prov = _provenance(monkeypatch, tmp_path, porcelain=None)
    assert prov["git_commit"] == "abc123"
    assert prov["git_dirty"] is None, (
        "the status call failed and the report claimed a clean tree"
    )


def test_git_dirty_still_reports_a_real_answer_when_git_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control, both ways."""
    assert _provenance(monkeypatch, tmp_path, porcelain="")["git_dirty"] is False
    assert _provenance(monkeypatch, tmp_path, porcelain=" M f.py")["git_dirty"] is True


# ── A11: device_count ─────────────────────────────────────────────────────────────────

def test_device_count_is_absent_when_the_question_was_never_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PIN (A11). `... else 0` read identically on a CPU-only host and on a CUDA host
    that saw no device, and the third refusal then named the wrong condition for the first."""
    import torch

    from mantis.diagnostics import cuda_build_guard as guard

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert guard.torch_build()["device_count"] is None
    assert guard.torch_build()["cuda_available"] is False


def test_a_cpu_only_host_is_refused_by_NAME_not_by_the_device_count_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence: the refusal an operator reads must name what actually happened."""
    import torch

    from mantis.diagnostics import cuda_build_guard as guard

    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(guard.CudaBuildRefusal) as exc:
        guard.assert_cuda_build()
    assert "is_available() is False" in str(exc.value)
    assert "device_count 0" not in str(exc.value)
