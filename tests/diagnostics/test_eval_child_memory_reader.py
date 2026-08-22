"""⊕ RECAL-PREP item 2 — the eval-child growth reader (R308(g)(ii)).

Written by ORACLE **before** the feature exists.

WHAT THIS TOOL IS FOR, in one sentence from the sitting it comes from: *a term measured by
watching until it looks flat is not a bound* (`RECAL_EXIT_2026-08-22.md` §11b). STEP 1d as the
procedure writes it takes a maximum over one eval round and carries it into a four-constant-term
budget; that procedure produced 0.881 GiB, then 1.1855, then was falsified at 3.5293 by the
strengthened STEP 4. The replacement is not a longer look — it is a STATED STOPPING RULE that a
tool applies and a record carries.

The defect each row is the ONLY witness to:

- **RD-01** — a verdict from a stopping rule nobody wrote down. `--plateau-rounds` and
  `--band-pct` have NO defaults; the tool prints the rule it applied beside the verdict.
- **RD-02** — "no data" reading as "converged". Zero rounds, too few rounds, and a file with no
  events of the expected kind are each a NAMED REFUSAL with rc 2, never a verdict. This is the
  half that would have caught the sitting's `peaks.py` before it produced 1 392 GiB.
- **RD-03** — a rule that cannot say GROWING. A classifier that answers PLATEAU on a climbing
  series is a classifier with one arm, and the term this tool exists for has grown on every
  occasion it has been measured.
- **RD-04** — a figure without its sampling limit. Every readout states the rounds observed and
  the wall seconds they cover, per the block's own convention (`larger governs`, and a limit
  stated beside every number).
- **RD-05** — a reader that silently drops the rounds whose child had no counters. An
  `available: false` round is REPORTED as unmeasured and excluded from the verdict by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.diagnostics.eval_child_memory import (
    EVENT,
    GROWING,
    PLATEAU,
    RC_REFUSED,
    InsufficientRoundsError,
    NoRoundsFoundError,
    classify,
    main,
    read_rounds_from_events,
)

GIB = 1024 ** 3


def _event(round_id: str, step: int, peak_gib: float, *, available: bool = True,
           wall: float = 60.0) -> str:
    peak = int(peak_gib * GIB)
    return json.dumps({
        "event": EVENT,
        "round_id": round_id,
        "step": step,
        "device_memory": {
            "available": available,
            "device": "cuda" if available else "cpu",
            "round_id": round_id,
            "phases": [
                {"phase": "round_start", "t_mono_sec": 0.0,
                 "max_memory_allocated_bytes": peak // 4 if available else None,
                 "max_memory_reserved_bytes": peak // 4 if available else None,
                 "memory_allocated_bytes": peak // 4 if available else None,
                 "memory_reserved_bytes": peak // 4 if available else None},
                {"phase": "round_end", "t_mono_sec": wall,
                 "max_memory_allocated_bytes": peak if available else None,
                 "max_memory_reserved_bytes": peak if available else None,
                 "memory_allocated_bytes": peak // 2 if available else None,
                 "memory_reserved_bytes": peak if available else None},
            ],
            "round_peak_allocated_bytes": peak if available else None,
            "round_peak_reserved_bytes": peak if available else None,
        },
    })


def _stream(*events: str) -> str:
    return "".join(e + "\n" for e in events)


# ── RD-01 / RD-03: the classifier, both arms, under one stated rule ─────────────────────
def test_rd03_a_flat_series_is_a_plateau():
    peaks = [1.00, 1.01, 0.99, 1.02, 1.00]
    assert classify(peaks, plateau_rounds=3, band_pct=5.0) == PLATEAU


def test_rd03_a_climbing_series_is_growing():
    peaks = [1.00, 1.20, 1.60, 2.40, 3.50]
    assert classify(peaks, plateau_rounds=3, band_pct=5.0) == GROWING


def test_rd03_the_sittings_own_series_is_growing_under_any_reasonable_band():
    """0.881 -> 1.186 -> 3.529: the three readings the sitting took, in order. A rule that
    calls this PLATEAU is not a rule."""
    assert classify([0.881, 1.186, 3.529], plateau_rounds=3, band_pct=25.0) == GROWING


def test_rd03_a_late_jump_after_a_long_flat_run_is_growing():
    """The failure mode STEP 1d actually hit: flat for the final 30% of a 24-minute drive,
    and then a 2.98x reading from a phase the drive never entered."""
    peaks = [1.0] * 10 + [3.0]
    assert classify(peaks, plateau_rounds=3, band_pct=5.0) == GROWING


def test_rd03_an_early_jump_that_then_settles_is_a_plateau():
    """Growth OUTSIDE the trailing window is history, not a live trend — otherwise no series
    that ever rose could ever converge, and the rule would have exactly one arm."""
    peaks = [1.0, 3.0, 3.0, 3.01, 2.99, 3.0]
    assert classify(peaks, plateau_rounds=3, band_pct=5.0) == PLATEAU


def test_rd01_the_band_is_a_real_parameter_in_both_directions():
    peaks = [1.0, 1.0, 1.0, 1.10]
    assert classify(peaks, plateau_rounds=2, band_pct=5.0) == GROWING
    assert classify(peaks, plateau_rounds=2, band_pct=25.0) == PLATEAU


def test_rd01_the_window_is_a_real_parameter_in_both_directions():
    peaks = [1.0, 3.0, 3.0, 3.0, 3.0]
    assert classify(peaks, plateau_rounds=2, band_pct=5.0) == PLATEAU
    assert classify(peaks, plateau_rounds=5, band_pct=5.0) == GROWING


# ── RD-02: refusals, never a verdict ─────────────────────────────────────────────────────
def test_rd02_too_few_rounds_refuses_rather_than_answering():
    with pytest.raises(InsufficientRoundsError) as exc:
        classify([1.0, 1.0], plateau_rounds=3, band_pct=5.0)
    assert "3" in str(exc.value)


def test_rd02_zero_rounds_refuses():
    with pytest.raises(InsufficientRoundsError):
        classify([], plateau_rounds=1, band_pct=5.0)


def test_rd02_a_stream_with_no_events_of_the_expected_kind_refuses_by_name(tmp_path):
    stream = _stream(json.dumps({"event": "iteration_complete", "step": 1}))
    with pytest.raises(NoRoundsFoundError) as exc:
        read_rounds_from_events(stream)
    assert EVENT in str(exc.value)


def test_rd02_an_empty_stream_refuses():
    with pytest.raises(NoRoundsFoundError):
        read_rounds_from_events("")


def test_rd02_a_non_json_line_is_skipped_but_an_all_noise_stream_still_refuses():
    """A run log carries lines that are not events; that is not a reason to guess. The
    refusal comes from finding no ROUNDS, not from finding a line it could not parse."""
    with pytest.raises(NoRoundsFoundError):
        read_rounds_from_events("not json at all\nstill not json\n")


# ── RD-05: unmeasured rounds are reported, not dropped ───────────────────────────────────
def test_rd05_rounds_whose_child_had_no_counters_are_kept_and_flagged():
    rounds = read_rounds_from_events(_stream(
        _event("r1", 10, 1.0),
        _event("r2", 20, 0.0, available=False),
        _event("r3", 30, 1.1),
    ))
    assert [r.round_id for r in rounds] == ["r1", "r2", "r3"]
    assert [r.available for r in rounds] == [True, False, True]
    assert rounds[1].peak_bytes is None


def test_rd05_the_verdict_is_taken_over_the_measured_rounds_only(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    events.write_text(_stream(
        _event("r1", 10, 1.0),
        _event("r2", 20, 0.0, available=False),
        _event("r3", 30, 1.0),
        _event("r4", 40, 1.0),
    ), encoding="utf-8")
    rc = main(["--events", str(events), "--plateau-rounds", "3", "--band-pct", "5"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert PLATEAU in out
    assert "unmeasured" in out.lower(), "an excluded round must be named, never dropped"


# ── RD-04: the sampling limit, beside every figure ──────────────────────────────────────
def test_rd04_the_readout_states_the_rule_it_applied_and_the_sample_it_applied_it_to(
    tmp_path, capsys,
):
    events = tmp_path / "events.jsonl"
    events.write_text(_stream(*[_event(f"r{i}", i * 10, 1.0, wall=90.0) for i in range(1, 6)]),
                      encoding="utf-8")
    rc = main(["--events", str(events), "--plateau-rounds", "3", "--band-pct", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "plateau_rounds=3" in out and "band_pct=5" in out
    assert "rounds_observed=5" in out
    assert "wall_sec" in out, "a figure with no sampling limit beside it is the thing this "\
                              "tool exists to stop"


def test_rd04_growing_exits_nonzero_so_a_sitting_can_gate_on_it(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    events.write_text(_stream(
        _event("r1", 10, 1.0), _event("r2", 20, 1.2),
        _event("r3", 30, 1.8), _event("r4", 40, 3.5),
    ), encoding="utf-8")
    rc = main(["--events", str(events), "--plateau-rounds", "3", "--band-pct", "5"])
    assert rc == 1
    assert GROWING in capsys.readouterr().out


def test_rd02_the_cli_refuses_with_rc_2_and_says_what_it_wanted(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    events.write_text(_stream(json.dumps({"event": "iteration_complete"})), encoding="utf-8")
    rc = main(["--events", str(events), "--plateau-rounds", "3", "--band-pct", "5"])
    assert rc == RC_REFUSED
    assert EVENT in capsys.readouterr().err


def test_rd02_the_cli_refuses_a_missing_file_with_rc_2(tmp_path, capsys):
    rc = main(["--events", str(tmp_path / "nope.jsonl"), "--plateau-rounds", "3",
               "--band-pct", "5"])
    assert rc == RC_REFUSED


def test_rd01_the_stopping_rule_arguments_have_no_defaults(capsys):
    """A default here would be a stopping rule nobody chose, applied to a mint-critical
    term. `SystemExit(2)` is argparse's own refusal, and it is the right one."""
    with pytest.raises(SystemExit) as exc:
        main(["--events", "irrelevant.jsonl"])
    assert exc.value.code == 2


# ── the marker channel is readable by the same tool ──────────────────────────────────────
def test_the_reader_accepts_the_child_marker_channel_too(tmp_path, capsys):
    """One tool, two transports: the structured event stream when a run wrote one, and the
    child's own stdout markers when the sitting only has a log."""
    from mantis.eval.child_memory import MARKER

    log = tmp_path / "burst.log"
    lines = []
    for i in range(1, 6):
        for phase, peak in (("round_start", 0.25), ("round_end", 1.0)):
            lines.append(f"{MARKER} " + json.dumps({
                "round_id": f"r{i}", "phase": phase, "t_mono_sec": 0.0 if phase == "round_start" else 90.0,
                "max_memory_allocated_bytes": int(peak * GIB),
                "max_memory_reserved_bytes": int(peak * GIB),
                "memory_allocated_bytes": int(peak * GIB),
                "memory_reserved_bytes": int(peak * GIB),
                "available": True, "device": "cuda",
            }))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = main(["--markers", str(log), "--plateau-rounds", "3", "--band-pct", "5"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert PLATEAU in out
    assert "rounds_observed=5" in out


def test_the_two_transports_are_not_both_accepted_at_once(tmp_path):
    with pytest.raises(SystemExit):
        main(["--events", "a.jsonl", "--markers", "b.log", "--plateau-rounds", "3",
              "--band-pct", "5"])


def test_the_module_is_runnable_as_a_module_entry_point():
    """CLAUDE.md's `python -m mantis.*` law: no loose script files."""
    import mantis.diagnostics.eval_child_memory as mod

    assert Path(mod.__file__).name == "eval_child_memory.py"
    assert hasattr(mod, "main")
