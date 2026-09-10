"""⊕ WP12-R Phase T (TARGET INTEGRITY) — CTR, PYTHON SURFACE LEG: the LAW-18
target-integrity counters are VISIBLE on the Python stats surface (DESIGN_T §3.6).

`mantis.selfplay.pool_hooks.RunnerStats` carries them, read from the engine runner by
`runner_stats(pool)`:

  * `export_offwindow_mass_moves` — moves whose export carried overflow mass;
  * `target_integrity_defects`    — the fatal-defect latch fire count (must read 0 in
    a healthy run; the name is fixed by the T-2 oracle bank — ORACLE_NOTES_T.md).

`gridls_zero_policy_rows` was the third and LEFT with R346(f): it counted §3.5 zero-row
fills per recorded CLUSTER row, a dense record-path quantity whose engine getter is gone.
Its `getattr(..., 0)` default would have kept it visible at a permanent 0 — visible and
FABRICATED, which is worse than absent, because a rate of 0 reads as "measured, none found".

The PRODUCER proofs (counters actually fire) are Rust-side
(`target_wire_carry.rs::ctr_export_offwindow_mass_moves_fires_on_a_dispersed_run`,
`::o4b_latch_stores_the_named_variant_and_halts_the_runner`). This leg pins the
surface: values thread through un-crosswired, and an idle lever stays VISIBLE at 0
(the chain_loss_with_fire_rate posture, losses.py:224-233). NOTE (recorded): the
`getattr(..., 0)` legacy-wheel default in `runner_stats` means surface visibility is
NOT a producer proof — that burden stays on the Rust legs (LAW-07).

PRE-FIX status at HEAD: RED (RunnerStats has none of the three fields).
Killer (PREREG_T §3): M-H (per-counter sub-runs; the surface legs red when the
mutated counter's value no longer threads through).
"""
from __future__ import annotations

from types import SimpleNamespace

from mantis.selfplay.pool_hooks import RunnerStats, runner_stats

_FIELDS = (
    "export_offwindow_mass_moves",
    "target_integrity_defects",
)


class _Pool:
    def __init__(self, runner: object) -> None:
        self._runner = runner


def test_runner_stats_threads_the_target_integrity_counters() -> None:
    runner = SimpleNamespace(
        export_offwindow_mass_moves=5,
        target_integrity_defects=9,
    )
    st = runner_stats(_Pool(runner))
    assert isinstance(st, RunnerStats)
    got = tuple(getattr(st, f) for f in _FIELDS)
    assert got == (5, 9), (
        f"counter values did not thread 1:1 through runner_stats (got {got}) — a "
        "crosswired or missing surface field hides the lever's fire-rate (LAW-18)"
    )


def test_idle_counters_are_visible_at_zero() -> None:
    st = runner_stats(_Pool(SimpleNamespace()))
    for f in _FIELDS:
        assert getattr(st, f) == 0, (
            f"idle counter {f!r} must be VISIBLE at 0 (a disabled/idle lever stays "
            "visible — the chain_loss_with_fire_rate posture)"
        )
