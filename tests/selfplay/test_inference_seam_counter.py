"""F-816-9 Phase C — the SEAM counter on the Python stats surface (R275(b), LAW-18).

`mantis.selfplay.pool_hooks.RunnerStats` carries `inference_failures_total`, read off the
engine runner by `runner_stats(pool)` exactly like the Phase-T target-integrity counters.

WHAT THIS LEG IS AND IS NOT. It pins the SURFACE: the value threads through un-crosswired
and an idle counter stays VISIBLE at 0. It is NOT a producer proof — `runner_stats`'
`getattr(..., 0)` legacy-wheel default means a missing engine getter reads as a real zero
here, which is exactly the reading LAW-07 refuses to accept as evidence. The producer burden
lives in Rust (`crates/mantis-selfplay/tests/search_seam_fatal.rs`: the counter fires on an
injected failure on BOTH arms, and does NOT fire on a clean drain shutdown), and the in-run
stream burden in `tests/train/test_inference_seam_events.py`.

MUTATION THAT REDS IT (M-SEAMPY): read `target_integrity_defects` into
`inference_failures_total` in `runner_stats` — the two conjuncts of the F-816-9 class then
report each other's counts, and the distinct values below are what sees it.
"""
from __future__ import annotations

from types import SimpleNamespace

from mantis.selfplay.pool_hooks import RunnerStats, runner_stats


class _Pool:
    def __init__(self, runner: object) -> None:
        self._runner = runner


def test_runner_stats_threads_the_inference_seam_counter() -> None:
    # DISTINCT values across the two conjuncts' counters: equal ones would be satisfied by a
    # crosswire, which is the only interesting way this can be wrong.
    runner = SimpleNamespace(inference_failures_total=4, target_integrity_defects=9)
    st = runner_stats(_Pool(runner))
    assert isinstance(st, RunnerStats)
    assert st.inference_failures_total == 4, (
        f"the seam counter did not thread 1:1 through runner_stats (got "
        f"{st.inference_failures_total}) — a crosswired or missing surface field hides the "
        "lever's fire-rate (LAW-18)"
    )
    assert st.target_integrity_defects == 9, (
        "the seam read clobbered the target-integrity counter — the two conjuncts must stay "
        f"separately readable (got {st.target_integrity_defects})"
    )


def test_the_idle_seam_counter_is_visible_at_zero() -> None:
    """The seam latch is run-fatal, so this reads 0 in every run that survives. That
    permanent zero is the posture — it is what distinguishes "no inference has failed" from
    a field with no producer at all."""
    st = runner_stats(_Pool(SimpleNamespace()))
    assert st.inference_failures_total == 0
