# ONE CLAIM with two halves that are deliberately not split across files: "the sweep's per-rung
# network is reproducible, and the control that says so can fail". The mechanism arm (the same
# seed builds the same net) and the instrument arm (`determinism_verdict`/`_hash_gate` separate
# AGREE from DIVERGED from REFUSED) are the two things F-RESIT-10 needs together — a reproducible
# net nobody checks is a claim, and a checker with nothing reproducible under it is a checker of
# noise. Each arm carries its own planted break in-file, for the reason
# `test_worker_sweep_reachability.py` states: a predicate and the proof it can fire must move
# together or the proof rots quietly.
# (No R8 justification is claimed: this file is under the 300-line cap. Sizes are derived by
# `wc -l`, never asserted.)
"""F-RESIT-10's repair, and the control that witnesses it — RE-SPECIFIED by R317(c).

**THE DEFECT, measured at the 2026-08-27 re-calibration re-sit.** `mantis.diagnostics.worker_sweep`
built a fresh `build_net(arch)` per rung from an UNSEEDED RNG, so every rung of the pre-registered
ladder raced a DIFFERENT random network. On an unbounded board a network's policy decides how far
stones spread, which decides the graph's node and edge counts, which decides what every fused
forward costs. So R309(f)'s knee rule — *the smallest rung within 95 % of the best passing rung's
throughput* — compared rungs on a column that was a function of `n_workers` **and an uncontrolled
draw**.

**The repair** is `build_sweep_net`: seed from the config's own `seed` immediately before the one
RNG consumer on that path. The module it calls MOVED to `mantis.util.determinism` to make that
possible — the sweep is guaranteed trainer-unreachable by import at any scope (R309(g),
`test_worker_sweep_reachability.py`), and importing anything under `mantis.train` was measured to
pull eight training modules into `sys.modules`.

**THE CHECK WAS THE DEFECT, NOT THE SEEDING (R317).** R315(c)(i) ordered a THROUGHPUT BAND
(sub-1%, same rung, same seed, twice) as the control — measured 0.5821% AGREE engine-side. Driven
LIVE on the box at RECAL-SITTING-3, the SAME check on the SAME rung came back 3.9258%, DIVERGED:
the band was a cross-regime carry, calibrated quiet and asked to certify a live-GPU drive whose
own within-drive round noise runs ~6% peak-to-peak. **The control is now net-parameter-hash
equality, no band** (R317(c)(i)) — this tests exactly what the repair above claims, with nothing
about wall-clock timing in it. The throughput spread is still computed and reported, but it no
longer gates anything (R317(c)(iii)), and a test below pins that it does not.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.diagnostics import worker_sweep as ws

_REPO = Path(__file__).resolve().parents[2]

#: A committed GRAPH config — the representation the sweep exists for. Read through the real
#: loader, never hand-built: a stub config would let this file pass while the production path
#: read a key that is not there.
_CONFIG = _REPO / "configs" / "smoke_gnn.yaml"


def _net_fingerprint(model: object) -> str:
    """A content hash of every parameter, key order fixed. Deliberately a SEPARATE
    implementation from `worker_sweep._net_param_hash` (R81: this is the oracle, not the
    mechanism re-run against itself)."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):        # type: ignore[attr-defined]
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def graph_arch():
    """The config and the arch the sweep would build for it, resolved the production way."""
    from mantis.model import arch_from_spec_and_config
    from mantis.selfplay.hparams import resolve_pool_encoding

    config = load_config(_CONFIG)
    raw = config.model_dump()
    resolved = resolve_pool_encoding(raw, arch=None)
    return config, arch_from_spec_and_config(resolved.registry_spec, raw)


# ── arm 1: the mechanism ─────────────────────────────────────────────────────────────────
def test_the_same_config_builds_a_BIT_IDENTICAL_network_every_time(graph_arch) -> None:
    """The repair itself: two `build_sweep_net` calls, same config, byte-identical parameters.

    This is the property the ladder's comparability rests on, and it is now also `_net_param_hash`
    itself — a second, independent hash (`_net_fingerprint`) is used here so the test does not
    validate the mechanism using the mechanism's own instrument."""
    config, arch = graph_arch
    first = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    second = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    assert first == second, (
        "two per-rung networks built from the same config must be bit-identical — that is the "
        "whole of F-RESIT-10's repair, and without it the knee rule ranks rungs on a column "
        f"carrying a resampled random term. Got {first[:16]} and {second[:16]}"
    )


def test_the_seeding_is_what_makes_it_identical_PLANTED_BREAK(graph_arch, monkeypatch) -> None:
    """LAW-07: the proof the check can fire. Remove the seeding and the fingerprints diverge."""
    config, arch = graph_arch
    monkeypatch.setattr(ws, "seed_everything", lambda _seed: None)
    first = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    second = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    assert first != second, (
        "with `seed_everything` neutered the two networks must DIFFER. They did not, which means "
        "this file's live row above is not witnessing the seeding at all"
    )


def test_the_seed_comes_from_the_CONFIG_and_not_from_a_literal(graph_arch, monkeypatch) -> None:
    """The seed passed to `seed_everything` is the config's own `seed`, re-read per build."""
    config, arch = graph_arch
    seen: list[int] = []
    monkeypatch.setattr(ws, "seed_everything", lambda seed: seen.append(seed))
    ws.build_sweep_net(config, arch, torch.device("cpu"))
    assert seen == [int(config.seed)], (
        f"the per-rung build must seed from the config's own seed ({int(config.seed)}); got {seen}"
    )


# ── arm 2: the control that reports it (R317(c)) ─────────────────────────────────────────
def _row(n_workers: int, value: float, net_hash: str | None = "h",
        verdict: str = ws.PLATEAU) -> dict[str, object]:
    return {"n_workers": n_workers, ws.PREREG_METRIC: value, "net_param_hash": net_hash,
            "verdict": verdict}


def test_EQUAL_hashes_AGREE_regardless_of_the_old_bands_throughput_spread() -> None:
    """R317(c)(i): the gate is the hash. This is the SITTING'S OWN measured pair — 276.999 vs
    267.3991 moves/min, 3.9258% apart, which DIVERGED under the old throughput band — and with
    equal hashes it must now AGREE: the throughput never controlled the answer to begin with."""
    control = ws.determinism_verdict(_row(4, 276.999, "same"), _row(4, 267.3991, "same"),
                                     metric=ws.PREREG_METRIC)
    assert control["verdict"] == ws.AGREE, (
        "equal net-parameter hashes must AGREE even at a throughput spread that failed the "
        f"retired 1% band — a control still gating on throughput would fail this; got {control}"
    )
    assert control["spread_pct"] == pytest.approx(3.5901, abs=1e-3), (
        "the spread is still COMPUTED and REPORTED (R317(c)(iii)) — it just does not gate"
    )


def test_UNEQUAL_hashes_DIVERGE_even_at_a_TINY_throughput_spread_PLANTED_BREAK() -> None:
    """The inverse: two drives 0.01% apart on throughput — which would have AGREED under any
    plausible band — must DIVERGE if their nets are not the same. This is the planted break for
    the re-specified gate: neuter the equality and the old band's favourite case now fails."""
    control = ws.determinism_verdict(_row(4, 276.999, "aaa"), _row(4, 276.972, "bbb"),
                                     metric=ws.PREREG_METRIC)
    assert control["verdict"] == ws.DIVERGED, (
        f"different net-parameter hashes must DIVERGE regardless of how close the throughput "
        f"reading is; got {control}"
    )


@pytest.mark.parametrize("verdict", [ws.REFUSED, ws.OOM, ws.RUNG_ERROR, ws.PRODUCER_DEAD])
def test_a_drive_that_is_not_a_MEASUREMENT_refuses_rather_than_agreeing(verdict: str) -> None:
    """REFUSED is never a verdict — the rule this tool carries everywhere, applied here."""
    control = ws.determinism_verdict(_row(4, 276.999, "same"),
                                     _row(4, 275.396, "same", verdict=verdict),
                                     metric=ws.PREREG_METRIC)
    assert control["verdict"] == ws.REFUSED
    assert control["spread_pct"] is None


def test_a_zero_throughput_drive_refuses_rather_than_dividing_by_it() -> None:
    """A rung that produced no moves ranks at 0; the spread would divide by it."""
    control = ws.determinism_verdict(_row(4, 0.0, "same"), _row(4, 275.396, "same"),
                                     metric=ws.PREREG_METRIC)
    assert control["verdict"] == ws.REFUSED


def test_the_control_refuses_TWO_DIFFERENT_RUNGS() -> None:
    """The control compares ONE rung with itself. Two rungs would be a ladder step wearing the
    control's name."""
    with pytest.raises(ValueError, match="ONE rung with itself"):
        ws.determinism_verdict(_row(4, 276.999, "same"), _row(8, 275.396, "same"),
                               metric=ws.PREREG_METRIC)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_NON_FINITE_ranking_value_refuses(bad: float) -> None:
    """NaN and +/-inf are values to `json.loads`; `select_knee` learned that the hard way and the
    control inherits the lesson rather than re-learning it."""
    with pytest.raises(ValueError, match="not a measurement"):
        ws.determinism_verdict(_row(4, 276.999, "same"), _row(4, bad, "same"),
                               metric=ws.PREREG_METRIC)


def test_a_drive_with_NO_HASH_refuses_the_gate_it_has_nothing_to_check() -> None:
    """A rung that OOM'd building the pool never reached `_net_param_hash`; comparing `None` to
    anything would be an absence read as an answer."""
    control = ws.determinism_verdict(_row(4, 276.999, None), _row(4, 275.396, "x"),
                                     metric=ws.PREREG_METRIC)
    assert control["net_hash_gate"]["verdict"] == ws.REFUSED
    assert control["verdict"] == ws.REFUSED


def test_the_bands_constant_is_SUPERSEDED_but_still_pinned_for_history() -> None:
    """R315(c)(i) pinned it; R317(c) supersedes it as a gate but the value stays on record."""
    assert ws.RULED_DETERMINISM_BAND_PCT == 1.0


# ── arm 2.5: the noise floor (R317(d)) ────────────────────────────────────────────────────
def test_the_knee_rule_requires_a_measured_noise_floor_not_a_default() -> None:
    """R317(d): `select_knee` no longer has an implicit floor. `0.0` recovers the un-amended
    rule and must be passed explicitly."""
    with pytest.raises(ValueError, match="noise_floor_rel_std"):
        ws.select_knee([], knee_pct=ws.RULED_KNEE_PCT, metric=ws.PREREG_METRIC,
                       noise_floor_rel_std=-0.01)


def test_the_amendment_can_only_pull_the_pick_toward_FEWER_workers() -> None:
    """R317(d)'s own safety property, measured over a case where the naive band would exclude
    rung 2: a positive noise floor must never REMOVE rung 2 from `within` once it already
    qualifies, and can only ADD smaller-or-equal rungs, never drop the picked rung upward."""
    rows = [{"n_workers": 2, ws.PREREG_METRIC: 91.0, "verdict": ws.PLATEAU},
            {"n_workers": 4, ws.PREREG_METRIC: 100.0, "verdict": ws.PLATEAU}]
    unamended = ws.select_knee(rows, knee_pct=95.0, metric=ws.PREREG_METRIC,
                               noise_floor_rel_std=0.0)
    amended = ws.select_knee(rows, knee_pct=95.0, metric=ws.PREREG_METRIC,
                             noise_floor_rel_std=0.10)
    assert unamended["picked"] == 4, "91 is below the 95-of-100 threshold under the base rule"
    assert amended["picked"] == 2, (
        "a 10% relative noise floor must widen `within` to admit rung 2 and the pick must move "
        f"to the SMALLER rung; got {amended}"
    )
    assert amended["adjusted_threshold"] < unamended["threshold"]


# ── arm 3: the CLI modes' refusals ───────────────────────────────────────────────────────
def test_the_determinism_mode_refuses_n_workers_1_which_the_prereg_REJECTS() -> None:
    assert ws.main(["--determinism-control", "1", "--config", "x", "--plan", "y"]) == ws.RC_REFUSED


def test_the_determinism_mode_refuses_inputs_it_does_not_read() -> None:
    assert ws.main(["--determinism-control", "4", "--config", "x", "--plan", "y",
                    "--select-only", "z"]) == ws.RC_REFUSED


@pytest.mark.parametrize("argv", [["--determinism-control", "4", "--config", "x"],
                                  ["--determinism-control", "4", "--plan", "y"],
                                  ["--determinism-control", "4"]])
def test_the_determinism_mode_refuses_without_both_config_and_plan(argv: list[str]) -> None:
    assert ws.main(argv) == ws.RC_REFUSED


def test_the_two_drive_modes_refuse_being_named_together() -> None:
    """R317(d): --noise-floor is a THIRD drive mode, disjoint from --determinism-control the same
    way --select-only is disjoint from both."""
    assert ws.main(["--determinism-control", "4", "--noise-floor", "4",
                    "--config", "x", "--plan", "y"]) == ws.RC_REFUSED


def test_the_noise_floor_mode_refuses_n_workers_1_which_the_prereg_REJECTS() -> None:
    assert ws.main(["--noise-floor", "1", "--config", "x", "--plan", "y"]) == ws.RC_REFUSED


@pytest.mark.parametrize("argv", [["--noise-floor", "4", "--config", "x"],
                                  ["--noise-floor", "4", "--plan", "y"],
                                  ["--noise-floor", "4"]])
def test_the_noise_floor_mode_refuses_without_both_config_and_plan(argv: list[str]) -> None:
    assert ws.main(argv) == ws.RC_REFUSED


def test_the_ladder_refuses_without_a_noise_floor_report() -> None:
    """R317(d): a ladder that has not measured a noise floor cannot select a pick, so it is
    refused BEFORE the seventy-minute walk, not after it."""
    assert ws.main(["--config", "configs/run5.yaml",
                    "--plan", "tools/worker_sweep_plan.toml"]) == ws.RC_REFUSED
