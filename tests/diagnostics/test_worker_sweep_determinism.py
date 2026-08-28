# ONE CLAIM with two halves that are deliberately not split across files: "the sweep's per-rung
# network is reproducible, and the control that says so can fail". The mechanism arm (the same
# seed builds the same net) and the instrument arm (`determinism_verdict` separates AGREE from
# DIVERGED from REFUSED) are the two things F-RESIT-10 needs together — a reproducible net nobody
# checks is a claim, and a checker with nothing reproducible under it is a checker of noise. Each
# arm carries its own planted break in-file, for the reason `test_worker_sweep_reachability.py`
# states: a predicate and the proof it can fire must move together or the proof rots quietly.
# (No R8 justification is claimed: this file is under the 300-line cap. Sizes are derived by
# `wc -l`, never asserted.)
"""F-RESIT-10's repair, and the control that witnesses it.

**THE DEFECT, measured at the 2026-08-27 re-calibration re-sit.** `mantis.diagnostics.worker_sweep`
built a fresh `build_net(arch)` per rung from an UNSEEDED RNG, so every rung of the pre-registered
ladder raced a DIFFERENT random network. On an unbounded board a network's policy decides how far
stones spread, which decides the graph's node and edge counts, which decides what every fused
forward costs. So R309(f)'s knee rule — *the smallest rung within 95 % of the best passing rung's
throughput* — compared rungs on a column that was a function of `n_workers` **and an uncontrolled
draw**.

**The size of it, measured rather than argued.** At a FIXED `n_workers = 4`, on one host, one
config, one posture: **276.999 moves/min** on the config's own seed against **444.20** on a lucky
unseeded draw — **1.60x** — while the ENTIRE ladder from 2 workers to 16 spanned 2.39x. The noise
was roughly 60 % of the signal, constant within a rung and resampled between rungs, which is the
worst possible shape for a rule that compares rungs. On the measured ladder the pick moved SIX
RUNGS depending on which net a rung happened to draw.

**The repair** is `build_sweep_net`: seed from the config's own `seed` immediately before the one
RNG consumer on that path. The module it calls MOVED to `mantis.util.determinism` to make that
possible — the sweep is guaranteed trainer-unreachable by import at any scope (R309(g),
`test_worker_sweep_reachability.py`), and importing anything under `mantis.train` was measured to
pull eight training modules into `sys.modules`. Moving the leaf was the only route that neither
weakened that ban nor gave the sweep a second seeding authority.

**What this file does NOT claim.** Seeding fixes the NETWORK. It does not make wall-clock
throughput bit-identical — a real self-play pool has OS scheduling in it — which is why the
sitting-level control is a BAND (`RULED_DETERMINISM_BAND_PCT`, R315(c)(i)'s own number, pinned in
source) and not an equality. The 2026-08-27 discriminator measured the residual with the network
held fixed at **0.58 %**, so the band is one the repair is known to clear with room.
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
    """A content hash of every parameter, key order fixed.

    STRUCTURE, NOT A SAMPLE (R296(f)): comparing one tensor, or a summary statistic, would pass
    for two networks that differ everywhere else. The whole state dict is hashed.
    """
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

    This is the property the ladder's comparability rests on. Before the repair these two hashes
    differed on every call, and nothing in the sweep's report said so."""
    config, arch = graph_arch
    first = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    second = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    assert first == second, (
        "two per-rung networks built from the same config must be bit-identical — that is the "
        "whole of F-RESIT-10's repair, and without it the knee rule ranks rungs on a column "
        f"carrying a resampled random term. Got {first[:16]} and {second[:16]}"
    )


def test_the_seeding_is_what_makes_it_identical_PLANTED_BREAK(graph_arch, monkeypatch) -> None:
    """LAW-07: the proof the check can fire. Remove the seeding and the fingerprints diverge.

    A determinism assertion that has never been shown to fail is indistinguishable from one that
    compares a constant with itself — and this one would pass trivially if `build_net` were
    deterministic for some reason other than the seed."""
    config, arch = graph_arch
    monkeypatch.setattr(ws, "seed_everything", lambda _seed: None)
    first = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    second = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    assert first != second, (
        "with `seed_everything` neutered the two networks must DIFFER. They did not, which means "
        "this file's live row above is not witnessing the seeding at all — either `build_net` "
        "became deterministic by another route, or the seam moved and the monkeypatch no longer "
        "reaches it. Either way the determinism claim is now unproven"
    )


def test_the_seed_comes_from_the_CONFIG_and_not_from_a_literal(graph_arch, monkeypatch) -> None:
    """The seed passed to `seed_everything` is the config's own `seed`, re-read per build.

    A literal here would survive a re-minted seed and go quietly stale — the transcription class
    this repo keeps finding. Asserted on the ARGUMENT, so it holds whatever the value is."""
    config, arch = graph_arch
    seen: list[int] = []
    monkeypatch.setattr(ws, "seed_everything", lambda seed: seen.append(seed))
    ws.build_sweep_net(config, arch, torch.device("cpu"))
    assert seen == [int(config.seed)], (
        f"the per-rung build must seed from the config's own seed ({int(config.seed)}); got {seen}"
    )


# ── arm 2: the control that reports it ───────────────────────────────────────────────────
def _row(n_workers: int, value: float, verdict: str = ws.PLATEAU) -> dict[str, object]:
    return {"n_workers": n_workers, ws.PREREG_METRIC: value, "verdict": verdict}


def test_two_drives_inside_the_band_AGREE() -> None:
    """The sitting's own discriminator numbers, which are why the band is 1 %."""
    control = ws.determinism_verdict(_row(4, 276.999), _row(4, 275.396),
                                     metric=ws.PREREG_METRIC,
                                     band_pct=ws.RULED_DETERMINISM_BAND_PCT)
    assert control["verdict"] == ws.AGREE
    assert control["spread_pct"] == pytest.approx(0.5821, abs=1e-3), (
        "the 2026-08-27 discriminator measured 276.999 against 275.396 with the network held "
        "fixed; this control must reproduce that spread, because it is the measurement the band "
        "was chosen against"
    )


def test_the_unseeded_DRAW_SPREAD_would_have_DIVERGED_PLANTED_BREAK() -> None:
    """The control must reject the defect it exists for: the 1.60x draw-to-draw variation.

    Not a synthetic pair — these are the two figures F-RESIT-10 was measured from, at the SAME
    worker count on the same host. A control that called these two drives 'agreeing' would be
    reporting determinism about the exact variation the repair removes."""
    control = ws.determinism_verdict(_row(4, 276.999), _row(4, 444.20),
                                     metric=ws.PREREG_METRIC,
                                     band_pct=ws.RULED_DETERMINISM_BAND_PCT)
    assert control["verdict"] == ws.DIVERGED, (
        "276.999 against 444.20 is the measured unseeded draw-to-draw spread and must NOT pass "
        f"a 1 % band; got {control}"
    )
    assert control["spread_pct"] > 50.0


@pytest.mark.parametrize("verdict", [ws.REFUSED, ws.OOM, ws.RUNG_ERROR, ws.PRODUCER_DEAD])
def test_a_drive_that_is_not_a_MEASUREMENT_refuses_rather_than_agreeing(verdict: str) -> None:
    """REFUSED is never a verdict — the rule this tool carries everywhere, applied here.

    Two numbers off a rung that OOM'd are not two measurements, and 'they agree' about them is
    the shape of every under-measurement this instrument family has been caught by."""
    control = ws.determinism_verdict(_row(4, 276.999), _row(4, 275.396, verdict=verdict),
                                     metric=ws.PREREG_METRIC,
                                     band_pct=ws.RULED_DETERMINISM_BAND_PCT)
    assert control["verdict"] == ws.REFUSED
    assert control["spread_pct"] is None


def test_a_zero_throughput_drive_refuses_rather_than_dividing_by_it() -> None:
    """A rung that produced no moves ranks at 0; the spread would divide by it."""
    control = ws.determinism_verdict(_row(4, 0.0), _row(4, 275.396),
                                     metric=ws.PREREG_METRIC,
                                     band_pct=ws.RULED_DETERMINISM_BAND_PCT)
    assert control["verdict"] == ws.REFUSED


def test_the_control_refuses_TWO_DIFFERENT_RUNGS() -> None:
    """The control compares ONE rung with itself. Two rungs would be a ladder step wearing the
    control's name, and it would 'pass' whenever two adjacent rungs happened to be close."""
    with pytest.raises(ValueError, match="ONE rung with itself"):
        ws.determinism_verdict(_row(4, 276.999), _row(8, 275.396),
                               metric=ws.PREREG_METRIC,
                               band_pct=ws.RULED_DETERMINISM_BAND_PCT)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_NON_FINITE_ranking_value_refuses(bad: float) -> None:
    """NaN and +/-inf are values to `json.loads`; `select_knee` learned that the hard way and the
    control inherits the lesson rather than re-learning it."""
    with pytest.raises(ValueError, match="not a measurement"):
        ws.determinism_verdict(_row(4, 276.999), _row(4, bad),
                               metric=ws.PREREG_METRIC,
                               band_pct=ws.RULED_DETERMINISM_BAND_PCT)


def test_the_band_is_the_RULING_S_and_is_pinned_in_SOURCE() -> None:
    """R315(c)(i) fixes the band by number. F-WS-3's lesson: a ruling's constant that a sitting
    can edit between two runs of the same tool has moved."""
    assert ws.RULED_DETERMINISM_BAND_PCT == 1.0


# ── arm 3: the CLI mode's refusals ───────────────────────────────────────────────────────
def test_the_mode_refuses_n_workers_1_which_the_prereg_REJECTS() -> None:
    """R309(f) rejects `n_workers = 1`; a control mode that drove it would be measuring the one
    value the picker refuses to pick."""
    assert ws.main(["--determinism-control", "1", "--config", "x", "--plan", "y"]) == ws.RC_REFUSED


def test_the_mode_refuses_inputs_it_does_not_read() -> None:
    """The `--select-only` discipline, carried: naming an input a mode does not read describes a
    run that did not happen."""
    assert ws.main(["--determinism-control", "4", "--config", "x", "--plan", "y",
                    "--select-only", "z"]) == ws.RC_REFUSED


@pytest.mark.parametrize("argv", [["--determinism-control", "4", "--config", "x"],
                                  ["--determinism-control", "4", "--plan", "y"],
                                  ["--determinism-control", "4"]])
def test_the_mode_refuses_without_both_config_and_plan(argv: list[str]) -> None:
    """Neither is defaulted, for the ladder's reason: a config this tool picked would measure a
    program nobody asked about, and a plan it picked would be a pre-registration nobody wrote."""
    assert ws.main(argv) == ws.RC_REFUSED
