# ONE CLAIM with two halves deliberately not split: the sweep's per-rung network is reproducible,
# and the control that says so can fail. Each arm carries its own planted break in-file, so a
# predicate and the proof it can fire move together.
"""The per-rung network is seeded, and the control that witnesses it.

THE DEFECT: the sweep built a fresh `build_net(arch)` per rung from an UNSEEDED RNG, so every
rung raced a DIFFERENT random network — and on an unbounded board the policy decides how far
stones spread, hence node and edge counts, hence what a fused forward costs. THE REPAIR is
`build_sweep_net`, seeding from the config's own `seed` immediately before the one RNG consumer,
through `mantis.util.determinism`, because importing anything under `mantis.train` pulls eight
training modules in and the sweep must stay trainer-unreachable.

THE CONTROL IS NET-PARAMETER-HASH EQUALITY, NO BAND: the retired throughput band measured 0.58%
engine-side and 3.93% on the box, against within-drive round noise of ~6% peak-to-peak.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.diagnostics import worker_sweep as ws

_REPO = Path(__file__).resolve().parents[2]

#: A committed GRAPH config, read through the real loader: a stub config would let this file pass
#: while the production path read a key that is not there.
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"


def _net_fingerprint(model: object) -> str:
    """A content hash of every parameter, key order fixed — a SEPARATE implementation from
    `worker_sweep._net_param_hash`, so this is the oracle and not the mechanism re-run."""
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


def test_the_same_config_builds_a_BIT_IDENTICAL_network_every_time(graph_arch) -> None:
    """The repair itself: two `build_sweep_net` calls, same config, byte-identical parameters,
    checked with an independent hash so the mechanism is not validated by its own instrument."""
    config, arch = graph_arch
    first = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    second = _net_fingerprint(ws.build_sweep_net(config, arch, torch.device("cpu")))
    assert first == second, (
        "two per-rung networks built from the same config must be bit-identical — that is the "
        "whole of F-RESIT-10's repair, and without it the knee rule ranks rungs on a column "
        f"carrying a resampled random term. Got {first[:16]} and {second[:16]}"
    )


def test_the_seeding_is_what_makes_it_identical_PLANTED_BREAK(graph_arch, monkeypatch) -> None:
    """The proof the check can fire: remove the seeding and the fingerprints diverge."""
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


def _row(n_workers: int, value: float, net_hash: str | None = "h",
        verdict: str = ws.PLATEAU) -> dict[str, object]:
    return {"n_workers": n_workers, ws.PREREG_METRIC: value, "net_param_hash": net_hash,
            "verdict": verdict}


def test_EQUAL_hashes_AGREE_regardless_of_the_old_bands_throughput_spread() -> None:
    """The gate is the hash. This is the sitting's own measured pair — 276.999 vs 267.3991
    moves/min, 3.9258% apart, which DIVERGED under the retired throughput band — and with equal
    hashes it must now AGREE."""
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
    """The inverse: two drives 0.01% apart on throughput, which any band would have called AGREE,
    must DIVERGE when their nets differ."""
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
    """The control compares ONE rung with itself; two rungs would be a ladder step wearing the
    control's name."""
    with pytest.raises(ValueError, match="ONE rung with itself"):
        ws.determinism_verdict(_row(4, 276.999, "same"), _row(8, 275.396, "same"),
                               metric=ws.PREREG_METRIC)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_NON_FINITE_ranking_value_refuses(bad: float) -> None:
    """NaN and +/-inf are values to `json.loads`, and the knee selector learned that the hard
    way."""
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
    """Superseded as a gate, but the value stays on record."""
    assert ws.RULED_DETERMINISM_BAND_PCT == 1.0


def test_the_knee_rule_takes_no_noise_scalar_and_refuses_a_rung_that_cannot_state_its_own() -> None:
    """The carried-noise assumption was measured FALSE, so each rung states its own rel-SE: no
    scalar left to pass, no default to fall to, and a rung without one is refused by name."""
    import inspect
    assert "noise_floor_rel_std" not in inspect.signature(ws.select_knee).parameters
    assert not hasattr(ws, "run_noise_floor") and not hasattr(ws, "read_noise_floor_report")
    with pytest.raises(ValueError, match="rung 2 carries no measured rel_se"):
        ws.select_knee([{"n_workers": 2, ws.PREREG_METRIC: 91.0, "verdict": ws.PLATEAU}],
                       knee_pct=ws.RULED_KNEE_PCT, metric=ws.PREREG_METRIC)


def test_the_widening_can_only_pull_the_pick_toward_FEWER_workers() -> None:
    """The safety property: a noisy rung can only ADD rungs to `within`, never remove one, and
    the pick is still the smallest member — so it moves toward fewer workers or stays put."""
    def rows(rel_se_2: float) -> list[dict]:
        return [{"n_workers": 2, ws.PREREG_METRIC: 91.0, "verdict": ws.PLATEAU,
                 f"{ws.PREREG_METRIC}_spread": {"rel_se": rel_se_2, "n_rounds": 5}},
                {"n_workers": 4, ws.PREREG_METRIC: 100.0, "verdict": ws.PLATEAU,
                 f"{ws.PREREG_METRIC}_spread": {"rel_se": 0.0, "n_rounds": 5}}]
    quiet = ws.select_knee(rows(0.0), knee_pct=95.0, metric=ws.PREREG_METRIC)
    noisy = ws.select_knee(rows(0.10), knee_pct=95.0, metric=ws.PREREG_METRIC)
    assert quiet["picked"] == 4, "91 is below the 95-of-100 threshold with no noise"
    assert noisy["picked"] == 2, (
        "a 10% rel-SE at rung 2 must widen `within` to admit it and the pick must move to the "
        f"SMALLER rung; got {noisy}"
    )
    assert noisy["adjusted_threshold"] < quiet["threshold"]


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
