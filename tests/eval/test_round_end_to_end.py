"""⊕ WP11-A — full headless eval round, end to end (mantis.eval.pipeline; integration-tier).

RED-at-import until IMPL writes `mantis.eval.pipeline` (+ worker.py, snapshot.py, bots).
Marked `@pytest.mark.integration` (CI integration tier, `make test.integration`) because —
unlike the other `tests/eval/*.py` suites, which fake the subprocess boundary to stay fast
and deterministic — THIS suite's whole point is to prove the real out-of-process worker
actually runs a real headless round on CPU: no `multiprocessing.get_context` patch here.
`worker_device="cpu"`.

FIX-PASS amendment (design-gap G-1..G-3, dispatcher ruling option (b)): the original tiny
`CnnArch(board_size=5, in_channels=4)` net was NOT a registered encoding and its tensors
never matched what the engine actually feeds a net through this path — `mantis.eval.worker`
runs inference via `LocalInferenceEngine` with NO `encoding_spec` override, so the wire
tensor it decodes is always the registered `"v6"` encoding's shape (`board_size=19,
n_planes=8` — verified directly against `crates/mantis-encoding/src/registry.toml`, and
empirically: a `board_size=5, in_channels=4` net dies with `RuntimeError: ... expected
input[2, 8, 19, 19] to have 4 channels`). This fixture now builds `encoding="v6"`
end-to-end (board, snapshot tag, RegimeKey stamps) and a REAL-ARCH net at v6's true dims
(`board_size=19, in_channels=8`), minimal width/depth (`filters=8, res_blocks=1`) for
speed — registry-true, so `Board.with_encoding_name` tensors match the net exactly, no
accidental shape coincidence.

Second dispatcher-ruled amendment (required by Part 4's revert of the first-round
dormant-rung top-up, deviation #3): `LadderState.initial()` marks ONLY rung INDEX 0 active;
every other rung starts dormant and activates only when its immediate predecessor's most
recent MEASURED round clears `activation_wr_lower_ci` (STATE §5's real chained law). The
resolvable stub (`bot="random"`) is therefore rung index 0 here — the ONE in-repo resolvable
rung, per DESIGN.md's census: "0 of 6 [ladder] rungs resolve locally" — so it plays from
round 1 without needing the reverted top-up. `sealbot_d5` sits behind it (index 1) and
never activates in this fixture's short run (its own predecessor's WR never needs to clear
the bar for the round to complete); it stays loud-skipped every round it IS active for
(no `MANTIS_BOT_SEALBOT` adapter at HEAD) — consistent with the 0/6 census, not a fixture
workaround. The real end-to-end round is exercised by the gate block (skipped here — no
`best_model` yet, run3 `run(best_model=None)` parity) + the random floor + the resolvable
ladder rung.

IMPL API pin introduced by this oracle: the routed round-result dict carries an ADDITIONAL
`"worker_pid"` key beyond the §c.2 shape (§c.2 is explicitly superset-stable: "consumers
must tolerate additions, never removals") — the concrete mechanism this suite uses to
assert "eval inference out-of-process" (a dispatch success criterion) without reaching
into pipeline internals.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.model import CnnArch, build_net

pytestmark = pytest.mark.integration


def _tiny_model(*, weight_seed: int) -> torch.nn.Module:
    # Registry-true "v6" dims (board_size=19, n_planes=8 — crates/mantis-encoding/src/
    # registry.toml `[encodings.v6]`); minimal width/depth (filters/res_blocks) for speed.
    # `weight_seed` is DETERMINISTIC-but-DIFFERENT per round: with n_sims=4 (a genuinely
    # shallow search) most individual games between two weak/untrained players end in the
    # ply-cap draw (`arena/match.py::_DEFAULT_MAX_PLIES=128` — the board is unbounded, a
    # 6-in-a-row is not guaranteed within any ply budget), so a handful of decisive games
    # per round is a low-probability, high-variance event — reproducible determinism (a
    # fixed seed per round, not "whatever torch's global RNG state happens to be") is what
    # keeps `test_second_round_scheduling_reflects_first_round_bt` from being flaky across
    # runs while still exercising the REAL worker/arena/BT path end to end.
    torch.manual_seed(weight_seed)
    arch = CnnArch(board_size=19, in_channels=8, filters=8, res_blocks=1)
    net = build_net(arch)
    net.arch = arch
    return net


def _eval_cfg() -> EvalConfig:
    rungs = [
        # index 0: `LadderState.initial()` starts ONLY the first rung ACTIVE (STATE §5's
        # real chained activation law, post deviation-#3-revert) — the resolvable stub
        # must be index 0 so it plays from round 1 without a top-up.
        LadderRung(name="resolvable_stub", bot="random", variant="raw", depth=None,
                   opponent_sims=None, opening_book="book_v1_s20260625_p4",
                   deploy_matched=True, games_max=20),
        # index 1: dormant behind the stub; never resolves (0/6 census) even if it later
        # activates — exercises the loud-skip path, not a fixture workaround.
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
    ]
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    ladder = LadderConfig(
        rungs=rungs, round_games=20, min_games_per_active_rung=10,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=2, bootstrap_resamples=200,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    return EvalConfig(
        random_model_sims=4, sealbot_model_sims=4, kraken_model_sims=4,
        strix_model_sims=4, random_floor_games=2, worker_device="cpu",
        round_timeout_sec=90.0, worker_kill_grace_sec=5.0, gate=gate, ladder=ladder,
    )


def _promotion_hooks(tmp_path: Path) -> DeployTagHooks:
    from types import SimpleNamespace

    return DeployTagHooks(
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        best_model_path=tmp_path / "best_model.pt",
        run_id="oracle_e2e_run",
        encoding="v6",
        save_anchor=lambda *a, **k: None,
        guarded_load=lambda *a, **k: None,
    )


def _build_pipeline(tmp_path: Path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    return build_eval_pipeline(
        eval_cfg=_eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=90.0,
            eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=90.0,
            terminal_eval_hard_cap_sec=90.0,
        ),
        encoding="v6",
        run_id="oracle_e2e_run",
        spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=_promotion_hooks(tmp_path),
    )


def _poll_until_complete(pipeline, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = pipeline.poll_completed()
        if result is not None:
            return result
        time.sleep(0.1)
    pytest.fail(f"eval round did not complete within {timeout}s")


def test_full_headless_round_end_to_end(tmp_path) -> None:
    pipeline = _build_pipeline(tmp_path)
    try:
        ack = pipeline.run_evaluation(_tiny_model(weight_seed=20260625), 1000, None,
                                       full_config={}, best_model_step=None)
        assert ack["kicked"] is True

        result = _poll_until_complete(pipeline, timeout=90.0)
        assert result["eval_broken"] is False
        assert "wr_sealbot" in result   # G-2 handshake: always present, even with no sealbot games
        assert "schedule_next" in result and result["schedule_next"]
        assert "bt" in result and result["bt"].get("ratings")
        assert isinstance(result["promoted"], bool)   # gate decision present in the routed shape

        rungs_played = {name: info for name, info in result["rungs"].items() if info["games"] > 0}
        assert rungs_played, "no rung recorded any games in a full round with a resolvable stub rung"
        assert "resolvable_stub" in rungs_played

        assert "worker_pid" in result
        assert result["worker_pid"] != os.getpid(), "eval inference must run out-of-process"
    finally:
        pipeline.stop()


def test_round_records_carry_regime_key_on_every_record(tmp_path) -> None:
    pipeline = _build_pipeline(tmp_path)
    try:
        pipeline.run_evaluation(_tiny_model(weight_seed=20260625), 1000, None,
                                 full_config={}, best_model_step=None)
        result = _poll_until_complete(pipeline, timeout=90.0)
        rungs_played = {name: info for name, info in result["rungs"].items() if info["games"] > 0}
        assert rungs_played
        regime_keys = [info["regime_key"] for info in rungs_played.values()]
        assert all(regime_keys), "every played rung's aggregate must carry a non-empty regime_key"
        # aggregate_rung raises MixedRegimeError on a mixed regime_key set (test_aggregate_regime.py
        # pins this directly) — a successfully-produced aggregate here is itself evidence every
        # underlying per-game record shared one canonical regime_key. Distinct rungs must not
        # collide on the same key (they differ in bot/opponent, so their regime_keys must differ).
        assert len(set(regime_keys)) == len(regime_keys)
    finally:
        pipeline.stop()


def test_second_round_scheduling_reflects_first_round_bt(tmp_path) -> None:
    # Two DIFFERENT deterministic weight seeds — empirically verified (this fix pass) to
    # be reproducibly decisive-outcome-yielding at this fixture's game count, so the BT
    # fit's p_hat genuinely differs between rounds instead of racing "will a 6-in-a-row
    # happen to form before the ply cap" on an unseeded net (see `_tiny_model` docstring).
    pipeline = _build_pipeline(tmp_path)
    try:
        pipeline.run_evaluation(_tiny_model(weight_seed=42), 1000, None,
                                 full_config={}, best_model_step=None)
        result1 = _poll_until_complete(pipeline, timeout=90.0)

        pipeline.run_evaluation(_tiny_model(weight_seed=1337), 2000, None,
                                 full_config={}, best_model_step=None)
        result2 = _poll_until_complete(pipeline, timeout=90.0)

        p_hat_1 = result1["bt"]["p_hat"]
        p_hat_2 = result2["bt"]["p_hat"]
        assert p_hat_1 != p_hat_2, "the second round's BT fit must reflect the first round's games"
        assert result1["schedule_next"] != result2["schedule_next"] or p_hat_1 != p_hat_2
    finally:
        pipeline.stop()
