# R8 justify: the four `monitor.drain.*` keys are ONE claim under R93 — a citation is
# verified by MUTATION, never by grep — and each oracle is the SAME drive (real
# `compose_run`, real minted config, one key distinguishable) with a different key set.
# They share the composition harness and the spied `build_eval_pipeline`; split across
# files, the shared drive would be copied four times and the R93 condition would be
# restated four times instead of held in one place.
"""`monitor.drain.*` reaches the consumer its registry entry NAMES — proved by MUTATION.

WPMINT Phase K-A, ruling R93, closing the DR-11 finding.

WHAT WAS WRONG. Four keys — `monitor.drain.{final_eval_drain_timeout_sec,
eval_final_drain_safety_factor, eval_final_drain_hard_cap_sec, terminal_eval_hard_cap_sec}` —
were minted into every `configs/*.yaml`, schema-validated by `DrainCapsConfig`, and claimed by
BOTH copies of `CONSUMER_REGISTRY` with a citation naming `drain_budget_sec` /
`_run_terminal_sync`. They reached neither. `config/resolve/monitor.py` did
`data.pop("drain")`, and `compose_run` built the real `DrainCaps` from a hardcoded `900.0`
plus three `StepCoordinatorConfig` terminal defaults. Every existing test passed: the registry
bijection is a key-SET diff, gates 7 and 12 are blind to an unconsumed key, and the citation
string named a REAL function — it just named one the value never got to.

WHY THIS FILE IS SHAPED AS A MUTATION. R93's binding condition is that a consumer-registry
citation be verified by MUTATION — set the knob, observe the consumer — never by grep,
"because a grep cannot tell a reader from a `pop`". A test that asserted `resolve_drain_caps`
returns what the config says would have been green throughout the entire defect: the resolver
is not the consumer. So each oracle below drives the REAL `compose_run` on a REAL minted
config with ONE key set to a distinguishable value, and reads the number off the object the
eval pipeline actually bounds its joins with.

WHERE THE OBSERVATION IS TAKEN, and why there. `build_eval_pipeline` is SPIED — the drive
captures the `coordinator_cfg_caps` it is handed and returns a stand-in pipeline, the same
instrument and the same reason as `tests/test_run_strict_composition.py`'s encoding oracle.
That argument object is exactly what `EvalPipeline` stores as `self._caps`, and the other
half of the journey (`_caps` -> `drain_budget_sec` / `_run_terminal_sync`'s `budget_sec`) is
already driven end-to-end against real subprocesses by
`tests/train/test_drain_hardcap_wiring.py`. So the two halves meet at the argument, and
neither is asserted from the other's side. `build_run_safety` and the anchor seed are faked
for the reasons those oracles state (a writable log tree, a watchdog thread, and a `.arch`
read on a fake model) — neither is on the wire under test.

Everything between the config FILE and that argument — the loader, `resolve_drain_caps`,
`_step_coordinator_config`, `StepCoordinatorConfig`, `compose_run`'s `DrainCaps` lift — runs
as production code, unpatched. In particular `_step_coordinator_config` is NOT monkeypatched
here, unlike the five composition drives that suppress terminal eval through it: it is the
subject.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import mantis.run
from mantis.config.resolve.drain import DrainCapsSpec, resolve_drain_caps
from mantis.eval.pipeline import drain_budget_sec
from mantis.train.coordinator.config import StepCoordinatorConfig

#: One distinguishable value per key. None is a shipped value (900/3/14400/14400), and the
#: safety factor is deliberately not a divisor of anything else here, so a `_caps` that
#: reported a stale or defaulted number could not accidentally match.
_DISTINGUISHABLE = {
    "final_eval_drain_timeout_sec": 137.0,
    "eval_final_drain_safety_factor": 7.0,
    "eval_final_drain_hard_cap_sec": 9999.0,
    "terminal_eval_hard_cap_sec": 4242.0,
}
_DRAIN_KEYS = tuple(_DISTINGUISHABLE)


#: The drive is bounded so it terminates; the three step-clock knobs move together because
#: the reachability validator spans them (DESIGN_S §6.6 MF-3).
_DRIVE_STEPS = 4


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self):
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None: ...
    def update_checkpoint_step(self, step: int) -> None: ...


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    # WPTS/TD-1 re-point (R90a): typed seam; the dead `train_step` fake is gone.
    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return {"w": "SENTINEL"}

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


def _composed_caps(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, **drain_over):
    """The `DrainCaps` a REAL `compose_run` hands `build_eval_pipeline`, for a config whose
    `monitor.drain` block carries `drain_over`."""
    import mantis.train.anchor as _anchor

    captured: dict = {}

    def _spy_build_eval_pipeline(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            # WP12-R Phase O: the TERMINAL call returns a ROUND RESULT, and the seam
            # now READS its `eval_broken_reason` (it used to discard it), so a double
            # that answers a kick ACK no longer models the contract it stands in for.
            run_evaluation=lambda *a, **k: {"eval_broken_reason": None},
            poll_completed=lambda: None, drain_pending=lambda: None,
            apply_gate_decision=lambda *a, **k: None, stop=lambda: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)
    monkeypatch.setattr(mantis.run, "build_eval_pipeline", _spy_build_eval_pipeline)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )

    config = smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               # WPTS/TD-1: the drive runs the real graph route; minted 256 batch is drag.
               "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1, "drain": drain_over},
        # WPMAIN/R120+R123: both are CONFIG facts now; `compose_run` has no parameter for
        # either, so the drive declares its posture where the config is built.
        eval_enabled=True, run_id="drain_wiring",
    )
    mantis.run.compose_run(
        config=config, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert "coordinator_cfg_caps" in captured, "the drive never reached build_eval_pipeline"
    return captured["coordinator_cfg_caps"]


# ── the four mutations, one per key (R93) ─────────────────────────────────────────────
@pytest.mark.parametrize("key", _DRAIN_KEYS)
def test_each_drain_key_changes_the_caps_the_eval_pipeline_bounds_its_joins_with(
    key, tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """Set ONE `monitor.drain` key to a distinguishable value; the composed `DrainCaps` must
    carry it. Before this phase every one of these four was green-at-the-config and dead at
    the consumer, so the parametrization is the finding's own shape: four keys, four
    demonstrations, no shared arm that could carry a sibling."""
    baseline = _composed_caps(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer)
    mutated = _composed_caps(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                             **{key: _DISTINGUISHABLE[key]})

    assert getattr(baseline, key) != _DISTINGUISHABLE[key], (
        f"the test value for {key} is not distinguishable from the minted one — this oracle "
        "would pass on a config that reached nothing"
    )
    assert getattr(mutated, key) == _DISTINGUISHABLE[key], (
        f"monitor.drain.{key} did not reach `DrainCaps`. That is DR-11 exactly: the key is "
        "minted, schema-validated and registry-claimed, and the run uses a code-side number "
        "instead"
    )
    unchanged = [other for other in _DRAIN_KEYS
                 if other != key and getattr(mutated, other) != getattr(baseline, other)]
    assert not unchanged, (
        f"setting {key} moved {unchanged} too — the four caps must arrive independently, or "
        "one key's citation is really another key's"
    )


def test_the_drain_budget_arithmetic_moves_with_the_config(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """The registry's cited consumer, driven: `drain_budget_sec` = `min(timeout * factor,
    hard_cap)`. Both branches of that `min` are exercised from the CONFIG, so the citation
    names an arithmetic the config genuinely feeds rather than a function it merely reaches."""
    safety_bound = _composed_caps(
        tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
        final_eval_drain_timeout_sec=10.0, eval_final_drain_safety_factor=2.0,
        eval_final_drain_hard_cap_sec=100.0,
    )
    assert drain_budget_sec(safety_bound) == 20.0

    hard_cap_bound = _composed_caps(
        tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
        final_eval_drain_timeout_sec=1000.0, eval_final_drain_safety_factor=100.0,
        eval_final_drain_hard_cap_sec=5.0,
    )
    assert drain_budget_sec(hard_cap_bound) == 5.0


def test_the_terminal_round_budget_is_the_configured_terminal_hard_cap(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """`_run_terminal_sync` passes `self._caps.terminal_eval_hard_cap_sec` as its
    `budget_sec` (eval/pipeline.py). Its registry entry cites that line, so the composed
    value must be the configured one and not the drain budget it sits beside."""
    caps = _composed_caps(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                          terminal_eval_hard_cap_sec=4242.0,
                          eval_final_drain_hard_cap_sec=5.0)
    assert caps.terminal_eval_hard_cap_sec == 4242.0
    assert drain_budget_sec(caps) == 5.0, (
        "the terminal cap and the drain budget are separate bounds; a wire that collapsed "
        "them would still satisfy a single-value assertion"
    )


# ── no second authority survives anywhere on the path (R1/LAW-08) ─────────────────────
def test_the_builder_takes_drain_caps_as_a_required_keyword_only_parameter() -> None:
    """The `stop_step`/`draw_rate_abort` rule (R83, MF-2 Attack B) applied to the third
    config-authored fact: a parameter DEFAULT would move the authority from the dataclass
    field to this signature, leaving every `dataclasses.fields()` assertion green while a
    caller that omitted the argument silently inherited a posture."""
    param = inspect.signature(mantis.run._step_coordinator_config).parameters.get("drain_caps")
    assert param is not None, (
        "`_step_coordinator_config` must take `drain_caps`: the four caps are "
        "`monitor.drain.*` and arrive from `resolve_drain_caps`, never from a literal here"
    )
    assert param.default is inspect.Parameter.empty, (
        f"drain_caps carries a parameter default ({param.default!r}) — the `900.0` did not "
        "die, it MIGRATED from the builder body to the builder signature"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("name", _DRAIN_KEYS)
def test_no_coordinator_drain_field_carries_a_code_side_default(name) -> None:
    """`StepCoordinatorConfig`'s four drain fields must be MISSING-defaulted. Three of them
    carried `DEFAULT_FINAL_EVAL_DRAIN_*` constants, which is what the run really used while
    the config was popped; the fourth constant had no reader at all."""
    import dataclasses

    field = {f.name: f for f in dataclasses.fields(StepCoordinatorConfig)}[name]
    assert (field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING), (
        f"StepCoordinatorConfig.{name} carries a code-side default ({field.default!r}) — the "
        "schema block is then not its only authority (R1)"
    )


def test_the_resolver_is_the_only_read_of_the_drain_block(smoke_run_config) -> None:
    """`resolve_drain_caps` returns exactly what the loaded config holds, field for field —
    the transport arm. A resolver that dropped or reordered a field would still satisfy the
    mutation oracles above for the key it happened to carry."""
    config = smoke_run_config(monitor={"drain": dict(_DISTINGUISHABLE)})
    spec = resolve_drain_caps(config.monitor)
    assert isinstance(spec, DrainCapsSpec)
    assert {name: getattr(spec, name) for name in _DRAIN_KEYS} == _DISTINGUISHABLE
