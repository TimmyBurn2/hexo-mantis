""">300 justify (R8): one block, one rejection corpus. The rejected payloads and the per-config
posture census are DATA whose reason text IS the assertion, so splitting them would separate a
rejection from the prereg pin it is judged against, and R5 bars the cross-test import that would
rejoin them.

What the draw-rate block can EXPRESS, and what each committed config says. The sibling authority
file asserts who may set the value; this one asserts what the type system ACCEPTS at all.

The defect each oracle is the ONLY witness to: a value outside the metric's own range in either
direction, on every key — a threshold `> 1.0` passes `gt=0`, audits ARMED and can NEVER fire, and
none of that is visible to the audit oracles, which only ever see values that already loaded; the
evidence bar's CEILING, a cross-SECTION rule against `selfplay.n_workers`, and its FLOOR, where
one drawn game would fire the abort; and a config that INHERITS its posture instead of stating it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

# `ruff --fix` re-sorts the `resolve.draw_rate` import into the third-party block while that
# module does not exist; it sits here, with its `mantis.*` siblings, where it belongs.
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.draw_rate import resolve_draw_rate_abort  # RED anchor (R80)
from mantis.config.schema import RunConfig
from mantis.util.constants import DRAW_RATE_WINDOW

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

#: Pre-registered run-scoped constants. NOT tunables: mint prereg is the only place they may
#: change, so they are written here as the pin that makes an in-place edit visible.
RUN5_PREREG = {"threshold": 0.25, "min_step": 25000, "N_pool_min": 50, "consec": 3}


#: The production evidence ceiling, `DRAW_RATE_WINDOW * selfplay.n_workers`, read off the live
#: config. Derived once so every arm below follows a re-minted `n_workers` instead of asserting a
#: literal that was only true at a particular pick.
_RUN5_EVIDENCE_CEILING = DRAW_RATE_WINDOW * load_config(
    CONFIGS_DIR / "run6.yaml").selfplay.n_workers


def _with_block(payload):
    """The production config's own minted dump with `train.draw_rate_abort` REPLACED WHOLESALE.

    Wholesale, not merged: the shared factory deep-merges section dicts, so a deliberately PARTIAL
    block would silently inherit the missing key and the "the three are inseparable" arm would
    assert nothing. Everything else in the payload is the committed file's.
    """
    dumped = load_config(CONFIGS_DIR / "run6.yaml").model_dump()
    dumped["train"]["draw_rate_abort"] = payload
    return RunConfig.model_validate(dumped)



def test_the_schema_cannot_express_a_value_OUTSIDE_the_metrics_own_range() -> None:
    """The schema cannot express a value outside the metric's own range, at either end.

    `gt=0` forecloses only the `<= 0` half. The pooled rate is a fraction in `[0, 1]` and the
    predicate is an UPPER bound, so ANY threshold `> 1.0` can never be met, is accepted, and reads
    ARMED — reachable by the natural percent slip, where an operator meaning 35% writes `35`.

    The hair-trigger residual is closed as a side effect of the pooled statistic: the compared
    value's smallest non-zero value at the bar is exactly `1/N_pool_min` at EVERY worker count, and
    the block requires `1/N_pool_min < threshold`. Boundaries are asserted on both sides, so the
    bound is the arithmetic and not a literal.
    """
    armed = dict(RUN5_PREREG)
    assert _with_block(armed).train.draw_rate_abort.threshold == 0.25
    assert _with_block({**armed, "threshold": 1.0}).train.draw_rate_abort.threshold == 1.0, (
        "1.0 is IN range — `le=1` is a ceiling on the metric's own maximum, not an exclusion"
    )
    assert _with_block(None).train.draw_rate_abort is None, (
        "`null` is the EXPLICIT off state (R79(1)): a word, not a number, and nobody types "
        "it by accident"
    )

    rejected = {
        "threshold 1e-300 — MF-1's disclosed hair-trigger, CLOSED by R92's floor rule":
            ({**armed, "threshold": 1e-300}, "N_pool_min"),
        "threshold 0.0 — the status-quo spelling of OFF, now a NAMED rejection":
            ({**armed, "threshold": 0.0}, "threshold"),
        "threshold -1.0 — below the metric's floor": ({**armed, "threshold": -1.0}, "threshold"),
        "threshold 2.0 — above the metric's ceiling; ARMED and can never fire":
            ({**armed, "threshold": 2.0}, "threshold"),
        "threshold 35 — the natural percent slip, minted green under `gt=0` alone":
            ({**armed, "threshold": 35}, "threshold"),
        "threshold .inf — the same class at the limit":
            ({**armed, "threshold": float("inf")}, "threshold"),
        "threshold true — a bool is not a fraction": ({**armed, "threshold": True}, "threshold"),
        # DERIVED, not transcribed: this read `51` while the pool was one worker, and a re-mint
        # made 51 perfectly reachable. The ceiling comes off the LIVE config, so the arm follows
        # any future pick instead of going quietly green.
        "N_pool_min one above the pool's OWN ceiling — unreachable evidence (R92's 4th axis)":
            ({**armed, "N_pool_min": _RUN5_EVIDENCE_CEILING + 1}, "N_pool_min"),
        "N_pool_min 0 — no pool ever banks fewer than zero games, so the bar is inert":
            ({**armed, "N_pool_min": 0}, "N_pool_min"),
        "min_step 0 — the ADJ-14 hair-trigger the R80 guards exist to close":
            ({**armed, "min_step": 0}, "min_step"),
        "min_step == train.max_train_steps — a guard the run never passes":
            ({**armed, "min_step": 1_000_000}, "min_step"),
        "a partial block — the three components are inseparable (R80)":
            ({"threshold": 0.25, "min_step": 25000}, "N_pool_min"),
        "the RETIRED key — `min_samples` is gone (R92) and must not load silently":
            ({**armed, "min_samples": 50}, "min_samples"),
        # Re-pointed rather than deleted once `consec` became an authored key: the property —
        # strictness reaches INSIDE the nested block, not just at the top level — is unchanged, so
        # the row needs a key that is genuinely not in the block.
        "an unknown inner key — extra='forbid' reaches inside the block too":
            ({**armed, "consec_rounds": 3}, "consec_rounds"),
        "consec 0 — a rule that needs zero consecutive observations is not a rule":
            ({**armed, "consec": 0}, "consec"),
    }
    for reason, (payload, key) in rejected.items():
        with pytest.raises(ValidationError) as caught:
            _with_block(payload)
        assert key in str(caught.value), (
            f"{reason}: the rejection must NAME the offending key {key!r} (R1: missing key = "
            f"error, unknown key = error, and the message is what an operator acts on); got "
            f"{caught.value}"
        )

    with pytest.raises(ValidationError) as caught:
        _with_block({**armed, "threshold": 35})
    assert "less than or equal to 1" in str(caught.value), (
        "the `35` rejection must come from the CEILING, not from some other bound that "
        f"happens to fire first — MF-1's whole finding is the open upper half; got "
        f"{caught.value}"
    )

    base = load_config(CONFIGS_DIR / "run6.yaml").model_dump()
    base["train"].pop("draw_rate_abort")
    with pytest.raises(ValidationError) as caught:
        RunConfig.model_validate(base)
    assert "draw_rate_abort" in str(caught.value), (
        "ABSENCE must be an error naming the key: `Field(default=...)` is the repo's own "
        "no-terminal-default idiom in this very class (`schema/train.py:41,43`). A key that "
        "may be omitted has a default somewhere, and that default is a second authority "
        "(R1/LAW-11)"
    )


def test_the_evidence_bar_must_be_reachable_within_the_pools_own_window() -> None:
    """The evidence bar must be reachable within the pool's own window.

    The ceiling is `DRAW_RATE_WINDOW * selfplay.n_workers`, and `selfplay` is a different SECTION,
    so it cannot be an `le=` on the field and lives on the ONE model that sees both. A bar above it
    is another "armed in the config, absent in effect" axis: the gate makes NO observation for the
    entire run while the row audits ARMED. Three arms, because two alone are satisfied by a
    re-spelled `le=`.
    """
    # THE PRECONDITION IS DERIVED, NOT PINNED. It used to assert a one-worker pool; a re-mint
    # changed that, so the arms move WITH the config. The property never depended on the pool
    # being one worker — only the literals did.
    ceiling = _RUN5_EVIDENCE_CEILING
    assert ceiling == DRAW_RATE_WINDOW * load_config(
        CONFIGS_DIR / "run6.yaml").selfplay.n_workers, "the ceiling is derived, never assumed"
    at_ceiling = _with_block({**RUN5_PREREG, "N_pool_min": ceiling})
    assert at_ceiling.train.draw_rate_abort.N_pool_min == ceiling, (
        "AT the ceiling the bar is satisfiable (the deques saturate exactly there), so it "
        "must load — a bound that also forbade the reachable value would disarm the abort"
    )

    with pytest.raises(ValidationError) as caught:
        _with_block({**RUN5_PREREG, "N_pool_min": ceiling + 1})
    assert "N_pool_min" in str(caught.value) and "n_workers" in str(caught.value), (
        "one game above the ceiling must be REJECTED, and the message must name BOTH keys: "
        "the operator cannot act on 'too big' without knowing what it is too big FOR; got "
        f"{caught.value}"
    )

    wider = load_config(CONFIGS_DIR / "run6.yaml").model_dump()
    wider["selfplay"]["n_workers"] = wider["selfplay"]["n_workers"] + 1
    wider["train"]["draw_rate_abort"] = {**RUN5_PREREG, "N_pool_min": ceiling + 1}
    assert RunConfig.model_validate(wider).train.draw_rate_abort.N_pool_min == ceiling + 1, (
        "the SAME value must be accepted once the pool is ONE worker wider: the bound is the "
        "PRODUCT `DRAW_RATE_WINDOW * selfplay.n_workers`, not a re-spelled "
        "`le=DRAW_RATE_WINDOW`. Without this arm the cross-section rule is indistinguishable "
        "from a field bound"
    )


def test_the_evidence_bar_cannot_be_so_small_that_one_drawn_game_fires() -> None:
    """The evidence bar cannot be so small that one drawn game fires the abort.

    The pooled rate's smallest non-zero value at the bar is `1/N_pool_min`, so at `N_pool_min = 4`
    with `threshold = 0.25` a SINGLE drawn game meets the threshold. The block closes it from
    values already inside it — no invented number — and the boundary is asserted on both sides, so
    the rule is the arithmetic `1/N_pool_min < threshold` and not a literal floor.
    """
    with pytest.raises(ValidationError) as caught:
        _with_block({**RUN5_PREREG, "N_pool_min": 4})
    assert "N_pool_min" in str(caught.value), (
        f"N_pool_min=4 at threshold 0.25 lets ONE drawn game in four fire a HARD ABORT — "
        f"1/4 = 0.25 >= 0.25. It must be rejected, naming the key; got {caught.value}"
    )
    assert _with_block({**RUN5_PREREG, "N_pool_min": 5}).train.draw_rate_abort.N_pool_min == 5, (
        "…and 5 must load: 1/5 = 0.2 < 0.25, so one drawn game is NOT enough. A floor that "
        "rejected both sides would be a policy number rather than the metric's arithmetic"
    )
    assert 1.0 / RUN5_PREREG["N_pool_min"] < RUN5_PREREG["threshold"], (
        "run5's own pre-registered pair must satisfy the rule with margin (0.02 vs 0.25) — "
        "if it ever did not, the armed production config would be unloadable"
    )


def test_every_config_states_its_draw_rate_posture_explicitly() -> None:
    """Every config STATES its draw-rate posture; none inherits one.

    Deliberate disarming stays legal for smoke configs, which is what the `null` spelling makes
    OBSERVABLE rather than inferable from absence. Enumerated through `discover_configs`, the ONE
    authority both gate 7 and gate 12 consume. Both directions are asserted, because "every config
    carries the key" is satisfied by a tree where every config is disarmed.
    """
    configs = discover_configs(CONFIGS_DIR)
    # The vacuity floor moves WITH the ruling that pruned `configs/` and not below it: three is
    # what the tree ships, so a fourth deletion still reds here.
    assert len(configs) >= 3, (
        f"the vacuity floor: {len(configs)} config(s) discovered. With none, every assertion "
        "below is true by having nothing to say (`silent_encoding_gate.py:70`'s "
        "MIN_SCANNED_FILES applied to the config set)"
    )

    postures: dict[str, object] = {}
    for path in configs:
        cfg = load_config(path)
        block = cfg.train.draw_rate_abort
        postures[path.name] = block
        resolved = resolve_draw_rate_abort(cfg.train)
        assert (resolved is None) == (block is None), (
            f"{path.name}: the ONE resolver must agree with the block it reads — a resolver "
            "that invents a posture is a second authority (R80)"
        )
        if block is not None:
            assert (resolved.threshold, resolved.min_step, resolved.N_pool_min,
                    resolved.consec) == (
                float(block.threshold), int(block.min_step), int(block.N_pool_min),
                int(block.consec)), (
                f"{path.name}: the resolver must carry the operator's terms through verbatim"
            )

    # The ONE armed production config, which CARRIES the four pre-registered constants rather than
    # re-authoring them. The pin is on the MINTED file, so an in-place edit of its armed block reds
    # here; gate 12 audits it by name.
    run6 = postures.pop("run6.yaml", None)
    assert run6 is not None, (
        "configs/run6.yaml is the ONE declared PRODUCTION config and must ARM the draw-rate "
        "row — a disarmed production config is rc 30 at gate 12 (R59/R61)"
    )
    assert (run6.threshold, run6.min_step, run6.N_pool_min, run6.consec) == (
        RUN5_PREREG["threshold"], RUN5_PREREG["min_step"], RUN5_PREREG["N_pool_min"],
        RUN5_PREREG["consec"]), (
        f"run6 CARRIES the four pre-registered constants unchanged; got {run6}. They are "
        f"RUN-SCOPED CONSTANTS pre-registered at mint prereg — R82's threshold, R85's "
        f"min_step, R92's evidence bar and R92's consec — and a dispatcher authors none "
        "(R1/R119). Changing one in place is R1's hand-varied config; it is re-minted with a "
        "recorded delta or not at all"
    )

    # Exactly ONE non-production config is ARMED — the preflight-rehearsal target, at its OWN
    # minted burst-scale guard values, NOT the production prereg constants; asserting the
    # distinction is what keeps those run-scoped. Every other one disarms DELIBERATELY, and `null`
    # is what makes that observable rather than forgotten.
    others = dict(postures)
    armed_smoke = others.pop("smoke_preflight_armed.yaml", None)
    assert armed_smoke is not None, (
        "configs/smoke_preflight_armed.yaml must ARM the draw-rate row — an armed rehearsal "
        "target that ships disarmed is refused at the preflight's own arming audit (rc 30) "
        "and its burst oracle goes red (R103)"
    )
    assert (armed_smoke.threshold, armed_smoke.min_step, armed_smoke.N_pool_min,
            armed_smoke.consec) != (
        RUN5_PREREG["threshold"], RUN5_PREREG["min_step"], RUN5_PREREG["N_pool_min"],
        RUN5_PREREG["consec"]), (
        "the armed smoke config must carry its OWN burst-scale guard values, never a copy of "
        "run5's pre-registered constants — those are run-scoped (R82/R85/R92)"
    )
    assert others and all(block is None for block in others.values()), (
        "every remaining non-production config disarms DELIBERATELY (R59), and `null` is what "
        f"makes that observable rather than forgotten: {others}"
    )
