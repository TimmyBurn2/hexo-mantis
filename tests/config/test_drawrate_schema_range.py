"""⊕ WPAX Phase D ORACLE — `CARD-DRAWRATE-KEY`: what the draw-rate block can EXPRESS, and
what each committed config actually says (DESIGN_D §1.1, §6.2, §6.3; MF-1 closed by R83).

RED-at-import until IMPL lands the delta: `mantis.config.resolve.draw_rate` is the ONE read
path R80 orders, and it does not exist at HEAD.

The sibling file `tests/config/test_drawrate_arming_authority.py` asserts who has AUTHORITY
over the value. This one asserts the two things that file cannot see: which values the type
system will ACCEPT at all, and what the five committed configs each declare.

The oracles, and the defect each is the ONLY witness to:

- O-D6 `test_the_schema_cannot_express_a_value_OUTSIDE_the_metrics_own_range` — **MF-1's
  class, in both directions, on all three keys**. A threshold `> 1.0` passes `gt=0`, audits
  ARMED, and can NEVER fire; a `min_samples` above the deque's own `maxlen` is permanently
  unsatisfiable; a `min_step` at or past `train.max_train_steps` is a guard the run never
  passes. All three are "armed in the config, absent in effect" — `schema/core.py:314-320`'s
  own words for the sibling defect it already forbids. Not caught by the audit oracles, which
  only ever see values that already loaded.
- O-D7 `test_every_config_states_its_draw_rate_posture_explicitly` — a config that INHERITS
  its posture instead of stating it, and a newly added config skipping the requirement.
  Enumerated through the ONE discovery authority (R71/R75), never a second glob.

R7 / gate 6: nothing here writes a `*.jsonl` and nothing is written inside the tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

# NOTE (ORACLE-WRITE): `ruff --fix` at HEAD re-sorts the `resolve.draw_rate` import into the
# third-party block, because the module it names does not exist yet. It sits here, with its
# `mantis.*` siblings, which is where it belongs the moment IMPL lands it.
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.draw_rate import resolve_draw_rate_abort  # RED anchor (R80)
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

#: R82/R85's pre-registered run-scoped constants. NOT tunables: mint prereg is "the only
#: place they may change", so they are written here as the pin that makes an in-place edit
#: visible instead of silent.
RUN5_PREREG = {"threshold": 0.25, "min_step": 25000, "min_samples": 50}


def _with_block(payload):
    """run5's own minted dump with `train.draw_rate_abort` REPLACED WHOLESALE, re-validated.

    Wholesale, not merged: the `smoke_run_config` factory deep-merges section dicts
    (`tests/conftest.py:60-66`), so a deliberately PARTIAL block would silently inherit the
    missing key from run5's own and the "the three are inseparable" arm would assert nothing.
    Everything else in the payload is the committed file's, so each row below varies exactly
    one thing.
    """
    dumped = load_config(CONFIGS_DIR / "run5.yaml").model_dump()
    dumped["train"]["draw_rate_abort"] = payload
    return RunConfig.model_validate(dumped)



# ── O-D6 — MF-1's class, in both directions, on all three keys ────────────────────────
def test_the_schema_cannot_express_a_value_OUTSIDE_the_metrics_own_range() -> None:
    """MF-1: loop 0 claimed "the schema cannot express a disarming number". FALSE — `gt=0`
    forecloses only the `<= 0` half. `recent_pool_draw_rate` is an unweighted mean of
    per-worker rates, i.e. a fraction in `[0, 1]` (`coordinator/config.py:141-146`), and the
    predicate is `all(value >= threshold)` — an UPPER bound (`rules.py:264`). So ANY
    threshold `> 1.0` can never be met, is accepted by `gt=0`, and reads ARMED to the
    manifest: "armed in the config, absent in effect", which is `schema/core.py:314-320`'s
    own words for the sibling defect it already forbids. It is reachable by the natural
    percent slip — an operator meaning 35% writes `35` — through the mint path.

    R83 closes it at both ends, and R71's class-fix law puts the same bound on the two other
    axes of the same defect: `min_samples > _DRAW_RATE_WINDOW` is permanently unsatisfiable
    (the 51-counterexample, O-D9), and `min_step >= train.max_train_steps` is a guard the run
    never passes.

    THE RESIDUAL, ASSERTED RATHER THAN GLOSSED: `1e-300` still loads. That is a
    maximum-sensitivity hair-trigger, not a disarm, and `le=1` does not address it. What
    addresses it is R80's `min_samples`: at 50 the smallest non-zero rate the estimator can
    report is `1/50 = 0.02`, so no threshold below `0.02` behaves differently from `0.02`
    (measured, O-D9). The arm is here so the residual is VISIBLE; if a later phase closes it
    at the type, this red is the correct signal to re-adjudicate the disclosure, not a bug.
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
    assert _with_block({**armed, "threshold": 1e-300}).train.draw_rate_abort.threshold == 1e-300, (
        "DISCLOSED RESIDUAL (MF-1 ancillary): the type does NOT close the hair-trigger end. "
        "`min_samples`'s 0.02 quantization and R82's mint prereg are what do"
    )

    rejected = {
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
        "min_samples 51 — permanently unsatisfiable: len(dq) is bounded by the deque's maxlen":
            ({**armed, "min_samples": 51}, "min_samples"),
        "min_samples 0 — no worker ever has fewer than zero games, so the guard is inert":
            ({**armed, "min_samples": 0}, "min_samples"),
        "min_step 0 — the ADJ-14 hair-trigger the R80 guards exist to close":
            ({**armed, "min_step": 0}, "min_step"),
        "min_step == train.max_train_steps — a guard the run never passes":
            ({**armed, "min_step": 1_000_000}, "min_step"),
        "a partial block — the three components are inseparable (R80)":
            ({"threshold": 0.25, "min_step": 25000}, "min_samples"),
        "an unknown inner key — extra='forbid' reaches inside the block too":
            ({**armed, "consec": 3}, "consec"),
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

    base = load_config(CONFIGS_DIR / "run5.yaml").model_dump()
    base["train"].pop("draw_rate_abort")
    with pytest.raises(ValidationError) as caught:
        RunConfig.model_validate(base)
    assert "draw_rate_abort" in str(caught.value), (
        "ABSENCE must be an error naming the key: `Field(default=...)` is the repo's own "
        "no-terminal-default idiom in this very class (`schema/train.py:41,43`). A key that "
        "may be omitted has a default somewhere, and that default is a second authority "
        "(R1/LAW-11)"
    )


# ── O-D7 — every config STATES its posture; none inherits one ─────────────────────────
def test_every_config_states_its_draw_rate_posture_explicitly() -> None:
    """R1: every config file is explicit and complete. A new required key means all five
    configs must carry it or fail gate 7 and the loader — and R59 is explicit that
    "deliberate disarming remains legal for smoke configs", which is what the `None`
    spelling makes OBSERVABLE rather than inferable from absence.

    Enumerated through `mantis.config.loader.discover_configs`, the ONE discovery authority
    both gate 7 and gate 12 consume (R71/R75). A second glob here would be exactly the
    divergence ADJ-13 F-1 was: a config the audit never sees because this file counted it
    differently.

    Both directions are asserted, because "every config carries the key" is satisfied by a
    tree where every config is disarmed, and "run5 is armed" is satisfied by a tree of one.
    """
    configs = discover_configs(CONFIGS_DIR)
    assert len(configs) >= 5, (
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
            assert (resolved.threshold, resolved.min_step, resolved.min_samples) == (
                float(block.threshold), int(block.min_step), int(block.min_samples)), (
                f"{path.name}: the resolver must carry the operator's terms through verbatim"
            )

    assert postures["run5.yaml"] is not None, (
        "configs/run5.yaml is the minted production config and the manifest's REQUIRED row "
        "audits it — a disarmed run5 is R59's whole subject"
    )
    run5 = postures["run5.yaml"]
    assert (run5.threshold, run5.min_step, run5.min_samples) == (
        RUN5_PREREG["threshold"], RUN5_PREREG["min_step"], RUN5_PREREG["min_samples"]), (
        f"run5's three values are RUN-SCOPED CONSTANTS pre-registered at mint prereg — R82's "
        f"threshold and R85's guards, "
        f"'the only place they may change'. Got {run5}, expected {RUN5_PREREG}. Changing one "
        "in place is R1's hand-varied config; it is re-minted with a recorded delta or not "
        "at all"
    )

    others = {name: block for name, block in postures.items() if name != "run5.yaml"}
    assert others and all(block is None for block in others.values()), (
        "the four non-production configs disarm DELIBERATELY (R59), and `null` is what makes "
        f"that observable rather than forgotten: {others}"
    )
