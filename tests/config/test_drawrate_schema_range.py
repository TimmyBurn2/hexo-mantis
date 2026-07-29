""">300 justify (R8), stated at this file's MEASURED size of 331 lines. The file was already
over the cap with NO justify at all (a pre-existing gap WPMINT Phase K-A flagged); WPMINT Phase
K-B touches it to author `train.draw_rate_abort.consec`, so it is written now rather than left.
One block, one rejection corpus: the ~15 rejected payloads and the per-config posture census are
DATA whose reason text IS the assertion — each row names the defect the bound closes and the
spelling an operator would actually write. Splitting them would separate a rejection from the
prereg pin it is judged against, and R5 bars the cross-test import that would rejoin them.

⊕ WPAX Phase D ORACLE — `CARD-DRAWRATE-KEY`: what the draw-rate block can EXPRESS, and
what each committed config actually says (DESIGN_D §1.1, §6.2, §6.3; MF-1 closed by R83).

RED-at-import until IMPL lands the delta: `mantis.config.resolve.draw_rate` is the ONE read
path R80 orders, and it does not exist at HEAD.

The sibling file `tests/config/test_drawrate_arming_authority.py` asserts who has AUTHORITY
over the value. This one asserts the two things that file cannot see: which values the type
system will ACCEPT at all, and what the five committed configs each declare.

The oracles, and the defect each is the ONLY witness to:

- O-D6 `test_the_schema_cannot_express_a_value_OUTSIDE_the_metrics_own_range` — **MF-1's
  class, in both directions, on all three keys**. A threshold `> 1.0` passes `gt=0`, audits
  ARMED, and can NEVER fire; an `N_pool_min` above `DRAW_RATE_WINDOW * selfplay.n_workers`
  is permanently unsatisfiable; a `min_step` at or past `train.max_train_steps` is a guard
  the run never passes. All are "armed in the config, absent in effect" —
  `schema/core.py`'s own words for the sibling defect it already forbids. Not caught by the
  audit oracles, which only ever see values that already loaded.
- **WPMINT Phase DS (R92) re-points the third key's arms.** `min_samples` is DELETED with the
  filtered-mean statistic it guarded, and `N_pool_min` takes its place with the SAME defect
  class on BOTH ends — `test_the_evidence_bar_must_be_reachable_within_the_pools_own_window`
  (the ceiling, a cross-SECTION rule against `selfplay.n_workers`, which is why it is not an
  `le=` on the field) and
  `test_the_evidence_bar_cannot_be_so_small_that_one_drawn_game_fires` (the floor, DR-9's
  class transferred). The behaviour those bounds describe is
  `tests/selfplay/test_drawrate_pooled_statistic.py`.
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
from mantis.util.constants import DRAW_RATE_WINDOW

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

#: R82/R85's pre-registered run-scoped constants. NOT tunables: mint prereg is "the only
#: place they may change", so they are written here as the pin that makes an in-place edit
#: visible instead of silent.
#: WPMINT Phase K-B (call K-b) adds `consec`, the FOURTH term. R92's prereg row already
#: NAMED `consec=3` among the values that stand; this phase moves who SAYS it — from a
#: `StepCoordinatorConfig` code-side default into the block the other three live in — so
#: the number is unchanged and this pin now covers all four.
RUN5_PREREG = {"threshold": 0.25, "min_step": 25000, "N_pool_min": 50, "consec": 3}


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
    forecloses only the `<= 0` half. `pooled_draw_rate` is `Sum(draws)/Sum(completed)`, i.e.
    a fraction in `[0, 1]`, and the predicate is `all(value >= threshold)` — an UPPER bound
    (`rules.py`). So ANY threshold `> 1.0` can never be met, is accepted by `gt=0`, and reads
    ARMED to the manifest: "armed in the config, absent in effect", which is
    `schema/core.py`'s own words for the sibling defect it already forbids. It is reachable
    by the natural percent slip — an operator meaning 35% writes `35` — through the mint path.

    R83 closes it at both ends, and R71's class-fix law puts the same bound on the other axes
    of the same defect: `min_step >= train.max_train_steps` is a guard the run never passes,
    and `N_pool_min > DRAW_RATE_WINDOW * selfplay.n_workers` is evidence the run can never
    bank (the two tests below this one).

    THE MF-1 RESIDUAL IS NOW CLOSED — a measured side effect of R92, recorded here rather
    than left for a reader to discover. Every earlier revision of this docstring disclosed
    that `1e-300` STILL LOADS: a maximum-sensitivity hair-trigger that `le=1` does not
    address, and (WPMINT DR-2, measured) that `min_samples` did not address either, because
    `1/min_samples = 0.02` bounded ONE WORKER's rate while the compared value was an
    unweighted MEAN over N included workers whose floor was `1/(min_samples * N)` —
    0.000625 at N=32, 0.0003125 at N=64, understated by a factor of N.

    Under R92 the compared value IS `Sum(draws)/Sum(completed)`, so its smallest non-zero
    value at the bar is exactly `1/N_pool_min` at EVERY worker count — and
    `_one_drawn_game_cannot_fire_the_abort` requires `1/N_pool_min < threshold`. On a
    one-worker pool `N_pool_min <= 50`, so `1/N_pool_min >= 0.02`, so **any threshold at or
    below 0.02 is now REJECTED AT LOAD**: it is a threshold a single drawn game would meet,
    which is a hair-trigger and not a threshold. That is MF-1's residual closed at the type
    on the axis DR-2 proved the old claim false on. The arm below asserts the REJECTION, and
    the boundary is asserted on both sides so the bound is the arithmetic and not a literal.
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
        "N_pool_min 51 — unreachable evidence on a 1-worker pool (R92's fourth axis)":
            ({**armed, "N_pool_min": 51}, "N_pool_min"),
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
        # WPMINT Phase K-B (call K-b) FLIPS this row: `consec` is an authored key now, so the
        # `extra='forbid'` claim needs a key that is genuinely not in the block. It is
        # re-pointed rather than deleted — the property (strictness reaches INSIDE the nested
        # block, not just at the top level) is exactly as load-bearing as it was, and DS left
        # this row named as the pin K would have to flip consciously.
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


# ── DS-7 — the evidence bar's CEILING (R92's fourth axis) ─────────────────────────────
def test_the_evidence_bar_must_be_reachable_within_the_pools_own_window() -> None:
    """WPMINT Phase DS (R92) — the bound that REPLACES `min_samples: le=DRAW_RATE_WINDOW`.

    `min_samples` carried a load-bearing `le=` (`util/constants.py`'s own words) and R92
    deletes the key. Deleting it would have deleted the safety property with it, so the bound
    moves — and it CANNOT move onto `N_pool_min` as a field bound, because the ceiling is
    `DRAW_RATE_WINDOW * selfplay.n_workers` and `selfplay` is a different SECTION. It lives
    in `schema/core.py::_draw_rate_evidence_bar_is_reachable`, the twin of the actor-sync
    rule, on the ONE model that sees both sections.

    A bar above the ceiling is the FOURTH "armed in the config, absent in effect" axis and
    the one R92's own change creates: `Sum(completed)` never reaches it, the gate makes NO
    observation for the entire run, and gate 12 audits the row ARMED.

    Three arms, because two of them alone are satisfied by a re-spelled `le=50`:
    the boundary on both sides at `n_workers: 1`, and the SAME value ACCEPTED once the worker
    count is raised. The behaviour the bound describes is
    `tests/selfplay/test_drawrate_pooled_statistic.py`.
    """
    assert load_config(CONFIGS_DIR / "run5.yaml").selfplay.n_workers == 1, (
        "harness precondition: run5 is a ONE-worker pool, which is what makes 50 the ceiling "
        "here. If this ever changes the two boundary arms below move with it"
    )
    at_ceiling = _with_block({**RUN5_PREREG, "N_pool_min": DRAW_RATE_WINDOW})
    assert at_ceiling.train.draw_rate_abort.N_pool_min == DRAW_RATE_WINDOW, (
        "AT the ceiling the bar is satisfiable (the deque saturates exactly there), so it "
        "must load — a bound that also forbade the reachable value would disarm the abort"
    )

    with pytest.raises(ValidationError) as caught:
        _with_block({**RUN5_PREREG, "N_pool_min": DRAW_RATE_WINDOW + 1})
    assert "N_pool_min" in str(caught.value) and "n_workers" in str(caught.value), (
        "one game above the ceiling must be REJECTED, and the message must name BOTH keys: "
        "the operator cannot act on 'too big' without knowing what it is too big FOR; got "
        f"{caught.value}"
    )

    wider = load_config(CONFIGS_DIR / "run5.yaml").model_dump()
    wider["selfplay"]["n_workers"] = 2
    wider["train"]["draw_rate_abort"] = {**RUN5_PREREG, "N_pool_min": DRAW_RATE_WINDOW + 1}
    assert RunConfig.model_validate(wider).train.draw_rate_abort.N_pool_min == (
        DRAW_RATE_WINDOW + 1), (
        "the SAME value must be accepted on a two-worker pool: the bound is the PRODUCT "
        "`DRAW_RATE_WINDOW * selfplay.n_workers`, not a re-spelled `le=DRAW_RATE_WINDOW`. "
        "Without this arm the cross-section rule is indistinguishable from a field bound"
    )


# ── DS-8 — the evidence bar's FLOOR (DR-9's class, transferred by R92) ────────────────
def test_the_evidence_bar_cannot_be_so_small_that_one_drawn_game_fires() -> None:
    """ADJ-14's own defect, re-expressed on R92's statistic. WPMINT DR-9 found `min_samples:
    1` — "the exact ADJ-14 defect value" — accepted and reading ARMED; `min_samples` is gone,
    but the class transfers verbatim.

    The pooled rate's smallest non-zero value at the bar is `1/N_pool_min`, so at
    `N_pool_min = 4` with `threshold = 0.25` a SINGLE drawn game meets the threshold. That is
    the one-game saturation R80 ordered closed, one statistic later.
    `DrawRateAbortConfig._one_drawn_game_cannot_fire_the_abort` closes it from values already
    inside the block — no invented number.

    Boundary on both sides at run5's own threshold, so the rule is the arithmetic
    `1/N_pool_min < threshold` and not a literal floor.
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
            assert (resolved.threshold, resolved.min_step, resolved.N_pool_min,
                    resolved.consec) == (
                float(block.threshold), int(block.min_step), int(block.N_pool_min),
                int(block.consec)), (
                f"{path.name}: the resolver must carry the operator's terms through verbatim"
            )

    assert postures["run5.yaml"] is not None, (
        "configs/run5.yaml is the minted production config and the manifest's REQUIRED row "
        "audits it — a disarmed run5 is R59's whole subject"
    )
    run5 = postures["run5.yaml"]
    assert (run5.threshold, run5.min_step, run5.N_pool_min, run5.consec) == (
        RUN5_PREREG["threshold"], RUN5_PREREG["min_step"], RUN5_PREREG["N_pool_min"],
        RUN5_PREREG["consec"]), (
        f"run5's four values are RUN-SCOPED CONSTANTS pre-registered at mint prereg — R82's "
        f"threshold, R85's min_step, R92's evidence bar and R92's consec (authored at "
        f"WPMINT Phase K-B, value unchanged), "
        f"'the only place they may change'. Got {run5}, expected {RUN5_PREREG}. Changing one "
        "in place is R1's hand-varied config; it is re-minted with a recorded delta or not "
        "at all"
    )

    others = {name: block for name, block in postures.items() if name != "run5.yaml"}
    assert others and all(block is None for block in others.values()), (
        "the four non-production configs disarm DELIBERATELY (R59), and `null` is what makes "
        f"that observable rather than forgotten: {others}"
    )
