"""The sweep plan is a PRE-REGISTRATION, and these rows are what makes that word mean something.

R309(f) closes with "no post-hoc movement of any of it". A rule that lives in a tool's argument
defaults is a rule nobody can date; `tools/worker_sweep_plan.toml` is committed before the
sitting so `git log` dates it against the numbers. This suite pins the two properties that make
the file load-bearing rather than decorative:

  * NOTHING IS DEFAULTED. A missing key is a `ValueError` naming the key, never a silent
    fallback. That is hard rule 1's shape applied to a tool.
  * NOTHING IS IGNORED. An unknown key is a `ValueError`, never dropped. A silently-ignored key
    is how a rule someone believed was in force turns out never to have been read — and this
    file is the only place the rule exists.

Plus the one refusal that is a RULE and not a shape check: a plan may not put `n_workers = 1`
back. R309(f) REJECTS it, and a rule that could be un-made by editing the file it is written in
is not a rule.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mantis.diagnostics.worker_sweep import METRICS, PLAN_SHAPE, load_plan

_PLAN = Path(__file__).resolve().parents[2] / "tools" / "worker_sweep_plan.toml"


def _write(tmp_path: Path, raw: dict) -> Path:
    """Round-trip a plan mapping back to TOML by hand — no toml writer is a dependency here."""
    lines: list[str] = []
    for section, block in raw.items():
        lines.append(f"[{section}]")
        for key, value in block.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, list):
                lines.append(f"{key} = [{', '.join(str(v) for v in value)}]")
            elif value is None:
                lines.append(f"{key} = \"\"")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    path = tmp_path / "plan.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def base() -> dict:
    return tomllib.loads(_PLAN.read_text(encoding="utf-8"))


# ══ the shipped plan is the one the sitting runs ═════════════════════════════════════════
def test_the_committed_plan_loads_and_carries_the_operator_ladder() -> None:
    """R309(f)'s bracket, verbatim in substance: (1..14], and 1 is ABSENT because it is
    REJECTED. If this row ever has to change, the change is a pre-registration change."""
    plan = load_plan(_PLAN)
    assert plan.rungs == (2, 4, 8, 12, 14)
    assert plan.knee_pct == 95.0
    assert plan.metric in METRICS
    assert plan.provenance["prereg_ruling"] == "R309(f)"
    assert plan.measured_rounds >= plan.plateau_rounds


# ══ P4 — every key is required ═══════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("section", "key"),
    [(s, k) for s, keys in PLAN_SHAPE.items() for k in keys],
)
def test_deleting_any_key_is_a_named_valueerror(base: dict, tmp_path: Path,
                                                section: str, key: str) -> None:
    """Every key, not a sample of them. A per-key row is what makes "no defaults" a measured
    property rather than a claim about the keys somebody remembered to check."""
    del base[section][key]
    with pytest.raises(ValueError, match=key):
        load_plan(_write(tmp_path, base))


def test_deleting_the_ladder_key_is_a_valueerror(base: dict, tmp_path: Path) -> None:
    """P4 called out by name in the packet: the ladder is the rule, and its absence is not a
    tool that walks a default ladder."""
    del base["ladder"]["rungs"]
    with pytest.raises(ValueError, match="rungs"):
        load_plan(_write(tmp_path, base))


def test_deleting_a_whole_section_is_a_valueerror(base: dict, tmp_path: Path) -> None:
    del base["stopping_rule"]
    with pytest.raises(ValueError, match="stopping_rule"):
        load_plan(_write(tmp_path, base))


# ══ nothing is ignored ═══════════════════════════════════════════════════════════════════
def test_an_unknown_key_is_refused_not_ignored(base: dict, tmp_path: Path) -> None:
    base["selection"]["knee_pct_v2"] = 90.0
    with pytest.raises(ValueError, match="knee_pct_v2"):
        load_plan(_write(tmp_path, base))


def test_an_unknown_section_is_refused_not_ignored(base: dict, tmp_path: Path) -> None:
    base["overrides"] = {"n_workers": 1}
    with pytest.raises(ValueError, match="overrides"):
        load_plan(_write(tmp_path, base))


# ══ P5 — the rule cannot be un-made by editing the file it is written in ═════════════════
def test_a_plan_cannot_smuggle_n_workers_1_back_into_the_ladder(base: dict,
                                                                tmp_path: Path) -> None:
    base["ladder"]["rungs"] = [1, 2, 4]
    with pytest.raises(ValueError, match="R309"):
        load_plan(_write(tmp_path, base))


def test_a_non_increasing_ladder_is_refused(base: dict, tmp_path: Path) -> None:
    base["ladder"]["rungs"] = [4, 2, 8]
    with pytest.raises(ValueError, match="increasing"):
        load_plan(_write(tmp_path, base))


def test_a_plan_that_cannot_reach_a_verdict_is_refused_at_load(base: dict,
                                                               tmp_path: Path) -> None:
    """The refusal is at LOAD, on the developer's machine, and not five hours into a box
    sitting — which is the whole reason the check is here and not in the verdict path."""
    base["rounds"]["measured_rounds"] = 2
    base["stopping_rule"]["plateau_rounds"] = 3
    with pytest.raises(ValueError, match="plateau_rounds"):
        load_plan(_write(tmp_path, base))


def test_an_out_of_range_knee_pct_is_refused(base: dict, tmp_path: Path) -> None:
    base["selection"]["knee_pct"] = 0.0
    with pytest.raises(ValueError, match="knee_pct"):
        load_plan(_write(tmp_path, base))


def test_an_unknown_ranking_metric_is_refused(base: dict, tmp_path: Path) -> None:
    base["selection"]["metric"] = "sims_per_sec"
    with pytest.raises(ValueError, match="metric"):
        load_plan(_write(tmp_path, base))


def test_an_extension_ceiling_below_the_ladder_is_refused(base: dict, tmp_path: Path) -> None:
    base["ladder"]["extension_max"] = 8
    with pytest.raises(ValueError, match="extension_max"):
        load_plan(_write(tmp_path, base))


def test_a_non_positive_round_length_is_refused(base: dict, tmp_path: Path) -> None:
    base["rounds"]["round_sec"] = 0
    with pytest.raises(ValueError, match="round_sec"):
        load_plan(_write(tmp_path, base))


# ══ THE PINS BITE — the producer test the pins did not have ══════════════════════════════
# MEASURED: `RULED_RUNGS`, `MAX_BAND_PCT` and `MIN_PLATEAU_ROUNDS` could be DELETED WHOLE and
# 151/151 tests stayed green. The committed plan satisfies every pin, so no row ever asked one to
# reject anything — and a pin nobody has seen reject anything is indistinguishable from no pin.
# That is `0bb4381`'s standard, which this packet cites twice while its own most-argued closure
# (design-review D-3, the ruling-constant-left-editable class) had no producer at all.

@pytest.mark.parametrize(
    ("section", "key", "value", "needle"),
    [
        # R309(g)'s base ladder. `[2, 4, 8, 12, 16]` is a plausible-looking edit: every rung is
        # >= 2, strictly increasing, and inside the extension ceiling — every OTHER check passes.
        ("ladder", "rungs", [2, 4, 8, 12, 16], "R309"),
        ("ladder", "rungs", [2, 4, 8, 12], "R309"),
        ("ladder", "rungs", [2, 4, 8, 12, 14, 16], "R309"),
        # R309(f)'s knee percent, IN RANGE and not the ruling's — the range check cannot see it.
        ("selection", "knee_pct", 94.0, "R309"),
        ("selection", "knee_pct", 90.0, "R309"),
        ("selection", "knee_pct", 99.0, "R309"),
        # Amendment A1's ranking metric: in the closed token set, and not the pre-registered one.
        ("selection", "metric", "games_per_min", "PRE-REGISTERED"),
        # The band's upper bound. 5.0 is the value the plan file argues AGAINST at length
        # (774 MiB/round on this card against a falsifier that fired on 343 MiB); 500 turns the
        # memory conjunct off entirely.
        ("stopping_rule", "band_pct", 5.5, "band_pct"),
        ("stopping_rule", "band_pct", 500.0, "band_pct"),
        # A one-round window is not a convergence test, and 0 used to load clean and die inside
        # `classify` hours later with a DIFFERENT error.
        ("stopping_rule", "plateau_rounds", 1, "plateau_rounds"),
        ("stopping_rule", "plateau_rounds", 0, "plateau_rounds"),
        # Types are not coerced: `int(2.9)` and `float("95.0")` both succeed, so a plan could
        # STATE one thing and RUN another.
        ("ladder", "rungs", [2.0, 4.0, 8.0, 12.0, 14.0], "integer"),
        ("stopping_rule", "plateau_rounds", "3", "integer"),
        ("selection", "knee_pct", "95.0", "number"),
        ("rounds", "round_sec", "120.0", "number"),
    ],
)
def test_every_pin_REJECTS_a_plan_that_moves_it(base: dict, tmp_path: Path, section: str,
                                                key: str, value: object, needle: str) -> None:
    base[section][key] = value
    with pytest.raises(ValueError, match=needle):
        load_plan(_write(tmp_path, base))


def test_the_pins_are_read_from_source_and_not_from_the_plan_file() -> None:
    """The other half: the committed plan must AGREE with the source pins, so the report's echo
    of the rule is the rule. If they ever diverge, the file the operator reads and the rule the
    tool applies are two different things."""
    from mantis.diagnostics.worker_sweep import (
        PREREG_METRIC,
        RULED_KNEE_PCT,
        RULED_RUNGS,
    )

    raw = tomllib.loads(_PLAN.read_text(encoding="utf-8"))
    assert tuple(raw["ladder"]["rungs"]) == RULED_RUNGS
    assert raw["selection"]["knee_pct"] == RULED_KNEE_PCT
    assert raw["selection"]["metric"] == PREREG_METRIC
