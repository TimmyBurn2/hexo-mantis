"""Who is allowed to say the draw-rate abort is armed, and on what terms.

The fact under single authority has three inseparable components (`threshold`, `min_step`,
`N_pool_min`) and therefore one nested block, `train.draw_rate_abort`, whose `None` is the
EXPLICIT off state. No boolean sits beside it, because a boolean could contradict it.

The defect each oracle is the ONLY witness to: a default authority surviving at ANY of three
layers — the dataclass field, the BUILDER SIGNATURE, or a `__post_init__` / `object.__setattr__`
resurrection on a frozen dataclass, extended to the family's other frozen dataclass where both
shapes survived the full tier; a `config_path` that does not resolve on a real `RunConfig`; and the
threading deleted / renamed / reordered with the source pin dying at the flip.

The SCHEMA half of the same block is `tests/config/test_drawrate_schema_range.py`: this file
asserts who has AUTHORITY over the value, that one what values are EXPRESSIBLE.

>300 justify (R8): three oracles over ONE manifest row, read through its three surfaces. The "not
caught by" column is only checkable while they sit together, and splitting would fork the row
lookup and the config load (R5 bars cross-test imports).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import tokenize
from pathlib import Path

import pytest

# `ruff --fix` re-sorts the `resolve.draw_rate` import into the third-party block while that
# module does not exist; it is placed here, with its `mantis.*` siblings, where it belongs.
from mantis.config.armed_aborts import (  # RED anchor #3 — ArmingSurfaceMissingError (F-4)
    MANIFEST,
    ArmedAbort,
    ArmingSurfaceMissingError,  # noqa: F401 — anchor: the delta is half-landed without it
    Status,
    audit_arming,
    exit_code_for_abort,
)
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import (  # RED anchor #1 — the ONE read path (R80)
    DrawRateAbortSpec,
    resolve_draw_rate_abort,  # noqa: F401 — anchor; its oracles live in the sibling files
)
from mantis.monitor.heartbeat import DRAW_RATE_COLLAPSE_EXIT_CODE
from mantis.run import _step_coordinator_config  # RED anchor #2 — MF-2 Attack B's surface
from mantis.train.coordinator.config import StepCoordinatorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
ROW_NAME = "draw_rate_collapse"

#: The builder's third config-authored parameter, from a MINTED block. This file is about
#: THRESHOLD authority, so the drain caps arrive derived rather than as four more literals.
_MINTED_DRAIN_CAPS = resolve_drain_caps(load_config(CONFIGS_DIR / "dev_example.yaml").monitor)

#: The builder's FOURTH config-authored parameter, from the same minted config and for the same
#: reason: the coordinator knobs arrive derived rather than as eighteen more literals.
_MINTED_KNOBS = resolve_coordinator_knobs(load_config(CONFIGS_DIR / "dev_example.yaml").train)
#: The builder's FIFTH config-authored parameter — `monitor.gate_interval`, the ARMING cadence.
_MINTED_GATE_INTERVAL = load_config(
    CONFIGS_DIR / "dev_example.yaml").monitor.gate_interval

#: Pre-registered run-scoped constants. NOT tunables: mint prereg is the only place they may
#: change, so they are written here as the pin that makes an in-place edit visible.
RUN5_PREREG = {"threshold": 0.25, "min_step": 25000, "N_pool_min": 50, "consec": 3}


def _load_tool():
    """Load the tool by absolute path — `tools/` is not a package and R5/LAW-17 bar a `sys.path`
    write."""
    spec = importlib.util.spec_from_file_location("preflight_mint_for_wpax_d", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _row(name: str = ROW_NAME) -> ArmedAbort:
    matches = [row for row in MANIFEST if row.name == name]
    assert len(matches) == 1, f"the manifest must carry exactly one {name!r} row; got {matches}"
    return matches[0]


def _walk(obj, path: str):
    """Walk a dotted path with plain `getattr`. Deliberately NOT the module's own `_dotted`: an
    oracle that navigates with the code under test cannot witness a navigation bug."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _complete_kwargs(spec) -> dict:
    """Every `StepCoordinatorConfig` field, read off an object the SHIPPED builder produced.
    Derived, never hand-written: a literal census would agree with the dataclass by maintenance
    rather than by construction, and the `TypeError` arm could go vacuous silently."""
    built = _step_coordinator_config(stop_step=11, draw_rate_abort=spec,
                                     drain_caps=_MINTED_DRAIN_CAPS,
                                     gate_interval=_MINTED_GATE_INTERVAL,
                                     knobs=_MINTED_KNOBS)
    return {field.name: getattr(built, field.name)
            for field in dataclasses.fields(StepCoordinatorConfig)}


def _source_without_comments_or_strings(path: Path) -> str:
    """`path`'s source with every COMMENT / STRING token blanked to spaces, geometry kept.

    Joining tokens with newlines destroys contiguity, so a multi-token pin could never be a
    substring. Blanking in place keeps the file's shape, so a pin genuinely IN THE CODE is still a
    substring while one retained only in a comment or docstring is not.
    """
    lines = path.read_text().splitlines(keepends=True)
    blank = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type not in blank:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                line = lines[row - 1]
                start = scol if row == srow else 0
                end = ecol if row == erow else len(line)
                masked = "".join(" " if ch != "\n" else "\n" for ch in line[start:end])
                lines[row - 1] = line[:start] + masked + line[end:]
    return "".join(lines)


def test_the_coordinator_threshold_has_NO_default_authority_ANYWHERE_so_the_config_is_its_only_one() -> None:  # noqa: E501 — DESIGN_D §7's name verbatim; the named RED is operator-binding
    """The literal must DIE, not MIGRATE: the config says `0.25` while the runtime uses something
    else, so the audit reads the config, goes green, and the run is disarmed.

    Three layers, each with its own defeat: the FIELD (`dataclasses.fields()` says MISSING,
    defeated by a parameter default); the BUILDER SIGNATURE, where the authority simply moves while
    every other assertion stays green; and NO RESURRECTION, since `object.__setattr__` inside
    `__post_init__` is legal on a frozen dataclass. The transport arm is last: a builder that
    accepts the parameter and then ignores it satisfies every signature and field assertion above
    while the config reaches nothing.
    """
    fields = {field.name: field for field in dataclasses.fields(StepCoordinatorConfig)}

    assert "draw_rate_threshold" not in fields and "draw_rate_min_step" not in fields, (
        "R65's dead literals must be DELETED, not overwritten: `draw_rate_threshold: float "
        "= 0.0` and `draw_rate_min_step: int = 0` are a second default authority over "
        "train.draw_rate_abort even when every caller replaces them (R1). Still present: "
        f"{sorted(set(fields) & {'draw_rate_threshold', 'draw_rate_min_step'})}"
    )
    # The field must be GONE and the term asserted at its new home. A coordinator field carrying
    # the term as well would be a SECOND authority beside `train.draw_rate_abort.consec`, exactly
    # the shape the dead `draw_rate_threshold: float = 0.0` had.
    assert "draw_rate_consec" not in fields, (
        "`draw_rate_consec` must be DELETED from StepCoordinatorConfig, not left beside the "
        "authored key: with `train.draw_rate_abort.consec` live, a same-fact field here is "
        "the duplicated-default class R1 kills, and a disarmed run would carry a term for an "
        "abort nobody armed (R80: the terms travel together)"
    )

    field = fields["draw_rate_abort"]
    assert (field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING), (
        "StepCoordinatorConfig.draw_rate_abort carries a code-side default "
        f"({field.default!r} / {field.default_factory!r}) — the config is then not its only "
        "authority and a caller that omits it silently gets an inherited posture (R1/R49)"
    )

    params = inspect.signature(_step_coordinator_config).parameters
    for name in ("stop_step", "draw_rate_abort"):
        assert name in params, (
            f"the builder must take {name!r} as a parameter: the CONFIG-authored values "
            "arrive from compose_run's resolvers, never from a literal inside the builder"
        )
        assert params[name].default is inspect.Parameter.empty, (
            f"{name} carries a parameter default ({params[name].default!r}) — R65's literal "
            "did not die, it MIGRATED from the dataclass field to the builder signature "
            "(MF-2 Attack B). `preflight_mint.py:990` is the live pressure for exactly this"
        )
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only, so a positional call site cannot silently supply "
            "the wrong config fact"
        )

    spec = DrawRateAbortSpec(threshold=0.5, min_step=3, N_pool_min=7, consec=2)
    complete = _complete_kwargs(spec)
    without = {key: value for key, value in complete.items() if key != "draw_rate_abort"}
    with pytest.raises(TypeError):
        StepCoordinatorConfig(**without)

    assert StepCoordinatorConfig(**{**complete, "draw_rate_abort": None}).draw_rate_abort is None, (
        "an EXPLICITLY disarmed value must survive construction verbatim — a __post_init__ / "
        "object.__setattr__ resurrection is a code-side default on a frozen dataclass, and "
        "it passes every `dataclasses.fields()` assertion above (MF-2 Attack A)"
    )
    assert StepCoordinatorConfig(**complete).draw_rate_abort is spec, (
        "an ARMED spec must survive construction by IDENTITY: a __post_init__ that rebuilds "
        "or normalises it is a second authority over the operator's own terms"
    )

    built = _step_coordinator_config(stop_step=11, draw_rate_abort=spec,
                                     drain_caps=_MINTED_DRAIN_CAPS,
                                     gate_interval=_MINTED_GATE_INTERVAL,
                                     knobs=_MINTED_KNOBS)
    assert built.draw_rate_abort is spec and built.stop_step == 11, (
        "the builder must hand ON both config-authored values. A builder that takes them as "
        "required parameters and then ignores them satisfies every signature and field "
        "assertion above while the config reaches nothing — this is the transport arm"
    )

    # ── the family's OTHER frozen dataclass ──────────────────────────────────────────
    # `DrawRateAbortSpec` is where the three VALUES live, so a default authority resurrected there
    # defeats every assertion above: the coordinator would faithfully carry a spec whose terms the
    # config never wrote. MEASURED: field defaults on all three keys, and a `__post_init__` +
    # `object.__setattr__` normalisation, BOTH left the full tier green.
    spec_fields = {field.name: field for field in dataclasses.fields(DrawRateAbortSpec)}
    assert set(spec_fields) == {"threshold", "min_step", "N_pool_min", "consec"}, (
        "the resolved spec must carry the block's keys and nothing else — a further field "
        "here is a term the schema block never authored. `consec` joined the set at WPMINT "
        "Phase K-B (call K-b): R80 assigned it to CARD-COORD-KNOBS and this is that card, so "
        f"it is authored INSIDE the block the other three live in; got {sorted(spec_fields)}"
    )
    for name, field in spec_fields.items():
        assert (field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING), (
            f"DrawRateAbortSpec.{name} carries a code-side default ({field.default!r} / "
            f"{field.default_factory!r}). The resolver would then build a spec the config "
            "did not fully author, and every arm above stays green while it happens (R1/R83)"
        )
    assert not hasattr(DrawRateAbortSpec, "__post_init__"), (
        "`DrawRateAbortSpec` is `frozen=True`, and `object.__setattr__` inside a "
        "`__post_init__` is legal on a frozen dataclass — a code-side default can be "
        "restored AFTER construction with `dataclasses.fields()` still reporting MISSING. "
        "This is R83's Attack A on the sibling class (MF-2's lesson one seam over)"
    )
    probe = DrawRateAbortSpec(threshold=0.5, min_step=3, N_pool_min=7, consec=2)
    assert dataclasses.asdict(probe) == {"threshold": 0.5, "min_step": 3, "N_pool_min": 7,
                                         "consec": 2}, (
        "the resolved terms must survive construction VERBATIM. The probe values are "
        "deliberately off-prereg (N_pool_min 7 is under DESIGN_DS's 50, min_step 3 is under "
        "R82's 25000, consec 2 is under R92's 3) so a normaliser that clamps toward the "
        "pre-registered numbers is "
        f"visible here rather than silently agreeing with run5; got {dataclasses.asdict(probe)}"
    )


def test_the_required_row_is_audited_against_a_REAL_RunConfig(smoke_run_config) -> None:
    """The row is audited against a REAL `RunConfig`, in both directions.

    Flipping the row to REQUIRED once raised an `AttributeError` that `main` collapses to rc 1 —
    the one outcome the tool's own docstring says cannot exist — and the ancestor could not catch
    it, having built its stub FROM the row's own `config_path`. Both directions, because a gate
    that only ever says PASS is as useless as one that only ever says FAIL.
    """
    row = _row()
    assert row.status is Status.REQUIRED, (
        "R65's flip is this phase's whole point: while the row is DEFERRED nothing audits "
        "the draw-rate abort on a production config"
    )
    assert row.owner is None, "__post_init__ forbids an owner on a REQUIRED row (§8.2)"
    assert row.config_path == "train.draw_rate_abort.threshold", (
        f"the row must name the block's own key, not a dataclass's ({row.config_path!r}). "
        "`train.step_coordinator.*` was RULED AGAINST (§2): it is named after a dataclass "
        "and invites the ~24 coordinator knobs R78 forecloses"
    )
    # Pinned to the CONSTANT, not to the literal 46: a row carrying its own number would be the
    # second authority the rule-name carrier exists to prevent, and an equality against `46` here
    # would pass just as happily against a hand-typed one.
    assert row.exit_code == DRAW_RATE_COLLAPSE_EXIT_CODE, (
        f"the row must carry the family's own authored constant, not a literal; got "
        f"{row.exit_code!r} against {DRAW_RATE_COLLAPSE_EXIT_CODE!r}"
    )
    assert exit_code_for_abort(row.name) == row.exit_code, (
        "and the resolver must ANSWER FROM THE ROW — it is what a process boundary calls, so "
        "if it can disagree with the manifest the manifest has stopped being the authority"
    )

    cfg = load_config(CONFIGS_DIR / "run6.yaml")
    audit = audit_arming(cfg)
    assert list(audit.disarmed) == [], (
        "configs/run6.yaml arms every REQUIRED row (actor-lag since R59, draw-rate at R82's "
        f"0.25); got {[r.name for r in audit.disarmed]}"
    )
    assert ROW_NAME in [r.name for r in audit.required], (
        "the flipped row must be in the audit's REQUIRED list — that list is what the "
        "evidence report publishes as `required_armed`"
    )
    # Stated positively: THIS row must not be among the deferred, because a draw-rate row quietly
    # demoted back to DEFERRED would stop gating the mint while every other assertion stayed green.
    # Other rows MAY be deferred — a live gate whose threshold nobody has pre-registered is.
    assert ROW_NAME not in [r.name for r in audit.deferred], (
        "the draw-rate row must never appear in the DEFERRED list: a deferred row prints and "
        "does not gate, so demoting this one un-does R65's whole flip silently. Other rows "
        "MAY be deferred (grad-norm is, by call K-c, because its threshold is un-pre-registered "
        f"— R84's class); got {[r.name for r in audit.deferred]}"
    )

    value = _walk(cfg, row.config_path)
    assert value == RUN5_PREREG["threshold"], (
        f"the row's dotted path must resolve on a real RunConfig to run5's minted value; "
        f"got {value!r}"
    )
    assert row.mechanism.is_armed(value) is True, (
        "R79(2): the row asserts a condition over the RESOLVED VALUE, never the existence "
        "of a schema field"
    )

    disarmed_cfg = smoke_run_config("run6.yaml", train={"draw_rate_abort": None})
    assert [r.name for r in audit_arming(disarmed_cfg).disarmed] == [ROW_NAME], (
        "run5 with the block explicitly disarmed must name THIS row and only this row — "
        "`Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(None)` is False through its "
        "non-numeric arm, with zero change to `Mechanism` (§1.1 reason 4)"
    )


def test_the_required_row_keeps_a_source_pin_bound_to_the_construction_site(tmp_path) -> None:
    """The newly-REQUIRED row keeps a source pin bound to the live construction site.

    The pin used to point at a literal and fire on its DELETION, so at the flip its subject stops
    existing; dropping it would leave the row with no tamper-evidence precisely as it starts gating
    a production mint, and would silently gut two live tests that open with "no pinned row means
    this test has no subject". What it proves is deletion / rename / reorder tamper-evidence over
    the SOURCE TEXT; the code-text arm closes the comment-retention defeat for this one pin.
    """
    # The load-bearing fact is that THIS row keeps ITS pin — not that it is the only pinned row,
    # since a DEFERRED row MUST carry one too. Keeping it is what stops the two tests that open
    # with "no pinned row means this test has no subject" going vacuous.
    pinned = [row for row in MANIFEST if row.source_pin is not None]
    assert ROW_NAME in [row.name for row in pinned], (
        "the flipped row must KEEP its pin: "
        "`__post_init__` (`armed_aborts.py`) constrains `owner` on a REQUIRED row and "
        "leaves `source_pin` unconstrained — the class docstring claiming otherwise "
        "was FALSE and R73 obliged its correction (N-1). Dropping the pin "
        "would silently empty `test_the_source_pin_scan_runs_inside_the_live_audit_path` and "
        f"`test_the_report_publishes_the_pins_the_scan_ACTUALLY_covered`; got {pinned!r}"
    )
    row = [candidate for candidate in pinned if candidate.name == ROW_NAME][0]
    rel, text = row.source_pin
    assert rel == "src/mantis/run.py", (
        "R79(3): the pin binds to the RESOLVED RUNTIME VALUE AT THE CONSTRUCTION SITE. "
        f"{rel!r} is not that site — `compose_run` is"
    )
    assert "resolve_draw_rate_abort" in text and "draw_rate_abort=" in text, (
        "the pinned text must be the THREADING itself, so that deleting the resolver call, "
        f"renaming it, or reordering the call past it all break the scan; got {text!r}"
    )

    pinned_file = REPO_ROOT / rel
    assert text in pinned_file.read_text(), f"the pin {text!r} must be present in {rel}"
    assert text in _source_without_comments_or_strings(pinned_file), (
        "the pinned text must live in CODE, not in a comment or a docstring. "
        "`verify_source_pins` is a whole-file substring scan, so a pin retained as a comment "
        "beside a changed call site passes gate 12 at rc 0 (REDTEAM_P.md:505-520). That "
        "defeat is inherited by every text pin in the repo; this arm closes it for this one"
    )
    assert list(TOOL.verify_source_pins(MANIFEST, repo_root=REPO_ROOT)) == [], (
        "every pinned text must still be present in the real tree — a pin that has already "
        "rotted at oracle-write time is a manifest bug, not a Phase D signal"
    )

    # The two tamper drives run against a manifest holding ONLY this row, so they still report an
    # exact list: with the whole manifest another row's pin would also break in a tree that holds
    # one file, and "exactly this row" would be unassertable. Restricting the data is the same
    # instrument, not a weaker one.
    only_this_row = (row,)
    tampered = tmp_path / "tampered"
    (tampered / rel).parent.mkdir(parents=True)
    (tampered / rel).write_text(pinned_file.read_text().replace(text, "# threading deleted\n"))
    assert [broken.name for broken in TOOL.verify_source_pins(
        only_this_row, repo_root=tampered)] == [row.name], (
        f"deleting the threading from {rel} must report exactly {row.name!r} broken — this "
        "is the tamper-evidence the flip would otherwise have lost"
    )

    absent = tmp_path / "absent"
    absent.mkdir()
    assert [broken.name for broken in TOOL.verify_source_pins(
        only_this_row, repo_root=absent)] == [row.name], (
        "a pinned file that does not exist must be reported broken, never skipped — "
        "'nothing to scan' is how a tamper-evidence gate goes silently vacuous (R56's "
        "asymmetry, `silent_encoding_gate.py:338-344`)"
    )

    ArmedAbort(name="probe", config_path=row.config_path, mechanism=row.mechanism,
               status=Status.REQUIRED, exit_code=None, owner=None,
               source_pin=row.source_pin, note="N-1: a pin on a REQUIRED row is LEGAL")
