# R8 >300 justify (343, re-measured at WPMINT DS-VERIFY; was 324 at WPAX Phase D, and the
# figure is restated at the file's MEASURED size rather than the size it was written for —
# `preflight_mint.py`'s header sets that precedent, and SF-7's rule is that a justification
# which is not true is worse than none): the manifest ROWS are data and their reason text IS the row —
# `note` is a live consumer's field, printed by gate 12 on every run, not a comment that
# can be trimmed. The two walkers below (`_dotted`, `audit_arming`) are ~35 lines of code
# carrying the F-4 named-arm and disarmed-short-circuit rationale; splitting them from the
# rows they walk would put "which aborts must arm" and the predicate that reads it on
# opposite sides of an import, which is the drift this module exists to prevent.
"""The armed-abort manifest — WHICH aborts a production config MUST arm (R61, DESIGN_P §8).

ONE authority, and it is DATA. A markdown register would need a parser, and the parser's
grammar becomes a second authority with its own failure modes (a row that parses to nothing
reads as "no such requirement"). A typed frozen dataclass is read by `import`, carries its
invariant in `__post_init__`, and cannot drift from a doc twin — so this module ships no doc
twin of the rows. The precedent is `config/resolve/composition.py:10-13`: "the rule is a
config-layer fact, so it lives in the config layer", and "which aborts a production config
must arm" is a config-layer fact of exactly that species.

SF-4 — THE LAYER BOUNDARY. This module makes ZERO filesystem calls. `PRODUCTION_CONFIGS`
holds repo-relative STRINGS (data); resolving them against a repo root, and reading a
`source_pin`'s pinned file, both live in `tools/ci_gates/preflight_mint.py`, where
`REPO_ROOT = Path(__file__).resolve().parents[2]` is structurally sound. A shipped package
that resolved `parents[3]` to the repo root would be depending on an editable install.
Pinned by `tests/config/test_armed_abort_manifest.py`,
`test_the_manifest_module_makes_no_filesystem_call`.

`wr_hard_abort_enabled` is ABSENT BY DECISION, not by oversight: it is the sealbot win-rate
abort, which ships WARN-ONLY by operator ruling G-3, and STATE §6 names the mint-blocking
pair as "draw-rate + actor-lag". A later reader must not "fix" it in.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Status(str, Enum):
    """REQUIRED rows are audited and gate; DEFERRED rows are printed loudly and do not."""

    REQUIRED = "required"
    DEFERRED = "deferred"


class Mechanism(str, Enum):
    """The predicate that decides "armed" for a row's value. DATA, not a branch on `name`.

    `audit_arming` never branches on a row's identity: `status` selects the list and
    `mechanism` selects the predicate, which is what makes Phase D's DEFERRED→REQUIRED flip
    a one-field data edit (§8.5, proven by O-7 rather than asserted).
    """

    CONFIG_BOOL = "config_bool"
    CONFIG_THRESHOLD_GT_ZERO = "config_threshold_gt_zero"

    def is_armed(self, value: Any) -> bool:
        """True iff `value` arms the abort. A real predicate in BOTH directions — a
        constant here would silently arm or disarm every row at once."""
        if self is Mechanism.CONFIG_BOOL:
            return value is True
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return float(value) > 0.0


@dataclass(frozen=True)
class ArmedAbort:
    """One row: an abort, the config surface that arms it, and its ownership posture.

    `owner` and `source_pin` are REQUIRED on a DEFERRED row; `owner` is FORBIDDEN on a
    REQUIRED one and `source_pin` is UNCONSTRAINED there. Each of the three rules
    `__post_init__` enforces is a way for a row to go invisible: an owner-less deferred row
    has nobody to chase, a pin-less one is not tamper-evident, and a required row carrying
    an owner reads as already-excused.

    N-1 (WPAX Phase D, R73): this sentence used to say `source_pin` was FORBIDDEN on a
    REQUIRED row. That was FALSE — `__post_init__` never constrained it — and acting on it
    would have dropped the draw-rate pin at Phase D's flip, leaving the newly-REQUIRED row
    with no tamper-evidence exactly as it started gating a production mint, and silently
    emptying the two pin-scan tests that stand on "no pinned row means this test has no
    subject". A REQUIRED row MAY keep a pin, and this one does.
    """

    name: str
    config_path: str
    mechanism: Mechanism
    status: Status
    exit_code: int | None
    owner: str | None
    source_pin: tuple[str, str] | None
    note: str

    def __post_init__(self) -> None:
        if self.status is Status.DEFERRED and not self.owner:
            raise ValueError(
                f"armed-abort row {self.name!r} is DEFERRED and carries no `owner`: "
                "deferred debt with no owner is debt nobody is chasing (R56)"
            )
        if self.status is Status.DEFERRED and not self.source_pin:
            raise ValueError(
                f"armed-abort row {self.name!r} is DEFERRED and carries no `source_pin`: "
                "a deferred row that is not tamper-evident rots into the status quo (§8.4)"
            )
        if self.status is Status.REQUIRED and self.owner:
            raise ValueError(
                f"armed-abort row {self.name!r} is REQUIRED and carries an `owner`: an "
                "owner on a required row reads as already-excused; drop the owner or "
                "declare the row DEFERRED"
            )


@dataclass(frozen=True)
class AuditResult:
    """What `audit_arming` publishes. `disarmed` is the only field that gates."""

    required: tuple[ArmedAbort, ...]
    deferred: tuple[ArmedAbort, ...]
    disarmed: tuple[ArmedAbort, ...]


#: The rows. R61 fixes the set, ADJ-08's census supplies the values, R65 fixes the statuses.
MANIFEST: tuple[ArmedAbort, ...] = (
    ArmedAbort(
        name="actor_lag",
        config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL,
        status=Status.REQUIRED,
        exit_code=45,
        owner=None,
        source_pin=None,
        note=(
            "The frozen-actor hard abort (exit 45). Armed on configs/run5.yaml since the "
            "R59 flip; disarming it on a production config is the run3 failure mode "
            "re-enabled."
        ),
    ),
    ArmedAbort(
        name="draw_rate_collapse",
        config_path="train.draw_rate_abort.threshold",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
        status=Status.REQUIRED,
        exit_code=None,
        owner=None,
        source_pin=(
            "src/mantis/run.py",
            "draw_rate_abort=resolve_draw_rate_abort(config.train)",
        ),
        note=(
            "The self-play draw-rate collapse hard abort. Armed on configs/run5.yaml at "
            "threshold 0.25 (R82) with min_step 25000 and N_pool_min 50 (R92's guards; the "
            "per-worker min_samples bar was DELETED with the filtered-mean statistic it "
            "guarded), all three pre-registered at mint prereg. The gated statistic is the "
            "pooled count-weighted rate Sum(draws)/Sum(completed) over the union of worker "
            "windows; below N_pool_min completed games the gate makes NO OBSERVATION "
            "(skip-counted), never a healthy 0.0. NOTE for the mint record (WPMINT DS-VERIFY, "
            "correcting the WITHDRAWN DR-8): consec=3 counts consecutive CHECKS at a stride of "
            "log_interval train steps, so at the shipped log_interval 1000 the three samples "
            "span 2000 steps. The earliest possible fire is nonetheless step 25000, because "
            "SAMPLING IS NOT GATED BY min_step — the history accumulates from step 1000, holds "
            "25 samples by step 25000, and min_step gates only the FIRE. DR-8's contrary claim "
            "(earliest fire 27000) was MEASURED FALSE and withdrawn. The pin binds to the THREADING at the "
            "construction site, so deleting it, renaming the resolver or reordering the "
            "call past it all break the R56 scan. exit_code is None because the gate stops "
            "the run cooperatively (shutdown.running = False) and NO distinct process exit "
            "code exists — a fired abort is indistinguishable by exit status from a clean "
            "run. That is TRUTHFUL, not an omission: R84 ratified it and opened "
            "CARD-ABORT-EXIT (pre-run5-mint, BLOCKING) as the one authorized flip."
        ),
    ),
)

#: WHICH configs the law binds — one authority. Repo-relative strings only; resolving them
#: is the tool's (SF-4).
#:
#: ADJ-13 N-1: `configs/run5.yaml`'s membership here is not decoration and is not free to
#: move. Nothing pinned it, so moving run5 to `EXEMPT_CONFIGS` with a written reason and
#: promoting an armed smoke config in its place kept gate 12 at **rc 0 with run5 disarmed** —
#: the partition stayed a partition and every exemption still carried a reason, so every
#: existing check was satisfied by the swap. The producer is
#: `test_run5_is_bound_BY_NAME_and_is_not_freely_exemptable`
#: — the run the operator is about to mint is audited BY NAME, and
#: exempting it is a red gate rather than a bookkeeping edit. (Recheck R-8: this citation
#: named a test that does not exist — a LAW-07 producer citation that greps to nothing is
#: LAW-07's own failure mode. Every citation in this pass was `grep`-verified.)
PRODUCTION_CONFIGS: tuple[str, ...] = ("configs/run5.yaml",)

#: The OTHER half of the same authority (MF-7). R59's "deliberate disarming remains legal for
#: smoke configs" used to be expressed by ABSENCE from `PRODUCTION_CONFIGS` — which made
#: "deliberately exempt" and "nobody remembered to list it" the SAME observable, and a
#: disarmed `configs/run6.yaml` dropped into the tree audited GREEN (measured: rc 0).
#:
#: So the exemption is now WRITTEN, and the two tuples must PARTITION the config set EXACTLY.
#: The tool hard-fails (rc 31) on either kind of drift: a config present on disk and
#: named by neither tuple, and a tuple naming a config that is not on disk. That is gate 11's
#: `KNOWN_DEBT` shape (`silent_encoding_gate.py:126,338-344`) applied to the config set —
#: registered debt whose staleness is itself a failure.
#:
#: **What "the config set" means, and why the earlier wording was FALSE** (ADJ-13 F-1). This
#: comment used to say `configs/*.yaml`, and the tool implemented exactly that — a flat
#: `*.yaml` glob — while gate 7 validated `**/*.yaml` + `**/*.yml`. So `configs/run6.yml` and
#: `configs/prod/run6.yaml` passed gate 7 and were never audited by gate 12: the claim below
#: was measured FALSE for two of the three ways to add a config, because MF-7's fix had been
#: fitted to the reviewer's `run6.yaml` rather than to the class. Discovery is now
#: `mantis.config.loader.discover_configs` — ONE authority, consumed by both gates (R71) —
#: and the declaration accepts subdirectory-relative paths, which it previously reported STALE
#: while the file sat on disk.
#:
#: **That was still not enough, and the recheck measured why** (R-2). Widening discovery by one
#: more extension leaves the boundary one extension further out: `configs/run6.txt` and
#: `configs/run6.YAML` were schema-valid, DISARMED on the required row, mintable, launchable —
#: and rc 0 from both gates. The asymmetry was never between two globs; it was that DISCOVERY
#: answered "is this a config" by extension while the LOADER answered it by CONTENT, so the
#: complement of every enumeration stayed launchable and invisible.
#:
#: **R75 rules which side closes it.** Not the loader — narrowing its accept-set was DECLINED,
#: and a run may be launched from a path of any shape. The protection is the **shared-authority
#: invariant**: whatever the loader accepts, the audit must see. `discover_configs` is therefore
#: name-agnostic — every path under `configs/` except a real directory, which `read_text`
#: refuses by type. With that, and only with that:
#:
#: Adding ANY file to `configs/` — at any name, at any depth — now FORCES a one-line declaration
#: here or in `PRODUCTION_CONFIGS`; it can no longer be forgotten into exemption, and there is
#: no longer a class of file the gates are silent about. The cost is deliberate: `configs/` may
#: hold only complete configs, so a stray note or an editor backup is a red gate rather than a
#: quiet resident of the audit root. The one limit still standing, stated rather than implied:
#: this binds `configs/`, and a loadable config OUTSIDE that directory is reachable by
#: `python -m mantis.run` without being discovered (CARD-CONFIG-DISCOVERY-ROOT) — `--config`
#: does audit it, shape-agnostically, which is what covers the mint path.
#:
#: `(repo-relative path, why it is exempt)`. The reason is data, printed by the tool on the
#: failure path, so an exemption cannot be a bare path nobody can justify later.
EXEMPT_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "configs/dev_example.yaml",
        "developer template, never minted for a run; disarmed at `:200` by design (R59). "
        "ADJ-13 N-3: this file is ALSO the mutation corpus's M1 row (the one real committed "
        "config that demonstrates the audit going red), and exempting it moved gate 12's "
        "red-capability on the real `configs/` tree onto `--config` — which had no producer "
        "in that direction until F-5's. Both halves are now pinned: the declared-production "
        "half by `test_naming_a_config_ADDS_scrutiny_and_never_replaces_the_production_set`'s "
        "bare drive, the `--config` half by "
        "`test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored`.",
    ),
    (
        "configs/smoke_gnn.yaml",
        "smoke config — bounded local drive, not a production run (R59).",
    ),
    (
        "configs/smoke_radius_curriculum.yaml",
        "smoke config — bounded local drive, not a production run (R59).",
    ),
    (
        "configs/sustained_kcluster.yaml",
        "not currently a production run config. WPAX Phase P wrote this row from the tree's "
        "own state (it is absent from PRODUCTION_CONFIGS at HEAD), NOT from an operator "
        "ruling — see CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM. If it is minted, its row moves "
        "to PRODUCTION_CONFIGS.",
    ),
)


class ArmingSurfaceMissingError(AttributeError):
    """A row's `config_path` does not resolve on a real `RunConfig` (RED-TEAM_P F-4).

    Subclasses `AttributeError` deliberately: `_dotted`'s failure has always been one, so
    every existing caller keeps its behaviour and nothing that catches `AttributeError`
    today starts leaking this one. What changes is that the failure is NAMED and carries
    the three things an operator needs — WHICH ROW is broken, WHAT PATH it declared, and
    WHICH SEGMENT of that path does not exist. Pydantic's `BaseModel.__getattr__` supplies
    only the last, and `preflight_mint.main`'s handler chain then lost even that, collapsing
    the whole class into rc 1 `PreflightInternalError` — the one outcome that tool's own
    docstring says cannot exist. The tool maps this to `PreflightManifestError`, rc 31.

    Written to the CLASS, not to one row (R71): the same route swallows a typo in ANY row's
    `config_path`.
    """


def _dotted(obj: Any, path: str, *, row: str = "<unnamed row>") -> Any:
    """Walk a dotted path into a validated config object.

    Two arms beyond the plain walk, and each is load-bearing:

    * a MISSING attribute raises `ArmingSurfaceMissingError` naming the row, the full path
      and the failing segment. `try/except AttributeError` PER SEGMENT rather than a
      `hasattr` pre-check, because only the former can say which segment failed and because
      the AttributeError comes from pydantic's `BaseModel.__getattr__`, not a plain lookup;
    * a `None` met MID-WALK is an EXPLICITLY DISARMED block and short-circuits to `None`.
      Without it a legitimately disarmed config — `train.draw_rate_abort: null`, the posture
      R59 permits for smoke configs and four of the five committed configs carry — would
      raise `'NoneType' object has no attribute 'threshold'` and fail gate 12 at rc 31
      instead of being reported DISARMED.

    DISCLOSED RESIDUAL: a typo AFTER a legitimately-`None` segment reports "disarmed"
    rather than raising, because the walk short-circuits before reaching it. It is caught
    where it gates — `PRODUCTION_CONFIGS` is run5 and run5 is ARMED, so the walk reaches the
    leaf and the typo raises. Both arms are pinned by
    `tests/tools/test_drawrate_arming_surface_named_failure.py`.
    """
    for part in path.split("."):
        if obj is None:
            return None
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ArmingSurfaceMissingError(
                f"armed-abort row {row!r} declares config_path {path!r}, but segment "
                f"{part!r} is absent on {type(obj).__name__}: a manifest row whose arming "
                "surface does not exist on a real RunConfig is a phantom gate input "
                "(R4 / LAW-07), and it must be a NAMED failure rather than the tool's own "
                "unnamed internal error"
            ) from exc
    return obj


def audit_arming(config: Any, *, manifest: tuple[ArmedAbort, ...] = MANIFEST) -> AuditResult:
    """Assertion (c): every REQUIRED row must be armed in `config`.

    Never branches on a row's `name` and never special-cases draw-rate: `status` selects
    the list, `mechanism` selects the predicate, and both are data. `manifest` is a keyword
    so O-7 can drive an in-memory copy with the deferred row flipped (§8.5).
    """
    required = tuple(row for row in manifest if row.status is Status.REQUIRED)
    deferred = tuple(row for row in manifest if row.status is Status.DEFERRED)
    disarmed = tuple(
        row for row in required
        if not row.mechanism.is_armed(_dotted(config, row.config_path, row=row.name))
    )
    return AuditResult(required=required, deferred=deferred, disarmed=disarmed)


__all__ = [
    "EXEMPT_CONFIGS",
    "MANIFEST",
    "PRODUCTION_CONFIGS",
    "ArmedAbort",
    "ArmingSurfaceMissingError",
    "AuditResult",
    "Mechanism",
    "Status",
    "audit_arming",
]
