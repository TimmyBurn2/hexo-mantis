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

    `owner` and `source_pin` are REQUIRED on a DEFERRED row and FORBIDDEN on a REQUIRED one.
    Each of the three ways those can disagree is a way for the deferred row to go invisible:
    an owner-less deferred row has nobody to chase, a pin-less one is not tamper-evident,
    and a required row carrying an owner reads as already-excused.
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
        config_path="train.step_coordinator.draw_rate_threshold",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
        status=Status.DEFERRED,
        exit_code=None,
        owner="R-TRAINCONFIG-SCHEMA / CARD-DRAWRATE-KEY (R65)",
        source_pin=(
            "src/mantis/train/coordinator/config.py",
            "draw_rate_threshold: float = 0.0",
        ),
        note=(
            "The arming surface DOES NOT EXIST as a config key yet — the threshold is an "
            "unauthored code-side default on StepCoordinatorConfig (ADJ-08). The pin goes "
            "RED the moment Phase D deletes that literal, which is the forcing function "
            "that makes this row's flip to REQUIRED unforgettable."
        ),
    ),
)

#: WHICH configs the law binds — one authority. Repo-relative strings only; resolving them
#: is the tool's (SF-4).
PRODUCTION_CONFIGS: tuple[str, ...] = ("configs/run5.yaml",)

#: The OTHER half of the same authority (MF-7). R59's "deliberate disarming remains legal for
#: smoke configs" used to be expressed by ABSENCE from `PRODUCTION_CONFIGS` — which made
#: "deliberately exempt" and "nobody remembered to list it" the SAME observable, and a
#: disarmed `configs/run6.yaml` dropped into the tree audited GREEN (measured: rc 0).
#:
#: So the exemption is now WRITTEN, and the two tuples must PARTITION `configs/*.yaml`
#: exactly. The tool hard-fails (rc 31) on either kind of drift: a config present on disk and
#: named by neither tuple, and a tuple naming a config that is not on disk. That is gate 11's
#: `KNOWN_DEBT` shape (`silent_encoding_gate.py:126,338-344`) applied to the config set —
#: registered debt whose staleness is itself a failure. Adding a config to `configs/` now
#: FORCES a one-line declaration here or in `PRODUCTION_CONFIGS`; it can no longer be
#: forgotten into exemption.
#:
#: `(repo-relative path, why it is exempt)`. The reason is data, printed by the tool on the
#: failure path, so an exemption cannot be a bare path nobody can justify later.
EXEMPT_CONFIGS: tuple[tuple[str, str], ...] = (
    (
        "configs/dev_example.yaml",
        "developer template, never minted for a run; disarmed at `:200` by design (R59).",
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


def _dotted(obj: Any, path: str) -> Any:
    """Walk a dotted path into a validated config object."""
    for part in path.split("."):
        obj = getattr(obj, part)
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
        row for row in required if not row.mechanism.is_armed(_dotted(config, row.config_path))
    )
    return AuditResult(required=required, deferred=deferred, disarmed=disarmed)


__all__ = [
    "EXEMPT_CONFIGS",
    "MANIFEST",
    "PRODUCTION_CONFIGS",
    "ArmedAbort",
    "AuditResult",
    "Mechanism",
    "Status",
    "audit_arming",
]
