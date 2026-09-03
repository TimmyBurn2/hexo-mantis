"""Launch-time warm-start seams (WP10 §a.6/§c.5 IMPROVE) — value-head + GNN-BC transfer.

>300 justify: combines three old modules (`training/warmstart_launch.py` 190,
`training/warmstart_value_head.py` 197, `training/gnn_warmstart.py` 144) into ONE file — the
DESIGN's "one concern, soft-cap permitting" — since they are one launch concern (seed a fresh
value/graph head from a prior artifact) whose seams belong together.

Behaviour-exact except the two ratified WP10 amendments:

  * **The personal-path default DIES (CLAUDE.md R1 + host-coupling ban).** The old absolute
    personal head-directory constant is DELETED; `head_dir` is now a REQUIRED explicit parameter /
    resolver key. Absent → a loud error, never a host-coupled path. `default_head_for_arm` returns
    only the RELATIVE arm path — the launch supplies `head_dir`.
  * **`load_value_head` loads weights-only** (LAW-12 — closes the pickle-exec hole the old
    `weights_only`-unset `torch.load` left open).

Three seams:
  - value-head warm-start (E1): overwrite ONLY the value-head tensors of a freshly-built
    (scalar OR dist65) net from a pre-registered head `.pt`, on a fresh/weights-only warm launch.
  - dist65-bins-seeded guard (E1): refuse to train a dist65 net whose bins are untrained/random.
  - GNN BC-prefit transfer: seed a fresh `GnnNet`'s representation+policy_head from a BC checkpoint
    (`mantis.model.load_representation_policy_from_bc`; value head untouched).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis.model import load_representation_policy_from_bc
from mantis.model.gnn import BcTransferReport

_LOG = logging.getLogger(__name__)


class WarmStartIdentityError(RuntimeError):
    """The declared warm-start checkpoint is not the net the config says it is (R332(d))."""

# Presence of a trained dist65 bin-logit tail marks a FULL GnnNet (vs a BC-prefit-only source).
_DIST65_BINS_KEY = "value_head.fc2_bins.weight"


# Presence of a trained dist65 bin-logit tail marks a FULL GnnNet (vs a BC-prefit-only source).
_DIST65_BINS_KEY = "value_head.fc2_bins.weight"


def _extract_state(raw: Any) -> dict[str, torch.Tensor]:
    """Pull the model state dict out of a loaded artifact — a bare `state_dict`, or a
    `{model_state: …}` / `{state_dict: …}` wrapper. Prefixes are left intact (the BC-transfer
    matcher handles wrapper prefixes itself)."""
    if isinstance(raw, dict):
        for key in ("model_state", "state_dict"):
            inner = raw.get(key)
            if isinstance(inner, dict):
                return inner
    return raw


# ══ THE VALUE-HEAD WARM-START ARM IS DELETED (AUDIT-1 F-19's dead-code half) ═══════════
# `resolve_warmstart_head_file`, `default_head_for_arm`, `load_value_head`,
# `maybe_warmstart_value_head` and `assert_dist65_bins_seeded` stood here — the predecessor
# engine's arm-A/arm-B head-seeding mechanism, ported and never entered. Measured at contact:
# zero production callers, and TWO reads of keys the schema does not have, each with a
# code-side default on an identity quantity — `combined_config.get("warm_start")` (the schema
# has `identity.warm_start`, a different block with different members) and
# `combined_config.get("value_head_type", "scalar")`, which is also one of the arch-sniff sites
# F-24 counts. Its own error text told the operator to set `warm_start.enabled` and
# `warm_start.head_dir`: config keys that do not exist.
#
# `assert_dist65_bins_seeded` went WITH the arm rather than being kept as a guard, because its
# third parameter is `warmstart_fired` — the return of `maybe_warmstart_value_head`. It is not
# a guard the BC entry below could adopt; it is a member of the mechanism that is gone.
#
# R332(d) DECIDED what the entry is, and it is the block below: a checkpoint named by path AND
# by the net hash it must turn out to be. This is the completion of that decision — the arm
# that was NOT chosen does not stay behind reading keys nobody mints.


# ══ GNN BC-prefit transfer — THE PRODUCTION ENTRY (R332(d), AUDIT-1 F-19) ══════════════
#: THE ONE CONFIG ROW naming a BC warm-start source — dotted, as `RunConfig` spells it. Read by
#: `resolve_bc_warm_start` and nowhere else, so the row has exactly one reader to change (the
#: `ARCH_KIND_ROW` pattern, for the same reason).
WARM_START_ROW = "identity.warm_start"


@dataclass(frozen=True)
class BcWarmStart:
    """A resolved BC warm-start source: the checkpoint, and the net it must turn out to be.

    FROZEN for the reason every resolved run-scoped constant in this tree is frozen — a value a
    consumer could rebind is a second authority with extra steps. Both members travel together
    because a path without its expected hash is the shape this row exists to make
    unconstructible.
    """

    checkpoint: Path
    net_hash: str


def resolve_bc_warm_start(combined_config: Mapping[str, Any]) -> BcWarmStart | None:
    """The declared BC warm-start source, or `None` when the config carries no row.

    `None` is NOT a default carrying a guess: it states "this config declares no warm start",
    which is what every run before the row did and what every committed config still says. A
    row that is PRESENT is fully specified by the schema (`WarmStartConfig` requires both
    members), so there is no `.get(key, fallback)` anywhere on this path.

    Mapping-typed rather than `RunConfig`-typed so `mantis.train` keeps no import edge it does
    not already have; the schema is what guarantees the shape.

    Raises:
        ValueError: the row is present but is not a mapping, or is missing a member — a config
            that reaches here in that state did not come through the one loader.
    """
    identity = combined_config.get("identity")
    if not isinstance(identity, Mapping):
        return None
    row = identity.get("warm_start")
    if row is None:
        return None
    if not isinstance(row, Mapping):
        raise ValueError(
            f"{WARM_START_ROW} is {type(row).__name__}, expected a mapping with `checkpoint` "
            "and `net_hash` (or `null` for no warm start)."
        )
    missing = [m for m in ("checkpoint", "net_hash") if not row.get(m)]
    if missing:
        raise ValueError(
            f"{WARM_START_ROW} is missing {missing}. Both members are REQUIRED by the schema, "
            "so a config reaching here without them did not come through `load_config` — and a "
            "checkpoint path with no expected net hash is exactly the unverified warm start "
            "this row exists to prevent."
        )
    return BcWarmStart(Path(str(row["checkpoint"])), str(row["net_hash"]))


def apply_bc_warm_start(model: Any, declared: BcWarmStart, *, spec: Any) -> BcTransferReport:
    """Seed a fresh graph net's representation+policy_head from the DECLARED BC checkpoint.

    THE SEQUENCE, and every step is a refusal point:

    1. the artifact is read through THE checkpoint loader, so its arch comes from its own STAMP
       (`checkpoints.stamped_arch_kind`, the one selector authority for an artifact) and never
       from a shape sniff or this run's config;
    2. the net that checkpoint IS gets rebuilt and hashed, and the hash must equal the
       `net_hash` the identity declared — otherwise the file at that path is not the artifact
       the prereg named, and the run refuses rather than training from whatever is there;
    3. only then does the transfer run, and it is the existing strict primitive: every
       `representation.*` / `policy_head.*` key must match on both sides, with a landed-verify
       pass. The value head is NEVER touched.

    Returns the transfer report (`loaded_keys`, `verified_tensors`).

    Raises:
        ValueError:        the resolved encoding is not a graph representation.
        FileNotFoundError: the declared checkpoint does not exist.
        WarmStartIdentityError: the checkpoint's net hash is not the declared one.
        RuntimeError:      a key mismatch or failed landed-verify in the transfer (F1 guard).
    """
    representation = getattr(spec, "representation", None)
    if representation != "graph":
        raise ValueError(
            f"{WARM_START_ROW} is declared but the resolved encoding "
            f"{getattr(spec, 'name', '?')!r} has representation={representation!r} (expected "
            "'graph') — the BC-prefit transfer is graph-only. Use `warm_start.*` for the CNN "
            "value-head-only E1 warm-start instead."
        )
    if not declared.checkpoint.exists():
        raise FileNotFoundError(
            f"{WARM_START_ROW}.checkpoint not found: {declared.checkpoint}."
        )

    from mantis.model import build_net
    from mantis.model.identity import net_param_hash
    from mantis.train.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION,
        load_checkpoint,
        load_legacy_weights,
    )

    raw = torch.load(declared.checkpoint, map_location="cpu", weights_only=True)
    is_v2 = isinstance(raw, dict) and raw.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
    ck = (
        load_checkpoint(declared.checkpoint)
        if is_v2
        else load_legacy_weights(declared.checkpoint, declared_encoding=getattr(spec, "name", None))
    )
    if ck.metadata.arch is None:
        raise WarmStartIdentityError(
            f"{declared.checkpoint}: the artifact's stamp resolves no arch, so the net it "
            "carries cannot be rebuilt and its identity cannot be checked."
        )
    source_net = build_net(ck.metadata.arch)
    source_net.load_state_dict(ck.model_state)
    actual = net_param_hash(source_net)
    if actual != declared.net_hash:
        raise WarmStartIdentityError(
            f"{WARM_START_ROW}: the checkpoint at {declared.checkpoint} has net_param_hash "
            f"{actual}, but the config declares {declared.net_hash}. The artifact at that path "
            "is NOT the one this run was pre-registered against — refusing to warm-start from "
            "it. Re-point the path, or re-mint the hash against the checkpoint of record."
        )

    if _DIST65_BINS_KEY in ck.model_state:
        _LOG.warning(
            "bc_warmstart_source_has_value_head checkpoint=%s "
            "(looks like a FULL net checkpoint, not a BC-prefit-only source; the value head "
            "stays fresh either way. If a full resume was intended, use --resume-from).",
            str(declared.checkpoint),
        )

    result = load_representation_policy_from_bc(model, dict(ck.model_state))
    _LOG.info(
        "bc_warmstart_loaded checkpoint=%s net_param_hash=%s loaded_keys=%d verified_tensors=%s",
        str(declared.checkpoint), actual, len(result["loaded_keys"]), result["verified_tensors"],
    )
    return result


def maybe_warmstart_gnn_from_bc(model: Any, combined_config: Mapping[str, Any], *, spec: Any) -> bool:
    """The launch hook: resolve `identity.warm_start` and apply it. True iff a transfer fired.

    Raises:
        ValueError:        the row is malformed, or the encoding is not a graph representation.
        FileNotFoundError: the declared checkpoint does not exist.
        WarmStartIdentityError: the checkpoint is not the declared net.
        RuntimeError:      a key mismatch / failed landed-verify (F1 guard).
    """
    declared = resolve_bc_warm_start(combined_config)
    if declared is None:
        return False
    apply_bc_warm_start(model, declared, spec=spec)
    return True


__all__ = [
    "WARM_START_ROW",
    "BcWarmStart",
    "WarmStartIdentityError",
    "apply_bc_warm_start",
    "resolve_bc_warm_start",
    "maybe_warmstart_gnn_from_bc",
]
