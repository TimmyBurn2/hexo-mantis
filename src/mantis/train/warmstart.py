"""Launch-time warm-start seams — value-head + GNN-BC transfer.

>300 justify: one launch concern, seeding a fresh value/graph head from a prior artifact, whose
seams belong in one file. `head_dir` is a REQUIRED explicit parameter — absent is a loud error,
never a host-coupled default path — and the load is weights-only. The BC-prefit seam seeds a
fresh `GnnNet`'s representation+policy_head and never touches the value head.
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
    """The declared warm-start checkpoint is not the net the config says it is."""

# Presence of a trained dist65 bin-logit tail marks a FULL GnnNet (vs a BC-prefit-only source).
_DIST65_BINS_KEY = "value_head.fc2_bins.weight"


_DIST65_BINS_KEY = "value_head.fc2_bins.weight"


def _extract_state(raw: Any) -> dict[str, torch.Tensor]:
    """Pull the model state dict out of a loaded artifact — bare, or a `{model_state: …}` /
    `{state_dict: …}` wrapper. Prefixes stay intact; the BC-transfer matcher handles them."""
    if isinstance(raw, dict):
        for key in ("model_state", "state_dict"):
            inner = raw.get(key)
            if isinstance(inner, dict):
                return inner
    return raw


#: THE ONE CONFIG ROW naming a BC warm-start source; `resolve_bc_warm_start` is its only reader.
WARM_START_ROW = "identity.warm_start"


@dataclass(frozen=True)
class BcWarmStart:
    """A resolved BC warm-start source: the checkpoint, and the net it must turn out to be.
    Frozen, and both members travel together: a path without its expected hash is the shape
    this row exists to make unconstructible."""

    checkpoint: Path
    net_hash: str


def resolve_bc_warm_start(combined_config: Mapping[str, Any]) -> BcWarmStart | None:
    """Return the declared BC warm-start source, or `None` when the config carries no row.

    `None` states "this config declares no warm start" rather than carrying a guess; a present
    row is fully specified by the schema, so no `.get(key, fallback)` appears on this path.
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

    Every step is a refusal point: the arch comes from the artifact's own STAMP, the rebuilt
    net's hash must equal the declared `net_hash`, and only then does the strict key-matched
    transfer run. The value head is NEVER touched.

    Raises:
        ValueError:        the resolved encoding is not a graph representation.
        FileNotFoundError: the declared checkpoint does not exist.
        WarmStartIdentityError: the checkpoint's net hash is not the declared one.
        RuntimeError:      a key mismatch or failed landed-verify in the transfer.
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
