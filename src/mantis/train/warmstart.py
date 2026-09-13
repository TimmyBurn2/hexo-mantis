"""Launch-time BC warm start: the `identity.warm_start` row, its one reader, and the seam that
copies EVERY tensor of the declared checkpoint onto a fresh net, re-initialises only the heads
`reinit` names, and refuses a live step-0 net whose hash is not the source's (R350(b)(i))."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis.model import load_from_bc
from mantis.model.gnn import BcTransferReport
from mantis.model.identity import net_param_hash

_LOG = logging.getLogger(__name__)


class WarmStartIdentityError(RuntimeError):
    """The declared warm-start checkpoint is not the net the config says it is, or the net
    that came out of the seam is not the net that went in."""


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
    """A resolved BC warm-start source: the checkpoint, the net it must turn out to be, and the
    heads put back to fresh after the copy. Frozen, and the members travel together: a path
    without its expected hash is the shape this row exists to make unconstructible."""

    checkpoint: Path
    net_hash: str
    reinit: tuple[str, ...]


def resolve_bc_warm_start(combined_config: Mapping[str, Any]) -> BcWarmStart | None:
    """Return the declared BC warm-start source, or `None` when the config carries no row; a
    present row is fully specified by the schema, so nothing here is defaulted.

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
    if missing or "reinit" not in row:
        raise ValueError(
            f"{WARM_START_ROW} is missing {missing or ['reinit']}. Every member is REQUIRED by "
            "the schema, so a config reaching here without them did not come through "
            "`load_config` — a checkpoint path with no expected net hash is exactly the "
            "unverified warm start this row exists to prevent, and an absent `reinit` would "
            "let the seam drop a head without the config saying so."
        )
    return BcWarmStart(Path(str(row["checkpoint"])), str(row["net_hash"]),
                       tuple(str(h) for h in row["reinit"]))


def apply_bc_warm_start(model: Any, declared: BcWarmStart, *, spec: Any) -> BcTransferReport:
    """Seed a fresh graph net from the DECLARED BC checkpoint: every tensor, then the heads
    `declared.reinit` names put back to the fresh init. Every step refuses: the arch comes from
    the artifact's STAMP, the source's hash must equal `net_hash`, the transfer is strict, and
    with an empty `reinit` the LIVE net must hash to `net_hash` (the step-0 witness).

    Raises:
        ValueError:        the resolved encoding is not a graph representation, or a `reinit`
                           entry names no tensor of the net.
        FileNotFoundError: the declared checkpoint does not exist.
        WarmStartIdentityError: the checkpoint's net hash is not the declared one, or the live
                           net after a full transfer does not hash to it.
        RuntimeError:      a key mismatch or failed landed-verify in the transfer.
    """
    representation = getattr(spec, "representation", None)
    if representation != "graph":
        raise ValueError(
            f"{WARM_START_ROW} is declared but the resolved encoding "
            f"{getattr(spec, 'name', '?')!r} has representation={representation!r} (expected "
            "'graph') — the BC transfer is graph-only."
        )
    if not declared.checkpoint.exists():
        raise FileNotFoundError(
            f"{WARM_START_ROW}.checkpoint not found: {declared.checkpoint}."
        )

    from mantis.model import build_net
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

    result = load_from_bc(model, dict(ck.model_state), reinit=declared.reinit)
    live = net_param_hash(model)
    if not declared.reinit and live != declared.net_hash:
        raise WarmStartIdentityError(
            f"{WARM_START_ROW}: the seam copied {len(result['loaded_keys'])} tensors from "
            f"{declared.checkpoint} with nothing to re-initialise, yet the live net hashes to "
            f"{live}, not the declared {declared.net_hash}. The step-0 net is not the source "
            "net — refusing to start from it."
        )
    _LOG.info(
        "bc_warmstart_loaded checkpoint=%s net_param_hash=%s live_net_hash=%s loaded_keys=%d "
        "reinit=%s verified_tensors=%s",
        str(declared.checkpoint), actual, live, len(result["loaded_keys"]),
        list(result["reinit_keys"]), result["verified_tensors"],
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
