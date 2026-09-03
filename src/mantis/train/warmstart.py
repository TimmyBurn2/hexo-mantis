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

# ── Arm-selection by value_head_type (RELATIVE to the launch-supplied head_dir) ────────
# Pre-registered selection (INV-D1 / R5): scalar arm <- arm_A_seed0, dist arm <- arm_B_seed0.
# These are RELATIVE paths only — the concrete directory is a launch parameter (`head_dir`),
# never a code-side default (the old absolute host-coupled default is DELETED, WP10 §a.6).
HEAD_FILE_BY_TYPE: dict[str, str] = {
    "scalar": "arm_A_seed0/head_A_seed0.pt",
    "dist65": "arm_B_seed0/head_B_seed0.pt",
}

# head_shape metadata string -> expected head_type
_SHAPE_TO_HEAD_TYPE = {"scalar": "scalar", "bin65": "dist65"}
_VALID_HEAD_TYPES = {"scalar", "dist65"}

# Presence of a trained dist65 bin-logit tail marks a FULL GnnNet (vs a BC-prefit-only source).
_DIST65_BINS_KEY = "value_head.fc2_bins.weight"


def _base_model(net: Any) -> Any:
    """Unwrap a `torch.compile` / DDP wrapper (`_orig_mod`) to reach the raw net whose
    value-head Parameters this module overwrites in place."""
    return getattr(net, "_orig_mod", net)


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


# ══ Value-head warm-start (E1) ═════════════════════════════════════════════════════════
def resolve_warmstart_head_file(head_dir: str, value_head_type: str) -> str:
    """Resolve the head `.pt` for ``value_head_type`` under the REQUIRED ``head_dir``.

    ``head_dir`` is an explicit launch parameter — there is NO code-side default (R1). A
    misconfigured ``head_dir`` fails LOUDLY at launch (never silently seeds nothing).

    Raises:
        ValueError:        unknown value_head_type.
        FileNotFoundError: the resolved head file does not exist.
    """
    rel = HEAD_FILE_BY_TYPE.get(value_head_type)
    if rel is None:
        raise ValueError(
            f"value_head_type={value_head_type!r} has no warm-start head mapping "
            f"(known: {sorted(HEAD_FILE_BY_TYPE)})."
        )
    head_file = str(Path(head_dir) / rel)
    if not Path(head_file).exists():
        raise FileNotFoundError(
            f"warm-start head file not found: {head_file} "
            f"(head_dir={head_dir!r}, value_head_type={value_head_type!r}). "
            "Check warm_start.head_dir points at the head-artifact directory."
        )
    return head_file


def default_head_for_arm(head_type: str) -> str:
    """The pre-registered RELATIVE head path for an arm's ``head_type`` (no directory).

    scalar -> arm_A_seed0/head_A_seed0.pt ; dist65 -> arm_B_seed0/head_B_seed0.pt. The launch
    joins this against its explicit ``head_dir`` — this NEVER returns an absolute host path."""
    rel = HEAD_FILE_BY_TYPE.get(head_type)
    if rel is None:
        raise ValueError(f"head_type={head_type!r} not in {sorted(_VALID_HEAD_TYPES)}")
    return rel


def load_value_head(net: Any, head_pt_path: str, head_type: str) -> None:
    """Seed ``net``'s value head from a head `.pt`; touches ONLY the value-head tensors.

    A scalar↔dist mismatch (`.pt` vs head_type, or head_type vs the net's built head) RAISES
    (C1 guard — no silent random-head fallback); loaded tensors are VERIFIED to have landed via
    ``allclose``. Loads weights-only (LAW-12).

    Raises:
        ValueError:   unknown head_type; scalar↔dist mismatch; any value-head shape mismatch.
        RuntimeError: post-load verification failed (a tensor did not land).
    """
    if head_type not in _VALID_HEAD_TYPES:
        raise ValueError(f"head_type={head_type!r} not in {sorted(_VALID_HEAD_TYPES)}")

    base = _base_model(net)

    net_has_bins = getattr(base, "value_fc2_bins", None) is not None
    net_head_type = getattr(base, "value_head_type", "scalar")
    if head_type == "dist65" and not net_has_bins:
        raise ValueError(
            f"head_type='dist65' but the net has no value_fc2_bins layer "
            f"(net value_head_type={net_head_type!r}). Build the net with "
            "value_head_type='dist65' before warm-starting a dist head."
        )
    if head_type == "scalar" and net_has_bins:
        raise ValueError(
            "head_type='scalar' but the net is a dist65 net (has value_fc2_bins). Seeding a "
            "scalar head onto a dist net would leave value_fc2_bins random. Match head_type to "
            f"the net's built head (net value_head_type={net_head_type!r})."
        )

    blob = torch.load(head_pt_path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "head_state" not in blob:
        raise ValueError(
            f"{head_pt_path}: not a head `.pt` (missing 'head_state' wrapper key)."
        )

    head_shape = blob.get("head_shape")
    # Absent/non-str head_shape resolves to None and lands in the raise below.
    pt_head_type = _SHAPE_TO_HEAD_TYPE.get(head_shape) if isinstance(head_shape, str) else None
    if pt_head_type is None:
        raise ValueError(
            f"{head_pt_path}: unknown head_shape={head_shape!r} "
            f"(expected one of {sorted(_SHAPE_TO_HEAD_TYPE)})."
        )
    if pt_head_type != head_type:
        raise ValueError(
            f"scalar/dist mismatch: head `.pt` head_shape={head_shape!r} (=> {pt_head_type!r}) "
            f"but head_type={head_type!r} was requested. Refusing to load — mismatched value-head "
            "kind would silently drop the trained bin/scalar weights (C1 regression)."
        )

    head_state: dict[str, torch.Tensor] = blob["head_state"]
    fc2_dst = base.value_fc2 if head_type == "scalar" else base.value_fc2_bins
    fc2_label = "value_fc2" if head_type == "scalar" else "value_fc2_bins"
    mapping = [
        ("fc1.weight", base.value_fc1.weight, "value_fc1.weight"),
        ("fc1.bias", base.value_fc1.bias, "value_fc1.bias"),
        ("fc2.weight", fc2_dst.weight, f"{fc2_label}.weight"),
        ("fc2.bias", fc2_dst.bias, f"{fc2_label}.bias"),
    ]

    for src_key, dst_param, label in mapping:
        if src_key not in head_state:
            raise ValueError(
                f"{head_pt_path}: head_state missing key {src_key!r} (have {sorted(head_state)})."
            )
        src_tensor = head_state[src_key]
        if tuple(src_tensor.shape) != tuple(dst_param.shape):
            raise ValueError(
                f"shape mismatch for {label}: head `.pt` {src_key} is {tuple(src_tensor.shape)} "
                f"but net expects {tuple(dst_param.shape)}."
            )
        with torch.no_grad():
            dst_param.data.copy_(src_tensor.to(device=dst_param.device, dtype=dst_param.dtype))

    for src_key, dst_param, label in mapping:  # post-load landed-verify (C1)
        src_tensor = head_state[src_key].to(device=dst_param.device, dtype=dst_param.dtype)
        if not torch.allclose(dst_param.data, src_tensor):
            raise RuntimeError(
                f"warm-start value-head verify FAILED for {label}: the tensor did not land "
                f"(post-copy mismatch). head_pt={head_pt_path}."
            )


def maybe_warmstart_value_head(
    trainer: Any,
    combined_config: dict[str, Any],
) -> bool:
    """Seed ``trainer.model``'s value head from the pre-registered head, IF warm_start is
    enabled AND this is a weights-only warm launch. Returns True iff the head was seeded.

    ``warm_start.head_dir`` is REQUIRED when enabled — absent → a loud ``ValueError`` (never a
    host-coupled default). RESUME GUARD: a FULL-checkpoint resume already restored the trained
    value head; re-seeding would corrupt it, so the hook skips + WARNs. Default-OFF: a
    byte-identical no-op when warm_start is disabled/absent.
    """
    ws_cfg = combined_config.get("warm_start") or {}
    if not isinstance(ws_cfg, dict) or not ws_cfg.get("enabled", False):
        return False

    head_dir = ws_cfg.get("head_dir")
    if not head_dir:
        raise ValueError(
            "warm_start.enabled is true but warm_start.head_dir is unset — cannot resolve the "
            "head to seed the value head (head_dir is a REQUIRED explicit parameter; there is no "
            "code-side default)."
        )

    loaded_from_full = getattr(trainer, "loaded_from_full_checkpoint", None)
    if loaded_from_full is None:
        _LOG.warning(
            "warmstart_value_head_skipped reason=no_checkpoint_loaded "
            "(warm_start.enabled but no checkpoint loaded — nothing to warm-start)."
        )
        return False
    if loaded_from_full:
        _LOG.warning(
            "warmstart_value_head_skipped reason=full_checkpoint_resume "
            "(the trained value head is already restored; re-seeding would corrupt it)."
        )
        return False

    value_head_type = combined_config.get("value_head_type", "scalar")
    head_file = resolve_warmstart_head_file(head_dir, value_head_type)
    load_value_head(trainer.model, head_file, value_head_type)

    arm = "A" if value_head_type == "scalar" else "B"
    _LOG.info(
        "warmstart_value_head_loaded arm=%s head_file=%s head_type=%s head_dir=%s",
        arm, head_file, value_head_type, str(head_dir),
    )
    return True


def assert_dist65_bins_seeded(
    trainer: Any,
    combined_config: dict[str, Any],
    warmstart_fired: bool,
) -> None:
    """Raise if a dist65 net's value_fc2_bins are untrained/random (neither loaded from the
    checkpoint NOR seeded by the warm-start hook). No-op for scalar nets, dist65 + warm-start ON,
    or a genuine dist65 checkpoint resume.

    Raises:
        RuntimeError: dist65 + scalar trunk (no bins in ckpt) + no warm-start.
    """
    value_head_type = combined_config.get("value_head_type", "scalar")
    if value_head_type != "dist65":
        return
    ckpt_had_bins = getattr(trainer, "ckpt_had_value_fc2_bins", True)
    if ckpt_had_bins:
        return
    if warmstart_fired:
        return
    raise RuntimeError(
        "dist65 value head has untrained/random value_fc2_bins and no warm-start seeded them — "
        "refusing to train. The loaded checkpoint is a SCALAR trunk (no value_fc2_bins.*) and "
        "warm_start.enabled is false (or warm-start was skipped). Fix: set warm_start.enabled=true "
        "and point warm_start.head_dir at the head-artifact directory, OR resume from a full "
        "dist65 checkpoint that already has trained bins."
    )


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
    "HEAD_FILE_BY_TYPE",
    "WARM_START_ROW",
    "BcWarmStart",
    "WarmStartIdentityError",
    "apply_bc_warm_start",
    "resolve_bc_warm_start",
    "assert_dist65_bins_seeded",
    "default_head_for_arm",
    "load_value_head",
    "maybe_warmstart_gnn_from_bc",
    "maybe_warmstart_value_head",
    "resolve_warmstart_head_file",
]
