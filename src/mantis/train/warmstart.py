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
from pathlib import Path
from typing import Any

import torch

from mantis.model import load_representation_policy_from_bc

_LOG = logging.getLogger(__name__)

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


# ══ GNN BC-prefit transfer (fresh graph launch) ════════════════════════════════════════
def maybe_warmstart_gnn_from_bc(
    model: Any,
    combined_config: dict[str, Any],
    *,
    spec: Any,
) -> bool:
    """Seed a FRESH ``GnnNet``'s representation+policy_head from a BC-prefit checkpoint declared
    at ``gnn_warm_start.checkpoint``. Returns True iff the transfer fired. Value head is NEVER
    touched (``load_representation_policy_from_bc`` transfers ONLY representation.*/policy_head.*).
    Default-OFF: a byte-identical no-op when ``gnn_warm_start`` is disabled/absent.

    Raises:
        ValueError:        enabled but ``spec.representation`` != "graph", or checkpoint unset.
        FileNotFoundError: the declared checkpoint does not exist.
        RuntimeError:      a key mismatch / failed landed-verify (F1 guard).
    """
    ws_cfg = combined_config.get("gnn_warm_start") or {}
    if not isinstance(ws_cfg, dict) or not ws_cfg.get("enabled", False):
        return False

    representation = getattr(spec, "representation", "grid")
    if representation != "graph":
        raise ValueError(
            "gnn_warm_start.enabled is true but the resolved encoding "
            f"{getattr(spec, 'name', '?')!r} has representation={representation!r} "
            "(expected 'graph') — the BC-prefit warm-start seam is graph-only. Use warm_start.* "
            "for the CNN value-head-only E1 warm-start instead."
        )

    ckpt_path_raw = ws_cfg.get("checkpoint")
    if not ckpt_path_raw:
        raise ValueError(
            "gnn_warm_start.enabled is true but gnn_warm_start.checkpoint is unset — cannot "
            "resolve the BC-prefit source checkpoint."
        )
    ckpt_path = Path(ckpt_path_raw)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"gnn_warm_start.checkpoint not found: {ckpt_path}.")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    bc_state_dict = _extract_state(raw)

    if _DIST65_BINS_KEY in bc_state_dict:
        _LOG.warning(
            "gnn_warmstart_source_has_value_head checkpoint=%s "
            "(looks like a FULL GnnNet checkpoint, not a BC-prefit-only source; the value head "
            "stays fresh either way. If a full resume was intended, use --checkpoint).",
            str(ckpt_path),
        )

    result = load_representation_policy_from_bc(model, bc_state_dict)
    _LOG.info(
        "gnn_warmstart_loaded checkpoint=%s loaded_keys=%d verified_tensors=%s",
        str(ckpt_path), len(result["loaded_keys"]), result["verified_tensors"],
    )
    return True


__all__ = [
    "HEAD_FILE_BY_TYPE",
    "assert_dist65_bins_seeded",
    "default_head_for_arm",
    "load_value_head",
    "maybe_warmstart_gnn_from_bc",
    "maybe_warmstart_value_head",
    "resolve_warmstart_head_file",
]
