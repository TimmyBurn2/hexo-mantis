"""The ONE checkpoint loader + envelope-v2 writer/reader (repo_design §6; WP10 §a.1/§c.1).

>300 justify: four old loaders (`training/checkpoints`, `training/trainer_ckpt_load`,
`eval/checkpoint_loader`, `viewer/model_loader`) collapse into this single module — the
highest-value structural win of WP10. It owns the envelope-v2 write path, the ONE read
path (v2 + the three legacy shapes), the immutable-stamp + provenance-reverify guards,
the O3b killed-prefix REJECT, the resume-precedence helpers, and the weights-strip path.

Zero-behavior-change doctrine: every reachable numeric op is a pure relocation. The
approved FAILURE-MODE amendments (immutable stamps, unstamped=failed save,
`weights_only=True` everywhere, persist-fatal, killed-prefix reject) change how failures
SURFACE, never a reachable numeric result. All shape-inference (`infer_*_hparams`,
`_build_min_max_model` sniff-reconstruct, `MODEL_HPARAM_DEFAULTS`) is DELETED — arch
travels on `metadata.arch` (a WP9 declared dataclass) → `build_net`, never re-derived.
"""
from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.model import (
    CnnArch,
    GnnArch,
    ModelArch,
    RepresentationMismatch,
    arch_from_spec_and_config,
    build_net,
)
from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)

# ── Envelope axis (DISTINCT from the config's own schema_version=1) ────────────────────
CHECKPOINT_SCHEMA_VERSION = 2

# WP9 O3b (load-bearing): state-dict key prefixes of FALSIFIED-and-DELETED branches. The
# loader REJECTS any state dict carrying one — it NEVER reconstructs a PMA/gpool pool.
KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")

# Persist-fatal counter (repo_design §11 / LAW-14): a swallowed persist failure is banned;
# a failed/quarantined write increments this, never `except: pass`.
persist_errors_total = 0


class CheckpointStampError(RuntimeError):
    """Unstamped save / re-stamp / provenance / content-hash / stamp-disagree failure."""


class DeclaredEncodingMismatchError(ValueError):
    """A caller-declared encoding disagrees with the checkpoint's own trusted stamp."""


# ── Envelope dataclasses (the in-memory view of a loaded envelope) ─────────────────────
@dataclass(frozen=True)
class CheckpointMetadata:
    encoding_name: str            # REQUIRED (LAW-11); no fallback
    run_id: str                   # provenance stamp (drives the filename); "" on a legacy read
    step: int
    commit_sha: str               # "unknown" outside a git checkout (never blocks a write)
    created_utc: str              # ISO-8601 Z; written ONCE, immutable; "" on a bare legacy read
    arch: ModelArch | None        # the WP9 declared dataclass — the SOLE arch source at load
    corpus_sha256: str | None = None


@dataclass(frozen=True)
class Checkpoint:
    schema_version: int
    kind: str                     # "full" | "weights"
    model_state: dict[str, torch.Tensor]
    metadata: CheckpointMetadata
    config: dict
    optimizer_state: dict | None = None
    scaler_state: dict | None = None
    scheduler_state: dict | None = None


@dataclass(frozen=True)
class LrProvenance:
    declared: float | None
    baked: float | None
    effective: float | None
    override_ignored: bool


# ── Stamp helpers ──────────────────────────────────────────────────────────────────────
def _resolve_commit_sha() -> str:
    """`git rev-parse HEAD` — best-effort; "unknown" outside a git checkout. Never raises;
    a metadata write must not be blocked by VCS state (repo_design §6 R3)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
            timeout=2.0,
        )
        return out.decode("ascii", errors="replace").strip() or "unknown"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def _now_iso() -> str:
    return (
        _datetime.datetime.now(_datetime.UTC).replace(tzinfo=None).isoformat() + "Z"
    )


def _arch_to_dict(arch: ModelArch) -> dict[str, Any]:
    """Serialize a declared arch dataclass to a plain dict of primitives (J14) so the whole
    v2 payload round-trips under `torch.load(weights_only=True)` — NOT a pickled dataclass.
    The `representation` field is the on-load discriminator ("grid"→CnnArch, "graph"→GnnArch)."""
    return dataclasses.asdict(arch)


def _arch_from_dict(d: Mapping[str, Any]) -> ModelArch:
    """Rehydrate a serialized arch dict back to CnnArch|GnnArch, dispatching on the
    `representation` discriminator (no shape-inference — the SOLE arch source is this dict)."""
    d = dict(d)
    rep = d.get("representation")
    if rep == "graph":
        return GnnArch(**d)
    if rep == "grid":
        ic = d.get("input_channels")
        if ic is not None:
            d["input_channels"] = tuple(int(x) for x in ic)
        return CnnArch(**d)
    raise RepresentationMismatch(
        f"serialized arch has representation={rep!r} — expected 'grid' or 'graph'."
    )


def _stamp_name(value: Any) -> Any:
    """Normalize an encoding stamp value: a `{'version'|'name': X}` dict → X; else the value."""
    if isinstance(value, dict):
        return value.get("version") or value.get("name")
    return value


def _wire_signature(spec: Any) -> tuple[int, int, int]:
    """The registry-spec input-surface tuple that determines tensor shapes (J6):
    `(plane count, feature_len, policy width)` = `(n_planes, state_stride, policy_logit_count)`.
    Two encodings have EQUAL wire signature iff all three match."""
    return (int(spec.n_planes), int(spec.state_stride), int(spec.policy_logit_count))


# ── Content hash + filename ────────────────────────────────────────────────────────────
def _hash_update(h: hashlib._Hash, obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        h.update(b"\x01T")
        t = obj.detach().cpu().contiguous()
        h.update(repr(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        try:
            h.update(t.numpy().tobytes())
        except (TypeError, RuntimeError, ValueError):
            h.update(t.to(torch.float64).numpy().tobytes())
    elif isinstance(obj, Mapping):
        h.update(b"\x01D")
        for k in sorted(obj.keys(), key=repr):
            h.update(repr(k).encode())
            _hash_update(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"\x01L")
        for x in obj:
            _hash_update(h, x)
    else:
        h.update(b"\x01P")
        h.update(repr(obj).encode())


def content_sha8(payload: Mapping[str, Any]) -> str:
    """Deterministic first-8-hex content hash over a key-ordered serialization of the whole
    v2 payload (model_state tensors + metadata + config + state blobs). Stable across
    save/load round-trips; a one-byte model_state mutation changes it (T-CK-09)."""
    h = hashlib.sha256()
    _hash_update(h, payload)
    return h.hexdigest()[:8]


def checkpoint_filename(run_id: str, step: int, sha8: str) -> str:
    return f"{run_id}_{step:08d}_{sha8}.ckpt"


# ── O3b reject ─────────────────────────────────────────────────────────────────────────
def _reject_killed_prefixes(model_state: Mapping[str, Any]) -> None:
    """REJECT any state dict carrying a killed-branch prefix (WP9 O3b / F-04/F-05). Fires on
    BOTH loader surfaces — a stamped v2 can STRUCTURALLY carry a killed key, so the scan runs
    on the v2 read path too. NEVER reconstructs a pool."""
    hit = [k for k in model_state if isinstance(k, str) and k.startswith(KILLED_PREFIXES)]
    if hit:
        raise RepresentationMismatch(
            f"checkpoint state_dict carries killed-branch keys {hit[:3]} (prefixes "
            f"{KILLED_PREFIXES}); the PMA cluster_pool / pma_global global_encoder / "
            "gpool_bias_branch branches were FALSIFIED and DELETED (WP9 O3b, F-04/F-05) — "
            "the loader REJECTS them and NEVER reconstructs a pool."
        )


# ── Metadata build (immutable stamp) ───────────────────────────────────────────────────
def _build_stamped_metadata(metadata_kwargs: Mapping[str, Any], step: int) -> dict[str, Any]:
    """Build the v2 metadata block, stamping `created_utc`/`commit_sha` ONCE. Refuses a
    `metadata_kwargs` carrying those immutable fields (a re-stamp from a loaded envelope —
    F-12/LAW-12) and an unresolvable `encoding_name` (LAW-11)."""
    md = dict(metadata_kwargs)
    if "created_utc" in md or "commit_sha" in md:
        raise CheckpointStampError(
            "re-stamp refused: created_utc/commit_sha are stamped exactly once by save "
            "(LAW-12); a metadata_kwargs carrying them re-derives a stamp from a loaded "
            "envelope (the self-perpetuating F-12 bug)."
        )
    enc = md.get("encoding_name")
    if not enc or not isinstance(enc, str):
        raise CheckpointStampError(
            f"unstampable: metadata.encoding_name is REQUIRED (LAW-11), got {enc!r}; "
            "an artifact that cannot be stamped cannot be written."
        )
    run_id = md.get("run_id")
    if not run_id or not isinstance(run_id, str):
        raise CheckpointStampError(f"unstampable: metadata.run_id is required, got {run_id!r}.")
    arch = md.get("arch")
    if arch is None:
        raise CheckpointStampError("unstampable: metadata.arch (the declared CnnArch/GnnArch) is required.")
    return {
        "encoding_name": enc,
        "run_id": run_id,
        "step": int(step),
        "commit_sha": _resolve_commit_sha(),
        "created_utc": _now_iso(),
        "arch": _arch_to_dict(arch),
        "corpus_sha256": md.get("corpus_sha256"),
    }


def _assemble_payload(
    kind: str,
    model_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    optimizer_state: Any,
    scaler_state: Any,
    scheduler_state: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "model_state": dict(model_state),
        "metadata": dict(metadata),
        "config": dict(config),
    }
    if kind == "full":
        payload["optimizer_state"] = optimizer_state
        payload["scaler_state"] = scaler_state
        payload["scheduler_state"] = scheduler_state
    return payload


# ── Write path ─────────────────────────────────────────────────────────────────────────
def _write_v2_payload(
    *,
    model_state: Mapping[str, Any],
    optimizer_state: Any,
    scaler_state: Any,
    scheduler_state: Any,
    step: int,
    config: Mapping[str, Any],
    metadata_kwargs: Mapping[str, Any],
    checkpoint_dir: str | Path,
    kind: str,
    allow_quarantine: bool,
) -> Path:
    global persist_errors_total
    # 1. config schema-validated on write (repo_design §6) — raises before any file exists.
    RunConfig.model_validate(dict(config))
    # 2. immutable stamp (unstampable → quarantine under the survive-run flag, else raise).
    try:
        metadata = _build_stamped_metadata(metadata_kwargs, step)
    except CheckpointStampError:
        if not allow_quarantine:
            raise  # T-CK-05/10/11 — an unstampable save writes nothing.
        return _write_quarantine(
            model_state, kind, config, optimizer_state, scaler_state, scheduler_state,
            step, metadata_kwargs, checkpoint_dir,
        )
    # 3. assemble → content hash → provenance filename → persist-fatal write.
    payload = _assemble_payload(
        kind, model_state, metadata, config, optimizer_state, scaler_state, scheduler_state,
    )
    sha8 = content_sha8(payload)
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / checkpoint_filename(metadata["run_id"], step, sha8)
    try:
        torch.save(payload, path)
    except Exception:
        persist_errors_total += 1  # LAW-14: count + abort, never `except: pass`.
        raise
    return path


def _write_quarantine(
    model_state: Mapping[str, Any],
    kind: str,
    config: Mapping[str, Any],
    optimizer_state: Any,
    scaler_state: Any,
    scheduler_state: Any,
    step: int,
    metadata_kwargs: Mapping[str, Any],
    checkpoint_dir: str | Path,
) -> Path:
    """Survive-run clause (repo_design §6 / C4.5): an unstampable save writes
    `<path>.quarantine` (NEVER a canonical `.ckpt`) and increments the persist counter."""
    global persist_errors_total
    md = dict(metadata_kwargs)
    q_meta = {
        "encoding_name": md.get("encoding_name") or "",
        "run_id": md.get("run_id") or "unknown",
        "step": int(step),
        "commit_sha": _resolve_commit_sha(),
        "created_utc": _now_iso(),
        "arch": _arch_to_dict(md["arch"]) if md.get("arch") is not None else None,
        "corpus_sha256": md.get("corpus_sha256"),
    }
    payload = _assemble_payload(
        kind, model_state, q_meta, config, optimizer_state, scaler_state, scheduler_state,
    )
    sha8 = content_sha8(payload)
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    qpath = cdir / (checkpoint_filename(q_meta["run_id"], step, sha8) + ".quarantine")
    torch.save(payload, qpath)
    persist_errors_total += 1
    return qpath


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scaler: Any,
    scheduler: Any,
    step: int,
    config: Mapping[str, Any],
    metadata_kwargs: Mapping[str, Any],
    checkpoint_dir: str | Path,
    kind: str = "full",
    allow_quarantine: bool = False,
) -> Path:
    """Write an envelope-v2 checkpoint `{run_id}_{step:08d}_{sha8}.ckpt`.

    Validates `config` against config-schema v1, stamps metadata ONCE (encoding_name
    REQUIRED → else CheckpointStampError), computes the content hash, and persist-fatally
    writes. A weights save carries model_state + metadata only (no optimizer/scaler/scheduler).
    """
    base_model = getattr(model, "_orig_mod", model)
    model_state = base_model.state_dict()
    if kind == "full":
        optimizer_state = optimizer.state_dict() if optimizer is not None else None
        scaler_state = scaler.state_dict() if scaler is not None else None
        scheduler_state = scheduler.state_dict() if scheduler is not None else None
    else:
        optimizer_state = scaler_state = scheduler_state = None
    return _write_v2_payload(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scaler_state=scaler_state,
        scheduler_state=scheduler_state,
        step=step,
        config=config,
        metadata_kwargs=metadata_kwargs,
        checkpoint_dir=checkpoint_dir,
        kind=kind,
        allow_quarantine=allow_quarantine,
    )


# ── Read path (v2) ─────────────────────────────────────────────────────────────────────
def _verify_provenance(
    path: Path, payload: Mapping[str, Any], metadata: Mapping[str, Any], expected_run_id: str | None
) -> None:
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3:
        raise CheckpointStampError(
            f"{path.name}: not a v2 provenance filename {{run_id}}_{{step:08d}}_{{sha8}}.ckpt."
        )
    fn_run_id, fn_step_str, fn_sha8 = parts
    md_run_id = metadata.get("run_id")
    md_step = metadata.get("step")
    if md_run_id != fn_run_id:
        raise CheckpointStampError(
            f"{path.name}: embedded metadata.run_id {md_run_id!r} disagrees with filename "
            f"run_id {fn_run_id!r} (provenance re-verify)."
        )
    try:
        # An absent metadata.step lands in the same mismatch-raise as a non-numeric one.
        step_ok = md_step is not None and int(md_step) == int(fn_step_str)
    except (TypeError, ValueError):
        step_ok = False
    if not step_ok:
        raise CheckpointStampError(
            f"{path.name}: embedded metadata.step {md_step!r} disagrees with filename step "
            f"{fn_step_str!r} (provenance re-verify)."
        )
    actual = content_sha8(payload)
    if actual != fn_sha8:
        raise CheckpointStampError(
            f"{path.name}: content hash {actual} disagrees with the filename sha8 {fn_sha8} "
            "— the payload was tampered with after stamping."
        )
    if expected_run_id is not None and md_run_id != expected_run_id:
        raise CheckpointStampError(
            f"{path.name}: run_id {md_run_id!r} != expected_run_id {expected_run_id!r}."
        )


def _rehydrate_metadata(metadata: Mapping[str, Any]) -> CheckpointMetadata:
    arch = _arch_from_dict(metadata["arch"]) if metadata.get("arch") is not None else None
    return CheckpointMetadata(
        encoding_name=metadata.get("encoding_name", ""),
        run_id=metadata.get("run_id", ""),
        step=int(metadata.get("step", 0)),
        commit_sha=metadata.get("commit_sha", ""),
        created_utc=metadata.get("created_utc", ""),
        arch=arch,
        corpus_sha256=metadata.get("corpus_sha256"),
    )


def _config_encoding(config: Any) -> Any:
    if isinstance(config, dict):
        ident = config.get("identity")
        if isinstance(ident, dict):
            return ident.get("encoding")
    return None


def load_checkpoint(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
    device: Any = None,
    declared_encoding: Any = None,
    decode_override: Any = None,
) -> Checkpoint:
    """Read a v2 envelope. `torch.load(weights_only=True)`; re-verify run_id + content-hash
    vs the filename (provenance); REJECT any killed-branch prefix (O3b); reconcile
    declared_encoding (assert) / decode_override (loud, never raises) / stamp sources
    (disagree → raise); re-validate config; resolve arch from `metadata.arch`. NEVER
    re-stamps, always weights-only (no pickle-exec fallback), NEVER auto-upgrades a
    legacy/bare payload."""
    path = Path(path)
    payload = torch.load(path, weights_only=True, map_location="cpu")

    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        got = payload.get("schema_version") if isinstance(payload, dict) else type(payload).__name__
        raise CheckpointStampError(
            f"{path.name}: not a v2 checkpoint envelope (schema_version={got!r}). A legacy or "
            "bare-state-dict payload must be read via load_legacy_weights — a v2 stamp is never "
            "auto-minted on read (LAW-12)."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointStampError(f"{path.name}: v2 envelope missing its metadata block.")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise CheckpointStampError(f"{path.name}: v2 envelope missing model_state.")

    _verify_provenance(path, payload, metadata, expected_run_id)
    _reject_killed_prefixes(model_state)

    if declared_encoding is not None and decode_override is not None:
        raise ValueError(
            "declared_encoding and decode_override are mutually exclusive — declared_encoding "
            "is an ASSERTION (raises on stamp disagreement) while decode_override is a "
            "deliberate cross-decode (never raises). Pass exactly one."
        )

    stamp_enc = metadata.get("encoding_name")
    config = payload.get("config")
    cfg_enc = _config_encoding(config)
    if stamp_enc is not None and cfg_enc is not None and _stamp_name(stamp_enc) != _stamp_name(cfg_enc):
        raise CheckpointStampError(
            f"{path.name}: stamp sources disagree: metadata.encoding_name={stamp_enc!r} vs "
            f"config.identity.encoding={cfg_enc!r} — refusing to silently pick a side."
        )

    if declared_encoding is not None:
        declared_name = _stamp_name(declared_encoding)
        stamp = _stamp_name(stamp_enc) if stamp_enc is not None else None
        if stamp is not None and declared_name != stamp:
            raise DeclaredEncodingMismatchError(
                f"declared_encoding={declared_name!r} disagrees with the checkpoint stamp "
                f"metadata.encoding_name={stamp!r} ({path.name}); refusing to silently override."
            )

    if decode_override is not None:
        override_name = _stamp_name(decode_override)
        stamp = _stamp_name(stamp_enc) if stamp_enc is not None else None
        if stamp is not None and override_name != stamp:
            _LOG.warning(
                "encoding_decode_override: checkpoint=%s stamp=%s decode_as=%s (override wins; "
                "stamp disagrees — never raises)", path.name, stamp, override_name,
            )
        else:
            _LOG.info(
                "encoding_decode_override: checkpoint=%s stamp=%s decode_as=%s",
                path.name, stamp, override_name,
            )

    if isinstance(config, dict):
        RunConfig.model_validate(config)  # config schema-validated on read (repo_design §6)

    kind = payload.get("kind")
    if not isinstance(kind, str):
        # The v2 writer always stamps kind ("full"/"weights"); a missing one is corruption.
        raise CheckpointStampError(f"{path.name}: v2 envelope missing its kind field.")

    return Checkpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=kind,
        model_state=model_state,
        metadata=_rehydrate_metadata(metadata),
        config=config if isinstance(config, dict) else {},
        optimizer_state=payload.get("optimizer_state"),
        scaler_state=payload.get("scaler_state"),
        scheduler_state=payload.get("scheduler_state"),
    )


# ── Read path (legacy / anchor import) — the THREE real pre-v2 shapes ──────────────────
def load_legacy_weights(
    path: str | Path,
    *,
    declared_encoding: Any = None,
    decode_override: Any = None,
) -> Checkpoint:
    """Distinct read surface for pre-v2 artifacts (bare state_dict / light envelope / full-v1).
    `torch.load(weights_only=True)`. Resolve arch from the declared/stamped `encoding_name` →
    registry spec → `arch_from_spec_and_config` (NEVER shape-sniffs — an unregistered encoding
    raises loudly). Apply the SAME O3b killed-prefix REJECT. Returns a Checkpoint with NO
    synthetic run_id/content-hash/created_utc (a legacy anchor is never re-stamped on read;
    LAW-12); a full-v1 envelope reads via the old→v2 field map (training_date→created_utc,
    model_architecture/variant→arch, train_config_path DROPPED)."""
    if declared_encoding is not None and decode_override is not None:
        raise ValueError("declared_encoding and decode_override are mutually exclusive.")

    path = Path(path)
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(raw, dict):
        raise CheckpointStampError(f"{path.name}: legacy payload is not a dict.")

    # Shape sniff (ONCE): envelope {model_state, ...} vs a BARE state dict.
    if isinstance(raw.get("model_state"), dict):
        model_state = raw["model_state"]
        raw_meta = raw.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        is_full = raw.get("optimizer_state") is not None and raw.get("scaler_state") is not None
    else:
        model_state = raw
        meta = {}
        is_full = False

    _reject_killed_prefixes(model_state)

    # Resolve the encoding: declared wins, else the embedded stamp. NEVER shape-sniffs.
    resolved_enc = _stamp_name(declared_encoding) if declared_encoding is not None else meta.get("encoding_name")
    if not resolved_enc or not isinstance(resolved_enc, str):
        raise CheckpointStampError(
            f"{path.name}: cannot resolve a legacy encoding (no declared_encoding, no "
            "metadata['encoding_name']); pass an explicit declared_encoding — never shape-sniff."
        )
    spec = lookup(resolved_enc)  # raises EncodingRegistryError (loud) on an unregistered name.

    raw_config = raw.get("config")
    embedded_config = raw_config if isinstance(raw_config, dict) else {}
    if embedded_config:
        RunConfig.model_validate(embedded_config)  # config snapshot re-validated
    arch = arch_from_spec_and_config(spec, embedded_config)

    # old v1 metadata → v2 field map (read-only; NEVER writes back / mints v2 provenance).
    metadata = CheckpointMetadata(
        encoding_name=resolved_enc,                          # verbatim (or declared)
        run_id="",                                           # SYNTHESIZED-NEVER on a legacy read
        step=int(raw.get("step", 0)) if isinstance(raw.get("step"), int) else 0,
        commit_sha=meta.get("commit_sha", ""),               # verbatim
        created_utc=meta.get("training_date", ""),           # training_date → created_utc (verbatim)
        arch=arch,                                           # resolved, not sniffed
        corpus_sha256=meta.get("corpus_sha256"),             # verbatim, optional
    )
    kind = "full" if is_full else "weights"
    return Checkpoint(
        schema_version=1,
        kind=kind,
        model_state=model_state,
        metadata=metadata,
        config=embedded_config,
        optimizer_state=raw.get("optimizer_state") if is_full else None,
        scaler_state=raw.get("scaler_state") if is_full else None,
        scheduler_state=raw.get("scheduler_state") if is_full else None,
    )


# ── Weights-only strip + re-stamp (the ONE sanctioned encoding-change/stamp path) ──────
def strip_and_restamp(
    src_path: str | Path,
    *,
    new_encoding: str,
    run_id: str,
    checkpoint_dir: str | Path,
    declared_encoding: Any = None,
    step: int = 0,
) -> Path:
    """Give a legacy/v2 source a FRESH single v2 stamp — gated on wire-signature equality
    (T-CK-21/33). Stamped ONCE from the declared encoding + arch, NEVER from a loaded config.
    A wire-signature mismatch (e.g. v6 8-plane vs v6_live2_ls 4-plane) raises."""
    src_path = Path(src_path)
    raw = torch.load(src_path, weights_only=True, map_location="cpu")
    if isinstance(raw, dict) and raw.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        model_state = raw["model_state"]
        old_enc = _stamp_name((raw.get("metadata") or {}).get("encoding_name"))
    elif isinstance(raw, dict) and isinstance(raw.get("model_state"), dict):
        model_state = raw["model_state"]
        old_enc = _stamp_name(declared_encoding) if declared_encoding is not None else \
            (raw.get("metadata") or {}).get("encoding_name")
    elif isinstance(raw, dict):
        model_state = raw
        old_enc = _stamp_name(declared_encoding)
    else:
        raise CheckpointStampError(f"{src_path.name}: unsupported source payload for strip_and_restamp.")

    _reject_killed_prefixes(model_state)
    if not old_enc:
        raise CheckpointStampError(
            f"{src_path.name}: cannot resolve the source encoding for the wire-signature gate; "
            "pass declared_encoding."
        )
    old_spec = lookup(old_enc)
    new_spec = lookup(new_encoding)
    if _wire_signature(old_spec) != _wire_signature(new_spec):
        raise CheckpointStampError(
            f"weights-strip refused: wire signature mismatch — {old_enc} {_wire_signature(old_spec)} "
            f"!= {new_encoding} {_wire_signature(new_spec)}. The strip succeeds only on equality "
            "(the ONE sanctioned encoding-change path)."
        )

    arch = arch_from_spec_and_config(new_spec, {})
    synth_config = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": 0,
        "identity": {"encoding": new_encoding, "representation": new_spec.representation},
        # WP11-A schema extension: eval.gate/eval.ladder are now required (design §c.1).
        # This synthetic config exists only to satisfy the schema-validate-on-write gate
        # for a strip/restamp utility payload — placeholder values, same posture as the
        # pre-existing seed=0/run_id=<caller> placeholders above.
        "eval": {
            "random_model_sims": 1, "sealbot_model_sims": 1, "kraken_model_sims": 1,
            "strix_model_sims": 1, "random_floor_games": 0, "worker_device": "cpu",
            "round_timeout_sec": 1.0, "worker_kill_grace_sec": 1.0,
            "gate": {
                "stride": 1, "screen_games": 1, "confirm_games": 1, "promotion_winrate": 0.55,
                "screen_confirm_lo": 0.44, "deploy_sims": 1, "opening_book": "book_v1_s20260625_p4",
                "bootstrap_resamples": 1, "min_distinct_per_pair": 1, "seed_base": 1,
            },
            "ladder": {
                "rungs": [{"name": "r0", "bot": "random", "variant": "raw", "depth": None,
                          "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                          "deploy_matched": True, "games_max": 1}],
                "round_games": 1, "min_games_per_active_rung": 1, "graduation_wr_lower_ci": 0.9,
                "graduation_consec_rounds": 1, "activation_wr_lower_ci": 0.5,
                "calibration_every_k_rounds": 1, "calibration_games": 1,
                "bootstrap_resamples": 1, "bootstrap_ci_level": 0.95,
                "bt_prior_games": 1.0, "bootstrap_seed": 1,
            },
        },
        # WPSC Phase 2 SC-A1/A2: `train:`/expanded `selfplay:` are now required RunConfig
        # sections — this synthetic config exists only to satisfy the schema-validate-on-write
        # gate for a strip/restamp utility payload; placeholder values, same posture as the
        # pre-existing seed=0/run_id=<caller> placeholders above (zero-behavior-change mint
        # values, DESIGN_P2.md §1.1/§1.2).
        "train": {
            "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "fp16": True,
            "amp_dtype": "fp16", "lr_schedule": "cosine", "total_steps": 1_000_000,
            "scheduler_t_max": None, "eta_min": 5e-4, "min_lr": None,
            "checkpoint_interval": 0, "actor_sync_cadence_steps": 1,
            "max_train_steps": 1_000_000,  # WPAX S-4: required run-length key
            # WPAX Phase D (R65/R80): required key, no code-side default. `None` is the
            # EXPLICIT disarmed posture, which is the correct placeholder for a payload
            # that is not a run: a synthetic config must never claim an armed abort.
            "draw_rate_abort": None,
            # WPMINT Phase K-B (CARD-COORD-KNOBS, R78/R80): the 19 step-coordinator knobs are
            # required `train.*` keys now. Placeholder values, same posture as the rest of this
            # payload — they are the template's own values because a synthetic config that is
            # not a run must not invent a different run shape.
            "eval_interval": 1000, "log_interval": 1000, "buffer_save_interval": 0,
            "min_buf_size": 1, "replay_capacity": 100_000, "replay_capacity_schedule": [],
            "training_steps_per_game": 1.0, "max_train_burst": 1, "batch_size": 256,
            "augment": False, "recency_weight": 0.0, "mixing_initial_w": 0.0,
            "mixing_min_w": 0.0, "mixing_decay_steps": 1.0, "hard_gn_threshold": 1e9,
            "hard_gn_min_steps": 3, "terminal_eval_enabled": True, "bot_batch_share": 0.0,
            "selfplay_stall_timeout_sec": 1800.0,
            "completed_q_values": False,
            "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
            "draw_reward": -0.5, "ply_cap_value": -0.5, "policy_prune_frac": 0.0,
            "entropy_reg_weight": 0.0, "aux_opp_reply_weight": 0.0,
            "uncertainty_weight": 0.0, "ownership_weight": 0.0, "threat_weight": 0.0,
            "aux_chain_weight": 0.0, "ply_index_weight": 0.0, "threat_pos_weight": 1.0,
        },
        # WPSC Phase 2 SC-A2: `selfplay:` gains mcts:/playout_cap: sub-blocks + many new
        # required scalars; `legal_move_radius_schedule` is GONE (DESIGN_P2.md §5); the
        # registry alone is the radius authority, so this synthetic weights-strip payload
        # never needed the key to begin with. `inference:` is a new required top-level
        # section. Placeholder values, same posture as the eval block above.
        "selfplay": {
            "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
            "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
            "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
            "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
            "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
            "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
            "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
            "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
            "instrumentation_enabled": False,
            "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                     "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                     "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                     "dirichlet_enabled": True},
            "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                            "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                            "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                            "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
        },
        "inference": {
            "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
            "compile_inference": False, "compile_inference_mode": "default",
            "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
        },
        # WPSC Phase 2 SC-A3: `monitor:` is now a required RunConfig section — placeholder
        # values, same posture as the eval/train/selfplay blocks above (DESIGN_P2.md §4.2).
        "monitor": {
            "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
            "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
            "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
            "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
            "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
            "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
            "axis_warn": 0.45, "axis_alert": 0.50,
            "heartbeat_deadline_train_step_sec": 1800.0,
            "heartbeat_deadline_inference_dispatch_sec": 1800.0,
            "heartbeat_deadline_selfplay_drain_sec": 1800.0,
            "heartbeat_deadline_eval_round_sec": 1800.0,
            "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
            "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
            "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
            "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
            "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
            "drain": {
                "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
                "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
            },
        },
    }
    return _write_v2_payload(
        model_state=model_state,
        optimizer_state=None,
        scaler_state=None,
        scheduler_state=None,
        step=step,
        config=synth_config,
        metadata_kwargs={"encoding_name": new_encoding, "run_id": run_id, "arch": arch},
        checkpoint_dir=checkpoint_dir,
        kind="weights",
        allow_quarantine=False,
    )


# ── Resume-precedence partners (legacy flat-config shape; §c.2) ────────────────────────
def apply_config_overrides_f1(
    baked: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
    declared_keys: frozenset | set | None,
    *,
    sink: Any = None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """Apply `overrides` onto the checkpoint-baked config with the CONFRES F1(A) defer rule.

    A DECLARED key wins (E0, incl. an explicit `null`); a base-inherited (non-declared) key
    that the checkpoint BAKED DEFERS to the baked value (+ a `resume_base_default_deferred_to_baked`
    warning through the injected sink when they differ). Returns `(resolved_config, deferred_keys)`.
    Weights-only path (baked is None) or a legacy call (declared_keys is None) → verbatim update
    (byte-pure), nothing deferred.
    """
    if baked is None or declared_keys is None:
        resolved = dict(baked or {})
        resolved.update(overrides)
        return resolved, frozenset()

    resolved = dict(baked)
    _launch_mechanism = (
        "total_steps", "scheduler_t_max", "torch_compile", "torch_compile_mode",
        "allow_fresh_scheduler",
    )
    declared = frozenset(declared_keys) | {k for k in _launch_mechanism if k in overrides}
    deferred: set[str] = set()
    for key, override_val in overrides.items():
        if key in declared:
            resolved[key] = override_val
            continue
        if key in baked:
            baked_val = baked[key]
            if override_val != baked_val:
                deferred.add(key)
                emit_via(
                    sink,
                    {
                        "event": "resume_base_default_deferred_to_baked",
                        "knob": key,
                        "base_default": override_val,
                        "checkpoint_baked": baked_val,
                    },
                )
        else:
            resolved[key] = override_val
    return resolved, frozenset(deferred)


def resolve_lr_provenance(
    declared: float | None,
    baked: float | None,
    effective: float | None,
    *,
    rel_tol: float = 1e-9,
) -> LrProvenance:
    """CONFRES S1 — loud declared-vs-baked LR on a full-checkpoint resume. `override_ignored`
    is True only when a declared lr is present AND differs (beyond `rel_tol`) from the
    checkpoint's baked initial lr (the operator asked for an lr the resume silently drops)."""
    override_ignored = (
        declared is not None
        and baked is not None
        and abs(float(declared) - float(baked)) > rel_tol * max(abs(float(baked)), 1.0)
    )
    return LrProvenance(
        declared=None if declared is None else float(declared),
        baked=None if baked is None else float(baked),
        effective=None if effective is None else float(effective),
        override_ignored=override_ignored,
    )


# ── Resume path (builds a Trainer — Slice 2 consumer; lazy `cls`) ──────────────────────
def resume_trainer(
    cls: type,
    path: str | Path,
    *,
    fallback_config: Mapping[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    declared_keys: frozenset | set | None = None,
    sink: Any = None,
    device: Any = None,
) -> Any:
    """Full/weights resume path (folded `trainer_ckpt_load.load_checkpoint`): build_net from
    `metadata.arch`, restore optimizer/scaler/scheduler/step on a full checkpoint under the
    F1(A)/E0 frozen-key rules; lr is resume-state-owned (loud on an ignored declared override).

    Slice-2 consumer: `cls` is the `mantis.train.trainer.core.Trainer` class (passed by the
    caller so this module has no top-level trainer edge). Gated by T-CK-18/19 at Slice 2.
    """
    path = Path(path)
    raw = torch.load(path, weights_only=True, map_location="cpu")
    is_v2 = isinstance(raw, dict) and raw.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
    ck = load_checkpoint(path, device=device) if is_v2 else load_legacy_weights(path)

    arch = ck.metadata.arch
    if arch is None:
        raise CheckpointStampError(f"{path.name}: no arch on the loaded metadata — cannot rebuild the net.")
    model = build_net(arch)
    # Dev#2 net-load parity: the OLD resume path (`trainer_ckpt_load.load_checkpoint` →
    # `_load_state_dict_strict`, lenient `strict=False`) loads a subset anchor leniently — a
    # real bare anchor is a strict SUBSET of the build_net key set (T-CK-25), so `strict=True`
    # would spuriously reject it. `strict=False` reproduces the old resume load mode exactly.
    model.load_state_dict(ck.model_state, strict=False)

    # CONFRES F1(A)/E0 (S-2, DESIGN_P2.md §6): actually APPLY config_overrides onto the
    # checkpoint-baked config instead of dropping it on the floor. `baked_config` is the
    # already-schema-validated `train.*`-nested snapshot; a DECLARED top-level key (e.g.
    # the whole `"train"` section) wins outright, a base-inherited (non-declared) key that
    # differs from baked DEFERS to baked + emits `resume_base_default_deferred_to_baked`.
    baked_config = ck.config if ck.config else None
    if config_overrides:
        resolved_config, deferred = apply_config_overrides_f1(
            baked_config, config_overrides, declared_keys, sink=sink,
        )
    else:
        resolved_config, deferred = (
            dict(baked_config) if baked_config else dict(fallback_config or {}),
            frozenset(),
        )
    config = resolved_config
    # Pass the DECLARED arch (metadata.arch) so the Trainer stamps the same arch on re-save
    # (never re-derives it); the sink threads through for resume-time events (T-CK-18).
    trainer = cls(model, config, arch=arch, checkpoint_dir=path.parent, device=device, sink=sink)
    trainer.f1_deferred_keys = deferred

    # CONFRES S1 (S-2): lr is resume-state-owned — a declared `lr` is never allowed to win
    # on a full checkpoint resume, but an operator who declared one anyway must be warned
    # loudly rather than silently ignored. `baked_lr` reads the NESTED `train.lr` (the flat
    # top-level `lr` key no longer exists post-SC-A1); `declared_lr` still comes from a bare
    # flat `"lr"` key in `config_overrides` when `"lr"` is a declared key (E0's own shape).
    declared_lr = (
        (config_overrides or {}).get("lr")
        if declared_keys and "lr" in declared_keys else None
    )
    baked_train = baked_config.get("train") if isinstance(baked_config, dict) else None
    baked_lr = baked_train.get("lr") if isinstance(baked_train, dict) else None
    lr_prov = resolve_lr_provenance(declared=declared_lr, baked=baked_lr, effective=trainer.hp.lr)
    if lr_prov.override_ignored:
        emit_via(sink, {
            "event": "resume_lr_override_ignored",
            "declared": lr_prov.declared, "baked": lr_prov.baked, "effective": lr_prov.effective,
        })

    is_full = ck.kind == "full"
    trainer.loaded_from_full_checkpoint = is_full
    if is_full:
        if ck.optimizer_state is not None:
            trainer.optimizer.load_state_dict(ck.optimizer_state)
        if ck.scaler_state is not None:
            trainer.scaler.load_state_dict(ck.scaler_state)
        if getattr(trainer, "scheduler", None) is not None:
            if ck.scheduler_state is None:
                allow_fresh = bool((config_overrides or {}).get("allow_fresh_scheduler", False))
                if not allow_fresh:
                    raise ValueError(
                        "scheduler_state missing from checkpoint; cannot resume. Pass "
                        "config_overrides['allow_fresh_scheduler']=True to rebuild the scheduler "
                        "from scratch."
                    )
                emit_via(sink, {"event": "scheduler_state_missing_fresh_start"})
            else:
                trainer.scheduler.load_state_dict(ck.scheduler_state)
        trainer.step = ck.metadata.step
    else:
        trainer.step = ck.metadata.step
    return trainer
