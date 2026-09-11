"""The ONE checkpoint loader + envelope-v2 writer/reader.

>300 justify: four old loaders collapse here because they share one envelope — the v2 write
path, the ONE read path, the stamp and provenance guards, the killed-prefix REJECT, the
resume-precedence helpers and the weights-strip path. All shape-inference is DELETED: arch
travels on `metadata.arch` -> `build_net`, never re-derived.
"""
from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis.config.schema import ARCH_SCOPED_KEYS, RunConfig
from mantis.encoding import lookup
from mantis.model import (
    ARCH_KINDS,
    GnnArch,
    ModelArch,
    RepresentationMismatch,
    build_net,
    declared_arch_kind,
    select_arch,
)
from mantis.train.bundle import atomic_write
from mantis.train.emit import emit_via
from mantis.util.git import head_sha

_LOG = logging.getLogger(__name__)

# Envelope axis, DISTINCT from the config's own schema_version=1.
CHECKPOINT_SCHEMA_VERSION = 2

# State-dict key prefixes of FALSIFIED-and-DELETED branches; the loader REJECTS any state
# dict carrying one and NEVER reconstructs a PMA/gpool pool.
KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")

# A swallowed persist failure is banned; a FAILED write increments this. The watchdog's
# persist-fatal rule is the literal `> 0` (rc 43), so ONLY run-fatal facts may feed it.
persist_errors_total = 0

# A quarantine write is the survive-run clause WORKING — deliberately NOT run-fatal — so it
# counts HERE. Feeding it to `persist_errors_total` would kill the run it exists to save.
quarantine_writes_total = 0


class CheckpointStampError(RuntimeError):
    """Unstamped save / re-stamp / provenance / content-hash / stamp-disagree failure."""


class DeclaredEncodingMismatchError(ValueError):
    """A caller-declared encoding disagrees with the checkpoint's own trusted stamp."""


class ResumeIdentityMismatchError(ValueError):
    """A resume's EFFECTIVE identity block differs from the checkpoint's own.

    The net is rebuilt from the checkpoint's STAMPED arch while the config claims another, and
    every later save re-stamps the arch it was handed, so nothing downstream can tell.
    """


class ResumeTargetSemanticsError(ValueError):
    """A resume changes what a STORED replay row MEANS.

    These leaves build no net; they decide whether a stored row is a visit-count distribution
    or a completed improved policy, and which loss applies to it. A resume restores the ring,
    so moving one trains old and new rows under one loss with no provenance to tell them apart.
    """


@dataclass(frozen=True)
class CheckpointMetadata:
    encoding_name: str            # REQUIRED; no fallback
    run_id: str                   # provenance stamp (drives the filename); "" on a legacy read
    step: int
    commit_sha: str               # "unknown" outside a git checkout (never blocks a write)
    created_utc: str              # ISO-8601 Z; written ONCE, immutable; "" on a bare legacy read
    arch: ModelArch | None        # the declared dataclass — the SOLE arch source at load
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


def _resolve_commit_sha() -> str:
    """Return `git rev-parse HEAD`, or "unknown" outside a git checkout; never raises."""
    return head_sha(Path(__file__).resolve().parent) or "unknown"


def _now_iso() -> str:
    return (
        _datetime.datetime.now(_datetime.UTC).replace(tzinfo=None).isoformat() + "Z"
    )


#: The serialized discriminator's key and the registry it selects on. `representation` alone
#: stopped discriminating once a second graph arch existed: a V2 stamp rehydrated as `GnnArch`.
_ARCH_KIND_KEY = "arch_kind"
#: THE registry, imported rather than restated: two copies of a discriminator is duplicate authority.
_ARCH_KINDS = ARCH_KINDS

#: What a stamp written BEFORE the discriminator existed resolves to, by representation — a
#: fact about history, not a default. A legacy dict that does not fit its target raises.
_LEGACY_BY_REPRESENTATION: dict[str, type] = {"graph": GnnArch}

#: The NON-BINDING placeholder pair for each arch-scoped block, in a table so the splice stays a
#: loop over `ARCH_SCOPED_KEYS` and a block added there without one fails loudly at the write.
_SYNTH_ARCH_SCOPED: dict[tuple[str, str], dict[str, int]] = {
    ("train", "microbatch_caps"): {"max_edges": 100_000_000, "max_nodes": 4_000_000},
    ("inference", "fused_graph_caps"): {
        "max_fused_edges": 57149441, "max_fused_nodes": 1785921,
    },
}


def _arch_to_dict(arch: ModelArch) -> dict[str, Any]:
    """Serialize a declared arch dataclass to plain primitives so the v2 payload round-trips
    under `torch.load(weights_only=True)`, carrying `arch_kind` as the load discriminator."""
    return {**dataclasses.asdict(arch), _ARCH_KIND_KEY: type(arch).__name__}


def _arch_from_dict(d: Mapping[str, Any]) -> ModelArch:
    """Rehydrate a serialized arch dict, dispatching on `arch_kind` — the SOLE arch source.

    Raises:
        RepresentationMismatch: unknown `arch_kind`; no `arch_kind` with a non-graph
            `representation`; or a legacy dict that does not fit the arch it names.
    """
    d = dict(d)
    kind = d.pop(_ARCH_KIND_KEY, None)
    if kind is not None:
        cls = _ARCH_KINDS.get(str(kind))
        if cls is None:
            raise RepresentationMismatch(
                f"serialized arch has {_ARCH_KIND_KEY}={kind!r}, which this build does not "
                f"know — known kinds are {sorted(_ARCH_KINDS)}. A checkpoint from a newer arch "
                "is REFUSED rather than rebuilt as the nearest thing that fits."
            )
    else:
        rep = d.get("representation")
        cls = _LEGACY_BY_REPRESENTATION.get(str(rep))
        if cls is None:
            raise RepresentationMismatch(
                f"serialized arch has representation={rep!r} and no {_ARCH_KIND_KEY} — "
                "expected 'graph'."
            )
    try:
        return cls(**d)
    except TypeError as exc:
        raise RepresentationMismatch(
            f"serialized arch does not fit {cls.__name__}: {exc}. A stamp is rehydrated as what "
            "it says it is or not at all — never coerced into the nearest member of the union."
        ) from exc


def stamped_arch_kind(metadata: Mapping[str, Any] | None, *, representation: str) -> str:
    """Return which arch kind an ARTIFACT carries, from its stamp alone and never a config. A
    pre-discriminator stamp resolves through `_LEGACY_BY_REPRESENTATION` — a fact about history,
    not a default.

    Raises:
        RepresentationMismatch: the stamp names an `arch_kind` this build does not know, or the
            representation is not 'graph'.
    """
    arch = metadata.get("arch") if isinstance(metadata, Mapping) else None
    kind = arch.get(_ARCH_KIND_KEY) if isinstance(arch, Mapping) else None
    if kind is not None:
        if str(kind) not in _ARCH_KINDS:
            raise RepresentationMismatch(
                f"stamp names {_ARCH_KIND_KEY}={kind!r}, which this build does not know — known "
                f"kinds are {sorted(_ARCH_KINDS)}. A checkpoint from a newer arch is REFUSED "
                "rather than rebuilt as the nearest thing that fits."
            )
        return str(kind)
    cls = _LEGACY_BY_REPRESENTATION.get(str(representation))
    if cls is None:
        raise RepresentationMismatch(
            f"representation={representation!r} — expected 'graph'; a stamp with no "
            f"{_ARCH_KIND_KEY} resolves only through the legacy-by-representation rule."
        )
    return cls.__name__


def _stamp_name(value: Any) -> Any:
    """Normalize an encoding stamp value: a `{'version'|'name': X}` dict → X; else the value."""
    if isinstance(value, dict):
        return value.get("version") or value.get("name")
    return value


def _wire_signature(spec: Any) -> tuple[int, int, int]:
    """Return the spec tuple that determines tensor shapes — `(n_planes, state_stride,
    policy_logit_count)`. Two encodings have EQUAL wire signature iff all three match."""
    return (int(spec.n_planes), int(spec.state_stride), int(spec.policy_logit_count))


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
    """Hash the whole v2 payload to 8 hex chars over a key-ordered serialization: stable across
    save/load round-trips, and a one-byte model_state mutation changes it."""
    h = hashlib.sha256()
    _hash_update(h, payload)
    return h.hexdigest()[:8]


def checkpoint_filename(run_id: str, step: int, sha8: str) -> str:
    return f"{run_id}_{step:08d}_{sha8}.ckpt"


def _reject_killed_prefixes(model_state: Mapping[str, Any]) -> None:
    """REJECT any state dict carrying a killed-branch prefix. Fires on BOTH loader surfaces —
    a stamped v2 can STRUCTURALLY carry a killed key — and NEVER reconstructs a pool."""
    hit = [k for k in model_state if isinstance(k, str) and k.startswith(KILLED_PREFIXES)]
    if hit:
        raise RepresentationMismatch(
            f"checkpoint state_dict carries killed-branch keys {hit[:3]} (prefixes "
            f"{KILLED_PREFIXES}); the PMA cluster_pool / pma_global global_encoder / "
            "gpool_bias_branch branches were FALSIFIED and DELETED (WP9 O3b, F-04/F-05) — "
            "the loader REJECTS them and NEVER reconstructs a pool."
        )


def _build_stamped_metadata(metadata_kwargs: Mapping[str, Any], step: int) -> dict[str, Any]:
    """Build the v2 metadata block, stamping `created_utc`/`commit_sha` ONCE; refuses
    `metadata_kwargs` carrying those (a re-stamp) or an unresolvable `encoding_name`."""
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
        raise CheckpointStampError("unstampable: metadata.arch (the declared arch dataclass) is required.")
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
    # 1. config schema-validated on write — raises before any file exists.
    RunConfig.model_validate(dict(config))
    # 2. immutable stamp (unstampable → quarantine under the survive-run flag, else raise).
    try:
        metadata = _build_stamped_metadata(metadata_kwargs, step)
    except CheckpointStampError:
        if not allow_quarantine:
            raise  # an unstampable save writes nothing.
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
        # Temp file -> fsync -> rename -> fsync(dir): a bare `torch.save(payload, path)` opens
        # the FINAL path, so a kill mid-write leaves a `.ckpt` whose content hash does not match.
        atomic_write(path, lambda handle: torch.save(payload, handle))
    except Exception:
        persist_errors_total += 1  # count + abort, never `except: pass`.
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
    """Write `<path>.quarantine` for an unstampable save and count it as a QUARANTINE, not a
    persist error — a nonzero persist count aborts the run, and this path is survivable."""
    global quarantine_writes_total
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
    atomic_write(qpath, lambda handle: torch.save(payload, handle))
    quarantine_writes_total += 1
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
    """Write an envelope-v2 checkpoint `{run_id}_{step:08d}_{sha8}.ckpt`: schema-validates
    `config`, stamps metadata ONCE, content-hashes and persist-fatally writes. A weights save
    carries model_state + metadata only."""
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
    """Read a v2 envelope, re-verifying provenance and refusing every silent repair:
    `weights_only=True`, run_id and content hash re-checked against the filename, killed-branch
    prefixes REJECTED, `declared_encoding` asserting, `decode_override` loud but never raising,
    and disagreeing stamp sources raising. Never re-stamps, never auto-upgrades."""
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
        RunConfig.model_validate(config)  # config schema-validated on read

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


# The read path for the THREE real pre-v2 shapes.
def load_legacy_weights(
    path: str | Path,
    *,
    declared_encoding: Any = None,
    decode_override: Any = None,
) -> Checkpoint:
    """Read a pre-v2 artifact on its own surface. Arch comes from the declared or stamped
    `encoding_name` -> registry spec -> the STAMP's arch kind -> `select_arch`, and is NEVER
    shape-sniffed; the returned Checkpoint carries no synthetic run_id, content hash or
    created_utc, because a legacy anchor is never re-stamped on read."""
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
    # The ARTIFACT's stamp is the arch authority; an embedded config that carries the selector
    # row and DISAGREES is two records of one artifact contradicting each other, so it raises.
    kind = stamped_arch_kind(meta, representation=str(spec.representation))
    declared_kind = declared_arch_kind(embedded_config)
    if declared_kind is not None and str(declared_kind) != kind:
        raise CheckpointStampError(
            f"{path.name}: the embedded config declares identity.arch_kind={declared_kind!r} but "
            f"the artifact's stamp resolves to {kind!r}; a legacy artifact's arch is its stamp's, "
            "and a config that says otherwise describes a different artifact."
        )
    # A stamp carrying the WHOLE dataclass is rehydrated verbatim. Load-bearing for the ANCHOR,
    # whose embedded config is empty: `select_arch` would there yield the field DEFAULTS, so an
    # anchor from a run with non-default widths would rebuild at the wrong shape.
    stamped_arch = meta.get("arch")
    if isinstance(stamped_arch, Mapping) and _ARCH_KIND_KEY in stamped_arch:
        arch = _arch_from_dict(stamped_arch)
    else:
        arch = select_arch(spec, embedded_config, arch_kind=kind)

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


# Weights-only strip + re-stamp: the ONE sanctioned encoding-change / re-stamp path.
def strip_and_restamp(
    src_path: str | Path,
    *,
    new_encoding: str,
    run_id: str,
    checkpoint_dir: str | Path,
    declared_encoding: Any = None,
    step: int = 0,
) -> Path:
    """Give a legacy/v2 source a FRESH single v2 stamp, gated on wire-signature equality and
    stamped ONCE from the declared encoding + arch, never from a loaded config."""
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

    # The SOURCE artifact's stamp is the arch across the strip — kind AND widths. Only a
    # pre-discriminator stamp, which has no arch to carry, resolves to its incumbent-era kind.
    raw_meta = raw.get("metadata") if isinstance(raw, dict) else None
    stamped_arch = raw_meta.get("arch") if isinstance(raw_meta, dict) else None
    if isinstance(stamped_arch, Mapping):
        arch = _arch_from_dict(stamped_arch)
        if str(getattr(arch, "representation", None)) != str(new_spec.representation):
            raise CheckpointStampError(
                f"{src_path.name}: the stamped arch is {type(arch).__name__} "
                f"({arch.representation}) but {new_encoding} is {new_spec.representation}; the "
                "strip carries an arch, never re-derives one for a different representation."
            )
    else:
        kind = stamped_arch_kind(None, representation=str(old_spec.representation))
        arch = select_arch(new_spec, {}, arch_kind=kind)
    synth_config = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": 0,
        # Every literal in this synthetic config is a required key with no schema default, at
        # zero-behaviour placeholder values: a stripped artifact boots no run and trains nothing.
        "eval_enabled": True,
        # `None` is the correct posture, not a placeholder-by-default: the allocator posture is
        # a property of the RUN the caps were fitted for, and this payload belongs to none.
        "allocator_posture": None,
        "identity": {"encoding": new_encoding, "representation": new_spec.representation},
        # `puct` is the value that agrees with this payload's own
        # `train.policy_target: raw_visit_distribution` — the two are one decision.
        "search": {"kind": "puct"},
        "eval": {
            "random_model_sims": 1, "sealbot_model_sims": 1,
            "random_floor_games": 0, "worker_device": "cpu",
            "round_timeout_sec": 1.0, "worker_kill_grace_sec": 1.0,
            "ply_cap_adjudication": None, "strength_floor": None,
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
        # `legal_move_radius_schedule` is gone; the registry alone is the radius authority.
        "train": {
            "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0,
            "ema": {"enabled": False, "decay": 0.999, "update_every": 10},
            "device": "cpu",
            "lr_schedule": "cosine", "total_steps": 1_000_000,
            "scheduler_t_max": None, "eta_min": 5e-4,
            "checkpoint_interval": 0, "actor_sync_cadence_steps": 1,
            "max_train_steps": 1_000_000,  # required run-length key
            # `None` is the EXPLICIT disarmed posture: a config that is not a run claims no abort.
            "draw_rate_abort": None,
            # The step-coordinator knobs, at the template's own numbers.
            "eval_interval": 1000, "log_interval": 1000,
            "min_buf_size": 1, "replay_capacity": 100_000, "replay_capacity_schedule": [],
            "training_steps_per_game": 1.0, "max_train_burst": 1, "batch_size": 256,
            # (`train.microbatch_caps` is ARCH-SCOPED and is spliced in below, on the graph
            # route only: an unconditional literal would write a graph cap into a grid config.)
            "augment": False, "recency_weight": 0.0, "hard_gn_threshold": 1e9,
            "hard_gn_min_steps": 3, "terminal_eval_enabled": True,
            "selfplay_stall_timeout_sec": 1800.0,
            "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
            "draw_reward": -0.5, "ply_cap_value": -0.5,
            "fast_policy_weight": 0.0,
        },
        "selfplay": {
            "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
            "c_visit": 50.0, "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
            "results_queue_cap": 10_000, "random_opening_plies": 0,
            "log_investigation_metrics": True,
            "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                     "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                     "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                     "dirichlet_enabled": True},
            "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                            "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                            "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
        },
        "inference": {
            "inference_batch_size": 64, "inference_max_wait_ms": 10,
            # (`inference.fused_graph_caps` is ARCH-SCOPED and is spliced in below, graph only.)
        },
        "monitor": {
            # The TEMPLATE value, which every committed config also mints as `train.log_interval`:
            # a placeholder must never be able to disagree with a real run's cadence.
            "gate_interval": 1000,
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
            # Literals at exactly the MINTED values (60/10/5), for the reason above.
            "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
        },
    }
    # `ARCH_SCOPED_KEYS` is the ONE authority on which blocks belong to which representation, so
    # RENAMING one breaks loudly at the `_SYNTH_ARCH_SCOPED` lookup rather than silently writing
    # a config the schema then refuses.
    for key in ARCH_SCOPED_KEYS:
        if new_spec.representation == key.arch:
            synth_config[key.section][key.field] = dict(
                _SYNTH_ARCH_SCOPED[(key.section, key.field)]
            )
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


# Resume-DIRECTIVE keys: consumed BY the resume machinery, never carried. None is a RunConfig
# key, and the carried config is re-validated at the ONE writer on every periodic save, so
# `resume_trainer` STRIPS exactly these — deliberately NOT a general unknown-key filter, since
# an unknown key from anywhere else must still reach the writer and raise there.
RESUME_DIRECTIVE_KEYS: frozenset[str] = frozenset({
    "allow_fresh_scheduler", "scheduler_t_max", "torch_compile", "torch_compile_mode",
    "total_steps", "resume_owned_launch_values",
})


def apply_config_overrides_f1(
    baked: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
    declared_keys: frozenset | set | None,
    *,
    sink: Any = None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """Apply `overrides` onto the checkpoint-baked config under the defer rule: a DECLARED key
    wins, including an explicit `null`, while a non-declared baked key defers to the baked value
    and emits an event when they differ; weights-only or legacy calls update verbatim."""
    if baked is None or declared_keys is None:
        resolved = dict(baked or {})
        _update_leafwise(resolved, overrides)
        return resolved, frozenset()

    resolved = dict(baked)
    # The keys that force-declare themselves into the merge ARE the resume directives — one authority.
    declared = frozenset(declared_keys) | {k for k in RESUME_DIRECTIVE_KEYS if k in overrides}
    deferred: set[str] = set()
    _merge_leafwise(resolved, overrides, "", declared, deferred, sink)
    return resolved, frozenset(deferred)


def _update_leafwise(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Verbatim update recursing into sections, so an owned leaf the builder dropped stays baked."""
    for key, override_val in overrides.items():
        current = target.get(key)
        if isinstance(override_val, Mapping) and isinstance(current, Mapping):
            merged = dict(current)
            _update_leafwise(merged, override_val)
            target[key] = merged
        else:
            target[key] = override_val


def _merge_leafwise(
    resolved: dict[str, Any], overrides: Mapping[str, Any], prefix: str,
    declared: frozenset[str], deferred: set[str], sink: Any,
) -> None:
    """The F1 rule per leaf, dotted: declared wins, a differing baked leaf defers, absent is set."""
    for key, override_val in overrides.items():
        path = f"{prefix}{key}"
        current = resolved.get(key)
        if path in declared or key in declared:
            resolved[key] = override_val
            continue
        if isinstance(override_val, Mapping) and isinstance(current, Mapping):
            merged = dict(current)
            _merge_leafwise(merged, override_val, f"{path}.", declared, deferred, sink)
            resolved[key] = merged
            continue
        if key in resolved:
            if override_val != current:
                deferred.add(path)
                emit_via(
                    sink,
                    {
                        "event": "resume_base_default_deferred_to_baked",
                        "knob": path,
                        "base_default": override_val,
                        "checkpoint_baked": current,
                    },
                )
        else:
            resolved[key] = override_val


def _leaf(config: Mapping[str, Any] | None, dotted: str) -> Any:
    """Read a dotted path off a nested mapping; None when any segment is absent."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def resolve_lr_provenance(
    declared: float | None,
    baked: float | None,
    effective: float | None,
    *,
    rel_tol: float = 1e-9,
) -> LrProvenance:
    """Report a declared LR the resume drops: `override_ignored` is True only when a declared lr
    is present AND differs beyond `rel_tol` from the checkpoint's baked initial lr."""
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


#: The identity leaves a resume may not move. `arch_kind` is OPTIONAL, so absence on both sides
#: is agreement and absence on ONE side is a move — hence a union rather than a fixed list.
_IDENTITY_LEAVES = ("encoding", "representation", "arch_kind")


#: The `(section, leaf)` pairs that decide what a STORED replay row MEANS. Not identity keys —
#: they change no net and have no stamp to fall back on, so they need their own guard.
#: `search.kind` decides the other: resuming a `puct` ring under `gumbel` restores visit
#: distributions and applies the completed-Q loss to them, and vice versa.
_TARGET_SEMANTICS_LEAVES: tuple[tuple[str, str], ...] = (
    ("train", "policy_target"),
    ("search", "kind"),
)


def _refuse_target_semantics_drift(
    path: Path,
    baked_config: Mapping[str, Any] | None,
    effective_config: Mapping[str, Any],
) -> None:
    """HALT when a resume changes what the replay ring's rows mean, compared AFTER
    `config_overrides` over the UNION, since a leaf present on one side only is a move. Returns
    silently when the checkpoint carries no baked config.

    Raises:
        ResumeTargetSemanticsError: any target-semantics leaf differs, naming leaf and values.
    """
    if not baked_config:
        return
    drift: list[str] = []
    for section, leaf in _TARGET_SEMANTICS_LEAVES:
        baked_section = baked_config.get(section)
        effective_section = effective_config.get(section)
        if not isinstance(baked_section, Mapping) or not isinstance(effective_section, Mapping):
            # A shape this guard cannot read and must not GUESS at; mint-time validation covers it.
            continue
        want, got = baked_section.get(leaf), effective_section.get(leaf)
        if want != got:
            drift.append(f"{section}.{leaf}: checkpoint={want!r}, resume={got!r}")
    if drift:
        raise ResumeTargetSemanticsError(
            f"{path.name}: the resuming run builds its policy targets differently from the "
            "run that filled this checkpoint's replay ring — "
            + "; ".join(drift)
            + ". A resume restores the ring (R345(b)(3)), so the old rows and the new ones "
            "would carry different meanings under one loss, and a row records no provenance "
            "that could tell them apart. These leaves are ONE decision at mint "
            "(`_policy_target_matches_the_search_kind`); a resume is not a place to re-take "
            "it. Resume with the checkpoint's target semantics, or start a new run."
        )


def _refuse_identity_drift(
    path: Path,
    baked_config: Mapping[str, Any] | None,
    effective_config: Mapping[str, Any],
    arch: ModelArch,
) -> None:
    """HALT when the resuming run's EFFECTIVE identity differs from the checkpoint's, compared
    AFTER `config_overrides` — the only point at which the run's own identity exists — and
    falling back to the STAMPED arch, a fact about the artifact rather than about its config.

    Raises:
        ResumeIdentityMismatchError: any identity leaf differs, naming the leaf and both values.
    """
    baked_identity = (baked_config or {}).get("identity")
    effective_identity = effective_config.get("identity")
    if not isinstance(baked_identity, dict) or not isinstance(effective_identity, dict):
        # Nothing to compare; `load_checkpoint`'s stamp-vs-config check still covers ENCODING.
        return
    # The artifact's side, read off the DECLARED dataclass: `type(arch).__name__` is exactly the
    # discriminator `_arch_to_dict` serialises. `declared_arch_kind` reads the OTHER side.
    stamped = {
        "representation": arch.representation,
        "arch_kind": type(arch).__name__,
    }
    drift: list[str] = []
    for leaf in _IDENTITY_LEAVES:
        want = baked_identity.get(leaf, stamped.get(leaf))
        got = effective_identity.get(leaf, stamped.get(leaf))
        if _stamp_name(want) != _stamp_name(got):
            drift.append(f"identity.{leaf}: checkpoint={want!r}, resume={got!r}")
    if drift:
        raise ResumeIdentityMismatchError(
            f"{path.name}: the resuming run's effective identity differs from the "
            f"checkpoint's — " + "; ".join(drift) + ". The net is rebuilt from the "
            "checkpoint's STAMPED arch, so a moved identity key produces a model that is not "
            "what the config claims and every later save re-stamps the disagreement (LAW-11). "
            "Resume with the checkpoint's identity, or start a new run."
        )


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
    # STRICTNESS IS KEYED ON THE KIND. A bare ANCHOR is a genuine SUBSET of the `build_net` key
    # set, so `strict=True` would reject a healthy one; on a FULL checkpoint a missing key means
    # the stamped arch and the rebuilt net disagree, and a blanket `strict=False` then trained a
    # partly random model while optimizer, scheduler and step all reported a continuation.
    strict = ck.kind == "full"
    incompatible = model.load_state_dict(ck.model_state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        # A subset anchor is expected to be missing keys; unexpected keys are not expected at all.
        _LOG.info(
            "anchor_load_lenient path=%s missing=%d unexpected=%s",
            path.name, len(incompatible.missing_keys), sorted(incompatible.unexpected_keys),
        )

    # A DECLARED top-level key wins outright; a base-inherited key that differs from baked
    # DEFERS to baked and emits `resume_base_default_deferred_to_baked`.
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
    # Drop the machinery's OWN directive keys — and only those — before the Trainer carries the
    # config into every future save; anything else non-schema still reaches the writer and raises.
    config = {k: v for k, v in resolved_config.items() if k not in RESUME_DIRECTIVE_KEYS}
    _refuse_identity_drift(path, baked_config, config, arch)
    _refuse_target_semantics_drift(path, baked_config, config)
    # Pass the DECLARED arch so the Trainer re-stamps it rather than re-deriving one.
    trainer = cls(model, config, arch=arch, checkpoint_dir=path.parent, device=device, sink=sink)
    trainer.f1_deferred_keys = deferred

    # lr is resume-state-owned: a declared `lr` never wins on a full-checkpoint resume, but an
    # operator who declared one is warned loudly rather than silently ignored.
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
    # The nested shape's owned leaves: every launch value the builder dropped that differs from
    # the baked one is said out loud, since the baked value is what this resume runs on.
    owned = (config_overrides or {}).get("resume_owned_launch_values") or {}
    for dotted, launch_val in owned.items():
        baked_val = _leaf(baked_config, dotted)
        if baked_val is not None and launch_val != baked_val:
            emit_via(sink, {
                "event": "resume_owned_launch_value_ignored", "knob": dotted,
                "declared": launch_val, "baked": baked_val,
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
