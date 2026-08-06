"""Best-model anchor lifecycle — atomic save, resilient load, quarantine (WP10 §a.5/§c.6 IMPROVE).

>300 justify: this owns the whole `best_model.pt` artifact lifecycle — atomic save with
round-trip verify + `.bak` rotation + provenance sidecar, the resilient best→.bak→bootstrap
load fallback with corrupt-anchor quarantine, the tensor-identity sha256 launch-pin, and the
anchor↔inference-model sync — one cohesive artifact-IO surface (the old `training/anchor.py`
ported whole). Splitting it would scatter the save/load/verify invariants that must move
together.

Three ratified WP10 amendments over a pure relocation:
  * **Representation off the DECLARED arch (§c.6, WP9 O3 census).** The DELETED arch-off-a-live-
    module representation sniff is REPLACED by the declared `representation` carried on the WP9
    arch dataclass (`AnchorState.representation`, `trainer.arch.representation`,
    `Checkpoint.metadata.arch.representation`) — nobody reads arch off a live `nn.Module`.
  * **weights-only everywhere (LAW-12).** `checkpoint_state_sha256` loads weights-only — the old
    pickle-exec load mode is gone; the round-trip verify was already weights-only.
  * **DAG-clean (repo_design §2).** `resolve_anchor` takes the `eval_pipeline` INJECTED (no
    top-level `train → eval` import); the anchor's from-disk read routes through the ONE loader
    (`mantis.train.checkpoints`), inheriting its O3b killed-prefix REJECT.

**(B) — the preserved corruption guard (dispatcher requirement; RED-TEAM #1).** WP10 killed the
shape-sniff arch, so `build_net(arch)` may emit a SUPERSET of a legitimate SUBSET anchor's keys
(a min/max baseline lacks the aux heads). The old eval loader's `strict=True` landing-guard
(`eval/checkpoint_loader.py::_build_min_max_model`, `strict=False` + explicit E1-C1 allclose
"value_fc2_bins.weight was NOT loaded" raise, and `_build_gnn_model` `strict=True` +
"load_state_dict did not land this tensor") would spuriously reject that subset. So the anchor
load uses **`strict=False` PLUS explicit validation** (`_guarded_load_state_dict`): the missing
keys must be a SUBSET of the known-optional aux heads, EVERY core tensor (trunk / policy / value)
MUST land, and unexpected keys MUST be empty. A checkpoint missing a REQUIRED core tensor RAISES
(never a silent random-head load — the old E1-C1 / F-12 hazard) — NOT a silent `strict=False` drop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis.model import RepresentationMismatch, build_net

_LOG = logging.getLogger(__name__)

# Repo-relative bootstrap candidates tried (in order) when no usable best_model.pt exists — an
# EXPLICIT default list (no host-coupled / personal path; R1). A launch may override it.
_BOOTSTRAP_ANCHOR_CANDIDATES: tuple[str, ...] = (
    "checkpoints/bootstrap_model_v6.pt",
    "checkpoints/bootstrap_model_v7full.pt",
)

# (B) corruption guard — state-dict key roots a legitimate SUBSET/min-max baseline anchor may
# lack (the aux training-only heads `build_net` always emits + the optional input-channel buffer).
# EVERYTHING ELSE is a CORE tensor (trunk / policy / value) that MUST land on load. Cited from the
# old eval loader's E1-C1 landing-guard (`_build_min_max_model` value-head allclose raise).
_OPTIONAL_HEAD_PREFIXES: tuple[str, ...] = (
    "opp_reply_conv.",    # aux opponent-reply head
    "opp_reply_fc.",
    "value_var.",         # value-uncertainty head
    "ownership_head.",    # ownership head
    "threat_head.",       # threat head
    "chain_head.",        # Q13 chain-length head (the surviving aux target)
    "ply_index_head.",    # ply-index head
    "input_channel_index",  # input-channel selector buffer (present only when input_channels set)
)


class AnchorLoadError(RuntimeError):
    """A from-disk anchor failed the (B) corruption guard: a required CORE tensor did not land,
    or the state dict carried keys the declared arch does not accept."""


#: The canonical anchor (best-model) filename, in ONE place.
#:
#: The anchor is the promotion INCUMBENT: every eval round scores the candidate against
#: whatever this path holds, and a promotion overwrites it. Read and write must therefore
#: name the same file, and they did not — `resolve_anchor` defaulted to a CWD-relative
#: `checkpoints/best_model.pt` while the promotion hook was handed the run's real
#: `<out-dir>/checkpoints/best_model.pt`. A run launched from anywhere but the repo root
#: consequently evaluated against one file and promoted into another.
CANONICAL_ANCHOR_FILENAME = "best_model.pt"


def canonical_anchor_path(checkpoint_dir: str | Path) -> Path:
    """The run's anchor path, derived from ITS checkpoint directory (R98)."""
    return Path(checkpoint_dir) / CANONICAL_ANCHOR_FILENAME


@dataclass
class AnchorState:
    """Resolved best-model anchor + provenance. `best_model` is None only when no eval pipeline
    is configured (the pre-refactor invariant). `representation` is the DECLARED discriminant
    ("grid"/"graph") read off the arch — never sniffed off a live module (§c.6)."""

    best_model: torch.nn.Module | None
    best_model_step: int | None
    best_model_path: Path
    representation: str


# ══ Atomic save + provenance ═══════════════════════════════════════════════════════════
def save_best_model_atomic(
    model: torch.nn.Module,
    path: Path,
    *,
    step: int | None = None,
    run_id: str | None = None,
    encoding: str | None = None,
) -> None:
    """Save ``state_dict()`` to ``path`` atomically with one-revision backup.

    Sequence: (1) write ``path.tmp``, (2) round-trip verify the tmp file loads (catches partial
    writes), (3) rotate any existing ``path`` → ``path.bak``, (4) rename ``path.tmp`` → ``path``.
    A kill between (3) and (4) leaves ``.bak`` as the recovery copy (``load_best_model_resilient``
    falls through to it).

    With ``step`` supplied (the promotion path) the payload is wrapped with provenance
    (``step``/``run_id``/``promoted``/``encoding``/``metadata.encoding_name``) and a
    ``.provenance.json`` sidecar is written, so a promoted anchor is greppable. With ``step=None``
    the legacy bare-``state_dict`` payload is written (back-compat with provenance-less callers).
    """
    path = Path(path)
    base = getattr(model, "_orig_mod", model)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    sd = base.state_dict()
    payload: Any
    if step is None:
        payload = sd
    else:
        payload = {
            "model_state": sd,
            "step": int(step),
            "run_id": run_id,
            "promoted": True,
            "encoding": encoding,
            "metadata": {"encoding_name": encoding} if encoding is not None else {},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)
    # Round-trip verify — torch.save is not atomic on some filesystems and a mid-write kill
    # produces exactly the truncated zip we defend against. Weights-only (LAW-12).
    torch.load(tmp, map_location="cpu", weights_only=True)
    if path.exists():
        path.replace(bak)
    tmp.replace(path)
    if step is not None:
        _write_provenance_sidecar(path, step=int(step), run_id=run_id, encoding=encoding)


def _write_provenance_sidecar(
    path: Path, *, step: int, run_id: str | None, encoding: str | None,
) -> None:
    """Write ``<path>.provenance.json`` (atomic) so a promoted anchor's identity is greppable
    without loading torch."""
    prov = {"step": step, "run_id": run_id, "encoding": encoding, "promoted": True}
    sidecar = path.with_name(path.name + ".provenance.json")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(prov, indent=2))
    tmp.replace(sidecar)


# ══ Tensor-identity hash + launch pin ══════════════════════════════════════════════════
def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Deterministic sha256 over a model ``state_dict`` (canonical sorted keys + raw tensor
    bytes) — stable across processes and save formats for identical weights. Canonicalises
    `_orig_mod.`/`module.` compile/DDP wrapper prefixes so a wrapped and an unwrapped copy of the
    same weights hash equal."""
    def _canon(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    h = hashlib.sha256()
    for canon_key, raw_key in sorted((_canon(k), k) for k in state_dict.keys()):
        h.update(canon_key.encode("utf-8"))
        value = state_dict[raw_key]
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _extract_stored_state(raw: Any) -> dict[str, Any]:
    """The MODEL weights stored in a loaded anchor payload — a bare `state_dict` or a
    `{model_state: …}` provenance wrapper."""
    if isinstance(raw, dict) and isinstance(raw.get("model_state"), dict):
        return raw["model_state"]
    return raw if isinstance(raw, dict) else {}


def checkpoint_state_sha256(path: Path) -> str:
    """sha256 of the MODEL weights STORED in an anchor file — the dtype/device-independent
    identity the launch pin compares against. Hashes the STORED state (matching the on-disk pin),
    NOT a live model whose runtime dtype diverges from disk. Weights-only (LAW-12)."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return state_dict_sha256(_extract_stored_state(raw))


def verify_launch_anchor_pin(
    *,
    expected_anchor_sha256: str | None,
    checkpoint_path: str | Path | None,
    trainer_step: int | None,
    run_id: str | None,
) -> None:
    """Verify the launch pin on the fresh-init path — hashing the STORED weights of the
    ``--checkpoint`` the fresh anchor is seeded from (dtype-invariant). No-op when no pin is set;
    FAILS CLOSED when a pin is set but no verifiable source exists (never launch on an UNVERIFIED
    incumbent)."""
    if expected_anchor_sha256 is None:
        return
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        seed_sha = checkpoint_state_sha256(Path(checkpoint_path))
        _LOG.info(
            "anchor_identity path=%s step=%s run_id=%s sha256=%s pinned=%s source=fresh_init_checkpoint",
            str(checkpoint_path), trainer_step, run_id, seed_sha, expected_anchor_sha256,
        )
        if seed_sha != expected_anchor_sha256:
            raise RuntimeError(
                f"anchor sha256 mismatch (fresh-init seed): the checkpoint {checkpoint_path} "
                f"resolved to {seed_sha} but the run config pinned {expected_anchor_sha256}. "
                "Refusing to launch. Pass the pinned incumbent as --checkpoint, or clear the pin."
            )
    else:
        raise RuntimeError(
            f"expected_anchor_sha256={expected_anchor_sha256} is pinned but there is no readable "
            f"--checkpoint to verify the fresh-init anchor against (checkpoint_path={checkpoint_path}). "
            "The launch incumbent would be UNVERIFIED — refusing to launch."
        )


# ══ Quarantine + the (B) corruption-guarded from-disk load ═════════════════════════════
def _quarantine_corrupt(path: Path) -> Path:
    """Move a corrupt anchor aside with a unique suffix so the next write does not overwrite it.
    Returns the destination path for logging."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = path.with_suffix(path.suffix + f".corrupt-{ts}")
    path.replace(dest)
    return dest


def _guarded_load_state_dict(
    model: torch.nn.Module, state: dict[str, Any]
) -> list[str]:
    """(B) corruption guard — load `state` into `model` with `strict=False` PLUS explicit
    validation that preserves the old eval-loader's landing guard.

    Since WP10 kills the shape-sniff, `build_net(arch)` emits a SUPERSET of a min/max baseline
    anchor's keys, so `strict=True` would spuriously reject a legitimate SUBSET anchor (T-CK-25).
    Instead: unexpected keys MUST be empty; the missing keys MUST all be known-optional aux heads;
    every CORE tensor (trunk / policy / value) MUST land. A checkpoint missing a required core
    tensor RAISES (never a silent random-head load — the old E1-C1 / F-12 hazard). Returns the
    (optional-only) missing keys for logging."""
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise AnchorLoadError(
            f"anchor state_dict carries {len(unexpected)} unexpected key(s) {unexpected[:5]} the "
            "declared arch does not accept — refusing to load a mismatched anchor (killed-branch "
            "prefixes are already rejected upstream by the ONE loader's O3b guard; any other "
            "unexpected key is corruption / an arch mismatch, not a subset baseline)."
        )
    missing = list(result.missing_keys)
    core_missing = [k for k in missing if not k.startswith(_OPTIONAL_HEAD_PREFIXES)]
    if core_missing:
        raise AnchorLoadError(
            f"anchor state_dict is missing REQUIRED core tensor(s) {core_missing[:5]} "
            "(trunk / policy / value) — a checkpoint missing a required core tensor MUST NOT "
            "silently load a random head (the old E1-C1 / F-12 hazard the eval loader's "
            "landing-guard existed to kill). Only these aux-head prefixes may be absent from a "
            f"legitimate min/max baseline anchor: {_OPTIONAL_HEAD_PREFIXES}."
        )
    return missing


def _build_anchor_model(
    path: Path,
    *,
    declared_encoding: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    """Read an anchor file through the ONE loader (weights-only + O3b killed-prefix REJECT +
    declared-arch resolution), build the net via `build_net(metadata.arch)`, and load its weights
    under the (B) corruption guard. Returns `(model, Checkpoint)`; the checkpoint's
    `metadata.arch.representation` is the anchor's DECLARED representation."""
    from mantis.train import checkpoints as _ck

    raw = torch.load(path, weights_only=True, map_location="cpu")
    is_v2 = isinstance(raw, dict) and raw.get("schema_version") == _ck.CHECKPOINT_SCHEMA_VERSION
    ck = (
        _ck.load_checkpoint(path, device=device)
        if is_v2
        else _ck.load_legacy_weights(path, declared_encoding=declared_encoding)
    )
    arch = ck.metadata.arch
    if arch is None:
        raise AnchorLoadError(f"{path}: no arch resolved for the anchor — cannot rebuild the net.")
    model = build_net(arch).to(device)
    _guarded_load_state_dict(model, ck.model_state)
    model.eval()
    return model, ck


def _try_load_anchor(
    candidate: Path,
    *,
    declared_encoding: str | None,
    device: torch.device,
    skip_encoding_mismatch: bool = False,
) -> tuple[torch.nn.Module, Path, int | None, str] | None:
    """Attempt to load ``candidate`` as an anchor. Returns
    ``(model, path, step, representation)`` on success, None on failure (corrupt zip, unreadable).

    A DeclaredEncoding disagreement RAISES by default (D-FORENSIC F1 — a configuration error, not
    corruption; it must not enter the quarantine/fresh-init machinery). ``skip_encoding_mismatch``
    restores skip-on-mismatch for FOREIGN multi-candidate bootstrap fallbacks."""
    if not candidate.exists():
        return None
    from mantis.train.checkpoints import DeclaredEncodingMismatchError

    try:
        model, ck = _build_anchor_model(candidate, declared_encoding=declared_encoding, device=device)
        representation = getattr(ck.metadata.arch, "representation", "grid")
        step = ck.metadata.step if ck.metadata.step else None
        return (model, candidate, step, representation)
    except DeclaredEncodingMismatchError:
        if skip_encoding_mismatch:
            _LOG.warning(
                "anchor_encoding_mismatch_skipped path=%s (foreign bootstrap candidate — skipping "
                "to the next candidate, by-design mismatch not lineage rot).", str(candidate),
            )
            return None
        _LOG.error(
            "anchor_encoding_mismatch path=%s (candidate encoding disagrees with the declared "
            "config encoding — refusing to fall through; re-stamp the anchor or fix `encoding:`).",
            str(candidate),
        )
        raise
    except AnchorLoadError:
        # A required-core-tensor miss / arch mismatch is a LOUD configuration error, not
        # recoverable corruption — surface it rather than silently quarantining a valid anchor.
        raise
    except Exception as exc:  # noqa: BLE001 — corrupt zip / unreadable file: fall through to next candidate
        _LOG.warning(
            "anchor_load_failed path=%s error=%s error_type=%s",
            str(candidate), exc, type(exc).__name__,
        )
        return None


def load_best_model_resilient(
    best_model_path: Path,
    *,
    declared_encoding: str | None,
    device: torch.device,
    bootstrap_candidates: tuple[str, ...] | None = None,
) -> tuple[torch.nn.Module, Path, int | None, str] | None:
    """Try best_model.pt → its .bak → bootstrap candidates. Returns
    ``(model, source_path, step, representation)`` or None if all fail.

    On corruption of ``best_model.pt`` the file is quarantined and the next candidate is tried."""
    candidates = bootstrap_candidates if bootstrap_candidates is not None else _BOOTSTRAP_ANCHOR_CANDIDATES

    # 1. Live anchor.
    if best_model_path.exists():
        ref = _try_load_anchor(best_model_path, declared_encoding=declared_encoding, device=device)
        if ref is not None:
            return ref
        quarantined = _quarantine_corrupt(best_model_path)
        _LOG.warning(
            "anchor_quarantined original=%s quarantined=%s (unreadable — falling through).",
            str(best_model_path), str(quarantined),
        )

    # 2. One-revision backup written by the previous atomic save.
    bak = best_model_path.with_suffix(best_model_path.suffix + ".bak")
    if bak.exists():
        ref = _try_load_anchor(bak, declared_encoding=declared_encoding, device=device)
        if ref is not None:
            _LOG.info("anchor_recovered_from_bak path=%s", str(bak))
            return ref

    # 3. Repo-relative bootstrap candidates — FOREIGN files: an encoding mismatch is by design
    #    for any non-matching variant, so skip instead of raising.
    for rel in candidates:
        cand = Path(rel)
        ref = _try_load_anchor(
            cand, declared_encoding=declared_encoding, device=device, skip_encoding_mismatch=True,
        )
        if ref is not None:
            _LOG.info("anchor_loaded_from_bootstrap path=%s", str(cand))
            return ref

    return None


# ══ resolve_anchor (injected eval_pipeline; representation off the declared arch) ══════
def _resolve_declared_encoding(config: Any) -> str | None:
    """The declared encoding NAME via THE one resolver (WPTS Phase P, ADJ-25/R104) — the
    private identity-first shape-read this function used to carry is dead, closing the
    nine-caller family WPBRIDGE collapsed.

    Absence is legal HERE and only here: a WP10-only launch hands the anchor a bare hparams
    dict that declares nothing, and `AnchorState` carries that truthfully as None. Only
    `MissingEncodingError` maps to None — a dual-shape declaration that DISAGREES raises
    `EncodingDeclarationConflictError` through this veneer (corrupt input must not degrade
    into "no declaration"), and an UNREGISTERED name raises from the registry lookup (the
    old read returned any string; the cross-check is the LAW-11 posture). The anchor-private
    `{'encoding': {'name': ...}}` form died with the private read.
    """
    if not isinstance(config, dict):
        return None
    from mantis.encoding.resolvers import MissingEncodingError, resolve_from_config

    try:
        return resolve_from_config(config).name
    except MissingEncodingError:
        return None


def resolve_anchor(
    *,
    trainer: Any,
    eval_pipeline: Any,
    anchor_state: Any = None,
    sink: Any = None,
    config: dict[str, Any] | None = None,
    device: torch.device | None = None,
    best_model_path: str | Path | None = None,
    declared_encoding: str | None = None,
    run_id: str | None = None,
    expected_anchor_sha256: str | None = None,
    bootstrap_candidates: tuple[str, ...] | None = None,
) -> AnchorState:
    """Resolve the best-model anchor (the DEPLOY tag — WP-UNFREEZE, R49: deploy state
    NEVER writes actor weights, at launch or any other time; the old ``inf_model``
    launch-time sync arm is deleted, not merely unused).

    INJECTED-collaborator contract (§c.6/§c.8): ``eval_pipeline`` is injected (no `train → eval`
    import); when None the anchor stays unresolved (the pre-refactor invariant). Everything else
    derives from ``trainer`` (``trainer.arch``/``.config``/``.device``/``.step``/
    ``.inference_state_dict()``) when not passed explicitly. The trainer/anchor
    cross-representation check compares DECLARED representations
    (``trainer.arch.representation`` vs the anchor's ``metadata.arch.representation``) —
    NEVER an arch-off-a-live-module sniff (§c.6).
    """
    config = config if config is not None else dict(getattr(trainer, "config", {}) or {})
    # A distinct name: rebinding the `device | None` parameter would keep its declared
    # Optional type; this local is a plain torch.device.
    resolved_device: torch.device = (
        device if device is not None else getattr(trainer, "device", torch.device("cpu"))
    )
    inf_representation = getattr(getattr(trainer, "arch", None), "representation", "grid")
    if declared_encoding is None:
        declared_encoding = _resolve_declared_encoding(config)
    # NO CWD FALLBACK (item 5(a)). This used to default to `Path("checkpoints/best_model.pt")`
    # when the caller passed nothing — and `train/loop.py` passed nothing, so the default was
    # the production path. The promotion WRITE side (`DeployTagHooks`) was meanwhile handed
    # the run's real `<out-dir>/checkpoints/best_model.pt`. Read and write therefore named
    # DIFFERENT FILES for any run not launched from the repo root: every round scored the
    # candidate against a stale or absent incumbent, and promotions landed somewhere the next
    # round would not look. A wrong incumbent is silently wrong — the round still completes,
    # still reports a win rate, still promotes — so this fails loud instead (LAW-11's spirit:
    # an absent identity is an error, never a default).
    if best_model_path is None:
        raise ValueError(
            "resolve_anchor requires an explicit best_model_path. There is no default: the "
            "old CWD-relative fallback let the anchor READ and the promotion WRITE name "
            "different files. Derive it with "
            "`mantis.train.anchor.canonical_anchor_path(checkpoint_dir)`."
        )
    bmp = Path(best_model_path)

    if eval_pipeline is None:
        return AnchorState(None, None, bmp, inf_representation)

    bmp.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_best_model_resilient(
        bmp, declared_encoding=declared_encoding, device=resolved_device,
        bootstrap_candidates=bootstrap_candidates,
    )
    if loaded is not None:
        best_model, best_source_path, best_model_step, anc_representation = loaded

        # Launch-pin identity: hash the STORED weights (dtype-invariant), hard-fail on a mismatch.
        anchor_sha = checkpoint_state_sha256(best_source_path)
        _LOG.info(
            "anchor_identity path=%s step=%s run_id=%s sha256=%s pinned=%s",
            str(best_source_path), best_model_step, run_id, anchor_sha, expected_anchor_sha256,
        )
        if expected_anchor_sha256 is not None and anchor_sha != expected_anchor_sha256:
            raise RuntimeError(
                f"anchor sha256 mismatch: best_model.pt resolved to {anchor_sha} but the run "
                f"config pinned {expected_anchor_sha256}. Refusing to launch (WRONG INCUMBENT, or "
                "a legitimate post-promotion resume — update the pin or clear it)."
            )

        # Persist a fallback-recovered anchor as the live best_model.pt for subsequent runs.
        if not bmp.exists():  # True after quarantine (rename, not delete)
            save_best_model_atomic(best_model, bmp)
            _LOG.info("anchor_persisted_from_fallback path=%s", str(bmp))

        # Cross-representation lineage check — DECLARED representations, never a module sniff (§c.6).
        if inf_representation != anc_representation:
            raise RepresentationMismatch(
                f"resolve_anchor: the trainer arch is representation={inf_representation!r} but "
                f"the resolved anchor at {bmp} is representation={anc_representation!r}. A "
                "cross-representation anchor is a wrong-lineage incumbent — namespace the "
                "per-lineage best_model_path."
            )
        trainer_step = getattr(trainer, "step", None)
        if best_model_step is not None and trainer_step is not None and trainer_step != best_model_step:
            _LOG.warning(
                "resume_anchor_step_mismatch trainer_step=%s best_model_step=%s "
                "(trainer.model and best_model.pt loaded from different steps — confirm intended).",
                trainer_step, best_model_step,
            )
        _LOG.info("best_model_loaded path=%s step=%s", str(bmp), best_model_step)
        return AnchorState(best_model, best_model_step, bmp, anc_representation)

    # No usable anchor anywhere — last-resort fresh init from trainer.model.
    verify_launch_anchor_pin(
        expected_anchor_sha256=expected_anchor_sha256,
        checkpoint_path=getattr(trainer, "checkpoint_source", None),
        trainer_step=getattr(trainer, "step", None),
        run_id=run_id,
    )
    _LOG.warning(
        "anchor_fresh_init_no_bootstrap tried=%s (no anchor or bootstrap available — initialising "
        "best_model.pt from current trainer.model).", list(bootstrap_candidates or _BOOTSTRAP_ANCHOR_CANDIDATES),
    )
    best_model = build_net(trainer.arch).to(resolved_device)
    best_model.load_state_dict(trainer.inference_state_dict())
    best_model.eval()
    save_best_model_atomic(best_model, bmp)
    best_model_step = getattr(trainer, "step", None)
    _LOG.info("best_model_initialized path=%s step=%s", str(bmp), best_model_step)
    return AnchorState(best_model, best_model_step, bmp, inf_representation)


__all__ = [
    "AnchorLoadError",
    "AnchorState",
    "checkpoint_state_sha256",
    "load_best_model_resilient",
    "resolve_anchor",
    "save_best_model_atomic",
    "state_dict_sha256",
    "verify_launch_anchor_pin",
]
