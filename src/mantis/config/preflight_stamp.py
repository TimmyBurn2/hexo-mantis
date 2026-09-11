"""The preflight stamp: written per passing preflight, demanded by `mantis.run` (R348(c))."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mantis.config.loader import config_identity_sha256
from mantis.config.schema import RunConfig
from mantis.util.git import head_sha, is_dirty
from mantis.util.mirror_receipts import MIRRORED_VERDICT

STAMP_SCHEMA_VERSION = 1
#: The two START halts a stamp must carry a reading for; a stamp missing either is refused.
START_HALT_READINGS = ("workspace", "cuda_build")


class PreflightStampRefusal(RuntimeError):
    """Base of every named reason `mantis.run` refuses to launch on the stamp's account."""


class PreflightStampMissingError(PreflightStampRefusal):
    """No passing preflight stamp exists for this config's identity hash."""


class PreflightStampTreeMismatchError(PreflightStampRefusal):
    """The stamp was written on a different tree than the one launching."""


class PreflightStampMalformedError(PreflightStampRefusal):
    """The stamp exists but does not carry what a stamp must carry."""


class PreflightStampUnmirroredError(PreflightStampRefusal):
    """The stamp's workspace reading does not say the run directory's mirror loop was proven."""


def stamp_dir() -> Path:
    """`$XDG_STATE_HOME/mantis/preflight`, or `~/.local/state/mantis/preflight` when unset."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "mantis" / "preflight"


def stamp_path(config_sha256: str) -> Path:
    """Where the stamp for one config identity lives."""
    return stamp_dir() / f"{config_sha256}.json"


def write_stamp(
    *, config: RunConfig, config_path: Path, tree_root: Path, halts: dict[str, Any],
    booted_config_sha256: str, burst_steps: int, report_path: Path, tier: str,
) -> Path:
    """Write the stamp for `config` (the minted identity) and return its path; `tier` is what the burst PROVED.

    Raises:
        PreflightStampMalformedError: a START halt reading is absent from `halts`.
        OSError: the store is unwritable.
    """
    missing = [name for name in START_HALT_READINGS if name not in halts]
    if missing:
        raise PreflightStampMalformedError(
            f"a stamp must carry a reading for every START halt; missing {missing}")
    sha = config_identity_sha256(config)
    stamp = {
        "schema_version": STAMP_SCHEMA_VERSION,
        "verdict": "pass",
        "config_sha256": sha,
        "config_path": str(config_path),
        "run_id": config.run_id,
        "booted_config_sha256": booted_config_sha256,
        "burst_steps": int(burst_steps),
        "tier": str(tier),
        "tree_sha": head_sha(tree_root),
        "tree_dirty": is_dirty(tree_root),
        "preflight_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "halts": {name: halts[name] for name in START_HALT_READINGS},
        "report": str(report_path),
    }
    path = stamp_path(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def clear_stamp(config: RunConfig) -> None:
    """Remove any stamp for `config`, so a preflight that does not pass leaves no stale pass."""
    path = stamp_path(config_identity_sha256(config))
    if path.exists():
        path.unlink()


def read_stamp(config_sha256: str) -> dict[str, Any]:
    """Load and shape-check the stamp for one config identity.

    Raises:
        PreflightStampMissingError: no stamp file exists.
        PreflightStampMalformedError: not a passing stamp of this schema under its own identity.
    """
    path = stamp_path(config_sha256)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PreflightStampMissingError(
            f"no preflight stamp for config identity {config_sha256} at {path}") from exc
    try:
        stamp = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightStampMalformedError(f"{path}: not JSON: {exc}") from exc
    if not isinstance(stamp, dict) or stamp.get("schema_version") != STAMP_SCHEMA_VERSION:
        raise PreflightStampMalformedError(
            f"{path}: not a schema-{STAMP_SCHEMA_VERSION} stamp")
    if stamp.get("verdict") != "pass":
        raise PreflightStampMalformedError(f"{path}: verdict is {stamp.get('verdict')!r}")
    if stamp.get("config_sha256") != config_sha256:
        raise PreflightStampMalformedError(
            f"{path}: records identity {stamp.get('config_sha256')!r}, filed under "
            f"{config_sha256}")
    halts = stamp.get("halts")
    if not isinstance(halts, dict) or any(name not in halts for name in START_HALT_READINGS):
        raise PreflightStampMalformedError(
            f"{path}: START halt readings absent; need {list(START_HALT_READINGS)}")
    return stamp


def require_preflight_stamp(config: RunConfig, *, tree_root: Path) -> dict[str, Any]:
    """The launch-time trap: return the passing stamp for `config` on THIS tree, or refuse.

    Raises:
        PreflightStampMissingError: no stamp exists for this config identity.
        PreflightStampMalformedError: the stamp does not carry what a stamp must carry.
        PreflightStampTreeMismatchError: the stamp names another HEAD, or this HEAD is unreadable.
        PreflightStampUnmirroredError: the workspace reading's verdict is not `MIRRORED`, so
            the preflight never saw the puller's receipts on this run directory (R349(b)).
    """
    sha = config_identity_sha256(config)
    stamp = read_stamp(sha)
    workspace = stamp["halts"]["workspace"]
    verdict = workspace.get("verdict") if isinstance(workspace, dict) else None
    if verdict != MIRRORED_VERDICT:
        raise PreflightStampUnmirroredError(
            f"{config.run_id} ({sha}) was preflighted with workspace verdict {verdict!r}, not "
            f"{MIRRORED_VERDICT!r}: the mirror loop was never proven on its run directory")
    here = head_sha(tree_root)
    if here is None:
        raise PreflightStampTreeMismatchError(
            f"the launching tree at {tree_root} has no readable HEAD, so the stamp for "
            f"{config.run_id} ({sha}, preflighted on {stamp.get('tree_sha')}) cannot be "
            "shown to cover it")
    if stamp.get("tree_sha") != here:
        raise PreflightStampTreeMismatchError(
            f"{config.run_id} ({sha}) was preflighted on tree {stamp.get('tree_sha')} at "
            f"{stamp.get('preflight_utc')}; this launch is on {here}. Re-run the preflight "
            "on this tree")
    return stamp
