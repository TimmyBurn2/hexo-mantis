"""Mirror receipts (R349(b)): a receipt beside an artifact carries the sha256 of the MIRRORED
bytes; verifying it recomputes the artifact's sha256 HERE, so a receipt covers exactly one byte
string and none can be honest about bytes that never left the box."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mantis.util.hashing import sha256_file

RECEIPT_SUFFIX = ".receipt.json"
RECEIPT_SCHEMA_VERSION = 1
#: The workspace reading's verdict a stamp must carry: the loop proven on this run directory.
MIRRORED_VERDICT = "MIRRORED"


class MirrorReceiptError(RuntimeError):
    """A receipt is absent, malformed, or does not match the artifact beside it."""


def receipt_path_for(artifact: str | Path) -> Path:
    """`<artifact>.receipt.json`, beside the artifact."""
    target = Path(artifact)
    return target.with_name(target.name + RECEIPT_SUFFIX)


def is_receipt(path: str | Path) -> bool:
    return str(path).endswith(RECEIPT_SUFFIX)


def write_receipt(artifact: str | Path, *, mirrored_sha256: str, mirrored_bytes: int,
                  cycle: int, mirror_id: str) -> Path:
    """Write `artifact`'s receipt beside it from the OFF-BOX copy's hash and size; return its path.

    Raises:
        OSError: the receipt could not be written.
    """
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact": Path(artifact).name,
        "sha256": mirrored_sha256,
        "bytes": int(mirrored_bytes),
        "cycle": int(cycle),
        "mirror_id": mirror_id,
        "mirrored_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = receipt_path_for(artifact)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_receipt(path: str | Path) -> dict[str, Any]:
    """Load and shape-check one receipt.

    Raises:
        MirrorReceiptError: absent, not JSON, or not a schema-1 receipt with its facts.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MirrorReceiptError(f"no receipt at {path}") from exc
    except OSError as exc:
        raise MirrorReceiptError(f"receipt {path} unreadable: {exc}") from exc
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MirrorReceiptError(f"receipt {path} is not JSON: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise MirrorReceiptError(f"receipt {path} is not a schema-{RECEIPT_SCHEMA_VERSION} receipt")
    for key in ("artifact", "sha256", "bytes", "mirrored_utc"):
        if key not in receipt:
            raise MirrorReceiptError(f"receipt {path} lacks {key!r}")
    return receipt


def verify_receipt(artifact: str | Path) -> dict[str, Any]:
    """The artifact hashes to what its receipt says the mirror holds; returns the receipt + reading.

    Raises:
        MirrorReceiptError: no/malformed receipt, artifact absent or renamed, or the bytes differ.
    """
    target = Path(artifact)
    receipt = read_receipt(receipt_path_for(target))
    if receipt["artifact"] != target.name:
        raise MirrorReceiptError(
            f"receipt beside {target.name} names {receipt['artifact']!r}")
    if not target.is_file():
        raise MirrorReceiptError(f"{target} has a receipt but the artifact is absent")
    size = target.stat().st_size
    if size != int(receipt["bytes"]):
        raise MirrorReceiptError(
            f"{target.name}: {size} bytes here, the mirror holds {receipt['bytes']} — "
            "the artifact changed after it was mirrored, or the mirror is short")
    digest = sha256_file(target)
    if digest != receipt["sha256"]:
        raise MirrorReceiptError(
            f"{target.name}: sha256 {digest} here, the mirror holds {receipt['sha256']} — "
            "the bytes off-box are not these bytes")
    return {**receipt, "verified_bytes": size, "verified_sha256": digest,
            "receipt": str(receipt_path_for(target))}


__all__ = [
    "MIRRORED_VERDICT", "MirrorReceiptError", "RECEIPT_SCHEMA_VERSION", "RECEIPT_SUFFIX",
    "is_receipt", "read_receipt", "receipt_path_for", "verify_receipt", "write_receipt",
]
