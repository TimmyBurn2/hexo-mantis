"""The resume sidecar — the per-stop facts envelope v2 cannot carry.

It sits BESIDE envelope v2 because the envelope is stamped ONCE with its `content_sha8` in its own
filename, so a new field changes the format, the hash and THE ONE LOADER. Its failure posture is
run-fatal rather than best-effort — a stop that cannot persist its ring must say so loudly, or the
resume silently refills from empty — and it is a RESUME INPUT, never a provenance record, so a
sidecar that disagrees with its checkpoint is REFUSED rather than reconciled.

>300 justify (R8): ONE subject — what a stop must record for its resume to be lawful. Splitting it
would put the hash that is written in one file and the hash that is checked in another.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Appended to the checkpoint's own filename, so a sidecar can never be mistaken for a checkpoint
#: and the checkpoint grammar still parses the stem it is derived from.
SIDECAR_SUFFIX = ".resume.json"

#: Bumped when a field's MEANING changes, never when one is added — an older sidecar missing a
#: field is refused by `from_dict`'s explicit read.
SIDECAR_VERSION = 1

_HASH_CHUNK = 1 << 20

#: The streams `capture_rng_streams` writes and `restore_rng_streams` honours — ONE authority, so
#: a stream can never be captured into a sidecar that nothing can restore.
_RESTORABLE_STREAMS = frozenset({"python", "numpy", "torch", "torch_cuda"})


class ResumeStateError(RuntimeError):
    """The sidecar is absent, unreadable, malformed, or disagrees with its checkpoint. Named
    rather than a bare `RuntimeError` because the supervisor's resumable-halt classification
    reads it and must tell it from the run-fatal classes around it."""


class RingIdentityError(ResumeStateError):
    """The persisted ring's re-derived sha256 disagrees with the sidecar's record. Separate from
    `ResumeStateError` so a planted break can assert the specific refusal rather than the
    family — a test that accepts any exception cannot tell the two apart."""


def sha256_file(path: str | Path) -> str:
    """The file's sha256, streamed. Raises OSError if it cannot be read."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path_for(checkpoint_path: str | Path) -> Path:
    """The sidecar that belongs to `checkpoint_path`. Derived, never configured."""
    p = Path(checkpoint_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def _tensor_to_b64(t: Any) -> str:
    """A torch ByteTensor of RNG state as base64. Plain bytes, no pickle."""
    return base64.b64encode(bytes(bytearray(t.tolist()))).decode("ascii")


def _b64_to_tensor(blob: str) -> Any:
    return torch.tensor(list(base64.b64decode(blob)), dtype=torch.uint8)


def capture_rng_streams() -> dict[str, Any]:
    """Snapshot the PYTHON-SIDE RNG streams `seed_everything` seeds.

    NO PICKLE ANYWHERE ON THIS PATH, as a security property: the sidecar is UNAUTHENTICATED, so a
    `pickle.loads` over it would hand code execution to anyone who can write one file into the
    run's checkpoint directory.

    DISCLOSED SCOPE: python, numpy, torch-cpu and torch-cuda, and NOT the ring's sampler, a Rust
    `StdRng` in the engine. Two launches of the same config reproduce their draws; what remains is
    stop/resume CONTINUITY, a deterministic REWIND to draw 1, whose closure needs `StdRng`'s
    ChaCha word position and is refused with grounds rather than deferred.
    """
    py_version, py_state, py_gauss = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    streams: dict[str, Any] = {
        "python": {"version": int(py_version), "state": [int(v) for v in py_state],
                   "gauss_next": None if py_gauss is None else float(py_gauss)},
        "numpy": {"bit_generator": str(np_name), "keys": [int(v) for v in np_keys],
                  "pos": int(np_pos), "has_gauss": int(np_has_gauss),
                  "cached_gaussian": float(np_cached)},
        "torch": {"state_b64": _tensor_to_b64(torch.get_rng_state())},
    }
    if torch.cuda.is_available():
        streams["torch_cuda"] = {
            "states_b64": [_tensor_to_b64(s) for s in torch.cuda.get_rng_state_all()],
        }
    return streams


def restore_rng_streams(streams: dict[str, Any]) -> list[str]:
    """Restore what `capture_rng_streams` captured. Returns the stream names restored.

    Rebuilds each native object FROM TYPED FIELDS, so a hostile sidecar can at worst produce a
    `ResumeStateError`. A stream present but unrestorable on THIS host is an error, not a skip.

    Raises:
        ResumeStateError: a stream name this build does not restore, or a malformed payload.
    """
    # THE NAME IS CHECKED BEFORE THE PAYLOAD IS TOUCHED: reporting a decode failure for an
    # unknown name would name the wrong defect.
    unknown = sorted(set(streams) - _RESTORABLE_STREAMS)
    if unknown:
        raise ResumeStateError(
            f"unknown rng stream(s) {unknown} in the sidecar — this build does not know how to "
            "restore them, and proceeding would resume into an undeclared sampling regime"
        )
    restored: list[str] = []
    for name, payload in streams.items():
        try:
            if name == "python":
                random.setstate((
                    int(payload["version"]),
                    tuple(int(v) for v in payload["state"]),
                    None if payload["gauss_next"] is None else float(payload["gauss_next"]),
                ))
            elif name == "numpy":
                np.random.set_state((
                    str(payload["bit_generator"]),
                    np.array(payload["keys"], dtype=np.uint32),
                    int(payload["pos"]), int(payload["has_gauss"]),
                    float(payload["cached_gaussian"]),
                ))
            elif name == "torch":
                torch.set_rng_state(_b64_to_tensor(payload["state_b64"]))
            elif name == "torch_cuda":
                if not torch.cuda.is_available():
                    raise ResumeStateError(
                        "sidecar carries a torch_cuda rng stream but this host has no cuda "
                        "device; resuming would silently change the sampling regime the run "
                        "was stopped in"
                    )
                torch.cuda.set_rng_state_all(
                    [_b64_to_tensor(b) for b in payload["states_b64"]]
                )
        except ResumeStateError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised as the named type, never swallowed
            raise ResumeStateError(f"rng stream {name!r} did not decode: {exc}") from exc
        restored.append(name)
    return restored


@dataclasses.dataclass(frozen=True)
class RingRef:
    """The persisted ring: where it is, what it hashed to, how many positions it held."""

    path: str
    sha256: str
    positions: int


@dataclasses.dataclass(frozen=True)
class ResumeState:
    """Everything a resume needs that the checkpoint does not already carry. `round_counter` is
    the field with no HEAD mechanism at all: the eval pipeline's counter resets to 0 on every
    launch, so without it a resumed run re-enters the promotion stride at the wrong phase."""

    version: int
    run_id: str
    step: int
    checkpoint_filename: str
    ring: RingRef | None
    round_counter: int
    last_p_hat: dict[str, float]
    anchor_sha256: str | None
    rng: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> ResumeState:
        """Rehydrate, reading every field EXPLICITLY — inventing a default is how a resume
        silently starts from a different state than it was told to.

        Raises:
            ResumeStateError: the payload is not a mapping, is a version this build does not
                read, or is missing a field.
        """
        if not isinstance(payload, dict):
            raise ResumeStateError(f"sidecar payload is {type(payload).__name__}, not an object")
        version = payload.get("version")
        if version != SIDECAR_VERSION:
            raise ResumeStateError(
                f"sidecar version {version!r} is not {SIDECAR_VERSION} — this build cannot read "
                "it, and reading it partially would resume into an undeclared state"
            )
        try:
            ring_raw = payload["ring"]
            ring = None if ring_raw is None else RingRef(
                path=ring_raw["path"], sha256=ring_raw["sha256"],
                positions=int(ring_raw["positions"]),
            )
            return cls(
                version=version,
                run_id=payload["run_id"],
                step=int(payload["step"]),
                checkpoint_filename=payload["checkpoint_filename"],
                ring=ring,
                round_counter=int(payload["round_counter"]),
                last_p_hat={str(k): float(v) for k, v in payload["last_p_hat"].items()},
                anchor_sha256=payload["anchor_sha256"],
                rng={str(k): v for k, v in payload["rng"].items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResumeStateError(f"sidecar is malformed: {exc}") from exc


def write_resume_state(state: ResumeState, checkpoint_path: str | Path) -> Path:
    """Write `state` atomically beside `checkpoint_path`, so an interrupted stop leaves either
    the previous sidecar or none. Returns the sidecar path.

    Raises:
        OSError: the sidecar could not be written — run-fatal at the caller, because a stop that
            cannot record its ring has not stopped resumably.
    """
    path = sidecar_path_for(checkpoint_path)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_resume_state(checkpoint_path: str | Path) -> ResumeState:
    """Read the sidecar belonging to `checkpoint_path` and cross-check it against it: a sidecar
    naming a different checkpoint is a pairing error that must die here rather than resume a ring
    into the wrong weights.

    Raises:
        ResumeStateError: the sidecar is absent, unreadable, malformed, or names a different
            checkpoint than the one being resumed.
    """
    path = sidecar_path_for(checkpoint_path)
    if not path.exists():
        raise ResumeStateError(
            f"no resume sidecar at {path} — the checkpoint exists but the state that makes it "
            "resumable (ring identity, round counter, rng streams) does not. Resuming would "
            "refill the ring from empty, which R343(c) forbids"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeStateError(f"resume sidecar {path} could not be read: {exc}") from exc
    state = ResumeState.from_dict(payload)
    actual = Path(checkpoint_path).name
    if state.checkpoint_filename != actual:
        raise ResumeStateError(
            f"resume sidecar names checkpoint {state.checkpoint_filename!r} but is being loaded "
            f"beside {actual!r} — a sidecar and a checkpoint that disagree are refused, never "
            "reconciled"
        )
    return state


def verify_ring(ring: RingRef) -> None:
    """Re-derive the persisted ring's sha256 and refuse a mismatch. THE COMPARISON IS THE POINT:
    a hash recorded and never checked is the phantom-gate class.

    Raises:
        RingIdentityError: the file is missing, unreadable, or hashes differently.
    """
    try:
        actual = sha256_file(ring.path)
    except OSError as exc:
        raise RingIdentityError(
            f"persisted ring {ring.path} could not be read for verification: {exc}"
        ) from exc
    if actual != ring.sha256:
        raise RingIdentityError(
            f"persisted ring {ring.path} hashes {actual} but the resume sidecar recorded "
            f"{ring.sha256} — the ring was modified after the stop that wrote it, and a resume "
            "onto an altered replay buffer is refused"
        )


__all__ = [
    "SIDECAR_SUFFIX",
    "SIDECAR_VERSION",
    "ResumeState",
    "ResumeStateError",
    "RingIdentityError",
    "RingRef",
    "capture_rng_streams",
    "load_resume_state",
    "restore_rng_streams",
    "sha256_file",
    "sidecar_path_for",
    "verify_ring",
    "write_resume_state",
]
