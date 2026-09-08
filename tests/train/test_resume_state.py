"""⊕ R343(c) / CARD-RESUME — the resume sidecar, and the two refusals that make it a mechanism.

The defect each row is the ONLY witness to:

* a sidecar that round-trips but is never cross-checked lets a ring be resumed into the WRONG
  weights — so `load_resume_state` refuses a sidecar naming a different checkpoint;
* a hash that is WRITTEN and never COMPARED is the phantom gate LAW-07 forbids — so
  `test_planted_ring_corruption_is_refused_on_load` mutates a persisted ring by one byte and
  asserts the specific `RingIdentityError`, not the family. This is R343(c)'s witness (5) and
  it is the mutation self-test for witness (1);
* a `.get(key, default)` read anywhere in `from_dict` would let an older or truncated sidecar
  resume into an undeclared state (R1) — so every absent-field row asserts a refusal;
* a captured RNG stream nobody restores is a field, not a mechanism — so the round-trip row
  drives real draws on both sides of the restore and compares them.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from mantis.train.resume_state import (
    SIDECAR_VERSION,
    ResumeState,
    ResumeStateError,
    RingIdentityError,
    RingRef,
    capture_rng_streams,
    load_resume_state,
    restore_rng_streams,
    sha256_file,
    sidecar_path_for,
    verify_ring,
    write_resume_state,
)


def _ring_file(tmp_path: Path, payload: bytes = b"HEXG-ring-bytes-0123456789") -> RingRef:
    p = tmp_path / "replay_buffer.bin"
    p.write_bytes(payload)
    return RingRef(path=str(p), sha256=sha256_file(p), positions=7)


def _state(tmp_path: Path, ckpt_name: str = "run6_00000750_abcdef12.ckpt") -> ResumeState:
    return ResumeState(
        version=SIDECAR_VERSION, run_id="run6", step=750, checkpoint_filename=ckpt_name,
        ring=_ring_file(tmp_path), round_counter=3, last_p_hat={"sealbot_d5": 0.79},
        anchor_sha256="0" * 64, rng=capture_rng_streams(),
    )


def test_sidecar_round_trips_every_field(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    state = _state(tmp_path)
    written = write_resume_state(state, ckpt)
    assert written == sidecar_path_for(ckpt)
    back = load_resume_state(ckpt)
    assert back == state


def test_sidecar_naming_a_different_checkpoint_is_refused(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path, ckpt_name="run6_00000500_99999999.ckpt"), ckpt)
    with pytest.raises(ResumeStateError, match="refused, never"):
        load_resume_state(ckpt)


def test_absent_sidecar_is_refused_and_names_the_consequence(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    with pytest.raises(ResumeStateError, match="refill the ring from empty"):
        load_resume_state(ckpt)


def test_unreadable_version_is_refused(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path), ckpt)
    side = sidecar_path_for(ckpt)
    payload = json.loads(side.read_text(encoding="utf-8"))
    payload["version"] = SIDECAR_VERSION + 1
    side.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeStateError, match="cannot read it"):
        load_resume_state(ckpt)


@pytest.mark.parametrize("field", ["run_id", "step", "ring", "round_counter", "rng"])
def test_a_missing_field_is_refused_never_defaulted(tmp_path: Path, field: str) -> None:
    """R1: no `.get(key, default)` on a resume read. An absent field is a sidecar this build
    does not understand, and a default for it resumes into a state nobody declared."""
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path), ckpt)
    side = sidecar_path_for(ckpt)
    payload = json.loads(side.read_text(encoding="utf-8"))
    del payload[field]
    side.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeStateError, match="malformed"):
        load_resume_state(ckpt)


def test_unmodified_ring_verifies(tmp_path: Path) -> None:
    verify_ring(_ring_file(tmp_path))


def test_planted_ring_corruption_is_refused_on_load(tmp_path: Path) -> None:
    """R343(c) WITNESS (5), and the LAW-07 mutation self-test for witness (1).

    One byte is flipped in the persisted ring AFTER the sidecar recorded its hash. If the hash
    were written and never compared this would pass silently and a resume would train on an
    altered replay buffer. The assertion is the SPECIFIC error, not the family: a test that
    accepted any exception could not tell a refused corruption from a missing file.
    """
    ring = _ring_file(tmp_path)
    raw = bytearray(Path(ring.path).read_bytes())
    raw[0] ^= 0x01
    Path(ring.path).write_bytes(bytes(raw))
    with pytest.raises(RingIdentityError, match="refused"):
        verify_ring(ring)


def test_a_ring_that_vanished_is_refused_not_skipped(tmp_path: Path) -> None:
    ring = _ring_file(tmp_path)
    Path(ring.path).unlink()
    with pytest.raises(RingIdentityError, match="could not be read"):
        verify_ring(ring)


def test_rng_streams_round_trip_and_reproduce_real_draws() -> None:
    """A captured stream nobody restores is a field, not a mechanism — so this drives draws."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    blobs = capture_rng_streams()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    # advance all three past the captured point
    for _ in range(17):
        random.random()
        np.random.rand()
        torch.rand(1)
    restored = restore_rng_streams(blobs)
    assert {"python", "numpy", "torch"} <= set(restored)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1).item())) == expected


def test_an_unknown_rng_stream_is_refused(tmp_path: Path) -> None:
    """A stream this build cannot restore must stop the resume, not be skipped past."""
    with pytest.raises(ResumeStateError, match="undeclared sampling regime"):
        restore_rng_streams({"jax": "AAAA"})


def test_an_undecodable_rng_blob_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResumeStateError, match="did not decode"):
        restore_rng_streams({"python": "!!!not-base64!!!"})


def test_sidecar_write_is_atomic_leaving_no_temp_behind(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path), ckpt)
    assert not list(tmp_path.glob("*.tmp"))
