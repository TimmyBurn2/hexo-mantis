"""The resume sidecar, and the refusals that make it a mechanism.

One defect per group of rows:

* a sidecar that round-trips but is never cross-checked resumes a ring into the WRONG weights;
* a hash that is written and never compared is a phantom gate, so one row flips a byte of a
  persisted ring and asserts the specific error, not the family;
* a `.get(key, default)` read in `from_dict` would resume an older sidecar into an undeclared
  state, so every absent-field row asserts a refusal;
* a captured RNG stream nobody restores is a field, not a mechanism, so the round-trip row
  drives real draws on both sides of the restore.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from mantis.train import resume_state
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
    """Prove a missing sidecar field is refused, never defaulted into a state nobody declared."""
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
    """Prove a ring corrupted after its hash was recorded is refused on load.

    The assertion names the specific error: accepting any exception could not tell a refused
    corruption from a missing file.
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
    """Prove the captured RNG streams restore, by comparing real draws either side."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    blobs = capture_rng_streams()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    # Advance all three past the captured point.
    for _ in range(17):
        random.random()
        np.random.rand()
        torch.rand(1)
    restored = restore_rng_streams(blobs)
    assert {"python", "numpy", "torch"} <= set(restored)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1).item())) == expected


def test_an_unknown_rng_stream_is_refused(tmp_path: Path) -> None:
    """Prove a stream this build cannot restore stops the resume rather than being skipped."""
    with pytest.raises(ResumeStateError, match="undeclared sampling regime"):
        restore_rng_streams({"jax": "AAAA"})


def test_an_undecodable_rng_blob_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResumeStateError, match="did not decode"):
        restore_rng_streams({"python": "!!!not-base64!!!"})


def test_the_sidecar_path_never_unpickles(tmp_path: Path) -> None:
    """Prove the resume module imports no pickle: the sidecar is unauthenticated input.

    Nothing signs this file, so unpickling it would hand code execution to anyone able to write
    one file into the checkpoint directory. Structural, because a behavioural test cannot see a
    `pickle.loads` that has not been reached yet.
    """
    src = Path(resume_state.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "pickle" not in imported, (
        "resume_state imports pickle — the sidecar is unauthenticated input and every RNG "
        "stream it carries is representable as JSON scalars and base64 bytes"
    )


def test_the_sidecar_is_plain_json_with_no_opaque_stream_blob(tmp_path: Path) -> None:
    """Prove the RNG streams are typed fields, not one opaque blob per stream.

    A single blob would parse as JSON while still being a serialized object graph.
    """
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path), ckpt)
    rng = json.loads(sidecar_path_for(ckpt).read_text(encoding="utf-8"))["rng"]
    assert set(rng["python"]) == {"version", "state", "gauss_next"}
    assert isinstance(rng["python"]["state"], list)
    assert set(rng["numpy"]) == {"bit_generator", "keys", "pos", "has_gauss", "cached_gaussian"}
    assert set(rng["torch"]) == {"state_b64"}


def test_sidecar_write_is_atomic_leaving_no_temp_behind(tmp_path: Path) -> None:
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"x")
    write_resume_state(_state(tmp_path), ckpt)
    assert not list(tmp_path.glob("*.tmp"))
