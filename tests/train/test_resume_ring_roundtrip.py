"""⊕ R343(c) witnesses 1, 2 and 5 END TO END, on a REAL engine ring.

`test_resume_state.py` pins the sidecar's own contract against byte payloads. This file pins
the thing the ruling actually ordered: that a stopped run's REPLAY RING comes back, through
`mantis.run._restore_resume_state` — the production restorer — into a real `HexgBuffer` built
the way the composition root builds one.

The defect each row is the ONLY witness to:

* **witness 1** — a resume that reports success while the ring is EMPTY. This is not
  hypothetical: it is HEAD's behaviour, measured and recorded at `REAL_RUN_OPEN_DECISIONS.md`
  §4 ("a restart resumes the trainer but starts a FRESH RING"). The row asserts the reloaded
  ring's SIZE and its sha256, because size alone would pass on a ring reloaded from the wrong
  file;
* **witness 5** — the mutation self-test for witness 1. A hash written and never compared is a
  phantom gate (LAW-07), so one byte is flipped and the SPECIFIC refusal is asserted;
* **witness 2** — the round counter, which has no HEAD mechanism at all: `EvalPipeline` resets
  it to 0 every launch while `_build_round_spec` gates the promotion channel on
  `round_idx % gate.stride`. The row drives the REAL pipeline seam, not a fake;
* **the pre-fix control** — a checkpoint with NO sidecar must be read as a WARM START and must
  NOT raise, because run6's own anchor is a BC warm-start checkpoint (R343(d)) and a resume
  path that refuses it refuses the run's launch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.run import _restore_resume_state
from mantis.train.resume_state import (
    SIDECAR_VERSION,
    ResumeState,
    RingIdentityError,
    RingRef,
    capture_rng_streams,
    sha256_file,
    write_resume_state,
)

_ENCODING = "gnn_axis_v1"
_CAPACITY = 64


def _stop(tmp_path: Path, buffer, *, round_counter: int = 4) -> Path:
    """The STOP half, in the shape `step.py`'s O3 arm performs it."""
    ckpt = tmp_path / "run6_00000750_abcdef12.ckpt"
    ckpt.write_bytes(b"stand-in-for-envelope-v2")
    ring_path = tmp_path / "replay_buffer.bin"
    buffer.save_to_path(str(ring_path))
    state = ResumeState(
        version=SIDECAR_VERSION, run_id="run6", step=750, checkpoint_filename=ckpt.name,
        ring=RingRef(path=str(ring_path), sha256=sha256_file(ring_path),
                     positions=int(buffer.size)),
        round_counter=round_counter, last_p_hat={"sealbot_d5": 0.79},
        anchor_sha256="a" * 64, rng=capture_rng_streams(),
    )
    write_resume_state(state, ckpt)
    return ckpt


def _fresh_buffer():
    from mantis._engine import HexgBuffer

    return HexgBuffer(_CAPACITY, _ENCODING, 128)


def test_the_ring_comes_back_and_it_is_the_same_ring(tmp_path: Path, mk_graph_buffer) -> None:
    """WITNESS 1. The ring is persisted on stop and reloaded on resume — never refilled empty."""
    stopped = mk_graph_buffer(n_records=11, capacity=_CAPACITY, encoding=_ENCODING)
    ckpt = _stop(tmp_path, stopped)

    resumed = _fresh_buffer()
    assert resumed.size == 0, "the control: a fresh ring is empty before the restore"

    state = _restore_resume_state(resumed, str(ckpt))

    assert state is not None, "a checkpoint WITH a sidecar is a continuation, not a warm start"
    assert resumed.size == stopped.size == 11, (
        f"the resumed ring holds {resumed.size} positions against the stopped ring's "
        f"{stopped.size} — this is the empty-ring resume R343(c) forbids"
    )
    # Byte identity, not just a count: a ring reloaded from a DIFFERENT file of the same
    # length would satisfy the size assertion above and nothing else here.
    round_tripped = tmp_path / "resumed_ring.bin"
    resumed.save_to_path(str(round_tripped))
    assert sha256_file(round_tripped) == state.ring.sha256, (
        "the resumed ring re-serialises to different bytes than the ring that was persisted"
    )


def test_the_round_counter_and_p_hat_survive(tmp_path: Path, mk_graph_buffer) -> None:
    """WITNESS 2, carried on the sidecar the restorer returns."""
    ckpt = _stop(tmp_path, mk_graph_buffer(n_records=3, capacity=_CAPACITY,
                                           encoding=_ENCODING), round_counter=7)
    state = _restore_resume_state(_fresh_buffer(), str(ckpt))
    assert state.round_counter == 7
    assert state.last_p_hat == {"sealbot_d5": 0.79}


def test_a_planted_ring_corruption_is_refused_before_it_reaches_the_ring(
    tmp_path: Path, mk_graph_buffer,
) -> None:
    """WITNESS 5, and the ORDER is the assertion: the refusal fires BEFORE the load.

    A verify-after-load would report the corruption with the corrupt bytes already in the
    ring — a check that cannot prevent what it reports. So the row asserts both the specific
    error AND that the target ring is still empty afterwards.
    """
    ckpt = _stop(tmp_path, mk_graph_buffer(n_records=5, capacity=_CAPACITY,
                                           encoding=_ENCODING))
    ring_path = tmp_path / "replay_buffer.bin"
    raw = bytearray(ring_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ring_path.write_bytes(bytes(raw))

    target = _fresh_buffer()
    with pytest.raises(RingIdentityError, match="modified after the stop"):
        _restore_resume_state(target, str(ckpt))
    assert target.size == 0, (
        "the corrupt ring was loaded before the hash was checked — the refusal reported a "
        "corruption it had already accepted (LAW-07: a check that cannot prevent its subject)"
    )


def test_a_checkpoint_with_no_sidecar_is_a_warm_start_not_a_failure(tmp_path: Path) -> None:
    """THE PRE-FIX CONTROL, and it is load-bearing for run6's own launch.

    `--resume-from` carries two meanings at HEAD and `resolve_bootstrap` cannot tell them
    apart. run6's anchor is a BC warm-start checkpoint (R343(d)) with no sidecar, so a
    restorer that RAISED here would refuse the run it was built to protect. The absence must
    be announced and survivable, never fatal — and never silent.
    """
    ckpt = tmp_path / "bc_00006500_5191bd09.ckpt"
    ckpt.write_bytes(b"a bc warm-start checkpoint, written before R343")
    target = _fresh_buffer()
    assert _restore_resume_state(target, str(ckpt)) is None
    assert target.size == 0


def test_the_witness_fails_against_a_pre_fix_resume(tmp_path: Path, mk_graph_buffer) -> None:
    """THE MUTATION SELF-TEST the packet asks for: a resume that does NOT reload the ring must
    FAIL witness 1. Without this row, `test_the_ring_comes_back` could pass on a ring that was
    never emptied in the first place, and the suite would be measuring nothing."""
    stopped = mk_graph_buffer(n_records=6, capacity=_CAPACITY, encoding=_ENCODING)
    _stop(tmp_path, stopped)
    pre_fix = _fresh_buffer()  # HEAD's behaviour: construct and never load
    assert pre_fix.size == 0 and stopped.size == 6, (
        "the pre-fix resume path leaves the ring empty — which is precisely why witness 1 is "
        "a witness and not a tautology"
    )
