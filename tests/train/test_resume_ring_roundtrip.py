# >300 justify (R8): one subject — a stopped run's replay ring — read from five angles. Split
# apart, a change to the restorer reds one file and leaves a stale disclosure green in another.
"""A stopped run's REPLAY RING comes back end to end, on a REAL engine ring.

`test_resume_state.py` pins the sidecar's own contract against byte payloads; this file drives
`mantis.run._restore_resume_state` — the production restorer — into a real `HexgBuffer` built
the way the composition root builds one: the ring reloads with its SIZE and its sha256 (size
alone passes on the wrong file, against a measured HEAD behaviour that started a FRESH RING),
one byte is flipped and the SPECIFIC refusal asserted, the round counter survives through the
REAL pipeline seam, and a checkpoint with NO sidecar is read as a WARM START rather than
refusing the launch.
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
    """The refusal fires BEFORE the load, so the row asserts the error AND the still-empty ring:
    a verify-after-load reports corruption it has already accepted."""
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
    """A checkpoint with no sidecar is a WARM START, not a failure: the run's own anchor is a BC
    warm-start checkpoint, so a restorer that RAISED here would refuse the run it protects."""
    ckpt = tmp_path / "bc_00006500_5191bd09.ckpt"
    ckpt.write_bytes(b"a bc warm-start checkpoint, written before R343")
    target = _fresh_buffer()
    assert _restore_resume_state(target, str(ckpt)) is None
    assert target.size == 0


def test_the_witness_fails_against_a_pre_fix_resume(tmp_path: Path, mk_graph_buffer) -> None:
    """MUTATION SELF-TEST: a resume that does NOT reload the ring must FAIL witness 1, or that
    witness could pass on a ring that was never emptied in the first place."""
    stopped = mk_graph_buffer(n_records=6, capacity=_CAPACITY, encoding=_ENCODING)
    _stop(tmp_path, stopped)
    pre_fix = _fresh_buffer()  # HEAD's behaviour: construct and never load
    assert pre_fix.size == 0 and stopped.size == 6, (
        "the pre-fix resume path leaves the ring empty — which is precisely why witness 1 is "
        "a witness and not a tautology"
    )


# The anchor pin must not make a promoting run unresumable.
def test_the_resume_pin_comes_from_the_SIDECAR_not_the_step_zero_config_pin() -> None:
    """The resume-time anchor pin comes from the SIDECAR, not the step-0 config pin.

    A run that PROMOTES rewrites `best_model.pt`, so the next launch's anchor legitimately
    differs from the config pin `resolve_anchor` asserts, and the guard refuses — measured on a
    real run, unresumable from its first promotion onward, with an error whose advice is a human
    act an unattended resume cannot perform. The fix, asserted against the composition root's
    source: the pin's SOURCE follows the launch mode, and neither mode is unpinned.
    """
    import ast

    src = Path(__import__("mantis.run", fromlist=["run"]).__file__).read_text(encoding="utf-8")
    assert "resumed_anchor_sha" in src, (
        "the composition root no longer derives a resume-time anchor pin — a promoting run is "
        "unresumable again"
    )
    tree = ast.parse(src)
    # The call must pass the DERIVED name, not the config's warm-start hash directly.
    passed = [
        kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "expected_anchor_sha256"
    ]
    assert passed, "nothing passes expected_anchor_sha256 — the pin is unwired"
    assert any(isinstance(v, ast.Name) and v.id == "expected_anchor" for v in passed), (
        "expected_anchor_sha256 must be the mode-dependent value, not the config pin inlined: "
        "inlining it is exactly what refused the post-promotion resume"
    )


def test_the_sidecar_records_the_anchor_in_the_GUARDS_denomination() -> None:
    """The sidecar must record the anchor in `checkpoint_state_sha256`, the denomination
    `resolve_anchor` compares against; a FILE sha256 reads plausibly and cannot be compared at
    all. Asserted structurally because the two values agree only on a real checkpoint."""
    import ast

    from mantis.train.coordinator import step as step_mod

    src = Path(step_mod.__file__).read_text(encoding="utf-8")
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_anchor_sha256"
    )
    body = ast.dump(fn)
    assert "checkpoint_state_sha256" in body, (
        "_anchor_sha256 must record the anchor in the denomination the launch pin compares "
        "against, or the resume cannot assert it"
    )
    assert "sha256_file" not in body, (
        "a FILE hash here is not comparable to the pin — that was the first cut's defect"
    )


_BATCH = 24
_PRE_STOP_DRAWS = 5


def _fill(buffer, n_records: int = 64) -> None:
    """Unique `outcome` per record, so `GraphTargets.outcomes` transcribes WHICH slots a draw
    took — the observable the sampler's seed is supposed to determine."""
    for i in range(n_records):
        k = 6 + (i % 7)
        stones = [(q, (q % 3) - 1, 1 if q % 2 == 0 else -1) for q in range(k)]
        buffer.push_graph_position(
            stones, [(-1, 0, 0.6), (k, 0, 0.4)], 1 if i % 2 == 0 else -1, 2, i % 50,
            True, -1.0 + 2.0 * i / (n_records - 1), True, 40, 10 + i,
        )


def _seeded_filled_ring(seed: int):
    from mantis._engine import HexgBuffer

    buffer = HexgBuffer(_CAPACITY, _ENCODING, 128)
    buffer.seed_sampler(seed)          # what `mantis.run._select_buffer` does at launch
    _fill(buffer)
    return buffer


def _draw(buffer) -> list[float]:
    return list(buffer.sample_graph_batch(_BATCH)[1].outcomes)


def test_witness_4_half_a_two_launches_of_one_config_now_draw_the_same_batches() -> None:
    """Two launches of the SAME config draw the same batch sequence, so the "uninterrupted run"
    counterfactual is a fixed object. MUTATION THAT REDS IT: make `seed_sampler` a no-op."""
    seed = 20260719
    assert _draw(_seeded_filled_ring(seed)) == _draw(_seeded_filled_ring(seed)), (
        "two launches of one config drew different batches"
    )
    assert _draw(_seeded_filled_ring(seed)) != _draw(_seeded_filled_ring(seed + 1)), (
        "the draw did not move with the seed — the assertion above would then be vacuous"
    )


def test_witness_4_half_b_a_resumed_ring_REWINDS_its_sample_stream(tmp_path: Path) -> None:
    """A resumed ring REWINDS its sample stream, deterministically: seeded from the same
    `config.seed`, its first draw equals the uninterrupted run's FIRST draw, not its (k+1)-th.
    Not a data defect — the indices repeat but the ring CONTENTS behind them have moved on.
    Closing it needs `StdRng`'s ChaCha word position, reachable only through rand's private
    backend. IF THIS ROW EVER REDS, the gap was closed and this disclosure must be rewritten.
    """
    seed = 20260719

    uninterrupted = _seeded_filled_ring(seed)
    first_draw = _draw(uninterrupted)
    for _ in range(_PRE_STOP_DRAWS - 1):
        _draw(uninterrupted)
    draw_after_the_stop_point = _draw(uninterrupted)

    stopped = _seeded_filled_ring(seed)
    for _ in range(_PRE_STOP_DRAWS):
        _draw(stopped)
    ring_path = tmp_path / "ring.bin"
    stopped.save_to_path(str(ring_path))

    resumed = _seeded_filled_ring(seed)
    restored = resumed.load_from_path(str(ring_path))
    assert restored > 0, "the ring must actually come back, or this measures nothing"
    resumed_draw = _draw(resumed)

    assert resumed_draw != draw_after_the_stop_point, (
        "the resumed stream CONTINUED the pre-stop stream — witness 4 as written now "
        "passes, and this row's disclosure is stale"
    )
    assert resumed_draw == first_draw, (
        "the resumed stream neither continued nor rewound — the residual is no longer the "
        "deterministic rewind this row documents"
    )
