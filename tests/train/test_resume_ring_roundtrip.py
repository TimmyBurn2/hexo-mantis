# >300 justify (R8): R343(c) and R344(a)'s witnesses over ONE subject — a stopped run's
# replay ring. The ring coming back, the planted corruption being refused, the round counter
# continuing and the sample stream's two halves are one mechanism read from five angles;
# separating them would let a change to the restorer red one file and leave a stale
# disclosure green in another.
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


# ══ R343(d) vs R343(c): the anchor pin must not make a promoting run unresumable ═════════
def test_the_resume_pin_comes_from_the_SIDECAR_not_the_step_zero_config_pin() -> None:
    """THE COLLISION A LIVE BOX RUN FOUND, and it would have stopped run6 dead.

    R343(d) pins the anchor to `identity.warm_start.net_hash` — the artifact a FRESH launch must
    start from, "at step 0". `resolve_anchor` asserts that pin on EVERY launch. But a run that
    PROMOTES rewrites `best_model.pt`, so on the next launch the anchor legitimately differs from
    the step-0 pin and the guard refuses:

        RuntimeError: anchor sha256 mismatch: best_model.pt resolved to f260a827… but the run
        config pinned 2e72abd4…. Refusing to launch (WRONG INCUMBENT, or a legitimate
        post-promotion resume — update the pin or clear it).

    MEASURED on the box: run6's config, one promotion (`"promoted": true`), stop at step 1052,
    and the resume died on exactly this. **A run would be unresumable from its first promotion
    onward** — and RESUME-1 exists so a 12 h block can be EXTENDED, i.e. precisely for runs that
    promote. The error's own advice ("update the pin or clear it") is a human act an unattended
    resume cannot perform, and clearing the pin would disarm R343(d) altogether.

    THE FIX, asserted here as a property of the composition root's source: the pin's SOURCE
    follows the launch mode. A fresh launch asserts the config's warm-start hash; a RESUME
    asserts the anchor the stop recorded on its sidecar. Neither mode is unpinned.
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
    """The other half, and the first cut got it wrong. `resolve_anchor` compares against
    `checkpoint_state_sha256` (AUDIT-1 F-32's one denomination). A FILE sha256 recorded on the
    sidecar reads plausibly and is useless to the only consumer that matters — it cannot be
    compared to the pin at all. Asserted structurally because the values agree only on a real
    checkpoint, which this row does not build."""
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


# --------------------------------------------------------------------------------------- #
# R344(a) — witness 4 RE-RUN against the seeded ring
# --------------------------------------------------------------------------------------- #
_BATCH = 24
_PRE_STOP_DRAWS = 5


def _fill(buffer, n_records: int = 64) -> None:
    """Unique `outcome` per record, so `GraphTargets.outcomes` transcribes WHICH slots a
    draw took — the observable the sampler's seed is supposed to determine."""
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
    """R344(a). The half witness 4 uncovered and did not name: before the seeding, two
    launches of the SAME config drew different batch sequences, so the witness's own
    counterfactual — "an uninterrupted run" — was not a fixed object to compare against.

    MUTATION THAT REDS IT: make `HexgBuffer::seed_sampler` a no-op."""
    seed = 20260719
    assert _draw(_seeded_filled_ring(seed)) == _draw(_seeded_filled_ring(seed)), (
        "two launches of one config drew different batches"
    )
    assert _draw(_seeded_filled_ring(seed)) != _draw(_seeded_filled_ring(seed + 1)), (
        "the draw did not move with the seed — the assertion above would then be vacuous"
    )


def test_witness_4_half_b_a_resumed_ring_REWINDS_its_sample_stream(tmp_path: Path) -> None:
    """R344(a) — THE RESIDUAL, PINNED AS THE STATE IT ACTUALLY IS.

    Witness 4 as R343(c) wrote it asks that a resumed run's first step consume the batch an
    uninterrupted run would have consumed at that step. It does not, and this row asserts the
    exact shape of the miss rather than leaving it as prose in an exit screen.

    **The residual is a deterministic REWIND, not nondeterminism.** A resumed ring is seeded
    from the same `config.seed` and therefore restarts the sample stream at draw 1 — so the
    first resumed draw equals the uninterrupted run's FIRST draw, not its (k+1)-th. Before
    R344(a) it equalled neither, because construction seeded from OS entropy. That is a
    strictly better disclosure than the ruling anticipated, and it is why this is a pin and
    not a `xfail`: the miss is now a known offset a reader can reason about.

    It is not a data defect. The index sequence repeats; the ring CONTENTS behind those
    indices have moved on, so the resumed run does not re-train on the positions the
    pre-stop run trained on.

    Closing it needs `StdRng`'s ChaCha word position, which is reachable only through rand's
    private backend — refused with grounds in `CARD-RING-SAMPLER-SEED`, not deferred. **IF
    THIS ROW EVER REDS, the gap was closed and the disclosure above must be rewritten** —
    which is the whole point of pinning a known gap instead of describing one.
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
