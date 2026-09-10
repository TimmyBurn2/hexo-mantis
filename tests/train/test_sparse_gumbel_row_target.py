"""R347(a)/(b) — the sparse Gumbel row's trainer side: the reconstructed tail and the
fast-arm policy weight.

WHAT THE ROW IS. Under Sequential Halving only `selfplay.gumbel_m` candidates are ever
visited, so the completed-Q target is exact on those m entries and, on every unvisited legal
action, is the recording prior times ONE scalar. The row therefore stores the m explicit
entries plus the tail mass alpha, and the trainer rebuilds the tail as
`alpha * its own DETACHED current prior, renormalized over the remaining legal set`.

THE DETACH IS THE MECHANISM. Built from a live `probs`, the tail would be a function of the
parameters and the CE gradient would pick up a second term pushing the prior toward whatever
it already is — a self-referential objective on every unvisited action, which is most of the
legal set. `test_the_tail_is_formed_from_the_detached_prior` is the planted break: it holds
the detached reference and the non-detached one side by side and requires the shipped
gradient to match the first and DIFFER from the second.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mantis.train.events import GUMBEL_TAIL_MASS_KEY, tail_mass_block
from mantis.train.losses import (
    _segment_softmax,
    graph_loss_denominators,
    graph_policy_row_weights,
    ragged_policy_ce,
)


def _fixture() -> dict[str, torch.Tensor]:
    """Two graphs of 5 and 4 legal nodes; the first carries a real tail, the second none."""
    logits = torch.tensor(
        [0.4, -0.2, 1.1, 0.0, -0.7, 0.9, 0.3, -0.5, 0.2], dtype=torch.float32
    )
    legal_offsets = torch.tensor([0, 5, 9], dtype=torch.long)
    # Graph 0: nodes 0 and 2 are the explicit entries and carry 0.7 of the mass; alpha 0.3.
    # Graph 1: fully explicit, alpha 0.
    policy_target = torch.tensor(
        [0.5, 0.0, 0.2, 0.0, 0.0, 0.4, 0.25, 0.2, 0.15], dtype=torch.float32
    )
    explicit_mask = torch.tensor([1, 0, 1, 0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    tail_mass = torch.tensor([0.3, 0.0], dtype=torch.float32)
    return {
        "logits": logits,
        "legal_offsets": legal_offsets,
        "policy_target": policy_target,
        "explicit_mask": explicit_mask,
        "tail_mass": tail_mass,
    }


def _rebuilt_target(f: dict[str, torch.Tensor], *, detached: bool) -> torch.Tensor:
    """The full target the reconstruction implies, built OUTSIDE `ragged_policy_ce`.

    Deliberately a second expression of the rule rather than a call into the first: a
    reference that reused the shipped code could not tell the detached form from the
    non-detached one, which is the whole subject here.
    """
    probs = _segment_softmax(f["logits"], f["legal_offsets"])
    prior = probs.detach() if detached else probs
    tail_prior = prior * (1.0 - f["explicit_mask"].to(prior.dtype))
    counts = f["legal_offsets"][1:] - f["legal_offsets"][:-1]
    seg = torch.repeat_interleave(torch.arange(2, dtype=torch.long), counts)
    denom = torch.zeros(2, dtype=tail_prior.dtype)
    denom.scatter_add_(0, seg, tail_prior)
    scale = f["tail_mass"] / denom.clamp_min(1e-12)
    return f["policy_target"] + tail_prior * scale[seg]


def _grad_of(loss_fn) -> torch.Tensor:
    f = _fixture()
    f["logits"] = f["logits"].clone().requires_grad_(True)
    loss = loss_fn(f)
    loss.backward()
    assert f["logits"].grad is not None
    return f["logits"].grad.clone()


def test_the_tail_reconstruction_reproduces_the_full_target() -> None:
    """The loss with `(explicit_mask, tail_mass)` equals the loss on the rebuilt full
    target — the reconstruction is a target substitution and nothing else."""
    f = _fixture()
    sparse = ragged_policy_ce(
        f["logits"], f["policy_target"], f["legal_offsets"],
        explicit_mask=f["explicit_mask"], tail_mass=f["tail_mass"],
    )
    full = ragged_policy_ce(
        f["logits"], _rebuilt_target(f, detached=True), f["legal_offsets"],
    )
    print(f"sparse-row loss {float(sparse):.9f} against rebuilt-target loss {float(full):.9f}")
    assert float(sparse) == pytest.approx(float(full), abs=1e-6)


def test_a_row_with_no_tail_is_byte_identical_to_the_stored_target() -> None:
    """alpha == 0 must add exactly nothing: the PUCT arm stores no tail and its loss may not
    move because the graph arm grew a reconstruction."""
    f = _fixture()
    zero_tail = torch.zeros_like(f["tail_mass"])
    with_recon = ragged_policy_ce(
        f["logits"], f["policy_target"], f["legal_offsets"],
        explicit_mask=f["explicit_mask"], tail_mass=zero_tail,
    )
    without = ragged_policy_ce(f["logits"], f["policy_target"], f["legal_offsets"])
    assert float(with_recon) == pytest.approx(float(without), abs=0.0)


def test_the_tail_is_formed_from_the_detached_prior() -> None:
    """PLANTED BREAK. Remove `.detach()` in `ragged_policy_ce`'s tail and this reds.

    The shipped gradient must equal the one a DETACHED-target reference produces and must
    NOT equal the one a live-target reference produces. Asserting only the first would pass
    under a reference that was itself wrong; asserting only the second would pass under any
    change at all. Both together name the mechanism.
    """
    shipped = _grad_of(lambda f: ragged_policy_ce(
        f["logits"], f["policy_target"], f["legal_offsets"],
        explicit_mask=f["explicit_mask"], tail_mass=f["tail_mass"],
    ))
    detached_ref = _grad_of(lambda f: ragged_policy_ce(
        f["logits"], _rebuilt_target(f, detached=True), f["legal_offsets"],
    ))
    live_ref = _grad_of(lambda f: ragged_policy_ce(
        f["logits"], _rebuilt_target(f, detached=False), f["legal_offsets"],
    ))

    gap_detached = float((shipped - detached_ref).abs().max())
    gap_live = float((shipped - live_ref).abs().max())
    print(
        f"max |grad| gap vs detached reference {gap_detached:.3e}; "
        f"vs non-detached reference {gap_live:.3e}"
    )
    assert gap_detached < 1e-6, (
        "the shipped tail is NOT formed from the detached prior — the gradient disagrees "
        f"with the detached reference by {gap_detached:.3e}"
    )
    assert gap_live > 1e-4, (
        "the shipped gradient is indistinguishable from one that back-propagates through "
        "the tail's own prior, so this fixture cannot see the detach at all and the test "
        f"proves nothing (gap {gap_live:.3e})"
    )


def test_supplying_one_half_of_the_reconstruction_pair_raises() -> None:
    f = _fixture()
    for kwargs in ({"explicit_mask": f["explicit_mask"]}, {"tail_mass": f["tail_mass"]}):
        with pytest.raises(ValueError, match="one argument in two parts"):
            ragged_policy_ce(f["logits"], f["policy_target"], f["legal_offsets"], **kwargs)


def test_a_graph_whose_legal_set_is_entirely_explicit_survives_a_positive_alpha() -> None:
    """No tail to spread over is a finite, contribution-free case — not a divide by zero."""
    f = _fixture()
    all_explicit = torch.ones_like(f["explicit_mask"])
    loss = ragged_policy_ce(
        f["logits"], f["policy_target"], f["legal_offsets"],
        explicit_mask=all_explicit, tail_mass=torch.tensor([0.3, 0.3]),
    )
    assert torch.isfinite(loss)


# ── R347(b): the fast-arm policy weight ───────────────────────────────────────────────
def test_the_row_weight_at_zero_is_the_binary_gate() -> None:
    ifs = np.array([1, 0, 1, 0], dtype=np.uint8)
    w = graph_policy_row_weights(ifs, 0.0)
    assert torch.equal(w, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_the_row_weight_lifts_the_fast_arm_and_the_denominator_follows() -> None:
    """The numerator's vector and the denominator's sum come from ONE evaluation of the
    rule; a second evaluation is how they would come to disagree."""
    ifs = np.array([1, 0, 1, 0], dtype=np.uint8)
    w = graph_policy_row_weights(ifs, 0.25)
    assert torch.equal(w, torch.tensor([1.0, 0.25, 1.0, 0.25]))
    p_den, _ = graph_loss_denominators(w, np.array([1, 1, 1, 1], dtype=np.uint8), 4)
    print(f"policy denominator at fast_policy_weight=0.25: {p_den}")
    assert p_den == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_a_negative_or_nonfinite_fast_policy_weight_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="fast_policy_weight"):
        graph_policy_row_weights(np.array([1, 0], dtype=np.uint8), bad)


# ── R347(a)/LAW-18: the in-run alpha reading ──────────────────────────────────────────
def test_the_tail_mass_block_reports_the_steps_own_distribution() -> None:
    block = tail_mass_block([0.0, 0.25, 0.5, 0.75, 1.0])[GUMBEL_TAIL_MASS_KEY]
    print(f"tail-mass block: {block}")
    assert block["n_rows"] == 5
    assert block["mean"] == pytest.approx(0.5)
    assert block["p50"] == pytest.approx(0.5)
    assert block["max"] == pytest.approx(1.0)
    assert block["rows_with_tail"] == 4


def test_a_step_with_no_graph_rows_omits_the_key_rather_than_keying_a_none() -> None:
    """An absence must not read as a measured zero (R249): a step with no rows has no alpha
    at all, so the key is omitted."""
    assert tail_mass_block([]) == {}


def test_a_puct_step_reports_a_measured_zero_rather_than_an_absence() -> None:
    block = tail_mass_block([0.0, 0.0, 0.0])[GUMBEL_TAIL_MASS_KEY]
    assert block["rows_with_tail"] == 0
    assert block["max"] == 0.0


# ── the LIVE producer, end to end ─────────────────────────────────────────────────────
def test_the_real_graph_trainer_step_publishes_the_tail_mass_reading(tmp_path) -> None:
    """LAW-07/LAW-18 producer test: the alpha block reaches the emitted `trainer_step`, and
    it carries what the ROWS carried.

    A block builder that is only unit-tested proves the arithmetic and not the wiring; this
    drives the production graph step and reads the event the sink actually received. The
    fixture plants ONE row with a real tail, so a block that reported a constant or dropped
    the per-row values would disagree with `max`.
    """
    import _microbatch_harness as H  # noqa: PLC0415 — the tests/train rootdir harness
    from mantis._engine import HexgBuffer
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import run_declared_train_step

    planted = 0.375
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i in range(8):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
        # The explicit half sums to 1 - alpha on the planted row: the bridge's own
        # distribution check is over explicit mass PLUS the tail (R347(a)).
        tail = planted if i == 0 else 0.0
        scale = 1.0 - tail
        policy = [(2, 0, 0.6 * scale), (1, 1, 0.4 * scale)]
        buf.push_graph_position(stones, policy, 1, 30, 2 + i, True,
                                1.0 if i % 2 == 0 else -1.0, True, 10 + i, -1, tail)

    # SEEDED, and the row this fixes is why. The ring samples WITH REPLACEMENT, so 8 draws
    # over 8 rows miss the single planted row about a third of the time — the assertion below
    # was a coin toss, and it lost one on a full-tier run. `H.SEED` is the harness's own seed
    # and draws the planted row; a re-seed that stopped drawing it reds here, which is the
    # right place for that to be noticed.
    buf.seed_sampler(H.SEED)
    replay = H.ReplayWireBuffer(buf, 8)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=0)
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=8, augment=False, recency_weight=0.0,
        recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
        sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)

    events = sink.named("trainer_step")
    assert len(events) == 1
    block = events[0][GUMBEL_TAIL_MASS_KEY]
    print(f"trainer_step {GUMBEL_TAIL_MASS_KEY}: {block}")
    assert block["n_rows"] == 8
    # `>= 1`, not `== 1`: the ring samples WITH REPLACEMENT, so the planted row can be drawn
    # more than once. The claim is that it reached the event at all, and that the reading is
    # the ROWS' rather than a constant — which `max` is what pins.
    assert block["rows_with_tail"] >= 1, (
        "the planted tail did not reach the event — the reading is not the rows'")
    assert block["max"] == pytest.approx(planted, abs=1e-6)
    assert block["p50"] == 0.0
