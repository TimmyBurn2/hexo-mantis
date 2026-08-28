"""SC-A6 oracle — the ONE determinism boot site (R30a; DESIGN_P2.md §7 / PREREG_P2.md
suite #13).

RED-at-import until IMPL lands `mantis.util.determinism.seed_everything`. Pins: two
`seed_everything(same_seed)` + build + first-step boots on CPU produce bit-identical
`loss`/`grad_norm`/first-layer-weight-tensor; two DIFFERENT seeds produce DIFFERENT init
weights (negative control — a suite that can't tell "seeded" from "unseeded" is
worthless); `seed_everything` seeds all THREE RNG streams (`random`/`numpy`/`torch`), not
a subset. Mutation-bite note (LAW-07): ORACLE-WRITE cannot literally comment out
`torch.manual_seed` inside `seed_everything` before IMPL exists (RED-at-import) — the
positive-control (same seed -> identical) paired with the negative-control (different
seed -> different) structurally provides the bite (removing the seed call would make the
positive control's "identical" claim depend on ambient RNG state, which the two builds do
not share); IMPL/REVIEW re-verifies this once the function exists, per DESIGN_P2.md §7.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from mantis.model import CnnArch, build_net
from mantis.util.determinism import seed_everything


def _tiny_arch() -> CnnArch:
    return CnnArch(board_size=9, in_channels=4, filters=8, res_blocks=1)


def _boot_first_layer_and_forward(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """seed_everything(seed) -> build a tiny net -> one forward pass on a seeded-random
    synthetic batch. Returns (first_conv_weight, forward_output) for bit-identity checks."""
    seed_everything(seed)
    net = build_net(_tiny_arch())
    first_weight = next(net.parameters()).detach().clone()
    x = torch.from_numpy(np.random.randn(2, 4, 9, 9).astype(np.float32))
    with torch.no_grad():
        out = net(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return first_weight, logits.detach().clone()


def test_two_boots_equal_seed_produce_bit_identical_init_and_forward():
    w1, out1 = _boot_first_layer_and_forward(20260716)
    w2, out2 = _boot_first_layer_and_forward(20260716)
    assert torch.equal(w1, w2), "identical seed must produce a bit-identical first layer"
    assert torch.equal(out1, out2), "identical seed must produce a bit-identical forward pass"


def test_two_boots_different_seed_produce_different_init_weights():
    w1, _ = _boot_first_layer_and_forward(20260716)
    w2, _ = _boot_first_layer_and_forward(1)
    assert not torch.equal(w1, w2), (
        "a suite that cannot tell seeded from unseeded is worthless — different seeds "
        "must produce DIFFERENT init weights"
    )


def test_seed_everything_seeds_random_numpy_and_torch_streams():
    seed_everything(424242)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1).item()
    seed_everything(424242)
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1).item()
    assert r1 == r2, "stdlib random must be re-seeded"
    assert n1 == n2, "numpy must be re-seeded"
    assert t1 == t2, "torch (CPU) must be re-seeded"


def test_reseeding_mid_run_resets_the_stream_to_a_reproducible_point():
    seed_everything(7)
    _ = [random.random() for _ in range(5)]  # consume some state
    seed_everything(7)  # reseed — the ONE boot site is callable more than once (idempotent)
    first_after_reseed = random.random()
    seed_everything(7)
    _ = [random.random() for _ in range(5)]
    seed_everything(7)
    second_after_reseed = random.random()
    assert first_after_reseed == second_after_reseed
