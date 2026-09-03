"""⊕ Suite B — collate BYTE-PARITY vs the captured old outputs (WP-SP, B-01 … B-06).

Written oracle-first against the dispatcher's old-side capture (#C1, wp/WPSP/CAPTURE_LOG.md)
BEFORE any port code. RED at import until IMPL writes `mantis.selfplay.graph_collate`.

No tolerance exists for B-01..B-05: the collate path is reshape/copy-only, so ANY byte
difference is real contract drift, not numerical noise (PREREG §3 Suite B). The capture's
own `verify_fixtures.py` reproduced these outputs bit-exactly from the reloaded npz, so a
mismatch here is a port defect, never a fixture artifact.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
from _retired_batch_fields import RETIRED_BATCH_FIELDS
from _wire_geometry import geometry_kwargs
import pytest
import torch

from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    collate_graph_batch,
    segment_softmax,
    stone_mask_from_batch,
)
from mantis.train.losses import _segment_softmax as train_segment_softmax

# The capture ran on CPU with `torch.set_num_threads(1)`; B-04/B-05 reproduce that regime.
CAPTURE_TORCH_THREADS = 1
# Seed for the B-06 property battery (test-local; B-06 compares two NEW-side copies).
DUPLICATION_BATTERY_SEED = 20260723


@pytest.fixture
def single_threaded_torch():
    """Pin torch to 1 CPU thread for the bit-parity arms, then restore the process default."""
    previous = torch.get_num_threads()
    torch.set_num_threads(CAPTURE_TORCH_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


#: The capture's geometry, READ OFF THE REGISTRY ROW it was built at (AUDIT-1 F-41), never typed.
GEOMETRY: dict[str, int] = geometry_kwargs()


def _collate(fields: dict[str, Any], **kw: Any):
    """The geometry is stated on every call — `collate_graph_batch` requires it since F-41."""
    return collate_graph_batch(GraphWirePayload(**fields), **{**GEOMETRY, **kw})


#: The ONE authority lives in `_retired_batch_fields`; a second copy here is the class this
#: mission keeps paying for. The capture `.npz` is NOT regenerated to drop retired keys — a
#: byte-parity capture whose bytes get rewritten when the code changes has stopped being a
#: capture. The extra key stays as evidence, and the retirement is asserted POSITIVELY below
#: rather than skipped, because a silent `continue` over an unmatched golden key is a check that
#: passes because it stopped checking.
_RETIRED_FIELDS = RETIRED_BATCH_FIELDS


def _assert_tensor_parity(batch, golden: dict[str, np.ndarray], label: str) -> None:
    for field, expected in golden.items():
        if field in _RETIRED_FIELDS:
            assert not hasattr(batch, field), (
                f"{label}.{field}: the batch produces a field this capture records as RETIRED. "
                "Either the retirement was reverted without updating this list, or a field was "
                "re-added under a retired name — both need saying out loud, not passing quietly"
            )
            continue
        actual = getattr(batch, field).detach().cpu().numpy()
        assert actual.shape == expected.shape, (
            f"{label}.{field}: shape {actual.shape} != {expected.shape}"
        )
        assert actual.dtype == expected.dtype, (
            f"{label}.{field}: dtype {actual.dtype} != {expected.dtype}"
        )
        assert np.array_equal(actual, expected), (
            f"{label}.{field}: byte parity lost vs the captured old collate output — the "
            "contract reader changed the bytes it hands the NN"
        )


# ═══ B-01 / B-02 / B-03 — collate output byte parity ══════════════════════════════════
def test_collate_output_byte_parity_b6(payload_fields, collated_golden):
    """B-01 — PASS iff all 12 output tensors of the B=6 mixed-spread wire are element-wise
    identical (f32 bit-exact, int exact) to the captured old outputs. FAIL = the port is not
    equivalent on the real multi-graph fused batch."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    _assert_tensor_parity(batch, collated_golden["b6"], "b6")


def test_collate_output_byte_parity_b1(payload_fields, collated_golden):
    """B-02 — PASS iff the degenerate single-graph batch is byte-identical to capture.
    FAIL = the B=1 path (node_offsets[0]==0 ⇒ local==global) drifted from the general one."""
    batch = _collate(payload_fields("b1"), expected_version=1, device="cpu", semantic="full")
    _assert_tensor_parity(batch, collated_golden["b1"], "b1")


@pytest.mark.parametrize("semantic", ["full", "off"])
def test_collate_empty_batch_parity(payload_fields, collated_golden, semantic):
    """B-03 — PASS iff B=0 collates SUCCESSFULLY and byte-identically under both semantic
    modes (the capture's two outputs were per-tensor sha-equal, so one golden serves both).
    FAIL = the empty-batch arm drifted, in shape, dtype, or existence."""
    batch = _collate(payload_fields("b0"), expected_version=1, device="cpu", semantic=semantic)
    _assert_tensor_parity(batch, collated_golden["b0"], f"b0[{semantic}]")


# ═══ B-04 / B-05 — hot-path bit parity ════════════════════════════════════════════════
def test_segment_softmax_bit_parity(payload_fields, hotpath_golden, single_threaded_torch):
    """B-04 — PASS iff `segment_softmax(captured_logits, batch.legal_offsets)` is BIT-exact
    against the captured old output (same torch minor, CPU, 1 thread). The logits vector is
    loaded from the capture, not regenerated, so the seed cannot drift. FAIL = the ragged
    softmax the Rust assembler normalizes against changed."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    logits = torch.from_numpy(hotpath_golden["logits"].copy())
    expected = hotpath_golden["segment_softmax"]

    probs = segment_softmax(logits, batch.legal_offsets).detach().cpu().numpy()
    assert probs.dtype == expected.dtype, f"dtype {probs.dtype} != {expected.dtype}"
    assert np.array_equal(probs, expected), (
        "segment_softmax output is not bit-identical to capture; PREREG permits NO tolerance "
        "here without a root-caused mechanism in IMPL_NOTES + dispatcher sign-off"
    )


def test_stone_mask_bit_parity(payload_fields, hotpath_golden, single_threaded_torch):
    """B-05 — PASS iff `stone_mask_from_batch(batch)` equals the captured mask exactly
    (60 True rows). FAIL = the value-head pooling subset changed, which silently re-weights
    every value target."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    expected = hotpath_golden["stone_mask"]

    mask = stone_mask_from_batch(batch).detach().cpu().numpy()
    assert mask.dtype == expected.dtype == np.bool_
    assert np.array_equal(mask, expected), "stone mask drifted vs capture"
    assert int(mask.sum()) == int(expected.sum()) == 60


# ═══ B-06 — the declared segment_softmax duplication pin (§c.6 ruling) ════════════════
def _normalized_body(fn_or_src: Callable[..., Any] | str) -> str:
    """AST dump of a function's BODY with the docstring dropped.

    Bodies only: the two copies are allowed to differ in name (`segment_softmax` vs
    `_segment_softmax`) and in signature annotations; they are NOT allowed to differ in a
    single statement. `ast.dump` omits line/col attributes by default, so formatting and
    comments cannot mask or manufacture a difference.
    """
    src = fn_or_src if isinstance(fn_or_src, str) else inspect.getsource(fn_or_src)
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef), "expected a single top-level function definition"
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    # Import statements are stripped before comparison: the self-play copy defers `import
    # torch` to call time (it must stay importable without torch), while the train copy relies
    # on a module-level import. That is a packaging difference, not an algorithmic one, so
    # comparing it would make this pin unsatisfiable by a verbatim port. Every remaining
    # statement is compared exactly — leg (iii)'s mutation self-test proves the pin still bites.
    body = [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def _battery() -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Property battery: single graph, many graphs, large magnitudes, count-1 segments."""
    gen = torch.Generator().manual_seed(DUPLICATION_BATTERY_SEED)
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    cases.append(("single_graph",
                  torch.randn(17, generator=gen, dtype=torch.float32),
                  torch.tensor([0, 17], dtype=torch.long)))
    counts = [3, 1, 8, 1, 12, 5]
    offsets = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long)
    cases.append(("many_graphs_with_count_1_segments",
                  torch.randn(int(offsets[-1]), generator=gen, dtype=torch.float32), offsets))
    cases.append(("large_magnitude",
                  torch.randn(20, generator=gen, dtype=torch.float32) * 500.0,
                  torch.tensor([0, 5, 20], dtype=torch.long)))
    cases.append(("all_count_1",
                  torch.randn(4, generator=gen, dtype=torch.float32),
                  torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)))
    return cases


def test_segment_softmax_train_duplication_pin():
    """B-06 — the §c.6 declared-duplication ruling made mechanical, in three legs:
    (i) numeric agreement of `selfplay.graph_collate.segment_softmax` with the inlined
    `train.losses._segment_softmax` over the property battery; (ii) normalized-AST equality
    of the two bodies; (iii) a mutation self-test proving leg (ii) actually bites.
    FAIL on any leg = the two copies can drift silently, which REVIEW-impl treats as a fail
    condition until the single-authority import lands."""
    # (i) numeric agreement
    for label, logits, offsets in _battery():
        ours = segment_softmax(logits, offsets)
        theirs = train_segment_softmax(logits, offsets)
        assert torch.equal(ours, theirs), f"{label}: the two segment_softmax copies disagree"
        seg_sums = torch.stack([
            ours[int(offsets[i]):int(offsets[i + 1])].sum()
            for i in range(len(offsets) - 1)
        ])
        assert torch.allclose(seg_sums, torch.ones_like(seg_sums), atol=1e-6), (
            f"{label}: per-segment probabilities must sum to 1"
        )

    # (ii) normalized-AST equality of the two bodies
    ours_body = _normalized_body(segment_softmax)
    theirs_body = _normalized_body(train_segment_softmax)
    assert ours_body == theirs_body, (
        "selfplay.graph_collate.segment_softmax and train.losses._segment_softmax have "
        "diverged; §c.6 rules them a declared, test-pinned duplication until train imports "
        "the selfplay authority"
    )

    # (iii) mutation self-test — leg (ii) must bite on a one-token perturbation
    mutated_src = inspect.getsource(train_segment_softmax).replace('"amax"', '"amin"')
    assert mutated_src != inspect.getsource(train_segment_softmax), (
        "mutation self-test could not perturb the source — the checker is not proven to bite"
    )
    assert _normalized_body(mutated_src) != ours_body, (
        "the AST comparison does NOT bite: a mutated body compared equal"
    )
