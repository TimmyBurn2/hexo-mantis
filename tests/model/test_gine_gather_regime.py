"""WP12-R D-FIX phase F1 (OF1-1, OF1-2) — the gather's MATERIALIZATION regime.

`_GINEConv.forward` gathers `[E, H]` node rows with `index_select`, which is
dtype-PRESERVING: it materializes in its RECEIVER's dtype, not in the autocast dtype.
Under the graph path's bf16 autocast (LAW-06) the receiver is the fp32 pre-norm tensor,
so the one tensor that scales with E is realized at 2x the width the regime implies —
the 8.94 GiB single allocation CARD-RUN5-GPU-OOM died on (MEASUREMENT_D §2/§3).

Why this file exists at all: `tests/model/test_amp_dtype.py` — the "regime-parity test"
LAW-06 names (`docs/registers/laws.md:37-39`) — is three tests on `amp_dtype_for`'s
string->dtype mapping. It instantiates no model, opens no autocast context and pins no
tensor's dtype. Before this file, NO test in this repository pinned ANY tensor's dtype on
the graph path. OF1-1 is the first, and it is an ALLOCATION property: no numeric oracle
can distinguish the fix from casting AFTER the gather (PREREG MA-3), which has identical
numerics and zero memory benefit.

Instrument: a `torch.overrides.TorchFunctionMode` records `(func, receiver.dtype,
result is receiver)` for every `index_select` / `index_add_` / `new_zeros` / `to` inside
a REAL `_GINEConv.forward`. The mode sits ABOVE the autocast dispatch key, so what it
records is the tensor as written in Python — exactly the allocation question. Nothing is
substituted: real conv, real autocast, torch's own dispatch.

Mutations this file kills (PREREG_DFIX §3): MA-1, MA-2, MA-3, MA-4, MA-5.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode

from mantis.model.gine import RepresentationNetwork, _GINEConv

# The ops whose RECEIVER dtype is the allocation fact. `to` is watched so the no-op
# property (OF1-2) is observed where it happens rather than asserted about torch.
_WATCHED = ("index_select", "index_add_", "new_zeros", "to")

_HIDDEN = 8
_N_NODES = 10
_N_EDGES = 4
_SEED = 20260803


@dataclass(frozen=True)
class _Event:
    name: str
    receiver: Tensor
    returned_receiver: bool


class _OpRecorder(TorchFunctionMode):
    """Records the receiver of every watched torch op executed inside the context."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[_Event] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        if name in _WATCHED and args and isinstance(args[0], Tensor):
            self.events.append(
                _Event(name=name, receiver=args[0], returned_receiver=result is args[0])
            )
        return result

    def named(self, name: str) -> list[_Event]:
        return [e for e in self.events if e.name == name]


def _conv_inputs() -> tuple[_GINEConv, Tensor, Tensor, Tensor]:
    """A real `_GINEConv` at the production shape: its `edge_in` is the ALREADY-projected
    hidden-dim edge tensor (`gine.py:84-86`), so `lin` is Linear(H->H)."""
    torch.manual_seed(_SEED)
    conv = _GINEConv(_HIDDEN, _HIDDEN)
    x = torch.randn(_N_NODES, _HIDDEN)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long
    )[:, :_N_EDGES]
    edge_attr = torch.randn(_N_EDGES, _HIDDEN)
    return conv, x, edge_index, edge_attr


def _run(enabled: bool) -> tuple[_OpRecorder, Tensor, _GINEConv, Tensor, Tensor, Tensor]:
    conv, x, edge_index, edge_attr = _conv_inputs()
    rec = _OpRecorder()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enabled), rec:
        out = conv(x, edge_index, edge_attr)
    return rec, out, conv, x, edge_index, edge_attr


# ── the instrument's own self-test (LAW-07: the trigger self-tests every run) ─────────
def test_recorder_observes_exactly_one_gather_and_one_scatter_per_conv() -> None:
    """Without this the dtype assertions below are satisfiable by recording NOTHING.

    Exactly one `index_select` (the gather) and one `index_add_` (the scatter) execute per
    `_GINEConv.forward` call with a non-empty edge set. `edge_index[0]` is `select`, not
    `index_select`, so it does not contaminate the count.
    """
    for enabled in (False, True):
        rec, _out, *_ = _run(enabled=enabled)
        assert len(rec.named("index_select")) == 1, (
            f"autocast={enabled}: expected exactly 1 index_select inside _GINEConv.forward, "
            f"got {len(rec.named('index_select'))} — the instrument is not seeing the gather"
        )
        assert len(rec.named("index_add_")) == 1, (
            f"autocast={enabled}: expected exactly 1 index_add_, "
            f"got {len(rec.named('index_add_'))}"
        )


# ── OF1-1 — the regime pin (RED at HEAD under autocast; this is the defect) ───────────
def test_gather_receiver_and_agg_are_bf16_under_bf16_autocast() -> None:
    """OF1-1, treatment regime. Under bf16 autocast the gather's receiver — and therefore
    the `[E, H]` tensor `index_select` materializes — must be bf16, and the accumulator
    `agg` it scatters into must match it.

    RED at HEAD: `x.index_select(0, src)` receives the fp32 pre-norm tensor.
    Kills MA-1 (revert to the HEAD form) and MA-3 (cast AFTER the gather — identical
    numerics, zero memory benefit, invisible to every numeric oracle).
    """
    rec, out, *_ = _run(enabled=True)
    gather = rec.named("index_select")[0]
    scatter = rec.named("index_add_")[0]
    assert gather.receiver.dtype is torch.bfloat16, (
        f"gather receiver dtype is {gather.receiver.dtype} under bf16 autocast; "
        "index_select preserves its receiver's dtype, so the [E, H] materialization is "
        "2x the width the bf16 regime implies (R179 / CARD-RUN5-GPU-OOM)"
    )
    assert scatter.receiver.dtype is gather.receiver.dtype, (
        f"agg dtype {scatter.receiver.dtype} != gather receiver dtype "
        f"{gather.receiver.dtype}; the accumulator must be built from the SAME tensor "
        "the gather reads (MA-4)"
    )
    assert out.dtype is torch.bfloat16, f"conv output dtype {out.dtype} under bf16 autocast"


def test_every_conv_in_the_representation_gathers_in_bf16() -> None:
    """OF1-1 through the REAL 4-layer `RepresentationNetwork`, not one isolated module.

    **NOT A ⊕ WITNESS ON CPU — and the earlier docstring said it was. Corrected here, in the
    file, not only in a log (R96 / the dispatcher-correction-propagation rule).**

    It read *"RED at HEAD"*. **Measured: GREEN at HEAD on CPU.** `gine.py` reverted to
    `88050d7`'s content (sha256 `f52ea88c5cbf93c8a120580d357463d6e6be98223a6e87b851559d05189fbc4b`,
    which is the R181 artifact's own `provenance.gine_py_sha256_HEAD`) on a scratch copy:
    this row PASSED and only `test_gather_receiver_and_agg_are_bf16_under_bf16_autocast`
    failed — `1 failed, 7 passed`.

    Where it is RED and where it cannot be:
      * **CUDA — RED at HEAD.** CUDA autocast promotes `layer_norm` to fp32
        (`MEASUREMENT_BF1` §2 recorded the receiver as `torch.float32` at HEAD and
        `torch.bfloat16` under F1, on a real GNN forward on the box), so the conv does
        receive an fp32 `x` and all four gathers are fp32 at HEAD.
      * **CPU — it CANNOT fail.** `nn.LayerNorm` is dtype-PRESERVING on CPU (probed
        directly: bf16 in -> bf16 out; fp32 in -> fp32 out) and `input_proj` is a `Linear`,
        so `x` is already bf16 before the loop. This row is green at HEAD, green under F1,
        and green under MA-1 / MA-3 / MA-4 — measured, all four.

    So: **kept as a regression pin and as a real CUDA pin, credited with NO ⊕ coverage on
    this machine.** The ⊕ witness for OF1-1 is
    `test_gather_receiver_and_agg_are_bf16_under_bf16_autocast`, which hands the conv an
    fp32 `x` directly and therefore reproduces the CUDA situation — it is the one that was
    RED at HEAD here, and it is the only one that was.

    The claim the row was written to carry — *"a fix reaching only one call path would pass
    the single-conv row and fail here"* — is TRUE ON CUDA and UNVERIFIED anywhere, since no
    such partial fix has been run on a GPU. It is not evidence on CPU.
    """
    torch.manual_seed(_SEED)
    net = RepresentationNetwork(in_dim=11, hidden=16, num_layers=4, edge_dim=5)
    x = torch.randn(24, 11)
    edge_index = torch.stack(
        (torch.arange(48) % 24, (torch.arange(48) * 7 + 3) % 24)
    ).to(torch.long)
    edge_attr = torch.randn(48, 5)
    rec = _OpRecorder()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True), rec:
        net(x, edge_index, edge_attr)
    gathers = rec.named("index_select")
    assert len(gathers) == 4, f"expected 4 gathers (one per layer), got {len(gathers)}"
    dtypes = [g.receiver.dtype for g in gathers]
    assert dtypes == [torch.bfloat16] * 4, f"per-layer gather receiver dtypes: {dtypes}"


def test_gather_receiver_and_agg_are_fp32_without_autocast() -> None:
    """OF1-1, reference regime. With autocast OFF nothing may change dtype: the deploy
    fp32 arm and `tests/model/test_forward_parity.py`'s 1e-6 goldens depend on it.

    Kills MA-2 (a hard-coded `x.to(torch.bfloat16)` fires in BOTH regimes).
    """
    rec, out, *_ = _run(enabled=False)
    gather = rec.named("index_select")[0]
    scatter = rec.named("index_add_")[0]
    assert gather.receiver.dtype is torch.float32, (
        f"gather receiver dtype is {gather.receiver.dtype} with autocast OFF — the cast "
        "became real in the fp32 regime, which is a silent precision change on deploy"
    )
    assert scatter.receiver.dtype is torch.float32, f"agg dtype {scatter.receiver.dtype}"
    assert out.dtype is torch.float32, f"conv output dtype {out.dtype} with autocast off"


# ── OF1-2 — the no-op property DESIGN §2.6's blast-radius argument rests on ───────────
def test_gather_receiver_is_the_input_tensor_without_autocast() -> None:
    """OF1-2 clause (i): with autocast off, `x.to(e.dtype) is x`.

    `Tensor.to` returns `self` when the dtype already matches, so the aligned tensor must
    be the SAME OBJECT the caller handed in — the observable form of the identity, and the
    reason the fp32 arm is bit-unchanged rather than merely close.

    Kills MA-5 (`x.clone().to(e.dtype)`: identity False, every numeric oracle green) and
    MA-2 (a hard-coded bf16 cast: identity False in the fp32 regime).
    """
    rec, _out, _conv, x, _ei, _ea = _run(enabled=False)
    gather = rec.named("index_select")[0]
    assert gather.receiver is x, (
        "the gather's receiver is not the tensor handed to forward — with autocast off "
        "the alignment allocated a copy, so F1 is not a no-op in the fp32 regime and "
        "DESIGN §2.6's whole blast-radius argument fails"
    )
    for ev in rec.named("to"):
        assert ev.returned_receiver, (
            f"a Tensor.to inside _GINEConv.forward returned a NEW tensor with autocast "
            f"off (receiver dtype {ev.receiver.dtype}); the fp32 arm must be an exact no-op"
        )


def test_conv_output_equals_head_form_expression_without_autocast() -> None:
    """OF1-2 clause (ii): with autocast off the conv output is `torch.equal` — bit-exact,
    not close — to a locally recomputed HEAD-form expression.

    The HEAD form is recomputed here from the same module and inputs, so this is a true
    before/after of the ARITHMETIC rather than a re-assertion of the fp32 goldens (which
    `test_forward_parity.py` holds unedited as OF1-5).
    """
    rec, out, conv, x, edge_index, edge_attr = _run(enabled=False)
    assert rec.named("index_select"), "instrument saw no gather"
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
        src, dst = edge_index[0], edge_index[1]
        msg = (x.index_select(0, src) + conv.lin(edge_attr)).relu()
        agg = x.new_zeros((x.shape[0], x.shape[1]))
        agg.index_add_(0, dst, msg)
        expected = conv.nn(agg + (1.0 + conv.eps) * x)
    assert torch.equal(out, expected), (
        "conv output differs from the HEAD-form expression with autocast off; max abs "
        f"diff {float((out - expected).abs().max()):.3e}. F1 is only licensed as a "
        "materialization change UNDER the bf16 policy — it may not move the fp32 arm"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_edge_branch_is_untouched(enabled: bool) -> None:
    """The `E == 0` branch keeps `x`'s dtype in both regimes (DESIGN §2.3, recorded so the
    asymmetry is not read as an oversight). Numerically identical either way — zeros."""
    conv, x, _ei, _ea = _conv_inputs()
    empty_index = torch.zeros((2, 0), dtype=torch.long)
    empty_attr = torch.zeros((0, _HIDDEN))
    rec = _OpRecorder()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enabled), rec:
        out = conv(x, empty_index, empty_attr)
    assert not rec.named("index_select"), "no gather may run on an empty edge set"
    assert out.shape == (_N_NODES, _HIDDEN)
