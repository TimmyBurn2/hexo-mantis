"""The gather's MATERIALIZATION regime.

`index_select` is dtype-PRESERVING: it materializes in its RECEIVER's dtype, not in the autocast
dtype, so under the graph path's bf16 autocast the fp32 pre-norm receiver realizes the one
tensor that scales with E at 2x the width the regime implies — the 8.94 GiB single allocation
the run5 GPU OOM died on. No numeric oracle can distinguish the fix from casting AFTER the
gather, so this is an ALLOCATION property, and it is the first test to pin any tensor's dtype on
the graph path. The instrument is a `TorchFunctionMode` recording `(func, receiver.dtype,
result is receiver)` for every watched op inside a REAL `_GINEConv.forward`; it sits ABOVE the
autocast dispatch key, so it records the tensor as written in Python.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode

from mantis.model.gine import RepresentationNetwork, _GINEConv

# The ops whose RECEIVER dtype is the allocation fact; `to` shows the no-op where it happens.
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


def test_recorder_observes_exactly_one_gather_and_one_scatter_per_conv() -> None:
    """Without this the dtype assertions below are satisfiable by recording NOTHING: exactly one
    `index_select` and one `index_add_` run per `_GINEConv.forward` with a non-empty edge set."""
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


def test_gather_receiver_and_agg_are_bf16_under_bf16_autocast() -> None:
    """Under bf16 autocast the gather's receiver — and so the `[E, H]` tensor `index_select`
    materializes — must be bf16, and the accumulator it scatters into must match. RED at HEAD,
    where `x.index_select(0, src)` receives the fp32 pre-norm tensor; also kills a cast placed
    AFTER the gather, which has identical numerics and zero memory benefit."""
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
    """The regime pin through the REAL 4-layer `RepresentationNetwork`, not one isolated module.

    NOT a coverage witness on CPU: `LayerNorm` is dtype-preserving there (probed) and
    `input_proj` is a Linear, so `x` is already bf16 before the loop and this row is green at
    HEAD and under every mutant. On CUDA autocast promotes `layer_norm` to fp32, the conv gets an
    fp32 `x`, and all four gathers are fp32 at HEAD: there it is a real pin.
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
    """Reference regime: with autocast OFF nothing may change dtype — the deploy fp32 arm and the
    1e-6 forward-parity goldens depend on it. A hard-coded bf16 cast would fire in BOTH."""
    rec, out, *_ = _run(enabled=False)
    gather = rec.named("index_select")[0]
    scatter = rec.named("index_add_")[0]
    assert gather.receiver.dtype is torch.float32, (
        f"gather receiver dtype is {gather.receiver.dtype} with autocast OFF — the cast "
        "became real in the fp32 regime, which is a silent precision change on deploy"
    )
    assert scatter.receiver.dtype is torch.float32, f"agg dtype {scatter.receiver.dtype}"
    assert out.dtype is torch.float32, f"conv output dtype {out.dtype} with autocast off"


def test_gather_receiver_is_the_input_tensor_without_autocast() -> None:
    """With autocast off, `x.to(e.dtype) is x`: `Tensor.to` returns `self` when the dtype already
    matches, so the fp32 arm is bit-unchanged. Kills `x.clone().to(...)`, whose identity is False
    while every numeric oracle stays green, and a hard-coded bf16 cast."""
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
    """With autocast off the conv output is `torch.equal` — bit-exact, not close — to a locally
    recomputed HEAD-form expression: a before/after of the ARITHMETIC, not of the goldens."""
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
    """The `E == 0` branch keeps `x`'s dtype in both regimes; numerically identical either way."""
    conv, x, _ei, _ea = _conv_inputs()
    empty_index = torch.zeros((2, 0), dtype=torch.long)
    empty_attr = torch.zeros((0, _HIDDEN))
    rec = _OpRecorder()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enabled), rec:
        out = conv(x, empty_index, empty_attr)
    assert not rec.named("index_select"), "no gather may run on an empty edge set"
    assert out.shape == (_N_NODES, _HIDDEN)
