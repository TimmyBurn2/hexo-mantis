"""The CPU message sum is the unchunked `index_select + add + relu + index_add_` bit for bit, forward and backward, on every caller's thread."""
from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from mantis._engine import Board, HexgBuffer, edge_vocabulary
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net, gine
from mantis.model.gine import gine_message_sum
from mantis.selfplay.graph_collate import collate_graph_batch, graph_wire_from_rust, stone_mask_from_batch

_REPO = Path(__file__).resolve().parents[2]

# (chunks, extra edges): E = chunks * the op's chunk size + extra, so every boundary is read off the op itself.
_EDGE_COUNTS = {"none": (0, 0), "one": (0, 1), "chunk-1": (1, -1), "chunk": (1, 0), "chunk+1": (1, 1),
                "two-chunks": (2, 0), "ragged-tail": (3, 17)}
# H 128 is production's width (the scatter's expanded-index kernel); H 8 takes torch's narrow-row kernel.
_REGIMES = {"fp32-h128": (torch.float32, 128), "bf16-h128": (torch.bfloat16, 128),
            "fp32-h8": (torch.float32, 8), "fp64-h8": (torch.float64, 8)}


def _reference(xs: Tensor, e: Tensor, src: Tensor, dst: Tensor, rowptr: Tensor | None, divisor: Tensor | None,
               code: Tensor | None = None) -> Tensor:
    """The CPU branch before chunking, transcribed: [E, H] temporaries and one `index_add_` over every edge."""
    acc = torch.promote_types(torch.float32, xs.dtype)
    msg = (xs.index_select(0, src) + (e if code is None else e.index_select(0, code.long()))).relu()
    agg = torch.zeros((xs.shape[0], xs.shape[1]), dtype=acc).index_add_(0, dst, msg.to(acc))
    return (agg if divisor is None else agg / divisor.to(acc)).to(xs.dtype)


def _n_edges(name: str) -> int:
    chunks, extra = _EDGE_COUNTS[name]
    return chunks * gine._CPU_CHUNK_EDGES + extra


def _case(n_edges: int, dtype: torch.dtype, h: int, *, coded: bool, with_div: bool,
          seed: int = 11) -> dict[str, Any]:
    """Nodes 0 and 1 receive nothing and the last node is a dummy whose in-edges run through every chunk."""
    gen = torch.Generator().manual_seed(seed)
    n = 300
    src = torch.randint(1, n, (n_edges,), generator=gen)
    dst = torch.randint(2, n - 1, (n_edges,), generator=gen)
    dst[::3] = n - 1
    return {"xs": torch.randn(n, h, generator=gen).to(dtype),
            "e": torch.randn(1024 if coded else n_edges, h, generator=gen).to(dtype),
            "src": src, "dst": dst,
            "div": torch.randint(1, 40, (n, 1), generator=gen).float() if with_div else None,
            "code": torch.randint(0, 91, (n_edges,), generator=gen, dtype=torch.uint8) if coded else None}


def _sum(fn: Any, c: dict[str, Any]) -> Tensor:
    return fn(c["xs"], c["e"], c["src"], c["dst"], None, c["div"], c["code"])


@pytest.mark.parametrize("regime", list(_REGIMES))
@pytest.mark.parametrize("with_div", [False, True], ids=["no-div", "div"])
@pytest.mark.parametrize("coded", [False, True], ids=["per_edge", "coded"])
@pytest.mark.parametrize("edges", list(_EDGE_COUNTS))
def test_the_forward_is_the_unchunked_sum_bit_for_bit(edges: str, coded: bool, with_div: bool, regime: str) -> None:
    """`torch.equal`, not a tolerance: a reordered fp32 sum moves the policy by ~1e-6, inside the forward golden's band."""
    dtype, h = _REGIMES[regime]
    c = _case(_n_edges(edges), dtype, h, coded=coded, with_div=with_div)
    got, want = _sum(gine_message_sum, c), _sum(_reference, c)
    assert got.dtype == want.dtype and torch.equal(got, want), (
        f"max |Δ| {float((got.double() - want.double()).abs().max()) if got.numel() else 0.0:.3g}")


@pytest.mark.parametrize("coded", [False, True], ids=["per_edge", "coded"])
@pytest.mark.parametrize("xs_dtype,e_dtype", [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
                         ids=["bf16-xs", "bf16-e"])
def test_mixed_input_dtypes_form_the_message_in_the_promoted_dtype(xs_dtype: torch.dtype, e_dtype: torch.dtype,
                                                                   coded: bool) -> None:
    c = _case(_n_edges("ragged-tail"), torch.float32, 128, coded=coded, with_div=True)
    c["xs"], c["e"] = c["xs"].to(xs_dtype), c["e"].to(e_dtype)
    got, want = _sum(gine_message_sum, c), _sum(_reference, c)
    assert got.dtype == want.dtype and torch.equal(got, want)


@pytest.mark.parametrize("regime", ["fp32-h128", "bf16-h128", "fp64-h8"])
@pytest.mark.parametrize("with_div", [False, True], ids=["no-div", "div"])
@pytest.mark.parametrize("edges", list(_EDGE_COUNTS))
def test_the_gradients_are_the_unchunked_sums_bit_for_bit(edges: str, with_div: bool, regime: str) -> None:
    """The registered backward against autograd through the transcribed unchunked expression, per-edge rows."""
    dtype, h = _REGIMES[regime]
    c = _case(_n_edges(edges), dtype, h, coded=False, with_div=with_div)
    grad = torch.randn(c["xs"].shape, generator=torch.Generator().manual_seed(3)).to(dtype)
    grads = []
    for fn in (gine_message_sum, _reference):
        xs, e = c["xs"].clone().requires_grad_(), c["e"].clone().requires_grad_()
        out = fn(xs, e, c["src"], c["dst"], None, c["div"])
        grads.append(torch.autograd.grad(out, (xs, e), grad))
    for name, got, want in zip(("xs", "e"), *grads, strict=True):
        assert torch.equal(got, want), f"grad_{name} differs from the unchunked path's"


@pytest.mark.parametrize("threads", [1, 4])
def test_the_sum_is_the_same_at_every_intra_op_thread_count(threads: int) -> None:
    c = _case(_n_edges("ragged-tail"), torch.float32, 128, coded=True, with_div=True)
    want = _sum(_reference, c)
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        got = _sum(gine_message_sum, c)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(got, want)


def test_a_reordered_sum_is_caught() -> None:
    """CONTROL: the same edges summed in reversed chunk order differ in the bits, so the rows above can see an order."""
    c = _case(_n_edges("ragged-tail"), torch.float32, 128, coded=False, with_div=False)
    order = torch.cat(torch.arange(c["src"].shape[0]).split(gine._CPU_CHUNK_EDGES)[::-1])
    reordered = _reference(c["xs"], c["e"][order], c["src"][order], c["dst"][order], None, None)
    assert not torch.equal(reordered, _sum(_reference, c))


def test_each_thread_owns_its_chunk_buffers() -> None:
    """Two live threads hold distinct storage and one thread reuses its own: no buffer is shared mutable state."""
    barrier = threading.Barrier(2)
    seen: dict[int, tuple[int, int]] = {}

    def grab(i: int) -> None:
        first = gine._chunk_buffers(128, torch.float32, torch.float32)[0].data_ptr()
        second = gine._chunk_buffers(128, torch.float32, torch.float32)[0].data_ptr()
        seen[i] = (first, second)
        barrier.wait(timeout=30)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen[0][0] == seen[0][1] and seen[1][0] == seen[1][1], "a thread re-allocated its buffers per call"
    assert seen[0][0] != seen[1][0], "two threads share one chunk buffer"


def test_concurrent_callers_each_get_their_own_exact_sum() -> None:
    """Several threads summing at once, as the ladder's game threads and a CPU trainer do: every result exact."""
    cases = [_case(_n_edges("ragged-tail"), torch.float32, 128, coded=i % 2 == 1, with_div=True, seed=i)
             for i in range(4)]
    wants = [_sum(_reference, c) for c in cases]
    wrong: list[int] = []

    def run(i: int) -> None:
        for _ in range(20):
            if not torch.equal(_sum(gine_message_sum, cases[i]), wants[i]):
                wrong.append(i)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(cases))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not wrong, f"threads {sorted(set(wrong))} read another caller's buffer"


@pytest.mark.parametrize("first_inference", [True, False], ids=["inference-first", "grad-first"])
def test_a_thread_alternating_inference_mode_and_grad_mode_sums_exactly(first_inference: bool) -> None:
    """A server call and a trainer call on one thread, either order: an inference-mode buffer would refuse the in-place write."""
    c = _case(_n_edges("chunk+1"), torch.float32, 128, coded=False, with_div=False)
    want = _sum(_reference, c)
    got: list[Tensor] = []
    failed: list[BaseException] = []

    def run() -> None:
        try:
            for inference in (first_inference, not first_inference):
                with torch.inference_mode(inference):
                    got.append(_sum(gine_message_sum, c).clone())
        except RuntimeError as ex:
            failed.append(ex)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert not failed, f"{failed[0]}"
    assert len(got) == 2 and all(torch.equal(g, want) for g in got)


@pytest.fixture(scope="module")
def ragged() -> tuple[dict[str, Any], Any, Any]:
    """The production config and encoding, and five real positions of one seeded game, 335 to 496 nodes, fused by the real buffer."""
    config = load_config(production_configs(_REPO)[0]).model_dump()
    spec = lookup(config["identity"]["encoding"])
    rng = random.Random(20261006)
    board = Board.with_encoding_name(spec.name)
    buf = HexgBuffer(8, spec.name, 8)
    for ply in (4, 10, 18, 28, 40):
        while board.ply < ply:
            near = sorted(board.legal_moves(), key=lambda m: max(abs(m[0]), abs(m[1]), abs(m[0] + m[1])))
            board.apply_move(*rng.choice(near[:60]))
        legal = board.legal_moves()
        buf.push_graph_position(board.get_stones(), [(legal[0][0], legal[0][1], 1.0)], board.current_player,
                                board.moves_remaining, board.ply, True, 0.0, True, 8)
    buf.seed_sampler(3)
    wire, _targets = buf.sample_graph_batch(5, augment=False)
    return config, spec, graph_wire_from_rust(wire)


def _net(config: dict[str, Any], spec: Any) -> torch.nn.Module:
    torch.manual_seed(20261006)
    return build_net(arch_from_spec_and_config(spec, config))


def _collate(spec: Any, payload: Any, *, coded: bool) -> Any:
    return collate_graph_batch(payload, device="cpu", semantic="off", trunk_size=spec.trunk_size,
                               win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
                               edge_feat_dim=spec.edge_feat_dim, coded_edges=coded)


def _forward(net: torch.nn.Module, spec: Any, batch: Any, *, coded: bool) -> tuple[Tensor, ...]:
    vocab = torch.from_numpy(np.asarray(edge_vocabulary(spec.win_length))).reshape(-1, 5) if coded else None
    edges = batch.edge_code if coded else batch.edge_attr
    return net.forward_batch(batch.x, batch.edge_index, edges, batch.legal_node_gather,
                             stone_mask_from_batch(batch), batch.node_offsets, edge_vocab=vocab)


@pytest.mark.parametrize("coded", [False, True], ids=["per_edge", "coded"])
def test_the_served_cpu_forward_is_unchanged_bit_for_bit(ragged, coded: bool,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The production net on real ragged graphs, served fp32 on the CPU: every output equal to the unchunked op's."""
    config, spec, payload = ragged
    net, batch = _net(config, spec).eval(), _collate(spec, payload, coded=coded)
    assert batch.edge_index.shape[1] > 10 * gine._CPU_CHUNK_EDGES, "the batch no longer spans many chunks"
    with torch.inference_mode():
        got = _forward(net, spec, batch, coded=coded)
        monkeypatch.setattr(gine, "gine_message_sum", _reference)
        want = _forward(net, spec, batch, coded=coded)
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(g, w), f"output {i}: max |Δ| {float((g - w).abs().max()):.3g}"


def test_the_cpu_training_gradients_are_unchanged_bit_for_bit(ragged, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer's per-edge forward and backward through layer recompute: every parameter gradient equal to the unchunked op's."""
    config, spec, payload = ragged
    net, batch = _net(config, spec).train(), _collate(spec, payload, coded=False)

    def grads() -> list[Tensor]:
        net.zero_grad(set_to_none=True)
        outputs = _forward(net, spec, batch, coded=False)
        sum(o.float().square().mean() for o in outputs).backward()
        return [p.grad.clone() for p in net.parameters() if p.grad is not None]

    got = grads()
    monkeypatch.setattr(gine, "gine_message_sum", _reference)
    want = grads()
    assert got and len(got) == len(want)
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True)), "a parameter gradient moved"
