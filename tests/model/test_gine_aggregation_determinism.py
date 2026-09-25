"""The graph aggregation is deterministic on CUDA, served and trained, within one bf16 ulp of fp64 at the op, and no less accurate end to end; (ii) reads the parent via MANTIS_PERF_CHECKPOINT / MANTIS_PERF_EVENTS."""
from __future__ import annotations

import importlib.util
from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from _gine_oracle import (SumFn, aggregating_with, capturing, exact_sum, fp64_sum, index_add_aggregation,
                          index_add_sum, message, synthetic_batch)
from mantis._engine import InferenceBatcher
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.model.gine import csr_edges, gine_message_sum
from mantis.selfplay.inference_server import InferenceServer
from mantis.train.checkpoints import load_checkpoint

_REPO = Path(__file__).resolve().parents[2]
_REPEATS = 5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the non-deterministic aggregation exists only on CUDA")


def _net() -> torch.nn.Module:
    config = load_config(production_configs(_REPO)[0]).model_dump()
    torch.manual_seed(20260925)
    return build_net(arch_from_spec_and_config(lookup(config["identity"]["encoding"]), config)).cuda()


def _assert_jitter_is_possible() -> None:
    assert not torch.are_deterministic_algorithms_enabled(), (
        "deterministic mode is on (leaked by an earlier test): the atomic path would not jitter, so nothing is read")


def _served(net: torch.nn.Module, b: dict[str, torch.Tensor], trunk: Any = None) -> tuple[torch.Tensor, ...]:
    kwargs = {} if trunk is None else {"trunk": trunk}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net.forward_batch(b["x"], b["edge_index"], b["edge_attr"], b["legal_index"],
                                 b["stone_mask"], b["node_offsets"], **kwargs)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_i_the_same_batch_serves_bit_identically(compiled: bool) -> None:
    """Dense random-init messages into a dummy hot spot: the regime where the bf16 atomics disagree with themselves."""
    _assert_jitter_is_possible()
    net, b = _net().eval(), synthetic_batch("cuda")
    torch._dynamo.reset()
    trunk = torch.compile(net.representation, dynamic=True) if compiled else None
    first, second = _served(net, b, trunk), _served(net, b, trunk)
    for name, x, y in zip(("logits", "value", "bins"), first, second, strict=True):
        assert torch.equal(x, y), f"{name} differs on a repeat: max |Δ| {float((x - y).abs().max()):.4g}"


def _grads(net: torch.nn.Module, b: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """A surrogate loss over the net's three outputs, so every aggregation's gradient is exercised."""
    net.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, value, bins = net.forward_batch(b["x"], b["edge_index"], b["edge_attr"],
                                                b["legal_index"], b["stone_mask"], b["node_offsets"])
    (logits.float().square().mean() + value.float().sum() + bins.float().square().mean()).backward()
    return [p.grad.clone() for p in net.parameters() if p.grad is not None]


def test_i_the_bf16_index_add_path_jitters_served_and_trained() -> None:
    """CONTROL: the atomic path the witnesses replaced must differ on a repeat here, or their greens prove nothing."""
    _assert_jitter_is_possible()
    served_net, b = _net().eval(), synthetic_batch("cuda")
    train_net, tb = _net().train(), synthetic_batch("cuda", n_graphs=16)
    with index_add_aggregation():
        served = [_served(served_net, b)[0] for _ in range(3)]
        trained = [_grads(train_net, tb) for _ in range(3)]
    assert any(not torch.equal(served[0], x) for x in served[1:]), "the bf16 index_add_ path served identically"
    assert any(not all(torch.equal(x, y) for x, y in zip(trained[0], t, strict=True)) for t in trained[1:]), (
        "the bf16 index_add_ path trained identically")


def test_i_the_same_batch_trains_bit_identically() -> None:
    """The trainer's forward and backward under bf16 autocast: every gradient equal on a repeat."""
    _assert_jitter_is_possible()
    net, b = _net().train(), synthetic_batch("cuda", n_graphs=16)

    def grads() -> list[torch.Tensor]:
        return _grads(net, b)

    first, second = grads(), grads()
    assert first and len(first) == len(second)
    worst = max(float((x - y).abs().max()) for x, y in zip(first, second, strict=True))
    assert worst == 0.0, f"a gradient differs on a repeat: max |Δ| {worst:.4g}"


def _real_batches(net: torch.nn.Module, config: dict[str, Any], n_batches: int) -> list[tuple]:
    """Real B-64 batches of distinct positions, captured from the real server exactly as it forwards them."""
    spec = importlib.util.spec_from_file_location("bench_server_under_test", _REPO / "tools" / "bench_server.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    pool = bench.positions_from_events(Path(os.environ["MANTIS_PERF_EVENTS"]),
                                       config["identity"]["encoding"], limit=20000)
    probe = bench.distinct_positions(pool, 64 * n_batches)
    got: list[tuple] = []
    forward = net.forward_batch

    def record(*args: Any, **kwargs: Any) -> Any:
        got.append(tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args))
        return forward(*args, **kwargs)

    net.forward_batch = record
    lspec = lookup(config["identity"]["encoding"])
    batcher = InferenceBatcher(encoding_spec=lspec)
    server = InferenceServer(net, torch.device("cuda"), config, batcher=batcher, encoding_spec=lspec,
                             compile_trunk=False)
    server.start()
    try:
        for i in range(n_batches):
            batcher.submit_graphs_and_wait(probe[64 * i: 64 * (i + 1)], 1)
    finally:
        server.stop()
        server.join(timeout=30.0)
        del net.forward_batch
    return got


def _outputs(net: torch.nn.Module, args: tuple, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, value, _bins = net.forward_batch(*args)
    return logits.float(), value.float().reshape(-1)


_NEEDS_PARENT = pytest.mark.skipif(
    not (os.environ.get("MANTIS_PERF_CHECKPOINT") and os.environ.get("MANTIS_PERF_EVENTS")),
    reason="LOUD SKIP — (ii) is pre-stated over the PARENT's real served batches; "
           "set MANTIS_PERF_CHECKPOINT and MANTIS_PERF_EVENTS")


@pytest.fixture(scope="module")
def parent() -> tuple[torch.nn.Module, list[tuple], list[tuple]]:
    """The parent net, its 8 real B-64 batches, and every GINE layer's `(xs, e, src, dst, n, divisor)` on them."""
    _assert_jitter_is_possible()
    config = load_config(production_configs(_REPO)[0]).model_dump()
    ck = load_checkpoint(Path(os.environ["MANTIS_PERF_CHECKPOINT"]))
    assert ck.metadata.arch is not None
    net = build_net(ck.metadata.arch)
    net.load_state_dict(ck.model_state)
    net = net.cuda().eval()
    batches = _real_batches(net, config, 8)
    assert len(batches) == 8, f"captured {len(batches)} forwards, (ii) is pre-stated over 8"
    captures: list[tuple] = []
    with capturing(captures):
        for args in batches:
            _outputs(net, args, amp=True)
    return net, batches, captures


AggFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, "torch.Tensor | None"], torch.Tensor]


def _production(xs: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, n: int,
                div: torch.Tensor | None) -> torch.Tensor:
    edge_index, e, rowptr = csr_edges(torch.stack((src, dst)), e, n)
    return gine_message_sum(xs, e, edge_index[0], edge_index[1], rowptr, div)


def _on_messages(sum_fn: SumFn, *, compile_sum: bool = False) -> AggFn:
    """A control summing the captured messages with `sum_fn`; only the sum compiles (Inductor keeps a fused add in fp32)."""
    if compile_sum:
        torch._dynamo.reset()
    fn = torch.compile(sum_fn, dynamic=True) if compile_sum else sum_fn
    return lambda xs, e, src, dst, n, div: fn(message(xs, e, src), dst, n, div)


def _ulp_bf16(ref: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place at `ref`'s magnitude: 8 significant bits."""
    _m, ex = torch.frexp(ref)
    return torch.ldexp(torch.ones_like(ref), ex - 8)


def _cuda(capture: tuple) -> tuple:
    return tuple(t.cuda() if torch.is_tensor(t) else t for t in capture)


def _band(agg_fn: AggFn, captures: list[tuple], *, compiled: bool) -> tuple[int, int, float]:
    """`(nonzero-at-zero outputs, outputs past one ulp, pooled max |agg − fp64|)` over the captures."""
    if compiled:
        torch._dynamo.reset()
    fn = torch.compile(agg_fn, dynamic=True) if compiled else agg_fn
    bad_zero = past_ulp = 0
    worst = 0.0
    for capture in captures:
        xs, e, src, dst, n, div = _cuda(capture)
        with torch.inference_mode():
            agg, ref = fn(xs, e, src, dst, n, div), fp64_sum(message(xs, e, src), dst, n, div)
        assert agg.dtype is torch.bfloat16, f"the aggregation returned {agg.dtype}, not bf16 storage"
        err = (agg.double() - ref).abs()
        zero = ref == 0
        bad_zero += int((agg[zero] != 0).sum())
        past_ulp += int((err[~zero] > _ulp_bf16(ref[~zero])).sum())
        worst = max(worst, float(err.max()))
    return bad_zero, past_ulp, worst


def _old_spread(captures: list[tuple]) -> float:
    """The bf16 `index_add_` aggregation's max range over five repeats of each capture, pooled."""
    spread = 0.0
    for capture in captures:
        xs, e, src, dst, n, div = _cuda(capture)
        msg = message(xs, e, src)
        with torch.inference_mode():
            reps = torch.stack([index_add_sum(msg, dst, n, div).float() for _ in range(_REPEATS)])
        spread = max(spread, float((reps.max(0).values - reps.min(0).values).max()))
    return spread


@_NEEDS_PARENT
def test_ii_a_the_instrument_is_live_the_old_sum_fails_and_the_exact_sum_passes(parent) -> None:
    """CONTROLS, before any verdict on production: the bf16 `index_add_` sum breaks the ulp bound and jitters; the fp32 oracle meets it all."""
    _net, _batches, captures = parent
    spread = _old_spread(captures)
    _z, old_past, _w = _band(_on_messages(index_add_sum), captures, compiled=False)
    print(f"(ii-a) control: bf16 `index_add_` sum past one ulp at {old_past} outputs, repeat spread {spread:.4g}")
    assert spread > 0.0 and old_past > 0, "the instrument is not live: the bf16 `index_add_` sum neither jitters nor breaks the bound"
    for compiled in (False, True):
        bad_zero, past, worst = _band(_on_messages(exact_sum, compile_sum=compiled), captures, compiled=False)
        print(f"(ii-a) control: exact sum compiled={compiled}: {bad_zero} / {past} violations, max |Δ| {worst:.4g}")
        assert bad_zero == 0 and past == 0 and worst <= spread


@_NEEDS_PARENT
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_ii_a_the_production_sum_is_within_one_ulp_of_fp64(parent, compiled: bool) -> None:
    """At the fused op: bf16 out, exact zeros, within one bf16 ulp of fp64, and no wider than the old path's jitter."""
    _net, _batches, captures = parent
    bad_zero, past, worst = _band(_production, captures, compiled=compiled)
    spread = _old_spread(captures)
    print(f"(ii-a) production compiled={compiled}: {bad_zero} nonzero-at-zero, {past} past one ulp, "
          f"max |Δ| {worst:.4g} vs the bf16 `index_add_` spread {spread:.4g}")
    assert bad_zero == 0 and past == 0, f"{bad_zero} nonzero-at-zero, {past} outputs past one bf16 ulp of fp64"
    assert worst <= spread, f"max |Δ| {worst:.4g} vs fp64 exceeds the bf16 `index_add_` repeat spread {spread:.4g}"


def _mean_logit_error(net: torch.nn.Module, batches: list[tuple], ref: list[tuple]) -> float:
    new = [_outputs(net, a, amp=True)[0] for a in batches]
    return float(torch.cat([(n - r[0]).abs() for n, r in zip(new, ref, strict=True)]).mean())


@_NEEDS_PARENT
@pytest.mark.parametrize("arm", ["production", "exact_control"])
def test_ii_b_the_mean_logit_error_is_no_worse_than_the_old_paths_worst(parent, arm: str) -> None:
    """End-to-end guard: pooled mean |Δlogit| vs fp32 <= the bf16 `index_add_` path's worst of five."""
    net, batches, _captures = parent
    ref = [_outputs(net, a, amp=False) for a in batches]
    with index_add_aggregation():
        old = [_mean_logit_error(net, batches, ref) for _ in range(_REPEATS)]
    if arm == "production":
        got = _mean_logit_error(net, batches, ref)
    else:
        with aggregating_with(exact_sum):
            got = _mean_logit_error(net, batches, ref)
    print(f"(ii-b) {arm}: mean |Δlogit| {got:.5g} vs the bf16 `index_add_` worst of {_REPEATS} {max(old):.5g} "
          f"(range {min(old):.5g}..{max(old):.5g})")
    assert got <= max(old), f"{arm}: mean |Δlogit| {got:.5g} > the bf16 `index_add_` worst {max(old):.5g}"
