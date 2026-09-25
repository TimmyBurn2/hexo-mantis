"""The graph aggregation is deterministic on CUDA, served and trained, within one bf16 ulp of fp64 at the op, and no less accurate end to end; (ii) reads the parent via MANTIS_PERF_CHECKPOINT / MANTIS_PERF_EVENTS."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from _gine_oracle import (SumFn, aggregating_with, capturing, exact_sum, fp64_sum, index_add_aggregation,
                          index_add_sum, synthetic_batch)
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net

_REPO = Path(__file__).resolve().parents[2]
_REPEATS = 5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the non-deterministic aggregation exists only on CUDA")


def _net() -> torch.nn.Module:
    config = load_config(production_configs(_REPO)[0]).model_dump()
    torch.manual_seed(20260925)
    return build_net(arch_from_spec_and_config(lookup(config["identity"]["encoding"]), config)).cuda()


def _served(net: torch.nn.Module, b: dict[str, torch.Tensor], trunk: Any = None) -> tuple[torch.Tensor, ...]:
    kwargs = {} if trunk is None else {"trunk": trunk}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net.forward_batch(b["x"], b["edge_index"], b["edge_attr"], b["legal_index"],
                                 b["stone_mask"], b["node_offsets"], **kwargs)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_i_the_same_batch_serves_bit_identically(compiled: bool) -> None:
    """Dense random-init messages into a dummy hot spot: the regime where the bf16 atomics disagree with themselves."""
    net, b = _net().eval(), synthetic_batch("cuda")
    torch._dynamo.reset()
    trunk = torch.compile(net.representation, dynamic=True) if compiled else None
    first, second = _served(net, b, trunk), _served(net, b, trunk)
    for name, x, y in zip(("logits", "value", "bins"), first, second, strict=True):
        assert torch.equal(x, y), f"{name} differs on a repeat: max |Δ| {float((x - y).abs().max()):.4g}"


def test_i_the_same_batch_trains_bit_identically() -> None:
    """The trainer's forward and backward under bf16 autocast: every gradient equal on a repeat."""
    net, b = _net().train(), synthetic_batch("cuda", n_graphs=16)

    def grads() -> list[torch.Tensor]:
        net.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value, bins = net.forward_batch(b["x"], b["edge_index"], b["edge_attr"],
                                                    b["legal_index"], b["stone_mask"], b["node_offsets"])
        (logits.float().square().mean() + value.float().sum() + bins.float().square().mean()).backward()
        return [p.grad.clone() for p in net.parameters() if p.grad is not None]

    first, second = grads(), grads()
    assert first and len(first) == len(second)
    worst = max(float((x - y).abs().max()) for x, y in zip(first, second, strict=True))
    assert worst == 0.0, f"a gradient differs on a repeat: max |Δ| {worst:.4g}"


def _real_batches(net: torch.nn.Module, config: dict[str, Any], n_batches: int) -> list[tuple]:
    """Real B-64 batches of distinct positions, captured from the real server exactly as it forwards them."""
    from mantis._engine import InferenceBatcher
    from mantis.selfplay.inference_server import InferenceServer

    spec = importlib.util.spec_from_file_location("bench_server_l2", _REPO / "tools" / "bench_server.py")
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
    """The parent net, its 8 real B-64 batches, and every GINE layer's `(msg, dst, n, divisor)` on those batches."""
    from mantis.train.checkpoints import load_checkpoint

    assert not torch.are_deterministic_algorithms_enabled(), (
        "deterministic mode is on (leaked by an earlier test): the pre-L2 path would not jitter, so (ii) reads nothing")
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


def _production_sum() -> SumFn:
    from mantis.model.gine import gine_aggregate

    return gine_aggregate


def _ulp_bf16(ref: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place at `ref`'s magnitude: 8 significant bits."""
    _m, e = torch.frexp(ref)
    return torch.ldexp(torch.ones_like(ref), e - 8)


def _band(sum_fn: SumFn, captures: list[tuple], *, compiled: bool) -> tuple[int, int, float]:
    """`(non-bf16 or nonzero-at-zero outputs, outputs past one ulp, pooled max |agg − fp64|)` over the captures."""
    if compiled:
        torch._dynamo.reset()
    fn = torch.compile(sum_fn, dynamic=True) if compiled else sum_fn
    bad_zero = past_ulp = 0
    worst = 0.0
    for msg, dst, n, div in captures:
        msg, dst = msg.cuda(), dst.cuda()
        div = None if div is None else div.cuda()
        with torch.inference_mode():
            agg, ref = fn(msg, dst, n, div), fp64_sum(msg, dst, n, div)
        assert agg.dtype is torch.bfloat16, f"the aggregation returned {agg.dtype}, not bf16 storage"
        err = (agg.double() - ref).abs()
        zero = ref == 0
        bad_zero += int((agg[zero] != 0).sum())
        past_ulp += int((err[~zero] > _ulp_bf16(ref[~zero])).sum())
        worst = max(worst, float(err.max()))
    return bad_zero, past_ulp, worst


def _old_spread(captures: list[tuple]) -> float:
    """The pre-L2 aggregation's max range over five repeats of each capture, pooled."""
    spread = 0.0
    for msg, dst, n, div in captures:
        msg, dst = msg.cuda(), dst.cuda()
        div = None if div is None else div.cuda()
        with torch.inference_mode():
            reps = torch.stack([index_add_sum(msg, dst, n, div).float() for _ in range(_REPEATS)])
        spread = max(spread, float((reps.max(0).values - reps.min(0).values).max()))
    return spread


@_NEEDS_PARENT
def test_ii_a_the_instrument_is_live_the_old_sum_fails_and_the_exact_sum_passes(parent) -> None:
    """CONTROLS, before any verdict on production: the pre-L2 sum breaks the ulp bound and jitters; the fp32 oracle meets it all."""
    _net, _batches, captures = parent
    spread = _old_spread(captures)
    _z, old_past, _w = _band(index_add_sum, captures, compiled=False)
    print(f"L2 (ii-a) control: pre-L2 sum past one ulp at {old_past} outputs, repeat spread {spread:.4g}")
    assert spread > 0.0 and old_past > 0, "the instrument is not live: the pre-L2 sum neither jitters nor breaks the bound"
    for compiled in (False, True):
        bad_zero, past, worst = _band(exact_sum, captures, compiled=compiled)
        print(f"L2 (ii-a) control: exact sum compiled={compiled}: {bad_zero} / {past} violations, max |Δ| {worst:.4g}")
        assert bad_zero == 0 and past == 0 and worst <= spread


@_NEEDS_PARENT
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_ii_a_the_production_sum_is_within_one_ulp_of_fp64(parent, compiled: bool) -> None:
    """At the op L2 changes: bf16 out, exact zeros, within one bf16 ulp of fp64, and no wider than the old path's jitter."""
    _net, _batches, captures = parent
    bad_zero, past, worst = _band(_production_sum(), captures, compiled=compiled)
    spread = _old_spread(captures)
    print(f"L2 (ii-a) production compiled={compiled}: {bad_zero} nonzero-at-zero, {past} past one ulp, "
          f"max |Δ| {worst:.4g} vs the pre-L2 spread {spread:.4g}")
    assert bad_zero == 0 and past == 0, f"{bad_zero} nonzero-at-zero, {past} outputs past one bf16 ulp of fp64"
    assert worst <= spread, f"max |Δ| {worst:.4g} vs fp64 exceeds the pre-L2 repeat spread {spread:.4g}"


def _mean_logit_error(net: torch.nn.Module, batches: list[tuple], ref: list[tuple]) -> float:
    new = [_outputs(net, a, amp=True)[0] for a in batches]
    return float(torch.cat([(n - r[0]).abs() for n, r in zip(new, ref, strict=True)]).mean())


@_NEEDS_PARENT
@pytest.mark.parametrize("arm", ["production", "exact_control"])
def test_ii_b_the_mean_logit_error_is_no_worse_than_the_old_paths_worst(parent, arm: str) -> None:
    """End-to-end guard: pooled mean |Δlogit| vs fp32 <= the pre-L2 path's worst of five; value, p99 and max are reported."""
    net, batches, _captures = parent
    ref = [_outputs(net, a, amp=False) for a in batches]
    with index_add_aggregation():
        old = [_mean_logit_error(net, batches, ref) for _ in range(_REPEATS)]
    if arm == "production":
        got = _mean_logit_error(net, batches, ref)
    else:
        with aggregating_with(exact_sum):
            got = _mean_logit_error(net, batches, ref)
    print(f"L2 (ii-b) {arm}: mean |Δlogit| {got:.5g} vs the pre-L2 worst of {_REPEATS} {max(old):.5g} "
          f"(range {min(old):.5g}..{max(old):.5g})")
    assert got <= max(old), f"{arm}: mean |Δlogit| {got:.5g} > the pre-L2 worst {max(old):.5g}"
