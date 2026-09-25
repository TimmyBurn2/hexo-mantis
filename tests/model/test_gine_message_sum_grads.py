"""The fused op's CUDA forward AND backward values: equal to the CPU op's math, and to fp64 autograd for fp32 inputs."""
from __future__ import annotations

import pytest
import torch

from mantis.model.gine import csr_edges, gine_message_sum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the Triton kernels exist only on CUDA")


def _case(h: int, dtype: torch.dtype, with_div: bool, seed: int = 7) -> dict[str, torch.Tensor | None]:
    """Nodes 0..3 receive nothing and nodes 4..7 send nothing; E is no multiple of the kernel's edge block."""
    gen = torch.Generator().manual_seed(seed)
    n, e_count = 300, 5_003
    src = torch.randint(8, n, (e_count,), generator=gen)
    dst = torch.randint(4, n, (e_count,), generator=gen)
    dst[src < 8] = 5
    return {"xs": torch.randn(n, h, generator=gen).to(dtype), "e": torch.randn(e_count, h, generator=gen).to(dtype),
            "src": src, "dst": dst,
            "div": torch.randint(1, 40, (n, 1), generator=gen).float() if with_div else None,
            "grad": torch.randn(n, h, generator=gen).to(dtype)}


def _run(c: dict, device: str, divisor_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = c["xs"].to(device).requires_grad_()
    e = c["e"].to(device).requires_grad_()
    div = None if c["div"] is None else c["div"].to(device) * divisor_scale
    edge_index, e_sorted, rowptr = csr_edges(torch.stack((c["src"], c["dst"])).to(device), e, xs.shape[0])
    out = gine_message_sum(xs, e_sorted, edge_index[0], edge_index[1], rowptr, div)
    gx, ge = torch.autograd.grad(out, (xs, e), c["grad"].to(device))
    return out.detach().cpu(), gx.cpu(), ge.cpu()


def _ulps(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a − b| in units of b's bf16 ulp (zeros compared exactly)."""
    _m, ex = torch.frexp(b.double())
    ulp = torch.ldexp(torch.ones_like(b.double()), ex - 8)
    return float(((a.double() - b.double()).abs() / ulp.clamp_min(1e-300)).max())


@pytest.mark.parametrize("h", [128, 24])
@pytest.mark.parametrize("with_div", [False, True], ids=["no-div", "div"])
def test_cuda_bf16_values_match_the_cpu_op(h: int, with_div: bool) -> None:
    """Per-edge gradients bit-equal (the same elementwise math); per-node sums within one bf16 ulp (fp32 order only)."""
    c = _case(h, torch.bfloat16, with_div)
    cuda, cpu = _run(c, "cuda"), _run(c, "cpu")
    assert torch.equal(cuda[2], cpu[2]), "grad_e differs from the CPU op's masked, divided gradient"
    assert _ulps(cuda[0], cpu[0]) <= 1.0 and _ulps(cuda[1], cpu[1]) <= 1.0


@pytest.mark.parametrize("with_div", [False, True], ids=["no-div", "div"])
def test_cuda_fp32_values_match_fp64_autograd(with_div: bool) -> None:
    """fp32 inputs: forward and both gradients within fp32 rounding of autograd over the plain expression in fp64."""
    c = _case(24, torch.float32, with_div)
    out, gx, ge = _run(c, "cuda")
    xs, e = c["xs"].double().requires_grad_(), c["e"].double().requires_grad_()
    msg = (xs.index_select(0, c["src"]) + e).relu()
    ref = torch.zeros(xs.shape, dtype=torch.float64).index_add(0, c["dst"], msg)
    ref = ref if c["div"] is None else ref / c["div"].double()
    rgx, rge = torch.autograd.grad(ref, (xs, e), c["grad"].double())
    for got, want in ((out, ref.detach()), (gx, rgx), (ge, rge)):
        assert float((got.double() - want).abs().max()) <= 1e-4 * max(1.0, float(want.abs().max()))


def test_a_planted_wrong_divisor_is_caught() -> None:
    """PLANTED BREAK: the CUDA op run with a doubled divisor must disagree with the CPU op, or the rows above are blind."""
    c = _case(128, torch.bfloat16, True)
    cuda, cpu = _run(c, "cuda", divisor_scale=2.0), _run(c, "cpu")
    assert not torch.equal(cuda[2], cpu[2]) and _ulps(cuda[1], cpu[1]) > 1.0
