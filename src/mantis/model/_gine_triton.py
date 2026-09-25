"""`mantis::gine_message_sum`'s CUDA kernels: one program per output row, fp32 sums, fixed blocks, no atomics; CUDA-only (no Triton on CPU)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_E = 32
_NUM_WARPS = 2


@triton.jit
def _message_sum_fwd(xs_ptr, e_ptr, src_ptr, rowptr_ptr, div_ptr, out_ptr, H,
                     BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr, ROUND_BF16: tl.constexpr,
                     HAS_DIV: tl.constexpr):
    v = tl.program_id(0)
    start = tl.load(rowptr_ptr + v)
    end = tl.load(rowptr_ptr + v + 1)
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j0 in range(start, end, BLOCK_E):
        j = j0 + tl.arange(0, BLOCK_E)
        m = j < end
        s = tl.load(src_ptr + j, mask=m, other=0)
        m2 = m[:, None] & hm[None, :]
        xv = tl.load(xs_ptr + s[:, None] * H + h[None, :], mask=m2, other=0.0).to(tl.float32)
        ev = tl.load(e_ptr + j[:, None] * H + h[None, :], mask=m2, other=0.0).to(tl.float32)
        pre = xv + ev
        if ROUND_BF16:
            pre = pre.to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(tl.where(m2, tl.maximum(pre, 0.0), 0.0), axis=0)
    if HAS_DIV:
        acc = tl.div_rn(acc, tl.zeros_like(acc) + tl.load(div_ptr + v))
    tl.store(out_ptr + v * H + h, acc.to(out_ptr.dtype.element_ty), mask=hm)


@triton.jit
def _message_grad(grad_ptr, xs_ptr, e_ptr, src_ptr, rowptr_ptr, div_ptr, gpre_ptr, H,
                  BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr, ROUND_BF16: tl.constexpr,
                  HAS_DIV: tl.constexpr):
    v = tl.program_id(0)
    start = tl.load(rowptr_ptr + v)
    end = tl.load(rowptr_ptr + v + 1)
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    g = tl.load(grad_ptr + v * H + h, mask=hm, other=0.0).to(tl.float32)
    if HAS_DIV:
        g = tl.div_rn(g, tl.zeros_like(g) + tl.load(div_ptr + v))
    g = g.to(gpre_ptr.dtype.element_ty).to(tl.float32)
    for j0 in range(start, end, BLOCK_E):
        j = j0 + tl.arange(0, BLOCK_E)
        m = j < end
        s = tl.load(src_ptr + j, mask=m, other=0)
        m2 = m[:, None] & hm[None, :]
        xv = tl.load(xs_ptr + s[:, None] * H + h[None, :], mask=m2, other=0.0).to(tl.float32)
        ev = tl.load(e_ptr + j[:, None] * H + h[None, :], mask=m2, other=0.0).to(tl.float32)
        pre = xv + ev
        if ROUND_BF16:
            pre = pre.to(tl.bfloat16).to(tl.float32)
        gp = tl.where(pre > 0.0, g[None, :], 0.0)
        tl.store(gpre_ptr + j[:, None] * H + h[None, :], gp.to(gpre_ptr.dtype.element_ty), mask=m2)


@triton.jit
def _gathered_sum(val_ptr, perm_ptr, rowptr_ptr, out_ptr, H,
                  BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr):
    u = tl.program_id(0)
    start = tl.load(rowptr_ptr + u)
    end = tl.load(rowptr_ptr + u + 1)
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j0 in range(start, end, BLOCK_E):
        j = j0 + tl.arange(0, BLOCK_E)
        m = j < end
        p = tl.load(perm_ptr + j, mask=m, other=0)
        m2 = m[:, None] & hm[None, :]
        acc += tl.sum(tl.load(val_ptr + p[:, None] * H + h[None, :], mask=m2, other=0.0).to(tl.float32), axis=0)
    tl.store(out_ptr + u * H + h, acc.to(out_ptr.dtype.element_ty), mask=hm)


def _div_arg(divisor: Tensor | None, like: Tensor) -> Tensor:
    return like.new_empty(0, dtype=torch.float32) if divisor is None else divisor.reshape(-1).float().contiguous()


def message_sum(xs: Tensor, e: Tensor, src: Tensor, rowptr: Tensor, divisor: Tensor | None) -> Tensor:
    """`out[v] = round(Σ_{dst-sorted edges j of v} relu(round(xs[src_j] + e_j)) / divisor[v])`, fp32 inside."""
    n, h = xs.shape
    out = torch.empty((n, h), dtype=xs.dtype, device=xs.device)
    if n:
        _message_sum_fwd[(n,)](xs.contiguous(), e.contiguous(), src, rowptr, _div_arg(divisor, xs), out, h,
                               BLOCK_H=triton.next_power_of_2(h), BLOCK_E=_BLOCK_E,
                               ROUND_BF16=xs.dtype == torch.bfloat16, HAS_DIV=divisor is not None,
                               num_warps=_NUM_WARPS)
    return out


def message_grads(grad: Tensor, xs: Tensor, e: Tensor, src: Tensor, rowptr: Tensor,
                  divisor: Tensor | None) -> tuple[Tensor, Tensor]:
    """`(grad_xs, grad_e)`: the masked per-edge gradient, then its fp32 sum over each source row, rounded once."""
    n, h = xs.shape
    block_h = triton.next_power_of_2(h)
    grad_e = torch.empty(e.shape, dtype=e.dtype, device=e.device)
    grad_xs = torch.empty(xs.shape, dtype=xs.dtype, device=xs.device)
    if n == 0 or src.numel() == 0:
        return grad_xs.zero_(), grad_e
    _message_grad[(n,)](grad.contiguous(), xs.contiguous(), e.contiguous(), src, rowptr, _div_arg(divisor, xs),
                        grad_e, h, BLOCK_H=block_h, BLOCK_E=_BLOCK_E, ROUND_BF16=xs.dtype == torch.bfloat16,
                        HAS_DIV=divisor is not None, num_warps=_NUM_WARPS)
    perm = torch.argsort(src, stable=True)
    src_rowptr = torch.zeros(n + 1, dtype=torch.long, device=src.device).index_add_(
        0, src + 1, torch.ones_like(src)).cumsum(0)
    _gathered_sum[(n,)](grad_e, perm, src_rowptr, grad_xs, h, BLOCK_H=block_h, BLOCK_E=_BLOCK_E,
                        num_warps=_NUM_WARPS)
    return grad_xs, grad_e
