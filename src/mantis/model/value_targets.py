"""Value-target codecs: a trajectory or a scalar in, a value-head training target out.

Codecs compose with `dist65` (the encoding codec) rather than replacing it, so a target change
cannot become an `if graph:` in the trainer. NOTHING HERE IS ARMED: no config key selects it
and no production call site calls it. Neither codec attacks the value blind spot — target fixes
on a frozen dense representation are already falsified and the deficit is a FEATURE problem.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from mantis.model.dist65 import N_VALUE_BINS, VALUE_SUPPORT


class ValueTargetError(ValueError):
    """A value-target codec was handed inputs it cannot construct a target from."""


def lambda_return_targets(
    values: Sequence[float] | torch.Tensor,
    movers: Sequence[int] | torch.Tensor,
    terminal_z: float,
    *,
    lam: float,
) -> torch.Tensor:
    """The λ-return value target for one trajectory, in each position's OWN mover's frame.

    Pure. `movers` is an argument rather than assumed alternation because a turn is TWO STONES,
    so the recursion negates at a handover and nowhere else; `values` are meant to be MCTS ROOT
    values, which breaks the paper's on-policy dependence.

    Args:
        values: per-index bootstrap values, each already in that index's own mover's frame.
        movers: per-index mover id; only INEQUALITY between adjacent entries is read.
        terminal_z: the game outcome in the LAST index's mover's frame.
        lam: the mixing weight, in [0, 1].

    Returns:
        A `(T,)` fp64 tensor of targets, index `i` in position `i`'s own mover's frame.

    Raises:
        ValueTargetError: `values` and `movers` have different lengths, either is empty, or
            `lam` is outside [0, 1].
    """
    v = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    m = torch.as_tensor(movers).reshape(-1)
    if v.numel() == 0:
        raise ValueTargetError(
            "an empty trajectory has no λ-return; a codec that returned an empty target for it "
            "would make a caller's bug indistinguishable from a terminal position"
        )
    if v.numel() != m.numel():
        raise ValueTargetError(
            f"values has {v.numel()} entries and movers has {m.numel()}; the mover of every "
            "index is what decides the sign, so a ragged pair cannot be resolved"
        )
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueTargetError(
            f"lam={lam} is outside [0, 1] — the mechanism's own range: λ mixes a bootstrap with "
            "a return, and a weight outside the simplex mixes nothing"
        )
    lam = float(lam)
    t = v.numel()
    out = torch.empty(t, dtype=torch.float64)
    out[t - 1] = float(terminal_z)
    for i in range(t - 2, -1, -1):
        sign = 1.0 if bool(m[i] == m[i + 1]) else -1.0
        out[i] = (1.0 - lam) * (sign * v[i + 1]) + lam * (sign * out[i + 1])
    return out


def scalar_to_hl_gauss(
    z: torch.Tensor, *, sigma: float, n_bins: int = N_VALUE_BINS
) -> torch.Tensor:
    """z (N,) in [-1,1] → (N, n_bins) fp32 HL-Gauss encoding over `dist65`'s own support.

    `sigma` is not free: the support has an ODD bin count so an exact-zero bin exists, and a
    kernel wide enough to help smears mass out of it. As `sigma → 0` this converges to
    `scalar_to_two_hot`.

    Args:
        z: scalars in [-1, 1]; values outside are clamped, as `scalar_to_two_hot` clamps.
        sigma: the kernel's standard deviation in SUPPORT units (not bins), strictly positive.
        n_bins: the support's bin count; defaults to `dist65`'s 65.

    Returns:
        `(N, n_bins)` fp32, each row summing to 1.

    Raises:
        ValueTargetError: `sigma` is not strictly positive, or `n_bins` disagrees with the
            imported support's length so the bin centres cannot be derived.
    """
    if not sigma > 0.0:
        raise ValueTargetError(
            f"sigma={sigma} is not strictly positive. A zero-width Gaussian is a point mass "
            "with no closed form here — the σ → 0 LIMIT is the two-hot encoding, and that "
            "encoding already exists as `scalar_to_two_hot`"
        )
    if n_bins == N_VALUE_BINS:
        support = VALUE_SUPPORT
    else:
        support = torch.linspace(-1.0, 1.0, n_bins)
    if support.numel() != n_bins:
        raise ValueTargetError(
            f"n_bins={n_bins} disagrees with the support of length {support.numel()}"
        )
    z = z.reshape(-1).detach().to(torch.float32).clamp(-1.0, 1.0)
    centres = support.to(z.device, torch.float32)
    d = (centres.unsqueeze(0) - z.unsqueeze(1)) / float(sigma)
    # Softmax over -d²/2 rather than exp-then-divide: the same normalised kernel, but it cannot
    # underflow to an all-zero row (then 0/0) at the narrow widths the σ → 0 witness drives.
    return torch.softmax(-0.5 * d * d, dim=-1)


__all__ = ["ValueTargetError", "lambda_return_targets", "scalar_to_hl_gauss"]
