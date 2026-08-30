"""VALUE-TARGET CODECS — target construction, behind the model contract, ARMED BY NOTHING.

`SEAM_V1_DESIGN` §2.3: **the loss assembles against codecs, not against arches.** A value-target
codec turns a trajectory or a scalar into the thing the value head is trained against; it
composes with `dist65` (the *encoding* codec) rather than replacing it. Keeping the two separate
is what stops a target change from becoming an `if graph:` in the trainer.

**LANDING IS NOT ARMING (R322(d)).** Nothing in this module is reachable from a live training or
serving path: no config key selects it, `mantis.train.losses` does not import it, and no
production call site calls it. It is proven by the conformance suite and selected by nothing.
Arming is the operator's run6 prereg. The conformance suite asserts the unreachability
structurally rather than trusting this paragraph.

**NEITHER CODEC HERE ATTACKS THE VALUE BLIND SPOT, and that is a register fact rather than
modesty.** F-35/F-36/F-37 falsified *target* fixes on a *frozen dense representation*, and F-35
concludes the deficit is a FEATURE problem. No function here may be proposed as a blind-spot
lever; F-35 has already answered that proposal.

Two codecs:

  * `lambda_return_targets` — the KLENT λ-return, re-typed as a CODEC (SCOUT-1 §2's finding).
    It is mover-relative and respects the COMPOUND TWO-STONE TURN, which is the half a
    transcription from a single-stone-per-turn paper gets wrong by construction (LAW-03).
  * `scalar_to_hl_gauss` — HL-Gauss re-binning into the same 65-bin support `dist65` uses, a
    strict generalisation of `scalar_to_two_hot` that converges to it as the kernel narrows.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from mantis.model.dist65 import N_VALUE_BINS, VALUE_SUPPORT


class ValueTargetError(ValueError):
    """A value-target codec was handed inputs it cannot construct a target from.

    A `ValueError` for `RepresentationMismatch`'s reason: a ragged trajectory or a kernel width
    outside the mechanism's own range is a caller ERROR, not a condition to recover from — and a
    codec that silently returned *something* would be a target nobody could trace.
    """


def lambda_return_targets(
    values: Sequence[float] | torch.Tensor,
    movers: Sequence[int] | torch.Tensor,
    terminal_z: float,
    *,
    lam: float,
) -> torch.Tensor:
    """The λ-return value target for one trajectory, in each position's OWN mover's frame.

    A PURE FUNCTION: no state, no RNG, no device dependence, no checkpoint. That is the property
    that lets this land and be proven before the mint, and it is why the codec is the seam
    citizenship SCOUT-1 §2 assigns it — calling it a loss term is what would put an `if graph:`
    back in the trainer.

    THE COMPOUND TURN IS THE WHOLE DIFFICULTY, and it is why `movers` is an argument rather than
    an assumed alternation. This game takes TWO STONES PER TURN, so the mover does not alternate
    every index; a λ-return transcribed from a single-stone-per-turn source flips sign on the
    wrong indices and the error is invisible in the aggregate. `movers[i]` names who is to move
    at index `i`, and the recursion negates exactly at a HANDOVER — where `movers[i+1]` differs
    from `movers[i]` — and nowhere else.

    THE RECURSION, stated so the endpoints are checkable:
        `G[T-1] = z_T` in the last mover's frame;
        `G[i]   = (1 - lam) * v[i+1]^ + lam * G[i+1]^`,
    where `^` is "expressed in position `i`'s frame": the value is negated iff a handover occurs
    between `i` and `i+1`. So `lam = 1` is the pure Monte-Carlo return to the terminal and
    `lam = 0` is the one-step bootstrap, which are the two values whose answer is known
    independently of this implementation.

    `values` are the bootstraps this codec mixes. Their PROVENANCE is the caller's and it is not
    a detail: the paper's construction captures `v̂` at acting time under the acting policy,
    which is inseparable from a cleared buffer; ours is meant to be fed the MCTS ROOT VALUE,
    which is a property of the position under the current net and is recomputed when the position
    is replayed. That substitution is what breaks the on-policy dependence, and it makes this a
    different object that shares a name — not a port (SCOUT-1 §2).

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

    A Gaussian kernel of standard deviation `sigma` centred on the target, evaluated at the bin
    centres and normalised to sum to 1. A STRICT GENERALISATION of `scalar_to_two_hot`: as
    `sigma → 0` the mass collapses onto the nearest bin(s) and the encoding converges to the
    two-hot one, which is this codec's structural witness and the falsifier for a wrong
    implementation (SCOUT-1 §4 card 1, witness (i)).

    THE SUPPORT IS `dist65`'S OWN, imported rather than reconstructed: a second `linspace` here
    would be a second authority over the bin centres, and the two would agree until one moved.

    THE ONE REAL RISK, and it is specific and testable rather than general: `dist65` has an ODD
    bin count **so that an exact-zero bin exists**, and a kernel wide enough to help smears mass
    out of it. `sigma` is therefore NOT free, and no width is minted here — the accept floor is
    a run6 prereg row (`plan/SEAM_B2_LEG3_PREREG.md` W-H2). This function is the instrument that
    lets that floor be set against measured numbers.

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
    # Softmax over -d²/2 rather than exp-then-divide: the two are the same normalised kernel,
    # and this one does not underflow to an all-zero row (then 0/0) at a narrow width, which is
    # exactly the regime the σ → 0 identity witness drives.
    return torch.softmax(-0.5 * d * d, dim=-1)


__all__ = ["ValueTargetError", "lambda_return_targets", "scalar_to_hl_gauss"]
