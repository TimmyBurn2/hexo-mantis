# >300 justify (R8): the two codecs' witnesses and the unarmedness checks are one unit — a
# witness that proves a codec correct while the codec is quietly wired into a live path proves
# the wrong thing, and the two halves are only meaningful read together.
"""T12 — the two VALUE-TARGET CODECS land UNARMED behind the contract (R322(d) Leg 3).

The witnesses were registered BEFORE any implementation line existed, in the governance
workspace at `plan/SEAM_B2_LEG3_PREREG.md`, and this file is what reads them. The scout's
sketched shapes are followed verbatim except where contact falsified a detail, and each such
divergence is disclosed at its own site rather than smoothed over.

**LANDING IS NOT ARMING**, and that is asserted structurally in this file rather than promised
in a docstring: no config key selects either codec, no live training or serving module imports
them, and the trainer's loss assembly is untouched. Arming is the operator's run6 prereg.

**NOTHING HERE IS A STRENGTH CLAIM in either direction, and neither codec attacks the value
blind spot.** F-35/F-36/F-37 falsified *target* fixes on a *frozen dense representation*, and
F-35's own conclusion is that the deficit is a FEATURE problem — so no row here may be cited as
a blind-spot lever. F-01 is the standing fence on static probes generally.

THE TWO COMPONENTS AND THEIR REGISTERED WITNESSES:

  * `lambda_return_targets` — W-L1 the pure-function golden, W-L2 the mover-sign pair, W-L3 the
    two known endpoints. The λ-return is a CODEC and not a loss term (SCOUT-1 §2), and W-L2 is
    the row that matters for THIS game: a codec transcribed from a single-stone-per-turn source
    fails the compound-turn sign test by construction (LAW-03).
  * `scalar_to_hl_gauss` — W-H1 the σ → 0 identity against `scalar_to_two_hot`, W-H2 the
    zero-bin mass table with its accept floor DEFERRED to run6 prereg. W-H3, the strength /
    calibration arm, is NOT RUN here: it needs a trained arm and a box, both excluded.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from mantis.model.dist65 import N_VALUE_BINS, VALUE_SUPPORT, scalar_to_two_hot
from mantis.model.value_targets import (
    ValueTargetError,
    lambda_return_targets,
    scalar_to_hl_gauss,
)

from _corpus import ConformanceRefusal

REPO = Path(__file__).resolve().parents[3]
GOLDEN = REPO / "tests" / "fixtures" / "value_targets" / "lambda_return_golden_v1.json"

#: The module under test, as a path — the unarmedness sweep needs the name a consumer would
#: import, not the object.
CODEC_MODULE = "mantis.model.value_targets"

#: The candidate kernel widths W-H2 tabulates. A GRID, not a choice: picking one is an arming
#: decision and it is the operator's at run6 prereg. Spanning three orders of magnitude so the
#: table shows the trade rather than a point.
_SIGMAS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)

#: `dist65`'s bin width on [-1, 1] with 65 bins — DERIVED from the imported support, never
#: typed, so a support change moves it here too.
_BIN_WIDTH = float((VALUE_SUPPORT[1] - VALUE_SUPPORT[0]).item())


class CodecIsArmed(ConformanceRefusal):
    """A codec that is supposed to be selected by nothing is reachable from a live path."""


# ═══ W-L1 — the pure-function golden ═════════════════════════════════════════════════════
def test_WL1_the_lambda_return_golden_reproduces_bit_for_bit(derived):
    """W-L1, as registered: a fixed trajectory in, a fixed target vector out, byte-frozen.

    NO TOLERANCE BAND, and that is the registered shape rather than strictness for its own
    sake: a codec that needs one is not a pure function, which is the property that lets this
    component be proven before the mint with no checkpoint, no strength gate and no box.
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    out = lambda_return_targets(
        inputs["values"], inputs["movers"], inputs["terminal_z"], lam=inputs["lam"]
    )
    derived("t12.wl1.targets", [float(x) for x in out.tolist()])
    assert out.dtype is torch.float64, "the golden is fp64; a narrower dtype cannot reproduce it"
    assert [float(x) for x in out.tolist()] == golden["targets"], (
        "the λ-return codec no longer reproduces its frozen golden"
    )


def test_WL1_the_golden_is_recomputed_and_not_merely_read(derived):
    """The half a golden test can lose silently: the same call twice, and a control that the
    fixture is not simply being echoed. Two invocations must agree, and the golden must NOT
    equal the input values — otherwise a codec that returned its input would pass."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    first = lambda_return_targets(inputs["values"], inputs["movers"], inputs["terminal_z"],
                                  lam=inputs["lam"])
    second = lambda_return_targets(inputs["values"], inputs["movers"], inputs["terminal_z"],
                                   lam=inputs["lam"])
    assert torch.equal(first, second), "the codec is not deterministic across two calls"
    assert golden["targets"] != inputs["values"], (
        "the frozen targets equal the input values, so an identity function would pass W-L1"
    )


def test_WL1_the_trajectory_carries_a_COMPOUND_turn_and_an_odd_final_turn(derived):
    """The golden's own premise, executed. A golden whose mover column alternated every index
    would not exercise the compound turn at all, and W-L2 below would be testing a case the
    frozen fixture never reaches."""
    movers = json.loads(GOLDEN.read_text(encoding="utf-8"))["inputs"]["movers"]
    lengths: list[int] = []
    current = 1
    for prev, nxt in zip(movers, movers[1:]):
        if prev == nxt:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    derived("t12.wl1.turn_lengths", lengths)
    assert max(lengths) >= 2, "no compound turn in the frozen trajectory"
    assert min(lengths) == 1, (
        "every turn in the frozen trajectory is complete, so the mid-turn ending a real game "
        "can have is not covered"
    )


# ═══ W-L2 — the mover-sign pair ══════════════════════════════════════════════════════════
def test_WL2_a_handover_at_k_flips_the_sign_after_k_and_nowhere_else(derived):
    """W-L2, as registered and EXACTLY as registered: two trajectories identical except that one
    has a turn handover at index `k` must produce targets differing in sign at exactly the
    indices after `k`, and nowhere else.

    THE CONSTRUCTION IS THE SUBTLE PART, and getting it wrong is how this witness first appeared
    to falsify its own sketch. `values` and `terminal_z` are stated in each index's OWN mover's
    frame. So "the same game, with the post-`k` positions attributed to the other mover" is not
    "the same numbers with a different mover column": the bootstraps after `k` and the terminal
    outcome are the SAME FACTS SEEN FROM THE OTHER SIDE, so they carry a minus sign too. Built
    that way — one absolute game, two mover attributions — the registered claim holds exactly,
    in both directions and with no tolerance. The first attempt held the numbers fixed and
    flipped only the mover column, which silently described a DIFFERENT game (one where the
    terminal outcome had changed hands), and it is recorded here because the near-miss is the
    instructive half: the sketch was right and the construction was wrong.
    """
    values = [0.10, -0.20, 0.35, 0.05, -0.45, 0.60, 0.15]
    k = 3
    movers_a = [0, 0, 1, 1, 1, 1, 1]
    movers_b = [0, 0, 1, 1, 0, 0, 0]
    assert movers_a[: k + 1] == movers_b[: k + 1], "the two mover columns differ before k"
    assert movers_a[k + 1:] != movers_b[k + 1:], "the two mover columns agree after k"
    z_a = 1.0
    # The SAME absolute game, re-attributed after k: every mover-relative quantity after k is
    # the same fact from the other side, so it carries a minus sign.
    values_b = [v if i <= k else -v for i, v in enumerate(values)]
    z_b = -z_a
    a = lambda_return_targets(values, movers_a, z_a, lam=0.7)
    b = lambda_return_targets(values_b, movers_b, z_b, lam=0.7)
    derived("t12.wl2.attribution_a", [float(x) for x in a.tolist()])
    derived("t12.wl2.attribution_b", [float(x) for x in b.tolist()])
    for i in range(k + 1, len(values)):
        assert float(a[i]) == -float(b[i]) and float(a[i]) != 0.0, (
            f"index {i} is AFTER the handover at k={k} and is not the exact negation of its "
            "counterpart; the two attributions describe one game seen from two sides"
        )
    for i in range(k + 1):
        assert float(a[i]) == float(b[i]), (
            f"index {i} is at or BEFORE the handover at k={k} and moved. The λ-return reaches "
            "forward, so a change after k must arrive through the frame conversion and cancel "
            "exactly here — an implementation that reads the mover wrongly does not cancel"
        )


def test_WL2_an_implementation_that_IGNORES_the_compound_turn_FAILS_this(derived):
    """The falsifier the registration names: *"A λ-return implementation that ignores the
    compound turn fails this by construction."* Executed against a deliberate one — a
    ply-alternating mover column, which is what a transcription from a single-stone-per-turn
    source produces — so the witness is shown to BITE rather than asserted to."""
    values = [0.10, -0.20, 0.35, 0.05, -0.45, 0.60, 0.15]
    compound = [0, 0, 1, 1, 0, 0, 1]
    alternating = [i % 2 for i in range(len(values))]
    derived("t12.wl2.alternating_movers", alternating)
    assert not torch.equal(
        lambda_return_targets(values, compound, 1.0, lam=0.7),
        lambda_return_targets(values, alternating, 1.0, lam=0.7),
    ), (
        "the compound-turn mover column and a ply-alternating one produce the SAME targets, so "
        "this codec is not reading the mover at all and W-L2 cannot fail"
    )


# ═══ W-L3 — the two known endpoints ══════════════════════════════════════════════════════
def test_WL3_lambda_one_is_the_pure_monte_carlo_return(derived):
    """λ = 1: the target is the terminal outcome carried back through the handovers, so every
    entry is ±|z| and the sign is the mover's. Computed independently of the recursion below —
    by counting handovers — which is what makes it a check rather than a restatement."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    movers, z = inputs["movers"], inputs["terminal_z"]
    expected = []
    for i in range(len(movers)):
        flips = sum(1 for a, b in zip(movers[i:], movers[i + 1:]) if a != b)
        expected.append(z * (-1.0) ** flips)
    out = lambda_return_targets(inputs["values"], movers, z, lam=1.0)
    derived("t12.wl3.lam1", [float(x) for x in out.tolist()])
    assert [float(x) for x in out.tolist()] == expected
    assert expected == golden["endpoints"]["lam_1"], "the frozen endpoint disagrees"


def test_WL3_lambda_zero_is_the_one_step_bootstrap(derived):
    """λ = 0: every non-terminal target is the NEXT value in this index's frame, and nothing
    else — no mixing, no reach past one step."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    values, movers, z = inputs["values"], inputs["movers"], inputs["terminal_z"]
    expected = [
        values[i + 1] * (1.0 if movers[i] == movers[i + 1] else -1.0)
        for i in range(len(values) - 1)
    ] + [z]
    out = lambda_return_targets(values, movers, z, lam=0.0)
    derived("t12.wl3.lam0", [float(x) for x in out.tolist()])
    assert [float(x) for x in out.tolist()] == expected
    assert expected == golden["endpoints"]["lam_0"], "the frozen endpoint disagrees"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"values": [], "movers": [], "terminal_z": 1.0, "lam": 0.5}, "empty trajectory"),
        ({"values": [0.1, 0.2], "movers": [0], "terminal_z": 1.0, "lam": 0.5}, "ragged"),
        ({"values": [0.1], "movers": [0], "terminal_z": 1.0, "lam": 1.5}, "outside"),
        ({"values": [0.1], "movers": [0], "terminal_z": 1.0, "lam": -0.1}, "outside"),
    ],
)
def test_the_codec_REFUSES_inputs_it_cannot_construct_a_target_from(kwargs, match):
    """PB-T12a. A codec that returned *something* for a ragged trajectory or an out-of-range λ
    would produce a target nobody could trace back to a defect."""
    with pytest.raises(ValueTargetError, match=match):
        lambda_return_targets(kwargs["values"], kwargs["movers"], kwargs["terminal_z"],
                              lam=kwargs["lam"])


# ═══ W-H1 — the σ → 0 identity ═══════════════════════════════════════════════════════════
def test_WH1_hl_gauss_converges_to_TWO_HOT_as_the_kernel_narrows(derived):
    """W-H1, as registered: for kernel width → 0 the HL-Gauss encoding must converge to the
    two-hot encoding bin-for-bin. The structural witness, and the falsifier for a wrong
    implementation.

    DISCLOSED DIVERGENCE, one line: the registered wording says "converge bin-for-bin", and the
    limit is EXACT only at a bin CENTRE — between two centres the two-hot encoding splits mass
    linearly by distance while a narrowing Gaussian collapses onto the NEARER bin, so the two
    disagree by construction off-centre no matter how small σ is. The witness is therefore
    driven at bin centres for the identity claim, and the off-centre behaviour is asserted
    separately as convergence to the NEAREST bin. Contact falsified the sketch's generality,
    not its mechanism, and the correction narrows the claim rather than relaxing the test.
    """
    centres = VALUE_SUPPORT.clone()
    narrow = scalar_to_hl_gauss(centres, sigma=_BIN_WIDTH / 100.0)
    two_hot = scalar_to_two_hot(centres)
    gap = (narrow - two_hot).abs().max().item()
    derived("t12.wh1.max_abs_gap_at_bin_centres", gap)
    assert gap < 1e-6, (
        f"at σ = bin_width/100 the HL-Gauss encoding differs from two-hot by {gap} at a BIN "
        "CENTRE; the two must coincide in the narrow limit or this is not a generalisation of "
        "the shipped codec"
    )


def test_WH1_off_centre_the_narrow_limit_is_the_NEAREST_bin(derived):
    """The other half of the disclosed correction, asserted rather than left implied."""
    off = torch.tensor([0.001, -0.007, 0.4999, -0.2531])
    narrow = scalar_to_hl_gauss(off, sigma=_BIN_WIDTH / 100.0)
    nearest = (VALUE_SUPPORT.unsqueeze(0) - off.unsqueeze(1)).abs().argmin(dim=-1)
    derived("t12.wh1.offcentre_argmax", narrow.argmax(dim=-1).tolist())
    assert torch.equal(narrow.argmax(dim=-1), nearest)
    assert (narrow.max(dim=-1).values > 0.999).all(), "the narrow limit is not concentrated"


def test_WH1_every_encoding_is_a_DISTRIBUTION_at_every_width(derived):
    """The property both codecs' consumer (a cross-entropy) requires, checked across the whole
    candidate grid: rows sum to 1, nothing is negative, nothing is NaN. A softmax gives this by
    construction — which is why the implementation uses one — and asserting it is what would
    catch a later rewrite that went back to exp-then-divide and underflowed at a narrow σ."""
    z = torch.linspace(-1.0, 1.0, 41)
    for sigma in _SIGMAS:
        enc = scalar_to_hl_gauss(z, sigma=sigma)
        assert enc.shape == (41, N_VALUE_BINS)
        assert torch.isfinite(enc).all(), sigma
        assert (enc >= 0).all(), sigma
        assert torch.allclose(enc.sum(dim=-1), torch.ones(41), atol=1e-6), sigma


# ═══ W-H2 — the zero-bin mass table, floor DEFERRED ══════════════════════════════════════
def test_WH2_the_zero_bin_mass_TABLE_is_produced_and_its_floor_is_DEFERRED(derived):
    """W-H2, as registered — the instrument and its table, and NOT the accept floor.

    `dist65` has an ODD bin count so that an EXACT-ZERO bin exists; a kernel wide enough to help
    smears mass out of it. Choosing the floor is choosing a candidate width, which is an arming
    decision and is a run6 prereg row (`SEAM_B2_LEG3_PREREG.md` W-H2). What this row asserts is
    only what makes the table trustworthy: the mass is monotone DECREASING in σ, so the operator
    is reading a trade and not noise, and the widest candidate has actually left the bin.

    The −0.5 row is ours and not the scout's: the draw / ply-cap label sits at −0.5, which is
    NOT a bin centre on a 65-bin support over [−1, 1], so the shipped two-hot codec already
    splits it across two bins. A label that is already split is where a smearing kernel is least
    visible, which is why it is tabulated beside the zero-bin row rather than assumed to behave
    the same way.
    """
    zero_bin = int((VALUE_SUPPORT.abs()).argmin().item())
    assert float(VALUE_SUPPORT[zero_bin]) == 0.0, (
        "there is no EXACT-ZERO bin on this support, so W-H2 has no subject — that is a change "
        "to `dist65`'s own bin count and is a ruling"
    )
    table = []
    for sigma in _SIGMAS:
        at_zero = scalar_to_hl_gauss(torch.tensor([0.0]), sigma=sigma)[0]
        at_draw = scalar_to_hl_gauss(torch.tensor([-0.5]), sigma=sigma)[0]
        table.append({
            "sigma": sigma,
            "sigma_in_bins": sigma / _BIN_WIDTH,
            "zero_bin_mass": float(at_zero[zero_bin]),
            "draw_label_top2_mass": float(at_draw.topk(2).values.sum()),
        })
    derived("t12.wh2.zero_bin_mass_table", table)
    derived("t12.wh2.bin_width", _BIN_WIDTH)
    derived("t12.wh2.accept_floor", "DEFERRED — run6 prereg row, operator's (R322(d))")
    masses = [row["zero_bin_mass"] for row in table]
    assert masses == sorted(masses, reverse=True), (
        f"zero-bin mass is not monotone decreasing in σ: {masses}. The table is meant to show a "
        "TRADE; a non-monotone column means the operator would be choosing against noise"
    )
    assert masses[0] > 0.99, "the narrowest candidate does not hold the zero bin"
    assert masses[-1] < 0.5, (
        "the widest candidate still holds most of the zero bin, so this grid does not span the "
        "risk W-H2 exists to measure"
    )


def test_WH2_no_kernel_WIDTH_is_minted_anywhere_in_the_tree(derived):
    """The arming half of W-H2, made structural. `sigma` has no default and no config key, so
    there is no width this repo has chosen — which is the state the deferral describes."""
    import inspect

    parameter = inspect.signature(scalar_to_hl_gauss).parameters["sigma"]
    assert parameter.default is inspect.Parameter.empty, (
        "`scalar_to_hl_gauss` grew a default σ; a default width is a minted width, and minting "
        "one is the run6 prereg row this leg deferred"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    derived("t12.wh2.sigma_has_no_default", True)


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_the_hl_gauss_codec_REFUSES_a_non_positive_width(bad):
    """PB-T12b. σ = 0 is the two-hot encoding's limit, not a value this function computes; a
    codec that returned NaNs for it would be a silent target corruption."""
    with pytest.raises(ValueTargetError, match="not strictly positive"):
        scalar_to_hl_gauss(torch.tensor([0.0]), sigma=bad)


# ═══ LANDING IS NOT ARMING — asserted structurally ═══════════════════════════════════════
def test_NO_live_module_imports_either_codec(derived):
    """The clause that governs this whole leg, executed rather than promised (R322(d)).

    An AST import census over `src/` and `tools/`: nothing outside the codec module itself may
    import it. Tests may — that is what proves it — and the conformance suite is where the proof
    lives, which is the whole shape of "proven by the suite, selected by nothing".
    """
    importers: list[str] = []
    for root in (REPO / "src", REPO / "tools"):
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel == "src/mantis/model/value_targets.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "") == CODEC_MODULE:
                    importers.append(rel)
                elif isinstance(node, ast.Import) and any(
                    alias.name == CODEC_MODULE for alias in node.names
                ):
                    importers.append(rel)
    derived("t12.live_importers", sorted(set(importers)))
    if importers:
        raise CodecIsArmed(
            f"{CODEC_MODULE} is imported by {sorted(set(importers))}. It landed UNARMED under "
            "R322(d) — proven by the suite, selected by nothing, armed by nothing — and arming "
            "it is the operator's run6 prereg. Delete this row in the same commit as the ruling "
            "that arms it."
        )


def test_the_model_PACKAGE_does_not_re_export_either_codec():
    """The public surface half: an exported codec is one import away from a live path, and the
    unarmedness above would then be a fact about today rather than a property."""
    import mantis.model as package

    for name in ("lambda_return_targets", "scalar_to_hl_gauss", "ValueTargetError"):
        assert name not in package.__all__, f"{name} is re-exported from mantis.model"


def test_the_TRAINER_loss_assembly_is_untouched():
    """The one place a value-target codec would actually bite. `mantis.train.losses` must not
    reach either function — that is the `if graph:`-in-the-trainer outcome the seam exists to
    prevent, and it is the specific thing SCOUT-1 §2 warns λ-returns would cause if typed as a
    loss term instead of a codec."""
    losses = (REPO / "src" / "mantis" / "train" / "losses.py").read_text(encoding="utf-8")
    for name in ("lambda_return_targets", "scalar_to_hl_gauss", "value_targets"):
        assert name not in losses, f"the trainer's loss module reaches {name}"


def test_the_unarmedness_census_can_FIRE(derived):
    """PB-T12c. LAW-07 on this file's own guard: the import census must be able to see an
    import, or "nothing imports it" is a statement about the census and not about the tree."""
    tree = ast.parse(f"from {CODEC_MODULE} import lambda_return_targets\n")
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == CODEC_MODULE
    ]
    assert found, "the census cannot see an import of the codec module even when given one"
    plain = ast.parse(f"import {CODEC_MODULE}\n")
    assert [
        node for node in ast.walk(plain)
        if isinstance(node, ast.Import) and any(a.name == CODEC_MODULE for a in node.names)
    ], "the census cannot see a plain `import` form"
