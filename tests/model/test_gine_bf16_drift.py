"""bf16 numeric drift against the fp32 reference (OF1-3, OF1-4).

Every gating reduction here is the MEDIAN form; the MAX form (OF1-3b, OF1-3c-max) is RETIRED
from gating and recorded only, because its identical-code CUDA null reaches 9.3367e-1, 2.33x
its own 4.0e-1 ABORT threshold. No band, envelope or threshold in this file was moved.

R8 justification: two arms of a real 4-layer GNN forward+loss+backward and eight pre-registered
statistics; the fixture and arms live in `_bf16_parity.py`, and the envelopes stay here beside
the rows they bound.

REGIME: every row runs on the DETERMINISTIC CPU path, where HEAD-vs-HEAD is bit-identical on
1140/1140 measured pairs, so each statistic's identical-code null is exactly 0. These rows
therefore CANNOT witness F1 — CPU F1-vs-HEAD is also bit-identical, 1200/1200, so F1 is an
exact no-op on this path. The green is on the bf16 REGIME's drift; the CUDA calibrated legs
live in `test_bf16_parity_nulldist.py`. DISCLOSED: the fixture's mean in-degree is ~8 against
run5's ~26.8, a gap in the direction that makes the effect SMALLER, so a real sampled run5
batch outranks every row here.

THREE-BAND DISCIPLINE: PASS / PASS-WITH-DISCLOSURE / ABORT. Only ABORT halts, so only it is
asserted; a middle-band result is not a failure and may not be summarised as "no abort fired".

WITNESS vs BOUND, measured against 3000 F1-vs-HEAD CUDA pairs: OF1-3a is the WITNESS (0.0%
overlap on all three fixtures); OF1-4a is a NARROW secondary (3.84x on the production fixture);
OF1-3c (85.1% overlap), OF1-3d (39.7% / 87.9%) and OF1-4b (62.1%) are BOUND only, as are
OF1-3e/OF1-3f with their CUDA null UNVERIFIED. A BOUND row's green says the bf16 regime's
deviation is inside the registered band, never that F1's effect is bounded.

MUTATION COVERAGE — THIS FILE KILLS NOTHING IN THE F1 MUTATION BANK ON THE CPU PATH. Measured
on a scratch copy: MA-1, MA-3 and MA-4 all PASS every row here, and every failure under all
three lands in `test_gine_gather_regime.py`, which carries the bank alone. `index_add_` does
raise on an fp32 `self` with a bf16 source, but only where the conv receives an fp32 `x`, and
here `nn.LayerNorm` is dtype-PRESERVING on CPU so the conv's `x` is already bf16.

UNVERIFIED: MA-4 through the production `RepresentationNetwork` on CUDA, where `layer_norm` IS
promoted to fp32 — there the raise is expected and the drift rows would red. Nobody has run it.
"""
from __future__ import annotations

import math
from pathlib import Path

import _bf16_parity as bp
import pytest
import torch

_REGIME = "DETERMINISTIC-PATH (CPU): reading is EXACT, identical-code null = 0 (1140/1140)"


def _band(value: float, envelope: float, threshold: float, *, lower_is_better: bool) -> str:
    """THREE outcomes, because the prereg preamble defines three.

    The inherited two-outcome form banded against `envelope` alone and never read `threshold`,
    so a reading past its own ABORT threshold printed PASS-WITH-DISCLOSURE. Invisible for the
    gating rows, but the RETIRED rows are not asserted — printing into the record is their
    entire remaining function, and they printed PASS at 1.556892e+00 against 4.0e-1.
    """
    if lower_is_better:
        if value > threshold:
            return "ABORT"
        return "PASS" if value <= envelope else "PASS-WITH-DISCLOSURE"
    if value < threshold:
        return "ABORT"
    return "PASS" if value >= envelope else "PASS-WITH-DISCLOSURE"


def _report(row: str, value: float, envelope: float, threshold: float, *,
            role: str, lower_is_better: bool = True) -> str:
    band = _band(value, envelope, threshold, lower_is_better=lower_is_better)
    line = (f"MEASURED {row}: value={value:.6e} envelope={envelope:.3e} "
            f"threshold={threshold:.3e} band={band} role={role} regime={_REGIME}")
    print(line)
    return line


@pytest.fixture(scope="module")
def arms() -> tuple[bp.Arm, bp.Arm, bp.Batch]:
    """(a) fp32 reference, (c) bf16 treatment, and the batch. Module-scoped: both arms are
    a full 4-layer forward+backward and every row below reads the same two."""
    net = bp.build_net()
    batch = bp.build_batch(bp.build_arch())
    ref = bp.run_arm(net, batch, autocast_enabled=False)
    treat = bp.run_arm(net, batch, autocast_enabled=True)
    print(f"MEASURED fixture: graphs={bp._N_GRAPHS} nodes={batch.x.shape[0]} "
          f"edges={batch.edge_index.shape[1]} mean_in_degree={batch.mean_in_degree:.3f} "
          f"(run5 measured ~26.8 — MEASUREMENT_D §3)")
    return ref, treat, batch


def test_policy_logit_median_relative_drift(arms) -> None:
    """OF1-3a — THE RE-POINTED PRIMARY WITNESS. PASS <= 2.0e-2; DISCLOSE (2.0e-2, 5.0e-2];
    ABORT > 5.0e-2, bands unchanged from the prereg. The one row measured to distinguish F1
    from nothing on all three box fixtures (0.0% overlap, 3675 null vs 3000 F1 pairs)."""
    ref, treat, _b = arms
    value = bp.median_form(treat.policy_logits, ref.policy_logits)
    line = _report("OF1-3a policy-logit median rel drift", value, 2.0e-2, 5.0e-2,
                   role="WITNESS (primary, R181 re-point)")
    assert value <= 5.0e-2, f"{line} — F1-ABORT-1: HALT to the architect (option iii revived)"


def test_bin_logit_median_relative_drift(arms) -> None:
    """OF1-3c (median reduction only). PASS <= 2.0e-2; DISCLOSE (2.0e-2, 5.0e-2]; ABORT.

    BOUND, NOT A WITNESS: 851 of 1000 genuine F1-vs-HEAD pairs read exactly 0.0 on the
    production fixture — 85.1% overlap with its own null. It bounds the regime, not F1."""
    ref, treat, _b = arms
    value = bp.median_form(treat.bin_logits, ref.bin_logits)
    line = _report("OF1-3c bin-logit median rel drift", value, 2.0e-2, 5.0e-2,
                   role="BOUND ONLY — 85.1% F1 overlap, cannot witness F1")
    assert value <= 5.0e-2, f"{line} — F1-ABORT-1: HALT"


def test_loss_relative_drift(arms) -> None:
    """OF1-3d, both graph losses. PASS <= 5.0e-2 each; DISCLOSE (5.0e-2, 1.5e-1]; ABORT.

    BOUND, NOT A WITNESS: F1-vs-HEAD overlap with the null is 39.7% (policy) and 87.9%
    (value), and the run-to-run spread over 50 CUDA runs of unmodified code is 181.8% and
    270.1% of the readings' own value. As a bound it is sound: ABORT sits 3.4e4x / 2.9e3x
    above the measured null max."""
    ref, treat, _b = arms
    for name, a, c in (("policy_loss", ref.policy_loss, treat.policy_loss),
                       ("value_loss", ref.value_loss, treat.value_loss)):
        value = abs(c - a) / abs(a)
        line = _report(f"OF1-3d {name} rel drift", value, 5.0e-2, 1.5e-1,
                       role="BOUND ONLY — 39.7%/87.9% F1 overlap, cannot witness F1")
        assert value <= 1.5e-1, f"{line} — F1-ABORT-1: HALT"


def _segment_argmax(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        int(logits[int(offsets[i]):int(offsets[i + 1])].argmax())
        for i in range(int(offsets.shape[0]) - 1)
    ])


def test_policy_argmax_agreement(arms) -> None:
    """OF1-3e. PASS >= 90%; DISCLOSE [80%, 90%); ABORT < 80%.

    The decision the policy head exports is its per-graph argmax, so a drift that leaves every
    logit close but flips the chosen move is what this row alone sees.

    CUDA NULL UNVERIFIED — the calibration measured five statistics and the argmax rows were
    not among them; only CPU bit-identity (1140/1140) licenses gating it here. On a 16-graph
    fixture its resolution is 6.25 pp, two graphs wide, so it is underpowered regardless."""
    ref, treat, batch = arms
    a = _segment_argmax(ref.policy_logits, batch.legal_offsets)
    c = _segment_argmax(treat.policy_logits, batch.legal_offsets)
    value = float((a == c).to(torch.float32).mean())
    line = _report("OF1-3e per-graph policy argmax agreement", value, 0.90, 0.80,
                   role="BOUND ONLY — CUDA null UNVERIFIED, never calibrated",
                   lower_is_better=False)
    assert value >= 0.80, f"{line} — F1-ABORT-1: HALT"


def test_bin_argmax_within_one_bin(arms) -> None:
    """OF1-3f. PASS >= 95%; DISCLOSE [90%, 95%); ABORT < 90%.

    **CUDA NULL UNVERIFIED (R181 disclosure)** — see OF1-3e's docstring; the same applies.
    """
    ref, treat, _b = arms
    delta = (treat.bin_logits.argmax(dim=-1) - ref.bin_logits.argmax(dim=-1)).abs()
    value = float((delta <= 1).to(torch.float32).mean())
    line = _report("OF1-3f dist65 argmax within +/-1 bin", value, 0.95, 0.90,
                   role="BOUND ONLY — CUDA null UNVERIFIED, never calibrated",
                   lower_is_better=False)
    assert value >= 0.90, f"{line} — F1-ABORT-1: HALT"


def test_gradient_cosine_similarity(arms) -> None:
    """OF1-4a. PASS >= 0.99; DISCLOSE [0.95, 0.99); ABORT < 0.95.

    SECONDARY WITNESS, NARROW: the cosine deficit discriminates on all three fixtures but on
    the production fixture by only 3.84x (null max 7.1952e-6, F1 min 2.7627e-5).

    Computed in float32 by `cosine_similarity`, the form the band was set on. Where EXACTNESS
    is the claim the assertion must be `torch.equal`, never `cos == 1.0`: the float64 deficit
    reads 1.1102e-16 on bit-identical gradients and the float32 form reads > 1.0."""
    ref, treat, _b = arms
    value = float(torch.nn.functional.cosine_similarity(
        treat.grads.unsqueeze(0), ref.grads.unsqueeze(0)
    ).squeeze())
    line = _report("OF1-4a parameter-gradient cosine", value, 0.99, 0.95,
                   role="WITNESS (secondary, NARROW: 3.84x on the production fixture)",
                   lower_is_better=False)
    assert value >= 0.95, f"{line} — F1-ABORT-1: HALT to the architect"


def test_gradient_norm_relative_difference(arms) -> None:
    """OF1-4b. PASS <= 5.0e-2; DISCLOSE (5.0e-2, 1.5e-1]; ABORT > 1.5e-1.

    Direction and magnitude are separate failures: a systematically shrunk gradient has cosine
    ~1 and rescales the effective learning rate. BOUND, NOT A WITNESS — 62.1% of genuine
    F1-vs-HEAD pairs fall at or below its null; as a bound the envelope is 62x that null max
    of 8.0303e-4."""
    ref, treat, _b = arms
    n_a = float(ref.grads.norm())
    value = abs(float(treat.grads.norm()) - n_a) / n_a
    line = _report("OF1-4b parameter-gradient norm rel difference", value, 5.0e-2, 1.5e-1,
                   role="BOUND ONLY — 62.1% F1 overlap, cannot witness F1")
    assert value <= 1.5e-1, f"{line} — F1-ABORT-1: HALT"


# Frozen census mapping each row in THIS file to the numeric literals its `assert` statements
# compare against. Derived from the prereg's ABORT thresholds, not from the code.
_GATING_ASSERT_CENSUS = {
    "test_policy_logit_median_relative_drift": {5.0e-2},
    "test_bin_logit_median_relative_drift": {5.0e-2},
    "test_loss_relative_drift": {1.5e-1},
    "test_policy_argmax_agreement": {0.80},
    "test_bin_argmax_within_one_bin": {0.90},
    "test_gradient_cosine_similarity": {0.95},
    "test_gradient_norm_relative_difference": {1.5e-1},
    # RETIRED row: its literals are the RETIREMENT'S GROUNDS (the max form's own ABORT
    # threshold, and the measured bf16 ulp), never a band applied to its own reading.
    "test_retired_max_form_rows_are_recorded_not_gating": {4.0e-1, 1.953125e-3},
}


def test_every_gating_row_still_asserts_its_registered_threshold() -> None:
    """Refuse a gating row that has silently become non-gating.

    Measured hole, reproduced before closing it: deleting the gating `assert` from OF1-3a left
    the tier fully green — the row still ran, still printed its MEASURED line, still reported
    PASS, and asserted nothing.

    Closed by AST: for every row it pins the exact numeric literals that row's `assert`
    statements compare against, so it REDs on a deleted assert, a changed threshold, a new row
    with no census entry, or a censused row deleted.

    Residue: it cannot catch an assert weakened while keeping its literal — the comparison
    operator and left operand are not pinned, and pinning the full expression would be a
    second copy of the row.
    """
    import ast

    source = Path(__file__).read_text()
    tree = ast.parse(source)
    census: dict[str, set[float]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name == "test_every_gating_row_still_asserts_its_registered_threshold":
            continue
        literals: set[float] = set()
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assert):
                continue
            for cmp_node in ast.walk(sub.test):
                if not isinstance(cmp_node, ast.Compare):
                    continue
                for operand in cmp_node.comparators:
                    if isinstance(operand, ast.Constant) and isinstance(
                        operand.value, (int, float)
                    ) and not isinstance(operand.value, bool):
                        literals.add(float(operand.value))
        census[node.name] = literals

    missing = set(_GATING_ASSERT_CENSUS) - set(census)
    unexpected = set(census) - set(_GATING_ASSERT_CENSUS)
    assert not missing, f"censused rows have vanished from this file: {sorted(missing)}"
    assert not unexpected, (
        f"new rows in this file are not in the frozen census: {sorted(unexpected)}. Add them "
        f"with their registered ABORT thresholds — a row outside the census is a row whose "
        f"assertions nothing guards."
    )
    for name, expected in _GATING_ASSERT_CENSUS.items():
        assert census[name] == expected, (
            f"{name}: asserted literals {sorted(census[name])} != registered "
            f"{sorted(expected)}. Either an assertion was removed (the row is now "
            f"NON-GATING while still printing PASS) or a threshold NUMBER was moved (R61). "
            f"Neither is a test-file edit; both go back to the prereg."
        )
    print(f"GATING-CENSUS {len(_GATING_ASSERT_CENSUS)} rows, every registered ABORT threshold "
          f"still asserted: { {k: sorted(v) for k, v in sorted(census.items())} }")


def test_retired_max_form_rows_are_recorded_not_gating(arms) -> None:
    """OF1-3b and OF1-3c-max: RETIRED from gating.

    This row prints the retired readings and asserts the RETIREMENT'S OWN GROUNDS, re-derived
    from the pinned artifact so the retirement is machine-checked: the identical-code CUDA null
    of each max-form statistic EXCEEDS the band that row was asked to police. If that ceases to
    be true the assertion REDs and the retirement returns to the architect.

    The max form is NOT blind to F1 — its F1-vs-HEAD minimum (1.3436e+0) sits above its
    identical-code maximum (9.3367e-1), 0.0% overlap at 1.44x separation. Its disqualification
    is worse: the null exceeds its own ABORT threshold, so it aborts on 24.82% of comparisons
    of a commit against itself.

    Second indictment, on the deterministic path: the 1.5e-1 PASS envelope demands an absolute
    error <= 1.5e-4 at the tensors' p90 scale, where ONE bf16 ulp measures 1.953125e-3 — about
    1/13 of a single ulp, unreachable by any bf16 regime on any device.
    """
    ref, treat, _b = arms
    doc = bp.load_nulldist()

    # The retired readings, printed and carried into the measurement report.
    for row, t, r, envelope, threshold in (
        ("OF1-3b policy-logit max rel drift", treat.policy_logits, ref.policy_logits,
         1.5e-1, 4.0e-1),
        ("OF1-3c bin-logit max rel drift", treat.bin_logits, ref.bin_logits, 1.5e-1, 4.0e-1),
    ):
        value = float(bp.rel(t, r).max())
        _report(row, value, envelope, threshold, role="RETIRED — recorded, NOT gating")
        assert math.isfinite(value), f"{row}: non-finite reading"

    # The retirement's grounds, re-derived from the per-pair columns at point of use.
    for stat, registered_abort, registered_pass in (
        ("policy_max_rel", 4.0e-1, 1.5e-1),
        ("bin_max_rel", 4.0e-1, 1.5e-1),
    ):
        cols = bp.cuda_pairs(doc, "null", stat)["prod27_run5shape"]
        null_max = max(cols)
        over_pass = sum(1 for v in cols if v > registered_pass) / len(cols)
        print(f"RETIREMENT-GROUNDS {stat}: identical-code CUDA null max={null_max:.6e} "
              f"over n={len(cols)} pairs; fraction over the registered PASS envelope "
              f"{registered_pass:.3e} = {100 * over_pass:.2f}%")
        assert null_max > registered_pass, (
            f"{stat}: the retirement's ground has vanished — the identical-code null "
            f"({null_max:.6e}) no longer exceeds the registered PASS envelope "
            f"({registered_pass:.3e}). Re-adjudicate the R181 retirement; do NOT re-arm "
            f"the row silently."
        )
    policy_null_max = max(bp.cuda_pairs(doc, "null", "policy_max_rel")["prod27_run5shape"])
    assert policy_null_max > 4.0e-1, (
        f"OF1-3b: the identical-code null ({policy_null_max:.6e}) no longer exceeds the "
        f"row's own ABORT threshold (4.0e-1) — R181's central finding must be re-measured."
    )

    # The bf16 spacing the second indictment rests on, measured rather than transcribed.
    quarter = torch.tensor([0.3], dtype=torch.bfloat16)
    nxt = (quarter.view(torch.int16) + 1).view(torch.bfloat16)
    ulp_p90 = float(nxt) - float(quarter)
    print(f"RETIREMENT-GROUNDS measured bf16 ulp on [0.25, 0.5) = {ulp_p90:.9e}; "
          f"the 1.5e-1 envelope demands <= 1.5e-4 there = 1/{ulp_p90 / 1.5e-4:.2f} of one ulp")
    assert ulp_p90 == 1.953125e-3, f"bf16 spacing on [0.25, 0.5) measured {ulp_p90!r}"
    assert 1.5e-4 < ulp_p90 / 10.0, (
        "the max form's envelope is no longer sub-ulp — re-adjudicate the retirement"
    )
