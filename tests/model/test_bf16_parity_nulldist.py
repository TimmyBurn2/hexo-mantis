"""WP12-R R181 — the re-pointed F1 parity oracle's GROUNDS: null calibration + mutation.

R8 justification: 300+ lines. R181 requires one artifact to carry four inseparable things —
the pinned null distribution, the detection floor DERIVED from it, and BOTH halves of the
two-sided mutation condition. Splitting them would let a derived constant live in a different
file from the measurement it is derived from, which is the exact failure R181 was ruled against.

R181, the mandate this file implements, verbatim on the operative clause:

    "the parity oracle re-points to a statistic that is ZERO on identical code (the median
     form measures 0.000e+00 on all pairs — it is the discriminating base), bounded by a
     null-calibrated envelope derived from the MEASURED HEAD-vs-HEAD distribution with
     stated margin; the null-distribution measurement commits as a pinned artifact and is
     the envelope's cited grounds (R69). ... granted under S-3 with the two-sided mutation
     condition: new statistic REDS under an injected real numerics change and GREENS on N
     fresh HEAD-vs-HEAD pairs. Where exactness matters, a deterministic-path (CPU) parity
     leg may assert equality outright — CUDA legs assert the calibrated bound, and each
     leg's label says which it is."

**R191 AMENDS R181's LAST CLAUSE, on measurement.** *"CUDA legs assert the calibrated bound"*
did not survive contact with a second GPU: the bound was calibrated on an RTX 5080 whose
identical-code null is exactly zero, and on an RTX 4060 the same statistic reads **0/15 pairs
zero, worst 1.395037e-02 — 14.0x that bound, and above F1's own maximum effect.** R191:

    "The CUDA parity leg runs under torch.use_deterministic_algorithms(True) and asserts
     EXACT equality on identical code ... an exact assertion beats any calibrated bound, is
     device-independent, and R181 already authorized the shape... Determinism is TEST-SCOPE
     ONLY — production keeps its kernels; the leg says so in its name."

**PER-DEVICE ENVELOPE CALIBRATION IS REJECTED (R191) and no hook for it exists here.**

REGIME LABELS. Every test below is labelled in its own name and docstring:

  * `..._cpu_exact_...`  — DETERMINISTIC PATH. Asserts EQUALITY outright (`torch.equal`).
                           Licensed by 1140/1140 measured bit-identical CPU null pairs.
                           Cannot witness F1 (CPU F1-vs-HEAD is also 1200/1200 identical).
  * `..._cuda_..._TEST_SCOPE_determinism` — asserts EXACT EQUALITY under
                           `torch.use_deterministic_algorithms(True)`. **TEST SCOPE ONLY:
                           production keeps its nondeterministic kernels.** LOUD SKIP
                           without a GPU — **NOT run by CI (venv is `torch 2.11.0+cpu`);
                           measured out-of-band on an RTX 4060 / `torch 2.11.0+cu130`,
                           results in IMPL_NOTES_R181_REPOINT §7C.**
  * `..._artifact_...`   — reads the pinned measurement; device-independent.

**THE PINNED ARTIFACT IS DEVICE-SPECIFIC (R191).** Its measurement stands; its generality
does not. See `test_artifact_null_is_device_specific_not_a_property_of_the_statistic`.

WHAT IS NOT CLAIMED. The re-pointed statistic is a MAJORITY-BIT-IDENTITY test: it reads
zero for any change leaving more than half the bf16 elements unchanged, at ANY magnitude.
That blind spot is not a footnote — `test_repointed_statistic_is_blind_to_a_minority_element
_change` asserts it, so that no reader can come to believe otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest

import _bf16_parity as bp
import pytest
import torch

# ── the DETECTION FLOOR (was: "the CUDA null envelope") ──────────────────────────────
# **RE-POINTED BY R191. This constant is NO LONGER A NULL BOUND on any device**, because
# both null legs now assert EXACT equality (CPU natively, CUDA under test-scope determinism).
# It survives in one role only: the value below which a reading carries no information, i.e.
# E3 "the discrimination floor" in MEASUREMENT_NULLDIST §7's own taxonomy — used by the two
# RED halves and by the blind-spot row, never to bound a null.
#
# Its derivation is re-executed live in `test_detection_floor_is_derived_and_its_null_role_is
# _withdrawn`. One of the three original constraints was WITHDRAWN as device-specific (C1,
# "above the measured null max = 0.0"); the two that survive are properties of bf16 and of
# F1's own effect, and I re-measured both on a SECOND GPU rather than assume they transfer.
# Margin 3 is not invented here: MEASUREMENT_NULLDIST §7.2 registered it for this measurement
# from its own tail ratios (worst max/p99 = 1.422, worst max/p50 = 7.99).
_DETECTION_FLOOR = 1.0e-3
_DERIVATION_MARGIN = 3.0

# N for R181's "GREENS on N fresh HEAD-vs-HEAD pairs". 6 fresh full forward+backward runs
# => 15 unordered pairs, the same all-pairs construction the box measurement used
# (MEASUREMENT_NULLDIST §2). Justified in the test's own docstring.
_N_NULL_RUNS = 6
_N_NULL_PAIRS = _N_NULL_RUNS * (_N_NULL_RUNS - 1) // 2

_EXPECTED_COMMIT = "982da03bae57758efc65c6cfe0d451d77f15561c"
_FIXTURES = ("synth8", "prod27_samesizes", "prod27_run5shape")
_PROD = "prod27_run5shape"

# `|a|` p50 of the fp32 reference policy logits, per fixture. The `synth8` value is
# DERIVED LOCALLY below from the real fp32 arm and is the BINDING one (smallest floor);
# the other two are transcribed box quantities (MEASUREMENT_BF1 §6.5) and are recorded
# only so the worst-case selection is visible.
_BOX_ABS_P50 = {"prod27_samesizes": 0.2050, "prod27_run5shape": 0.2102}


def _bf16_ulp(x: float) -> float:
    """One bf16 spacing at `x`, MEASURED from torch rather than computed from a binade
    formula. MEASUREMENT_NULLDIST §4.2 and MEASUREMENT_BF1 §6.5 both state this quantity
    as 2^(e-8); the true bf16 spacing is 2^(e-7). See
    `test_measured_bf16_spacing_corrects_the_reports_quantisation_arithmetic`."""
    a = torch.tensor([x], dtype=torch.bfloat16)
    nxt = (a.view(torch.int16) + 1).view(torch.bfloat16)
    return float(nxt) - float(a)


@pytest.fixture(scope="module")
def runs() -> tuple[object, bp.Batch, list[bp.Arm], bp.Arm]:
    """`_N_NULL_RUNS` fresh bf16-arm runs of IDENTICAL CODE, plus one fp32 reference arm.

    Every run is a full 4-layer forward + loss + backward on the PREREG_DFIX §1 fixture.
    `torch.autograd.grad` is used throughout, so no run can reach another through `.grad`.
    """
    net = bp.build_net()
    batch = bp.build_batch(bp.build_arch())
    bf16 = [bp.run_arm(net, batch, autocast_enabled=True) for _ in range(_N_NULL_RUNS)]
    fp32 = bp.run_arm(net, batch, autocast_enabled=False)
    return net, batch, bf16, fp32


# ── the pinned artifact — R69's cited grounds ────────────────────────────────────────
def test_artifact_pinned_nulldist_identity_and_shape() -> None:
    """The oracle's cited grounds are present, sha-pinned, and the shape they claim.

    `bp.load_nulldist()` RAISES on absence or sha drift — it never skips. Grounds that
    cannot be read are not grounds (R69, LAW-07).
    """
    doc = bp.load_nulldist()
    assert doc["artifact_id"] == "WP12R-R181-NULLDIST-v1"
    assert doc["schema_version"] == 1
    assert "R181" in doc["governing_ruling"]
    assert doc["provenance"]["commit_sha"] == _EXPECTED_COMMIT
    assert doc["provenance"]["torch"] == "2.11.0+cu128"
    assert doc["provenance"]["deterministic_algorithms"].startswith("NOT enabled")
    n_null = sum(c["null_pairs_treat_columnar"]["n_pairs"] for c in doc["cuda"]["fixtures"].values())
    n_alt = sum(c["alt_pairs_treat_columnar"]["n_pairs"] for c in doc["cuda"]["fixtures"].values())
    assert set(doc["cuda"]["fixtures"]) == set(_FIXTURES)
    assert (n_null, n_alt) == (3675, 3000), f"artifact shape drift: {n_null=} {n_alt=}"
    print(f"GROUNDS pinned artifact {bp.NULLDIST_PATH.name} sha256={bp.NULLDIST_SHA256} "
          f"cuda_null_pairs={n_null} cuda_alt_pairs={n_alt}")


def test_loader_raises_on_sha_drift(tmp_path) -> None:
    """**LAW-07 producer test for `NullDistArtifactError`, half 1 of 2: sha drift.**

    Six tests derive their grounds through `load_nulldist`, so its refusal to accept a
    changed artifact is a safety property — and until this row it had no producer test.
    `REVIEW_IMPL_DFIX_A` check 10 named the gap; this closes it. The gap mattered because
    the whole of this WP has been finding gates whose producers did not exist (R4/LAW-07).

    THE CONTROL runs first, so the row cannot pass by raising on everything: a byte-exact
    copy at a DIFFERENT path loads and returns the real artifact.

    THE MUTATION is deliberately the weakest one available — **one trailing whitespace
    byte.** The mutated file is still valid JSON and still parses to a semantically
    identical document; only its bytes differ. A loader that gated on "does it parse" or
    "does it have the right keys" would pass this. The sha gate must refuse it, and it must
    refuse it BEFORE the parse.
    """
    good = tmp_path / "byte_exact_copy.json"
    good.write_bytes(bp.NULLDIST_PATH.read_bytes())
    control = bp.load_nulldist(good)
    assert control["artifact_id"] == "WP12R-R181-NULLDIST-v1", (
        "CONTROL FAILED: the loader rejects a byte-exact copy, so this row's RED would "
        "prove nothing"
    )

    drifted = tmp_path / "one_byte_added.json"
    drifted.write_bytes(bp.NULLDIST_PATH.read_bytes() + b" ")
    assert json.loads(drifted.read_text()) == control, (
        "the mutation must stay semantically identical, or it is not the weak case"
    )
    with pytest.raises(bp.NullDistArtifactError) as caught:
        bp.load_nulldist(drifted)
    message = str(caught.value)
    assert "sha256 drift" in message
    assert bp.NULLDIST_SHA256 in message, "the message must name the EXPECTED sha"
    assert hashlib.sha256(drifted.read_bytes()).hexdigest() in message, (
        "the message must name the sha it actually SAW, or a drift cannot be diagnosed"
    )
    print(f"LAW-07 PRODUCER sha-drift: control loads; +1 byte (still valid JSON, "
          f"semantically identical) -> {type(caught.value).__name__}: {message}")


def test_loader_raises_on_absent_artifact_and_never_skips(tmp_path) -> None:
    """**LAW-07 producer test for `NullDistArtifactError`, half 2 of 2: absence.**

    The half that matters more. A missing oracle bank that SKIPS is the phantom-gate class
    this repo has been paying for all WP — the tier stays green and the grounds are gone.
    So this row does not merely assert that something is raised: it **catches a skip
    explicitly and converts it into a failure**, which is the only way to machine-check
    *"fails, never skips"* from inside a test.

    **TWO SKIP PATHS EXIST. ONE IS CLOSED HERE; THE OTHER IS STRUCTURALLY OUT OF REACH, and
    saying so is the point of this paragraph — a check whose whole subject is "never skips"
    must not imply it covers skips it cannot see.**

      1. **A skip raised from INSIDE the loader — CLOSED.** Both families are caught:
         `pytest.skip.Exception` (`_pytest.outcomes.Skipped`) **and** `unittest.SkipTest`,
         which is a DIFFERENT class that `pytest.skip.Exception` does not cover. Measured
         before the fix: a loader raising `unittest.SkipTest` made this row report
         `1 skipped` **silently**, attributed to `_pytest/unittest.py`. Measured after: it
         reports FAILED. Practically unreachable — nothing in this repo raises
         `unittest.SkipTest` from a plain helper — but a hole in this particular check is
         not the place to rely on "practically".
      2. **A `skip` / `skipif` MARKER placed on this row itself — NOT CLOSED, and it cannot
         be, from here.** Nothing in-test can observe its own non-execution: if this
         function never runs, no assertion inside it runs either. The collected-count gate
         does not close it either — a marked-skip row still COLLECTS, so `2568` is unchanged
         and the count stays at its floor. **Only a reader of the tier's skip list catches
         it.** Measured, not reasoned: with `@pytest.mark.skip` on this row the suite
         reports it skipped and every other row stays green. This is an inherent limit of
         in-test self-checking and it is recorded rather than papered over.
    """
    missing = tmp_path / "not_here" / "measurement_raw_R181_NULLDIST.json"
    assert not missing.exists()
    try:
        bp.load_nulldist(missing)
    except bp.NullDistArtifactError as exc:
        message = str(exc)
        assert "unreadable" in message
        assert str(missing) in message, "the message must name the path it looked for"
        assert isinstance(exc.__cause__, OSError), "the OSError cause must be chained, not eaten"
        print(f"LAW-07 PRODUCER absence: {type(exc).__name__}: {message}")
    except (pytest.skip.Exception, unittest.SkipTest):
        pytest.fail(
            "load_nulldist SKIPPED on an absent artifact. It must FAIL: a skipping loader "
            "leaves the tier green with no grounds behind it (R4/LAW-07)."
        )
    else:
        pytest.fail("load_nulldist RETURNED on an absent artifact instead of raising")

    # The type itself is a real error, not a skip/outcome wearing an error's name. Both skip
    # families are named here for the same reason they are both caught above.
    assert not issubclass(
        bp.NullDistArtifactError,
        (pytest.skip.Exception, pytest.fail.Exception, unittest.SkipTest),
    )


def test_artifact_null_is_exactly_zero_on_every_cuda_pair_ON_THE_RTX_5080() -> None:
    """R181's premise, re-derived from the per-pair columns: the median form is EXACTLY
    0.000000e+00 on 3675/3675 identical-code CUDA pairs **measured on an RTX 5080
    (sm_120, torch 2.11.0+cu128)**.

    **THE DEVICE IS IN THE NAME ON PURPOSE.** A test's name is its most quotable sentence —
    it appears in every run log, every failure report and every evidence table anyone pastes
    without opening the file. This row's earlier name asserted the result *"on every cuda
    pair"*, full stop, which is the exact falsehood R191 was raised to end: on an RTX 4060 the
    same statistic on the same fixture reads **0/15 pairs zero, worst 1.395037e-02**.

    **What this row establishes and what it does NOT.** It establishes that the artifact's
    measurement is faithfully reproduced from its own per-pair columns — the artifact is
    sound and is not retracted. It does **not** establish that the statistic is zero on
    identical code anywhere else, and it is **not** the discriminating base any more: that
    role now belongs to EXACT equality under test-scope determinism, which needs no device
    qualifier. See `test_artifact_null_is_device_specific_not_a_property_of_the_statistic`.
    """
    doc = bp.load_nulldist()
    cols = bp.cuda_pairs(doc, "null", "policy_median_rel")
    total = 0
    for fx in _FIXTURES:
        v = cols[fx]
        total += len(v)
        assert max(v) == 0.0, f"{fx}: identical-code null max is {max(v):.6e}, not zero"
        assert all(x == 0.0 for x in v)
    assert total == 3675
    print(f"GROUNDS policy median-form identical-code null: 0.000000e+00 on {total}/{total} "
          f"CUDA pairs across {len(_FIXTURES)} fixtures")


def test_artifact_repointed_statistic_separates_f1_from_nothing() -> None:
    """The other half of the base: against 3000 genuine F1-vs-HEAD pairs the same statistic
    never reads below 7.489964e-3, and 0.0% of them fall at or below the null maximum."""
    doc = bp.load_nulldist()
    null_c = bp.cuda_pairs(doc, "null", "policy_median_rel")
    alt_c = bp.cuda_pairs(doc, "alt", "policy_median_rel")
    alt_min = min(min(alt_c[fx]) for fx in _FIXTURES)
    for fx in _FIXTURES:
        overlap = sum(1 for v in alt_c[fx] if v <= max(null_c[fx])) / len(alt_c[fx])
        assert overlap == 0.0, f"{fx}: {100 * overlap:.1f}% of F1 pairs sit inside the null"
    every = [v for fx in _FIXTURES for v in null_c[fx] + alt_c[fx]]
    nonzero = [v for v in every if v > 0.0]
    assert min(nonzero) == alt_min
    print(f"GROUNDS F1-vs-HEAD minimum={alt_min:.6e} over 3000 pairs; over all {len(every)} "
          f"measured values the statistic took no value in (0, {alt_min:.6e}) — it is "
          f"quantised, so the gap is structural and not a sampling accident")


def test_artifact_null_is_device_specific_not_a_property_of_the_statistic() -> None:
    """**R191: the pinned artifact's MEASUREMENT stands; its GENERALITY is withdrawn.**

    The artifact is not retracted and nothing in it is disputed — 3675/3675 identical-code
    pairs really did read exactly `0.000000e+00` on an RTX 5080. What was wrong was reading
    that as a property of the STATISTIC. It is a property of that GPU's kernel scheduling.

    **The entailment this row asserts, which is the mechanism:** a median of exactly zero
    means at least half the elements are bit-identical — that is what a median IS. So the
    artifact's own `policy_median_rel` column ENTAILS **>= 50%** element-level bit-identity
    on the 5080. (The artifact stores no element-level fraction; its `bitidentical_*` columns
    are WHOLE-TENSOR flags and read 0.0 on the production fixture — i.e. the tensors always
    differ and the median reads zero anyway. The >= 50% figure is derived here from the
    median column, not read from a stored one, and this row says so rather than citing a
    column that does not exist.)

    **Measured on an RTX 4060 Laptop (sm_89, torch 2.11.0+cu130), same fixture, same code,
    nondeterministic — the second GPU this statistic ever met:**

        mean in-degree  8: 0/15 pairs zero, min non-zero 1.298431e-02, elem bit-ident 13.5%
        mean in-degree 15: 0/15 pairs zero, min non-zero 8.848618e-03, elem bit-ident 16.2%
        mean in-degree 27: 0/15 pairs zero, min non-zero 7.489964e-03, elem bit-ident 19.1%

    12.0%-19.1% against the 5080's >= 50%: the two devices sit either side of the median, so
    the same statistic reads exactly zero on one and never zero on the other. **That is why
    no envelope calibrated on one device may be asserted on another, and why per-device
    calibration is REJECTED rather than attempted (R191).**
    """
    doc = bp.load_nulldist()
    med = bp.cuda_pairs(doc, "null", "policy_median_rel")[_PROD]
    frac = doc["cuda"]["fixtures"][_PROD]["null_bitidentical_frac_treat"]
    assert max(med) == 0.0
    assert frac["bitidentical_policy"] == 0.0, (
        "the artifact's whole-tensor flag is expected to be 0.0 on the production fixture — "
        "the tensors always differ and the median reads zero anyway, which is the whole point"
    )
    assert doc["provenance"]["device_name"] == "NVIDIA GeForce RTX 5080"
    assert doc["provenance"]["device_capability"].startswith("sm_120")
    print(f"DEVICE-SPECIFIC the pinned null was measured on "
          f"{doc['provenance']['device_name']} / {doc['provenance']['device_capability']} / "
          f"torch {doc['provenance']['torch']}; its median-zero entails >=50% element "
          f"bit-identity THERE. RTX 4060 / cu130 measures 12.0%-19.1% and 0/15 zeros. "
          f"The measurement stands; its generality does not.")


def test_detection_floor_is_derived_and_its_null_role_is_withdrawn(runs) -> None:
    """**THE DERIVATION, RE-POINTED BY R191.** One constraint withdrawn, two re-verified on
    a second GPU. The constant no longer bounds a null on any device.

    Constraint 1 — ABOVE the measured null. **WITHDRAWN AS DEVICE-SPECIFIC.** It read
    "identical-code CUDA null max = 0.000000e+00, margin infinite". True on the 5080, false
    on the 4060 (0/15 zeros, worst 1.395037e-02). It was never binding — a multiplicative
    margin over zero is vacuous — so withdrawing it changes no number, only what may be
    claimed. **The CUDA null is now asserted EXACT under test-scope determinism instead.**

    Constraint 2 — BELOW the quantisation floor (**BINDING**). The statistic is the median of
    `|X-Y| / (|a| + 1e-3)` over bf16-valued tensors, so a non-zero value requires the median
    element to differ by at least one bf16 ulp at its own magnitude; its smallest attainable
    non-zero value is `ulp(|a|_p50) / (|a|_p50 + 1e-3)`, with `|a|_p50` DERIVED from the live
    fp32 arm and `ulp` MEASURED from torch. **This is a property of bf16, not of a GPU — and
    I did not assume that.** Re-measured on the RTX 4060 at three mean in-degrees, the
    smallest non-zero reading was 1.298e-2 / 8.849e-3 / 7.490e-3, all above the 3.87e-3 floor.
    **C2 SURVIVES on a second device.**

    Constraint 3 — BELOW the measured alternative's minimum, 7.489964e-3 over 3000
    F1-vs-HEAD pairs on the 5080. Corroborated on the 4060: at production in-degree its
    smallest non-zero reading is **7.489964e-3, the same value to every digit** — which is
    what a quantised statistic should do and is strong independent support for C2.

    Stated margin: **3**, applied to the binding constraint, NOT invented here —
    MEASUREMENT_NULLDIST §7.2 registered it from its own tail ratios (worst max/p99 = 1.422,
    worst max/p50 = 7.99). Rounded DOWN to the decade, which tightens rather than loosens.
    """
    _net, _batch, _bf16, fp32 = runs
    doc = bp.load_nulldist()

    abs_p50 = float(fp32.policy_logits.abs().median())
    floors = {"synth8": _bf16_ulp(abs_p50) / (abs_p50 + 1e-3)}
    for fx, p50 in _BOX_ABS_P50.items():
        floors[fx] = _bf16_ulp(p50) / (p50 + 1e-3)
    binding = min(floors.values())
    binding_fx = min(floors, key=lambda k: floors[k])

    null_max = max(max(v) for v in bp.cuda_pairs(doc, "null", "policy_median_rel").values())
    alt_min = min(min(v) for v in bp.cuda_pairs(doc, "alt", "policy_median_rel").values())

    print(f"DERIVATION |a|_p50(synth8, derived live)={abs_p50:.6f} "
          f"ulp={_bf16_ulp(abs_p50):.9e} floors={ {k: f'{v:.4e}' for k, v in floors.items()} }")
    print(f"DERIVATION binding constraint = quantisation floor on {binding_fx} = {binding:.6e}; "
          f"margin {_DERIVATION_MARGIN:g} => {binding / _DERIVATION_MARGIN:.6e}; "
          f"rounded down to the decade => detection floor {_DETECTION_FLOOR:.3e}")
    print(f"DERIVATION C1 (above the null max, {null_max:.3e} on the 5080) is WITHDRAWN as "
          f"device-specific — the CUDA null is asserted EXACT under determinism instead")
    print(f"DERIVATION realised margins: vs quantisation floor = "
          f"{binding / _DETECTION_FLOOR:.2f}x; "
          f"vs F1 minimum ({alt_min:.6e}) = {alt_min / _DETECTION_FLOOR:.2f}x")

    # C1 is NOT asserted as a bound any more. The artifact's value is still read, so that a
    # reader sees exactly what was withdrawn and can check it is the 5080 number.
    assert null_max == 0.0, "the artifact's own 5080 null; withdrawn as a BOUND, not as a fact"
    assert _DETECTION_FLOOR <= binding / _DERIVATION_MARGIN, (
        f"detection floor {_DETECTION_FLOOR:.3e} no longer clears the quantisation floor "
        f"{binding:.6e} at margin {_DERIVATION_MARGIN:g}"
    )
    assert _DETECTION_FLOOR <= alt_min / _DERIVATION_MARGIN, (
        f"detection floor {_DETECTION_FLOOR:.3e} no longer clears F1's own minimum "
        f"{alt_min:.6e} at margin {_DERIVATION_MARGIN:g}"
    )


def test_measured_bf16_spacing_corrects_the_reports_quantisation_arithmetic() -> None:
    """**DISAGREEMENT WITH THE CALIBRATION'S ARITHMETIC, recorded rather than absorbed.**

    MEASUREMENT_NULLDIST §4.2 states one bf16 ulp as 2.441406e-4 at |a|_p50 = 0.1128 and
    4.882812e-4 at 0.2050/0.2102; MEASUREMENT_BF1 §6.5 states it as 9.766e-4 on [0.25, 0.5)
    with the working `2^-2 * 2^-8`. All three are HALF the true bf16 spacing: bfloat16 has
    8 significand bits, so for x in [2^e, 2^(e+1)) the spacing is 2^(e-7), not 2^(e-8).

    **The numbers the reports reach are not affected and both errors run conservative.**
    The true quantisation floors are 2x LARGER than stated (the empty gap around zero is
    twice as wide, so the 1.0e-3 detection floor is twice as safe), and the max form's band is
    ~1/13 of a ulp rather than ~1/6 (a stronger indictment, not a weaker one). The
    disagreement is on the arithmetic; the dispositions stand.

    **`reported` below deliberately hard-codes the WRONG published figures.** It pins the
    disagreement, not the truth: the day `MEASUREMENT_NULLDIST` §4.2 / `MEASUREMENT_BF1` §6.5
    are corrected in the workspace, this test REDs — and that RED means *"the reports have
    been fixed, retire this row"*, not *"torch changed"*. The two exact pins at the bottom
    (`measured[...] == ...`) are what would catch an actual torch/`_bf16_ulp` regression.
    """
    measured = {x: _bf16_ulp(x) for x in (0.1128, 0.2050, 0.2102, 0.30)}
    reported = {0.1128: 2.441406e-4, 0.2050: 4.882812e-4, 0.2102: 4.882812e-4, 0.30: 9.766e-4}
    for x, ulp in measured.items():
        print(f"BF16-ULP x={x} measured={ulp:.9e} reported={reported[x]:.6e} "
              f"ratio={ulp / reported[x]:.4f}")
        assert ulp / reported[x] == pytest.approx(2.0, rel=1e-3), (
            f"the 2x correction no longer holds at x={x}: measured {ulp:.9e}"
        )
    assert measured[0.1128] == 4.8828125e-4
    assert measured[0.30] == 1.953125e-3


def test_artifact_four_statistics_fail_discrimination_and_are_declassified() -> None:
    """**THE LOUD NEGATIVE.** Four of the five statistics the calibration was asked to
    calibrate cannot tell a genuine F1-vs-HEAD change from a comparison of a commit against
    itself, on the production-shaped fixture. This test asserts that failure so it cannot be
    quietly forgotten, and names each row's disposition.

    Disposition, uniform and stated: each row KEEPS its registered band as a BOUND on the
    bf16 regime's drift — its null sits orders of magnitude below its band, so the bound is
    resolvable — and LOSES any claim to witness F1. `test_gine_bf16_drift.py` carries
    `role=BOUND ONLY` on each of them. Nothing was dropped and no band was moved; what was
    removed is a claim, and the coverage reduction is stated in both files' docstrings and
    in PREREG_DFIX §1.
    """
    doc = bp.load_nulldist()
    expected = {
        "bin_median_rel": 0.851, "policy_loss_rel": 0.397,
        "value_loss_rel": 0.879, "grad_norm_rel": 0.621,
    }
    for stat, want in expected.items():
        null_v = bp.cuda_pairs(doc, "null", stat)[_PROD]
        alt_v = bp.cuda_pairs(doc, "alt", stat)[_PROD]
        overlap = sum(1 for v in alt_v if v <= max(null_v)) / len(alt_v)
        print(f"DECLASSIFIED {stat} on {_PROD}: {100 * overlap:.1f}% of {len(alt_v)} genuine "
              f"F1-vs-HEAD pairs sit at or below the identical-code null max "
              f"({max(null_v):.6e}) — BOUND ONLY, cannot witness F1")
        assert overlap == pytest.approx(want, abs=5e-4), f"{stat}: overlap {overlap}"
        assert overlap > 0.0, (
            f"{stat}: overlap is now zero — the declassification's ground has vanished and "
            f"the row may be a witness again. Re-adjudicate; do not re-promote silently."
        )
    # The one row that survives, asserted beside the four that do not.
    surv = bp.cuda_pairs(doc, "alt", "policy_median_rel")[_PROD]
    assert all(v > 0.0 for v in surv)
    print("SURVIVING WITNESS on the production fixture: policy_median_rel (0.0% overlap). "
          "1 of 10 registered statistics. That is the coverage after the R181 re-point.")


# ── R181's two-sided mutation condition — BOTH halves ────────────────────────────────
def test_mutation_green_cpu_exact_null_over_n_fresh_head_vs_head_pairs(runs) -> None:
    """**MUTATION CONDITION, HALF 1 of 2: GREENS on N fresh HEAD-vs-HEAD pairs.**

    REGIME: DETERMINISTIC PATH (CPU). Asserts **EQUALITY OUTRIGHT** — `torch.equal` on all
    three tensors and exact `==` on both loss scalars — which R181 licenses where exactness
    matters and which is strictly stronger than any bound. Grounds: 1140/1140 measured CPU
    null pairs bit-identical (pinned artifact, `cpu.*.null_bitidentical_frac_treat`).

    **N = 15 pairs from 6 fresh runs.** Justification, stated rather than assumed:
      * The CPU claim is BINARY (bit-identity), not a tail quantile, so no order statistic
        is needed — MEASUREMENT_NULLDIST §2's own reason for using fewer CPU runs.
      * 15 pairs cost ~3 s in the default tier; the CUDA arm that needed n = 1225 is not
        runnable here at any N.
      * Power, stated exactly: any source of non-determinism firing with per-pair
        probability p is missed with probability (1-p)^15, so 15 pairs give >= 95% power
        against any p >= 18.1%. Below that they are underpowered, and with
        `torch.set_num_threads(1)` a CPU non-determinism would be systematic rather than
        rare — which is what 15 pairs are sized against.
      * DISCLOSED LIMIT: these 15 pairs are WITHIN-PROCESS. Cross-process pairing is
        covered only by the box artifact (MEASUREMENT_NULLDIST §3.4 measured the two
        sub-distributions indistinguishable, cross-process maxima <= 12% larger).

    **This green cannot witness F1** and is not offered as one: CPU F1-vs-HEAD is likewise
    1200/1200 bit-identical. It proves the instrument has a zero false-positive rate here.
    """
    _net, _batch, bf16, _fp32 = runs
    pairs = 0
    for i in range(_N_NULL_RUNS):
        for j in range(i + 1, _N_NULL_RUNS):
            a, b = bf16[i], bf16[j]
            assert torch.equal(a.policy_logits, b.policy_logits), f"pair ({i},{j}) policy"
            assert torch.equal(a.bin_logits, b.bin_logits), f"pair ({i},{j}) bin"
            assert torch.equal(a.grads, b.grads), f"pair ({i},{j}) grads"
            assert a.policy_loss == b.policy_loss and a.value_loss == b.value_loss
            assert bp.median_form(a.policy_logits, b.policy_logits) == 0.0
            pairs += 1
    assert pairs == _N_NULL_PAIRS
    print(f"MUTATION-GREEN cpu-exact: {pairs}/{pairs} fresh HEAD-vs-HEAD pairs bit-identical "
          f"on policy_logits, bin_logits and grads, and exactly equal on both losses "
          f"(N = {_N_NULL_RUNS} runs, all-pairs). Regime label: EQUALITY ASSERTED OUTRIGHT.")


def _cuda_batch() -> tuple[object, bp.Batch]:
    net = bp.build_net().cuda()
    batch = bp.build_batch(bp.build_arch())
    dev = bp.Batch(**{k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                      for k, v in vars(batch).items()})
    return net, dev


def test_test_scope_determinism_does_not_leak_to_sibling_tests() -> None:
    """Producer for `deterministic_algorithms()`'s RESTORE (LAW-07).

    `torch.use_deterministic_algorithms` is PROCESS-GLOBAL. A leak would silently change the
    numerics of every test that ran after the CUDA legs — an ordering-dependent failure that
    presents as a flaky oracle, the hardest kind to diagnose. This row runs on CPU, so unlike
    the two CUDA legs below **it actually executes in CI**, and it is the only part of the
    determinism machinery that does.

    It pins RESTORE, not merely clear: entered from an already-enabled ambient it must leave
    the mode enabled. A `finally: use_deterministic_algorithms(False)` would pass a
    naive check and fail this one.
    """
    ambient = torch.are_deterministic_algorithms_enabled()
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    with bp.deterministic_algorithms():
        assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.are_deterministic_algorithms_enabled() == ambient, (
        "deterministic_algorithms() LEAKED the mode into the enclosing test session"
    )
    assert ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == had_cublas, (
        "deterministic_algorithms() leaked CUBLAS_WORKSPACE_CONFIG"
    )
    torch.use_deterministic_algorithms(True)
    try:
        with bp.deterministic_algorithms():
            pass
        assert torch.are_deterministic_algorithms_enabled() is True, (
            "the context CLEARS instead of RESTORING — an enabled ambient was lost"
        )
    finally:
        torch.use_deterministic_algorithms(ambient)
    assert torch.are_deterministic_algorithms_enabled() == ambient
    print("DETERMINISM-CONTEXT restore verified in both directions; no leak")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LOUD SKIP — the CUDA parity legs need a GPU; the CI venv is "
                           "torch 2.11.0+cpu. NOT verified by CI; measured out-of-band on an "
                           "RTX 4060 / torch 2.11.0+cu130 — IMPL_NOTES_R181_REPOINT §7C.")
def test_mutation_green_cuda_exact_null_under_TEST_SCOPE_determinism() -> None:
    """**MUTATION CONDITION, HALF 1, CUDA leg. REGIME: EXACT EQUALITY** (R191).

    **The name says TEST_SCOPE because determinism is an instrument, not the production
    regime.** run5 trains with the fast nondeterministic `index_add_` — that is the whole
    reason F1 exists — and nothing in `src/mantis/` enables determinism.

    **WHY THIS REPLACED A CALIBRATED BOUND — a defect this file used to have.** The previous
    version asserted `median_form <= 1.0e-3`, an envelope derived from the RTX 5080's
    identical-code null of exactly 0.000000e+00 on 3675/3675 pairs. **That null is a property
    of THAT GPU's kernels, not of the statistic.** Measured on an RTX 4060 (sm_89,
    torch 2.11.0+cu130), same fixture, nondeterministic:

        0/15 pairs zero; worst median form 1.395037e-02
        = 14.0x the envelope, and ABOVE F1's own maximum measured effect (1.365076e-02)

    The shipped leg would have failed on the second GPU it ever met, and its envelope could
    not separate the null from the defect it exists to witness. Element-level bit-identity is
    the mechanism: a median of exactly zero ENTAILS >= 50% of elements identical (that is
    what a median is); the 4060 measures 12.0%-19.1% across mean in-degrees 8 / 15 / 27.

    **Under determinism the same net on the same GPU is BIT-IDENTICAL** — measured 15/15
    `torch.equal`, element bit-identity 1.0000 — so this leg asserts equality outright, which
    is strictly stronger than any bound and needs no per-device calibration.

    **PER-DEVICE ENVELOPE CALIBRATION IS REJECTED (R191):** a treadmill of one constant per
    (GPU x driver x torch), each one a number nobody re-measures. There is deliberately no
    hook for it here.

    If determinism ever REJECTS an op on this path, torch raises and this leg FAILS loudly.
    It does not skip and must not be converted into one. (Measured on the 4060: it does not
    reject `index_add_` — the recorded observation PREREG_DFIX's OF2-11 asks for.)
    """
    with bp.deterministic_algorithms():
        net, dev = _cuda_batch()
        arms = [bp.run_arm(net, dev, autocast_enabled=True, device="cuda")
                for _ in range(_N_NULL_RUNS)]
    pairs = 0
    for i in range(_N_NULL_RUNS):
        for j in range(i + 1, _N_NULL_RUNS):
            a, b = arms[i], arms[j]
            assert torch.equal(a.policy_logits, b.policy_logits), f"pair ({i},{j}) policy"
            assert torch.equal(a.bin_logits, b.bin_logits), f"pair ({i},{j}) bin"
            assert torch.equal(a.grads, b.grads), f"pair ({i},{j}) grads"
            assert a.policy_loss == b.policy_loss and a.value_loss == b.value_loss
            pairs += 1
    assert pairs == _N_NULL_PAIRS
    print(f"MUTATION-GREEN cuda-exact (TEST-SCOPE determinism): {pairs}/{pairs} fresh "
          f"HEAD-vs-HEAD pairs BIT-IDENTICAL on all three tensors and both scalars. "
          f"Regime label: EQUALITY ASSERTED OUTRIGHT, no envelope.")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LOUD SKIP — the CUDA parity legs need a GPU; the CI venv is "
                           "torch 2.11.0+cpu. NOT verified by CI; measured out-of-band on an "
                           "RTX 4060 / torch 2.11.0+cu130 — IMPL_NOTES_R181_REPOINT §7C.")
def test_mutation_red_cuda_injected_change_under_TEST_SCOPE_determinism() -> None:
    """**MUTATION CONDITION, HALF 2, CUDA leg.** The GREEN half alone proves nothing: under
    determinism a statistic that always read zero would pass it too. This half shows the leg
    can still SEE a real numerics change once the scheduler noise is removed.

    Same injection as the CPU RED half — the autocast dtype forced `bfloat16 -> float16`, a
    LAW-06 regime violation executed through the production forward. Measured on the RTX 4060
    under determinism: **not `torch.equal`; median form 1.258558e-02**, 12.6x the detection
    floor and inside F1's own measured range.

    Together the two CUDA halves bracket the leg: it cannot pass by being blind (this row)
    and it cannot pass by being noisy (the exact-equality row above).
    """
    with bp.deterministic_algorithms():
        net, dev = _cuda_batch()
        base = bp.run_arm(net, dev, autocast_enabled=True, device="cuda")
        inj = bp.run_arm(net, dev, autocast_enabled=True, dtype=torch.float16, device="cuda")
    assert not torch.equal(inj.policy_logits, base.policy_logits), (
        "the injected fp16 regime produced bit-identical policy logits on CUDA — the "
        "injection is not a numerics change on this build and the CUDA RED half is VOID"
    )
    value = bp.median_form(inj.policy_logits, base.policy_logits)
    print(f"MUTATION-RED cuda (TEST-SCOPE determinism): injected fp16 regime -> median form="
          f"{value:.6e} = {value / _DETECTION_FLOOR:.1f}x the detection floor "
          f"{_DETECTION_FLOOR:.3e}")
    assert value > _DETECTION_FLOOR, (
        f"THE STATISTIC IS BLIND TO AN INJECTED REAL NUMERICS CHANGE ON CUDA "
        f"({value:.6e} <= {_DETECTION_FLOOR:.3e}). The re-point is not licensed on this device."
    )


def test_mutation_red_injected_numerics_change_reds_the_repointed_statistic(runs) -> None:
    """**MUTATION CONDITION, HALF 2 of 2: REDS under an injected REAL numerics change.**

    A statistic satisfying only the GREEN half is the same failure in a new direction; the
    retired max-form row reds on a real change perfectly well and was still useless.

    THE INJECTION, and why it is real rather than a data tweak: the autocast dtype of the
    treatment arm is changed from `bfloat16` to `float16`. That is a violation of LAW-06's
    pinned graph-path dtype executed through the *production* forward — the same code, the
    same tensors, one regime constant different. It is not a perturbation applied to the
    outputs.

    ITS SIZE, stated so it is not mistaken for an easy target: it is NOT tuned. Measured, it
    lands at ~1.21e-2 on the policy median form — inside F1's own measured CUDA range
    [7.4900e-3, 1.3651e-2], re-read from the pinned artifact here. So the RED half is
    exercised at the effect size R181 actually cares about, not at an exaggerated one.

    Both regimes are asserted, each labelled:
      * DETERMINISTIC (CPU): the injected arm is NOT `torch.equal` to the base arm —
        equality asserted outright, and it fails, which is the point.
      * DETECTION FLOOR (not a calibrated bound — the label was stale after R191 renamed
        the constant): the median form exceeds `_DETECTION_FLOOR`. This is where the floor's
        numeric value gets exercised on real tensors in an environment with no GPU. It does
        NOT substitute for the CUDA legs above, which assert EXACT equality.
    """
    net, batch, bf16, _fp32 = runs
    base = bf16[0]
    inj = bp.run_arm(net, batch, autocast_enabled=True, dtype=torch.float16)
    doc = bp.load_nulldist()
    alt = [v for fx in _FIXTURES for v in bp.cuda_pairs(doc, "alt", "policy_median_rel")[fx]]

    assert not torch.equal(inj.policy_logits, base.policy_logits), (
        "the injected fp16 regime produced bit-identical policy logits — the injection is "
        "not a numerics change on this build and the RED half is VOID"
    )
    value = bp.median_form(inj.policy_logits, base.policy_logits)
    bin_value = bp.median_form(inj.bin_logits, base.bin_logits)
    print(f"MUTATION-RED injected fp16 regime (LAW-06 violation): policy median form="
          f"{value:.6e} = {value / _DETECTION_FLOOR:.1f}x the detection floor "
          f"{_DETECTION_FLOOR:.3e}; bin median form={bin_value:.6e}; "
          f"F1's own measured CUDA range=[{min(alt):.6e}, {max(alt):.6e}]")
    assert value > _DETECTION_FLOOR, (
        f"THE RE-POINTED STATISTIC IS BLIND TO AN INJECTED REAL NUMERICS CHANGE "
        f"({value:.6e} <= {_DETECTION_FLOOR:.3e}). R181's two-sided condition FAILS and "
        f"the re-point is not licensed."
    )
    assert min(alt) <= value <= max(alt), (
        f"the injection has drifted outside F1's own measured effect range "
        f"[{min(alt):.6e}, {max(alt):.6e}] and no longer probes the size that matters"
    )


def test_repointed_statistic_is_blind_to_a_minority_element_change(runs) -> None:
    """**THE RE-POINTED ORACLE'S OWN BLIND SPOT, asserted so it survives into the record.**

    MEASUREMENT_NULLDIST §9 declared this UNVERIFIED: *"How large a change confined to a
    minority of elements it would miss is NOT measured."* It is measured here, and the
    answer is exact rather than approximate, because it is a property of the median and not
    of the fixture:

      **the median form reads exactly 0.0 for ANY change confined to at most 50% of the
      elements, at ANY magnitude.**

    The re-pointed oracle saw F1 because F1 perturbs essentially every element. It would
    not see a change confined to a minority of them, and no envelope derived from the null
    distribution fixes that. This test is the producer for that sentence (LAW-07).
    """
    _net, _batch, bf16, _fp32 = runs
    base = bf16[0].policy_logits
    n = base.numel()
    for frac, expect_zero in ((0.49, True), (0.51, False)):
        k = int(frac * n)
        mutated = base.clone()
        mutated.reshape(-1)[:k] += 1.0e3
        value = bp.median_form(mutated, base)
        print(f"BLIND-SPOT {100 * frac:.0f}% of {n} elements perturbed by 1e3: "
              f"median form reads {value:.6e}")
        if expect_zero:
            assert value == 0.0, "the blind spot has moved — re-measure it, do not assume it"
        else:
            assert value > _DETECTION_FLOOR
