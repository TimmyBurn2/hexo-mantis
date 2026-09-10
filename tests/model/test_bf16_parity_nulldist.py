"""The re-pointed F1 parity oracle's GROUNDS: null calibration + mutation.

>300 justify (R8): one artifact carries four inseparable things — the pinned null distribution,
the detection floor derived from it, and BOTH halves of the two-sided mutation condition.
Splitting them would put a derived constant in a different file from the distribution it comes
from.

Regime labels are in every test name. `..._cpu_exact_...` asserts EQUALITY outright on the
deterministic path; `..._cuda_..._TEST_SCOPE_determinism` asserts exact equality under
`torch.use_deterministic_algorithms(True)`, which is TEST SCOPE ONLY (production keeps its
nondeterministic kernels) and skips loudly without a GPU; `..._artifact_...` reads the pinned
measurement and is device-independent. That artifact is DEVICE-SPECIFIC — its measurement
stands, its generality does not — and per-device envelope calibration is rejected. The statistic
is a MAJORITY-BIT-IDENTITY test: it reads zero for any change leaving more than half the bf16
elements unchanged, at any magnitude.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest

import _bf16_parity as bp
import pytest
import torch

# The DETECTION FLOOR: the value below which a reading carries no information. It is NOT a null
# bound on any device — both null legs assert EXACT equality — and its derivation is re-executed
# live in `test_detection_floor_is_derived_and_its_null_role_is_withdrawn`. Margin 3 is not
# invented here: it comes from the measurement's own tail ratios (worst max/p99 = 1.422, worst
# max/p50 = 7.99), rounded down to the decade.
_DETECTION_FLOOR = 1.0e-3
_DERIVATION_MARGIN = 3.0

# N for "GREENS on N fresh HEAD-vs-HEAD pairs": 6 fresh full forward+backward runs => 15
# unordered pairs, the same all-pairs construction the box measurement used.
_N_NULL_RUNS = 6
_N_NULL_PAIRS = _N_NULL_RUNS * (_N_NULL_RUNS - 1) // 2

_EXPECTED_COMMIT = "982da03bae57758efc65c6cfe0d451d77f15561c"
_FIXTURES = ("synth8", "prod27_samesizes", "prod27_run5shape")
_PROD = "prod27_run5shape"

# `|a|` p50 of the fp32 reference policy logits, per fixture. The `synth8` value is DERIVED
# LOCALLY below from the real fp32 arm and is the BINDING one (smallest floor); the other two are
# transcribed box quantities, recorded only so the worst-case selection is visible.
_BOX_ABS_P50 = {"prod27_samesizes": 0.2050, "prod27_run5shape": 0.2102}


def _bf16_ulp(x: float) -> float:
    """One bf16 spacing at `x`, measured from torch rather than computed from a binade formula:
    the reports state 2^(e-8), while the true bf16 spacing is 2^(e-7)."""
    a = torch.tensor([x], dtype=torch.bfloat16)
    nxt = (a.view(torch.int16) + 1).view(torch.bfloat16)
    return float(nxt) - float(a)


@pytest.fixture(scope="module")
def runs() -> tuple[object, bp.Batch, list[bp.Arm], bp.Arm]:
    """`_N_NULL_RUNS` fresh bf16-arm runs of IDENTICAL CODE, plus one fp32 reference arm.

    Every run is a full 4-layer forward + loss + backward; `torch.autograd.grad` is used
    throughout, so no run can reach another through `.grad`.
    """
    net = bp.build_net()
    batch = bp.build_batch(bp.build_arch())
    bf16 = [bp.run_arm(net, batch, autocast_enabled=True) for _ in range(_N_NULL_RUNS)]
    fp32 = bp.run_arm(net, batch, autocast_enabled=False)
    return net, batch, bf16, fp32


def test_artifact_pinned_nulldist_identity_and_shape() -> None:
    """The oracle's cited grounds are present, sha-pinned, and the shape they claim —
    `bp.load_nulldist()` RAISES on absence or sha drift and never skips."""
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
    """`NullDistArtifactError` fires on sha drift, with a control that loads first so the row
    cannot pass by raising on everything. The mutation is the weakest one available — one
    trailing whitespace byte — so the file still parses to a semantically identical document and
    the sha gate must refuse it BEFORE the parse."""
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
    """`NullDistArtifactError` fires on ABSENCE, and a skip is caught explicitly and converted
    into a failure, since a missing oracle bank that skips leaves the tier green with no grounds.

    Both skip families are caught (`pytest.skip.Exception` and `unittest.SkipTest`, a different
    class). A `skip`/`skipif` MARKER on this row itself is NOT closed and cannot be from here:
    nothing in-test observes its own non-execution, and a marked-skip row still collects."""
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

    # The type itself is a real error, not a skip wearing an error's name; both skip families are
    # named here for the same reason they are both caught above.
    assert not issubclass(
        bp.NullDistArtifactError,
        (pytest.skip.Exception, pytest.fail.Exception, unittest.SkipTest),
    )


def test_artifact_null_is_exactly_zero_on_every_cuda_pair_ON_THE_RTX_5080() -> None:
    """The median form reads exactly 0.000000e+00 on 3675/3675 identical-code CUDA pairs measured
    on an RTX 5080 (sm_120, torch 2.11.0+cu128).

    The device is in the name on purpose: on an RTX 4060 the same statistic on the same fixture
    reads 0/15 pairs zero, worst 1.395037e-02. This row shows the artifact reproduces its own
    per-pair columns; it is no longer the discriminating base."""
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
    """The pinned artifact's MEASUREMENT stands; its GENERALITY is withdrawn.

    A median of exactly zero means at least half the elements are bit-identical, so the
    artifact's `policy_median_rel` column entails >= 50% element-level bit-identity on the 5080
    (its stored `bitidentical_*` columns are whole-tensor flags and read 0.0, so the figure is
    derived from the median column rather than read off a stored one). Measured on an RTX 4060
    Laptop (sm_89, torch 2.11.0+cu130), same fixture, same code, nondeterministic:

        mean in-degree  8: 0/15 pairs zero, min non-zero 1.298431e-02, elem bit-ident 13.5%
        mean in-degree 15: 0/15 pairs zero, min non-zero 8.848618e-03, elem bit-ident 16.2%
        mean in-degree 27: 0/15 pairs zero, min non-zero 7.489964e-03, elem bit-ident 19.1%

    The two devices sit either side of the median, which is why no envelope calibrated on one
    device may be asserted on another and why per-device calibration is rejected.
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
    """The detection floor's derivation: one constraint withdrawn as device-specific, two
    re-verified on a second GPU. The constant no longer bounds a null on any device.

    C1 (above the measured null) is WITHDRAWN: true on the 5080, false on the 4060 (0/15 zeros,
    worst 1.395037e-02), and never binding, since a multiplicative margin over zero is vacuous.
    C2 (below the quantisation floor) BINDS — the smallest attainable non-zero value is
    `ulp(|a|_p50) / (|a|_p50 + 1e-3)`, and on the 4060 the smallest non-zero readings were
    1.298e-2 / 8.849e-3 / 7.490e-3, all above the 3.87e-3 floor. C3 (below the alternative's
    minimum, 7.489964e-3 over 3000 F1-vs-HEAD pairs) is corroborated on the 4060 to every digit.
    Stated margin 3, from the measurement's own tail ratios."""
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

    # C1 is NOT asserted as a bound any more; the artifact's value is still read so a reader sees
    # exactly what was withdrawn and can check it is the 5080 number.
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
    """The reports' bf16 ulp arithmetic is HALF the true spacing: bfloat16 has 8 significand bits,
    so for x in [2^e, 2^(e+1)) the spacing is 2^(e-7), not 2^(e-8).

    Both errors run conservative, so the reports' dispositions stand and only the arithmetic is
    disputed. `reported` below deliberately hard-codes the WRONG published figures: a RED here
    means the reports were corrected and this row retires, not that torch changed."""
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
    """THE LOUD NEGATIVE: four of the five calibrated statistics cannot tell a genuine F1-vs-HEAD
    change from a commit compared against itself on the production-shaped fixture. Each row KEEPS
    its registered band as a BOUND on the bf16 regime's drift and LOSES any claim to witness F1;
    no band was moved, what was removed is a claim."""
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


def test_mutation_green_cpu_exact_null_over_n_fresh_head_vs_head_pairs(runs) -> None:
    """MUTATION CONDITION, HALF 1 of 2: GREENS on N fresh HEAD-vs-HEAD pairs.

    DETERMINISTIC PATH (CPU), asserting EQUALITY OUTRIGHT on grounds of 1140/1140 measured CPU
    null pairs bit-identical. N = 15 pairs from 6 fresh runs, because the CPU claim is BINARY:
    non-determinism firing with per-pair probability p is missed with probability (1-p)^15, so 15
    pairs give >= 95% power against any p >= 18.1%. DISCLOSED LIMIT: these pairs are
    WITHIN-PROCESS. This green cannot witness F1 — CPU F1-vs-HEAD is 1200/1200 bit-identical."""
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
    """Producer for `deterministic_algorithms()`'s RESTORE.

    `torch.use_deterministic_algorithms` is PROCESS-GLOBAL, so a leak would change the numerics of
    every test after the CUDA legs. This row runs on CPU, so unlike them it executes in CI, and it
    pins RESTORE rather than clear: entered from an enabled ambient it must leave the mode on."""
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
    """MUTATION CONDITION, HALF 1, CUDA leg. REGIME: EXACT EQUALITY.

    TEST_SCOPE is in the name because determinism is an instrument, not the production regime:
    nothing in `src/mantis/` enables it. It replaced a calibrated bound (`median_form <= 1.0e-3`)
    derived from the RTX 5080's identical-code null of exactly zero — a property of that GPU's
    kernels. On an RTX 4060 (sm_89, torch 2.11.0+cu130), nondeterministic: 0/15 pairs zero, worst
    median form 1.395037e-02, 14.0x that envelope and above F1's own maximum effect
    (1.365076e-02). Under determinism the same net on the same GPU is BIT-IDENTICAL (15/15
    `torch.equal`). If determinism ever REJECTS an op here, torch raises and this leg FAILS
    loudly; it does not skip."""
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
    """MUTATION CONDITION, HALF 2, CUDA leg: the leg can still SEE a real numerics change once the
    scheduler noise is removed, which the green half alone cannot show.

    Same injection as the CPU RED half — the autocast dtype forced `bfloat16 -> float16` through
    the production forward. On the RTX 4060 under determinism: not `torch.equal`, median form
    1.258558e-02, 12.6x the detection floor."""
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
    """MUTATION CONDITION, HALF 2 of 2: REDS under an injected REAL numerics change.

    The injection is the treatment arm's autocast dtype changed from `bfloat16` to `float16` — a
    violation of the pinned graph-path dtype executed through the PRODUCTION forward, not a
    perturbation of the outputs. It is NOT tuned: it lands at ~1.21e-2 on the policy median form,
    inside F1's own measured CUDA range [7.4900e-3, 1.3651e-2]. Both regimes are asserted, and the
    `_DETECTION_FLOOR` limb is where the floor's value gets exercised without a GPU."""
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
    """The re-pointed oracle's own BLIND SPOT: the median form reads exactly 0.0 for ANY change
    confined to at most 50% of the elements, at ANY magnitude. It saw F1 because F1 perturbs
    essentially every element, and no envelope derived from the null distribution fixes that."""
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
