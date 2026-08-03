"""WP12-R D-FIX phase F1 (OF1-3, OF1-4) — bf16 numeric drift against the fp32 reference.

**R181 RE-POINT.** Every gating reduction in this file is the MEDIAN form. The MAX form
(OF1-3b, OF1-3c-max) is RETIRED from gating and recorded only, on measured grounds that are
re-derived from the pinned artifact in `test_retired_max_form_rows_are_recorded_not_gating`.
R181 verbatim: *"a band is never moved on a statistic that cannot distinguish its subject
from nothing"* — the max form's identical-code null on CUDA reaches 9.3367e-1, 2.33x its own
4.0e-1 ABORT threshold, so its verdict is produced by `index_add_` atomics, not by the change.
**No band, envelope or threshold in this file was moved** (R61); the max rows' bands are
recorded as unreachable rather than widened, which is what R181 required.

R8 justification: 300+ lines. Two arms of a real 4-layer GNN forward+loss+backward and eight
pre-registered statistics; the fixture and the arms now live in `_bf16_parity.py` (shared
with the null-calibration file) and the envelopes stay here beside the rows they bound.

**REGIME LABEL — read before reading any row (R181's labelling clause).** Every row in this
file runs on the DETERMINISTIC (CPU) path. Two consequences, both measured, neither inferred:

  * The reading is EXACT and reproducible bit-for-bit. CPU HEAD-vs-HEAD is bit-identical on
    1140/1140 measured pairs (pinned artifact, `cpu.fixtures.*.null_bitidentical_frac_*`),
    so every statistic's identical-code null here is exactly 0 and no calibrated bound is
    needed. **These rows may assert equality outright where equality is the claim.**
  * **These rows cannot witness F1.** CPU F1-vs-HEAD is ALSO bit-identical, 1200/1200
    measured — F1 is an exact no-op on the CPU full-net path (IMPL_NOTES_DFIX_A §3.1). The
    green here is a green on the bf16 REGIME's drift, never on F1's effect. The CUDA
    calibrated legs live in `test_bf16_parity_nulldist.py` and loud-skip off-GPU.

F1 aligns the gather to the autocast dtype, so under bf16 autocast the gathered node rows,
their sum with the edge projection, the ReLU'd message AND the `index_add_` accumulator all
move fp32 -> bf16 (DESIGN §1.2). bf16 carries 8 significand bits; a per-node sum of ~27
non-negative terms at run5's mean in-degree has a relative error of ~sqrt(n)*eps typical.
That is an argument, not a measurement — these rows are the measurement, and their bands
were registered in PREREG_DFIX §1 BEFORE any number existed.

DISCLOSED, and it is the reason no F1 verdict is complete from this file alone: the fixture's
mean in-degree is ~8 against run5's ~26.8 (MEASUREMENT_D §3), a ~3x gap running in the
direction that makes the effect SMALLER. BF1-4 — a box row on a real sampled run5 batch — is
the production-distribution instrument and it OUTRANKS every row here (PREREG_DFIX §2, §9).

THREE-BAND DISCIPLINE (PREREG_DFIX preamble). Each row has PASS / PASS-WITH-DISCLOSURE /
ABORT. Only the ABORT threshold halts, so that is what is asserted; the measured value and
its band are printed for the measurement report. A middle-band result is not a failure and
may not be summarised as "no abort fired".

WITNESS vs BOUND — the R181 coverage statement, stated here so no reader can miss it.
Measured discrimination against 3000 F1-vs-HEAD CUDA pairs (pinned artifact, re-derived in
`test_bf16_parity_nulldist.py::test_four_statistics_fail_discrimination`):

  | row                    | role after the re-point                                   |
  |------------------------|-----------------------------------------------------------|
  | OF1-3a policy median   | **WITNESS** — 0.0% overlap on all three fixtures           |
  | OF1-4a gradient cosine | secondary witness, NARROW (3.84x on the production fixture)|
  | OF1-3c bin median      | BOUND only — 85.1% overlap, cannot witness F1              |
  | OF1-3d both losses     | BOUND only — 39.7% / 87.9% overlap, cannot witness F1      |
  | OF1-4b gradient norm   | BOUND only — 62.1% overlap, cannot witness F1              |
  | OF1-3e / OF1-3f argmax | BOUND only — **CUDA null UNVERIFIED**, never calibrated    |
  | OF1-3b / OF1-3c max    | **RETIRED from gating** — null exceeds its own ABORT        |

A BOUND row's green says *"the bf16 regime's deviation from fp32 is inside the registered
band"*. It does NOT say *"F1's effect is bounded"*, and it may not be cited for that.
**Exactly one of the ten registered statistics is a sound F1 witness on the production
fixture. That is less coverage than the original inventory claimed, and it is the finding.**

Arms: (a) reference = autocast DISABLED, all fp32. (c) treatment = autocast bfloat16.
Same net, same weights, same inputs, same masks.

MUTATION COVERAGE — **THIS FILE KILLS NOTHING IN THE F1 MUTATION BANK ON THE CPU PATH.**

Stated flatly because it took two corrections to get here and both were the same failure —
a coverage claim inferred instead of measured.

  * The ORIGINAL claim was *"Mutations killed: MA-1, MA-3, MA-4"*.
  * The R181 re-point narrowed it to *"MA-4 IS killed here — `index_add_` raises in the
    forward, so every row in this file reds"*. **That replacement was itself false**, and it
    was inferred from the gather-regime file's raise rather than measured here.
  * **Measured, all three, on a scratch copy of `src/` + `tests/` with the live tree
    untouched (sha re-verified after every run):**

        MA-1 (revert the gather receiver)     drift+nulldist: ALL PASS
        MA-3 (cast AFTER the gather)          drift+nulldist: ALL PASS
        MA-4 (agg = x.new_zeros(...))         drift+nulldist: ALL PASS

    (The pass COUNT is deliberately not quoted here. It was written as `17 passed, 1 skipped`
    and was already stale two rounds later at `21 passed, 2 skipped` — a number that has to
    be re-edited every time a sibling row is added is a number that will eventually be wrong
    and read as evidence. The claim that matters is "none of the three reds this file", and
    that is what is stated.)

    Every failure under all three lands in `test_gine_gather_regime.py` — the dtype row,
    plus the instrument self-test under MA-1 and MA-4. **`test_gine_gather_regime.py`
    carries the entire F1 mutation bank alone.**

MECHANISM, probed directly rather than reasoned about. `index_add_` *does* raise
(`RuntimeError: index_add_(): self (Float) and source (BFloat16) must have the same scalar
type`, reproduced in isolation) — but only where the conv receives an **fp32** `x`. In this
file's arms the conv is reached through `RepresentationNetwork.forward`, where `input_proj`
is a `Linear` (bf16 out under CPU autocast) and `nn.LayerNorm` is **dtype-PRESERVING on CPU**
(measured: bf16 in -> bf16 out, where fp32 in -> fp32 out). So the conv's `x` is already
bf16, all four recorded gather receivers are bf16, `x.new_zeros` matches `msg`, and MA-4
neither raises nor moves a statistic. This is `IMPL_NOTES_DFIX_A` §3.1's finding — *"F1 is an
exact no-op on the CPU full-net path"* — carried one step further than it had been.

**CONSEQUENCE, wider than this file: `DESIGN_DFIX` §5.1's mutation column is FALSIFIED IN
FULL** for these rows (it credits OF1-3 with MA-1/MA-3 and OF1-4 with MA-1/MA-3/MA-4).
`PREREG_DFIX` §3 was right all along — MA-1 *"OF1-3/4 unchanged"*, MA-3 *"OF1-3/4 GREEN"*,
MA-4 *"which outcome fires is UNVERIFIED ... IMPL records which"* — so no prereg row is
wrong; the design's §5.1 is. Corrected there under R96, not only here.

**UNVERIFIED:** MA-4's behaviour through the production `RepresentationNetwork` **on CUDA**,
where `layer_norm` IS promoted to fp32 and the conv therefore does receive an fp32 `x`. There
the raise is expected and the drift rows would red. Nobody has run it. Owed to the box.

The drift rows are a band on the bf16 regime. They are not a dtype pin and never were.
"""
from __future__ import annotations

import math
from pathlib import Path

import _bf16_parity as bp
import pytest
import torch

_REGIME = "DETERMINISTIC-PATH (CPU): reading is EXACT, identical-code null = 0 (1140/1140)"


def _band(value: float, envelope: float, threshold: float, *, lower_is_better: bool) -> str:
    """THREE outcomes, because PREREG_DFIX's preamble defines three.

    The inherited two-outcome form banded against `envelope` alone and never read
    `threshold`, so a reading PAST its own ABORT threshold printed
    `band=PASS-WITH-DISCLOSURE`. For the seven gating rows that was invisible (the `assert`
    fires on the threshold regardless), but the two RETIRED rows are not asserted —
    **printing into the record is their entire remaining function**, and they printed
    `PASS-WITH-DISCLOSURE` at 1.556892e+00 against a 4.0e-1 threshold, a 3.9x breach. A
    record that reads as a pass at the one place it exists to be honest is the PREREG
    preamble's own named failure: *"it puts R61's tune-to-green trap at the reporting
    step."* Fixed here, not inherited forward.
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


# ── OF1-3 — forward drift ────────────────────────────────────────────────────────────
def test_policy_logit_median_relative_drift(arms) -> None:
    """OF1-3a — **THE RE-POINTED PRIMARY WITNESS** (R181). PASS <= 2.0e-2; DISCLOSE
    (2.0e-2, 5.0e-2]; ABORT > 5.0e-2. Bands unchanged from PREREG_DFIX §1.

    This is the one row whose statistic is measured to distinguish F1 from nothing on all
    three box fixtures (0.0% overlap, 3675 null pairs vs 3000 F1 pairs). Its calibrated
    CUDA null self-test lives in `test_bf16_parity_nulldist.py`.
    """
    ref, treat, _b = arms
    value = bp.median_form(treat.policy_logits, ref.policy_logits)
    line = _report("OF1-3a policy-logit median rel drift", value, 2.0e-2, 5.0e-2,
                   role="WITNESS (primary, R181 re-point)")
    assert value <= 5.0e-2, f"{line} — F1-ABORT-1: HALT to the architect (option iii revived)"


def test_bin_logit_median_relative_drift(arms) -> None:
    """OF1-3c (median reduction only). PASS <= 2.0e-2; DISCLOSE (2.0e-2, 5.0e-2]; ABORT.

    **BOUND, NOT A WITNESS (R181).** On `prod27_run5shape` 851 of 1000 genuine F1-vs-HEAD
    pairs read exactly 0.0 on this statistic — 85.1% overlap with its own identical-code
    null. A zero null is worthless when the alternative is also zero. The band is retained
    because the statistic still BOUNDS the bf16 regime's drift (its null is 0, so the
    2.0e-2 band is fully resolvable); it may not be cited as evidence about F1.
    """
    ref, treat, _b = arms
    value = bp.median_form(treat.bin_logits, ref.bin_logits)
    line = _report("OF1-3c bin-logit median rel drift", value, 2.0e-2, 5.0e-2,
                   role="BOUND ONLY — 85.1% F1 overlap, cannot witness F1")
    assert value <= 5.0e-2, f"{line} — F1-ABORT-1: HALT"


def test_loss_relative_drift(arms) -> None:
    """OF1-3d, both graph losses. PASS <= 5.0e-2 each; DISCLOSE (5.0e-2, 1.5e-1]; ABORT.

    **BOUND, NOT A WITNESS (R181).** Measured F1-vs-HEAD overlap with the identical-code
    null on the production fixture: `policy_loss` 39.7%, `value_loss` 87.9%; and the
    readings' own run-to-run spread over 50 CUDA runs of UNMODIFIED code is 181.8% and
    270.1% of their own value. As a bound the rows are sound — the 1.5e-1 ABORT sits
    3.4e4x / 2.9e3x above the measured null max — and that is the only claim they carry.
    """
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

    The decision the policy head actually exports is its per-graph argmax, so a drift that
    leaves every logit close but flips the chosen move is the failure mode this row —
    and only this row — sees.

    **CUDA NULL UNVERIFIED (R181 disclosure).** The R181 calibration measured five
    statistics; the two argmax rows were NOT among them, so this row has no measured
    identical-code null on CUDA. On the deterministic CPU path its null is exactly 0 by
    bit-identity (1140/1140), which is what licenses gating it HERE. It may not be read as
    a CUDA row until its null is measured. MEASUREMENT_BF1 §0 further records that on a
    16-graph fixture the row's resolution is 6.25 pp — two graphs wide — so the instrument
    is underpowered even where its null is known.
    """
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


# ── OF1-4 — gradient drift (a forward that stays close can still corrupt the update) ──
def test_gradient_cosine_similarity(arms) -> None:
    """OF1-4a. PASS >= 0.99; DISCLOSE [0.95, 0.99); ABORT < 0.95.

    **SECONDARY WITNESS, NARROW (R181).** The cosine DEFICIT (1 - cos) does discriminate
    F1 from nothing on all three fixtures, but on the production fixture by only 3.84x
    (null max 7.1952e-6, F1 min 2.7627e-5). That caps any usable margin at M < 3.84 and is
    why this is a secondary, not a replacement for OF1-3a.

    Computed in float32 by `torch.nn.functional.cosine_similarity`, which is the form the
    registered band was set on and is left unmoved. **Where EXACTNESS is the claim the
    assertion must be `torch.equal`, never `cos == 1.0`**: the float64 deficit reads
    1.1102e-16 on provably bit-identical gradients (pinned artifact,
    `cpu.fixtures.prod27_run5shape.null_order_stats_treat.grad_cos_deficit_f64`), and the
    float32 form reads > 1.0. The null self-test in `test_bf16_parity_nulldist.py` honours
    that; this row does not need to, because it is a band and not an equality.
    """
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

    Direction and magnitude are separate failures: a systematically shrunk gradient has
    cosine ~1 and rescales the effective learning rate.

    **BOUND, NOT A WITNESS (R181).** 62.1% of genuine F1-vs-HEAD pairs on the production
    fixture fall at or below this statistic's own identical-code null. As a bound it is
    sound (the 5.0e-2 envelope is 62x the measured null max, 8.0303e-4).
    """
    ref, treat, _b = arms
    n_a = float(ref.grads.norm())
    value = abs(float(treat.grads.norm()) - n_a) / n_a
    line = _report("OF1-4b parameter-gradient norm rel difference", value, 5.0e-2, 1.5e-1,
                   role="BOUND ONLY — 62.1% F1 overlap, cannot witness F1")
    assert value <= 1.5e-1, f"{line} — F1-ABORT-1: HALT"


# ── the meta-guard: a gating row may not silently become non-gating ──────────────────
# Frozen census. Maps each row in THIS file to the numeric literals its `assert` statements
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
    """**Closes RED-TEAM's "a gating row can silently become non-gating".**

    Measured hole, reproduced before closing it: deleting the gating `assert` from OF1-3a —
    the primary witness, the one row R181's whole re-point exists to install — left the tier
    at **74 passed, 2 skipped**. Nothing anywhere detected a de-fanged assertion. The row
    still ran, still printed its `MEASURED …` line, still reported PASS, and asserted nothing.

    That is the phantom-gate class (R4/LAW-07) in its purest form, and it is worse here than
    usual because the printed line makes the row LOOK like it is gating.

    This census closes it by AST — the `PREREG_DFIX` OF2-9 precedent, which censuses readers
    rather than grepping. For every row it pins the exact set of numeric literals the row's
    `assert` statements compare against, so it REDs on:

      * an `assert` deleted or replaced by `pass` (the literal disappears);
      * a threshold NUMBER changed — i.e. R61's tune-to-green, caught at the literal;
      * a new row added to the file without a census entry (`unexpected` below);
      * a censused row deleted (`missing` below).

    It cannot catch an assert weakened while keeping its literal (e.g. `<=` flipped to `>=`,
    or the left operand swapped for a constant). That residue is stated rather than implied
    away: the comparison OPERATOR and the left operand are not pinned. Closing that too would
    mean pinning each row's full expression, which is a second copy of the row and drifts.
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


# ── the retired rows — RECORDED, NOT GATING (R181) ───────────────────────────────────
def test_retired_max_form_rows_are_recorded_not_gating(arms) -> None:
    """**OF1-3b and OF1-3c-max: RETIRED from gating under R181's S-3 re-point grant.**

    This row does not assert the retired readings against their registered bands. It
    prints them, and it asserts the RETIREMENT'S OWN GROUNDS, re-derived here from the
    pinned artifact so the retirement is machine-checked rather than remembered:

      the identical-code CUDA null of each max-form statistic EXCEEDS the band that row
      was asked to police.

    If that ever ceases to be true the assertion REDs and the retirement returns to the
    architect — which is the correct behaviour, because the retirement is a measurement
    and not a preference.

    **Stated without softening, because it is the honest reading of the data:** the max
    form is NOT blind to F1. On `prod27_run5shape` its F1-vs-HEAD minimum (1.3436e+0)
    sits above its identical-code maximum (9.3367e-1) — 0.0% overlap, 1.44x separation.
    Its disqualification is different and worse: **its null is larger than its own ABORT
    threshold**, so it aborts on 24.82% of comparisons of a commit against itself. A gate
    that REDs a quarter of the time on identical code returns a verdict about the
    scheduler, not about the change (R181: *"it measures index_add_ atomics"*).

    Second, independent indictment, on the deterministic path this file runs on: the
    1.5e-1 PASS envelope demands an absolute error <= 1.5e-4 at the tensors' own p90
    scale, where ONE bf16 ulp measures 1.953125e-3 (`torch` bfloat16 spacing on
    [0.25, 0.5), re-derived below). That is ~1/13 of a single ulp — arithmetically
    unreachable by any bf16 regime, on any device. IMPL_NOTES_DFIX_A §4.1 measured 5 of 5
    independent batch draws breaching, with F1 a measured no-op across the edit.
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
