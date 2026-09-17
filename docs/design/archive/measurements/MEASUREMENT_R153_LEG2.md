# MEASUREMENT — R153 LEG 2: run5 exposure through the PRODUCTION expand

Instrument: `crates/mantis-search/tests/r153_leg2_ls_target_mass.rs`. Verdict rule frozen in
`PREREG_R153_LEG2.md` before the probe first ran. 960 positions (480 per encoding), 2026-07-31.

# VERDICT: **RUN5 EXPOSURE CONFIRMED**

---

## 1. Pre-registered aborts — all cleared

| abort | condition | status |
|---|---|---|
| 1 | sample must reach >361 legal | CLEARED — max `n_legal` 7 364 |
| 2 | `exported_mass > 1.0 + 1e-6` | never fired |
| 3 | non-reproducible at a fixed seed | never fired |
| 4 | **no off-window CHILD exists** → a zero would be structural | **CLEARED — 467/480 positions carry off-window children** |

Abort 4 is the one that makes this leg admissible. It was written precisely so a REFUTED
verdict could not be returned by an instrument that failed to create the thing under test —
leg 1's failure mode wearing leg 2's label. It cleared, so these drops are real.

## 2. Results — production expand path, per R155's instrument clause

| encoding | expand (production) | positions | affected | median (affected) | p90 | ≥50% lost | **degenerate (≥99%)** | max |
|---|---|---|---|---|---|---|---|---|
| **`gnn_axis_v1`** (run5 mints this) | `expand_and_backup_ls_at` | 480 | **314 (65.4%)** | **0.2349** | 0.7181 | **53** | **12** | **1.000000** |
| `v6_live2_ls` | `expand_and_backup_ls` | 480 | **420 (87.5%)** | 0.6577 | 1.0000 | 236 | **144** | 1.000000 |

Unattributed drops: **zero** on both. Every dropped child was an off-window cell failing
`is_covered`.

`production_path: true` for both rows (R155 clause). `v6w25` remains leg 1's result
(`expand_and_backup`, production-valid, zero drop).

## 3. The inversion, stated plainly

| encoding | leg 1 (dense — NOT production) | leg 2 (production) |
|---|---|---|
| `gnn_axis_v1` | 0/480 affected | **314/480, up to 100%** |
| `v6_live2_ls` | 3/480, max 4.0% | **420/480, up to 100%** |

Leg 1's `gnn_axis_v1` zero was **entirely an artifact of the wrong expand path**, exactly as
its own §4 warned and exactly as R155 anticipated. Under the dense expand, off-window cells
get `sort_prior = 0.0` and are truncated by the 192-child cap, so no off-window child exists
and there is nothing to drop. Under the production legal-set expand they are real children
with real priors — and then the exporter's coverage gate discards them.

Per R155, leg 1's `gnn_axis_v1` zero stays labelled **non-production forever**. It is not
superseded by this table and not merged into it. Leg 1's `v6_live2_ls` number is retired to
"exporter behaviour, non-production"; the production number is this leg's.

## 4. What CONFIRMED means concretely for run5

run5 mints `gnn_axis_v1` and trains on the exported visit distribution. Measured, that target:

- loses visit mass on **65.4% of positions**;
- on affected positions loses a **median 23.5%**, p90 **71.8%**;
- loses **half or more** on 53 positions;
- is **entirely zero on 12 positions** — `dense` all zeros, `overflow` empty, nothing left.

The degenerate rows are a *qualitatively* different failure from partial truncation. An
all-zero policy target is not a truncated distribution; it carries no gradient signal and,
depending on the loss, is either a no-op row or a malformed one. That case was not anticipated
by ADJ-10's original framing ("a minority of off-window mass") and is the sharpest single
finding of this leg.

The operator's R153 framing is now measured: *"run5 training on silently truncated targets is
R138's handicap moved from eval to learning."* On the eval side R138 measured 53.2% of the
action set dropped. On the learning side the exposure is **broader** (65.4% of positions
affected) and, at the tail, total.

## 5. Magnitude scales with dispersion — R156's hypothesis KILLED in passing

Recorded per R156 (*"if the fix's behaviour confirms or kills the hypothesis in passing,
record it; do not build an experiment for it"*). No experiment was built for this.

Leg 1's three drops sat at `n_legal` 193–235, which is what suggested a cap-boundary mechanism.
Under the production expand the drop is **not** confined to the cap boundary: it begins around
`n_legal` 362 and grows monotonically with dispersion —

`n_legal` 362 → 0.060 · 439 → 0.101 · 528–583 → 0.181 · 616 → 0.403 · deep tail → 1.000

**The cap-boundary hypothesis is refuted for the production path.** The governing variable is
how much of the legal set lies outside cluster coverage, which grows with dispersion.

This does not weaken R156's flip-set mandate — it **sharpens** it. The deep-tail dispersed rows
are now the HIGH-MAGNITUDE region rather than a completeness afterthought, and the 193–235 band
is the low-magnitude edge. Both remain mandatory; their roles are the reverse of what leg 1
suggested.

## 6. Pre-registered consequences now in force (R155)

1. **Mint stays BLOCKED** on this axis until the fix lands.
2. **The fix rides this stack**, and its scope re-verifies run5's exporter.
3. **`RUN5_MINT_PREREG.md` gains a dropped-mass row** — owner + status + these grounds.
4. Fix flip-set bound by R156: the **193–235 `n_legal` band**, sparse-coverage early plies, AND
   deep-tail dispersed rows — now known to be the severe end.
5. Exported-target parity oracle required: export == raw visit distribution on the fixture.

---

## 7. PARAMETER CORRECTION — re-measured at run5's ACTUAL target-generation sim count

**Self-disclosed defect in §2's numbers.** §2 ran at `N_SIMS = 150` — that is `deploy_sims`
(`configs/run5.yaml:30`), the EVAL count. run5 generates TRAINING TARGETS at
`selfplay.mcts.n_simulations: 50` (`:176`, `fast_sims: 50` at `:185`). A training-target
question measured at the eval sim count is R155's path-parity class one level down — at
PARAMETERS rather than paths — and it is the lesson F-RT-1 already charged this WP: *pin a
mutation's PARAMETERS and FIXTURE, not only its SITE.*

The verdict is unchanged (the mechanism is geometric and every drop was attributed per child).
The MAGNITUDES change, and **they get worse at run5's real setting.**

### Re-measured, `N_SIMS = 50` — THESE are the prereg row's grounds

| encoding | affected | median (affected) | p90 | ≥50% lost | **degenerate (≥99%)** | max |
|---|---|---|---|---|---|---|
| **`gnn_axis_v1`** (run5) | **256/480 (53.3%)** | **0.4898** | **1.0000** | **114** | **37** | 1.000000 |
| `v6_live2_ls` | **379/480 (79.0%)** | **1.0000** | 1.0000 | 277 | **195** | 1.000000 |

### 150 vs 50 sims — fewer positions hit, each hit far harder

| metric (`gnn_axis_v1`) | 150 sims (deploy — reference only) | **50 sims (run5's actual)** |
|---|---|---|
| positions affected | 65.4% | 53.3% |
| median loss among affected | 23.5% | **49.0%** |
| p90 | 71.8% | **100%** |
| ≥50% lost | 53 | **114** |
| **degenerate rows** | 12 | **37 (3.1×)** |

**Mechanism (consistent with §5, not a new hypothesis):** fewer sims visit fewer children, so
the visited set is smaller and more concentrated. When that concentrated set falls outside
cluster coverage, there is no surviving in-window mass to dilute the loss — so the whole target
vanishes rather than part of it. Lower sim counts make the *degenerate* outcome more likely,
not less.

### What this means for run5, stated at run5's own parameters

- **53.3%** of self-play positions export a target missing visit mass.
- Among those, the **median position loses about half** its mass.
- **37 of 480 positions — 7.7% of ALL sampled positions — export an ALL-ZERO target.**

Per DESIGN_R158 §1.5 those 37 rows are not no-ops: they contribute zero CE while remaining in
the loss denominator, so they **dilute** every other position's gradient in the batch.

The 150-sim table in §2 is retained as the deploy-sims reference and is explicitly **NOT** the
grounds for the `RUN5_MINT_PREREG` dropped-mass row. R157's "12 all-zero rows" is superseded by
**37** at run5's parameters; the named class is unchanged and its size is larger.
