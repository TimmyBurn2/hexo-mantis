# MEASUREMENT — R153 characterization of dropped training-target mass

Instrument: `crates/mantis-search/tests/r153_target_mass.rs`. Verdict rule frozen in
`PREREG_R153.md` **before** the probe first ran. 1 440 positions, run 2026-07-31.

---

## 1. Pre-registered aborts — all cleared

| abort | condition | status |
|---|---|---|
| 1 | sample must reach the **>361-legal** regime | **CLEARED** — max `n_legal` 7 364 (`gnn_axis_v1`), 5 166 (`v6_live2_ls`), 12 906 (`v6w25`) |
| 2 | `exported_mass > 1.0 + 1e-6` (double-count) | never fired |
| 3 | non-reproducible at a fixed seed | never fired — equality assert passed per encoding |

**Abort 1 fired on the FIRST run and was honoured.** The initial sample was games only and
reached `n_legal` 234 — nowhere near the tail. The dispersed tail probe was added and the run
repeated. The first run's numbers are superseded, not merged.

## 2. Results

| encoding | positions | affected | max `dropped_mass` | max `n_legal` |
|---|---|---|---|---|
| **`gnn_axis_v1`** (run5 mints this) | 480 | **0** | 0.000000 | 7 364 |
| `v6_live2_ls` (ruled dense control arm) | 480 | **3** | **0.040268** | 5 166 |
| `v6w25` | 480 | **0** | 0.000000 | 12 906 |

The three affected positions, all reproducible:

| ply | `n_legal` | `n_children` | `dropped_mass` | dropped children |
|---|---|---|---|---|
| 3 (dispersed) | 198 | 192 | 0.040268 | 6 |
| 4 (game) | 194 | 192 | 0.026846 | 4 |
| 5 (game) | 234 | 192 | 0.020134 | 3 |

Unattributed drops: **zero**. Every dropped child was an off-window cell failing `is_covered`.

## 3. VERDICT under the pre-registered rule — CONFIRMED

§4 criterion 1 — `dropped_mass > 1e-6` on ≥1 position: **TRUE** (3 positions).
§4 criterion 2 — systematic, every drop attributable to `is_covered`, reproducible: **TRUE**.

**→ CONFIRMED. The export gate demonstrably drops visit mass from the exported target, and
the drop is a deterministic function of geometry, not noise.** Per R153 the size (max 4.0%) is
recorded as urgency, not as guilt: the class defect stands regardless.

This also settles the documented contradiction empirically. `policy.rs:166-168` describes the
behaviour correctly; `records.rs:481`'s documented semantics is the AUTHORITY (R153, R34), and
the export diverges from it.

## 4. LIMITATION FOUND IN THIS INSTRUMENT — production exposure is NOT established

**Disclosed rather than glossed, because it changes what this measurement may be used for.**

The probe fills every tree with `MCTSTree::expand_and_backup` — the **DENSE** expand. But
`runner/params.rs:67` sets `legal_set = matches!(spec.policy_pool, LegalSetScatterMax)` and
`:76` forces it true for graph specs, so in production:

| encoding | production expand | did the probe drive it? |
|---|---|---|
| `gnn_axis_v1` | **legal-set** (`expand_and_backup_ls_at`) | **NO** — dense |
| `v6_live2_ls` | **legal-set** | **NO** — dense |
| `v6w25` | dense | **YES** |

Consequences, stated exactly:

1. **`v6w25`'s zero-drop result IS a production-path result.** It stands.
2. **`gnn_axis_v1`'s zero-drop result is NOT.** It does not establish that run5's training
   target is drop-free. Under the dense expand, off-window cells get `sort_prior = 0.0`
   (`backup.rs:97`) and are truncated out by the 192-child cap, so they never become children
   and there is no mass to drop — the zero may be an artifact of measuring the wrong expand.
   **The mint question is therefore still OPEN, and mint-blocking status stands.**
3. **The three confirmed drops were also measured on the dense expand**, so they establish the
   EXPORT GATE's behaviour — which is what §4 asks — but not `v6_live2_ls`'s production rate.

The verdict in §3 is unaffected: it is a claim about `get_policy_ls`, and `get_policy_ls` is
the same function on both paths. What is not yet answered is how often production trees
contain a visited, off-window, uncovered child.

## 5. Owed — the legal-set leg

A second leg driving `expand_and_backup_ls_at` (dense + coord-keyed overflow + builder centre,
as `search_drive.rs` does) over the same position sample, for `gnn_axis_v1` and `v6_live2_ls`.
Only that answers "does run5 train on silently truncated targets".

Until it runs, the honest statement is: **the class defect is confirmed; run5's exposure to it
is unmeasured.** Recorded this way so no one reads "gnn_axis_v1: 0 affected" as clearance.

## 6. Mechanism note — hypothesis, NOT a finding

All three drops sit at `n_legal` 194–234, just above the 192-child cap, at early plies (3–5) —
and NOT in the extreme tail (no drop at `n_legal` 5 166). A plausible reading is that the cap
admits marginal off-window cells only when `n_legal` barely exceeds it, while at high dispersion
the top-192 are all near the centre and in-window; early plies also have few cluster centres and
so less coverage. **This is an untested hypothesis.** It is recorded because it predicts where a
fix must be checked, and it is labelled so it is not mistaken for a measured mechanism.
