# PREREG — R153 LEG 2: run5 exposure through the PRODUCTION expand path

**Written and frozen BEFORE leg 2 is executed**, per R155. Leg 1's limitation is the reason
this leg exists: leg 1 drove the DENSE expand for two encodings whose production path is a
legal-set expand, so its zero for `gnn_axis_v1` is not a clearance and never becomes one.

Frozen at: **2026-07-31**, before `r153_leg2_ls_target_mass.rs` first executed.

---

## 1. R155's standing instrument clause — encoded here, not just quoted

> *"A drop measurement clears an encoding ONLY when driven through that encoding's PRODUCTION
> expand path (LAW-03 false-clear corollary made mechanical)."*

**Mechanical form, binding on this leg and every oracle this card ships:** each measured
encoding is paired with its production expand, the pairing is asserted in the instrument
itself, and a result carries a `production_path: true|false` label derived from that pairing —
never from the author's intent. A row labelled `false` may state a behaviour of the exporter;
it may never clear the encoding.

Production dispatch, read from `runner/search_drive.rs:322-339` and `params.rs:67,76`:

| encoding | `legal_set` | `is_graph` | PRODUCTION expand | measured by |
|---|---|---|---|---|
| `gnn_axis_v1` | true (forced, D2) | true | `expand_and_backup_ls_at` | **LEG 2** |
| `v6_live2_ls` | true (`LegalSetScatterMax`) | false | `expand_and_backup_ls` | **LEG 2** |
| `v6w25` | false | false | `expand_and_backup` | leg 1 — production-valid, stands |

**Leg 1's `gnn_axis_v1` zero stays labelled non-production forever** (R155). It is not
superseded by leg 2 and not merged with it; the two answer different questions.

## 2. Instrument

`crates/mantis-search/tests/r153_leg2_ls_target_mass.rs`. Same sample generator as leg 1 —
the identical games (seeds `20260731`, `8675309`, `42`, 128 plies) plus the identical
dispersed tail (96 plies, farthest-from-window-centre) — so the position sets are comparable
row for row. Same `N_SIMS = 150`, `TEMPERATURE = 1.0`, `TOL = 1e-6`.

**What changes is only the expand**, which is the whole point:

- **No-drop uniform legal-set policy per leaf.** Mass `1/n_legal` on EVERY legal move, split
  into `dense` (in-window, by `window_flat_idx`) and `overflow` (off-window, coord-keyed).
  This mirrors the producer's contract (`assemble_ls_from_gnn_probs`: dense + overflow sum to
  1, validated always-on). It is what makes off-window cells become real children with real
  priors — the condition leg 1's dense expand structurally could not create.
- `centers[i] = boards[i].window_center()` — the builder centre the production assert pins.
- `trunk_sz = spec.trunk_size` for the graph arm (production asserts `agg_trunk_sz ==
  spec.trunk_size`); `expand_and_backup_ls` for the grid legal-set arm takes no centre.

Measured per position, exactly as leg 1: `dropped_mass = 1 - (Σdense + Σoverflow)` from
`get_policy_ls`, `dropped_children`, `n_legal`, `n_children`, `ply`, and per-child
off-window/`is_covered` attribution.

## 3. VERDICT RULE — fixed here, before the data

The question is narrow and is NOT the class question (already CONFIRMED by leg 1, R153):

**RUN5 EXPOSURE CONFIRMED** iff, for `gnn_axis_v1` driven through `expand_and_backup_ls_at`:
`dropped_mass > 1e-6` on **at least one** sampled position, with every dropped child
attributable to an off-window cell failing `is_covered`, reproducible at a fixed seed.

**RUN5 EXPOSURE REFUTED** iff `dropped_mass <= 1e-6` on **every** `gnn_axis_v1` position
across both regimes (games + dispersed tail), with the tail confirmed to reach the
>361-legal regime.

`v6_live2_ls` is measured on the same leg and reported, but it does not decide run5's
exposure — run5 does not mint it. Its production-path result replaces leg 1's dense-path
result for that encoding, and leg 1's is retired to "exporter behaviour, non-production".

**Explicitly NOT verdict criteria:** the size of the drop; whether the drop is smaller than
leg 1's; whether it appears only near the cap boundary. Per R156 the cap-boundary hypothesis
stays UNTESTED and is not consumed as explanation here.

## 4. Pre-registered consequences (R155)

- **REFUTED** → mint unblocks **on this axis only**. The fix still rides this stack: the class
  is confirmed independently (R153), and refutation of run5's exposure is not acquittal of the
  exporter.
- **CONFIRMED** → the fix's scope re-verifies run5's exporter, and `RUN5_MINT_PREREG.md` gains
  a dropped-mass row. Mint stays blocked until the fix lands.

Either way the fix's flip-set is bound by R156: the **193–235 `n_legal` band**, sparse-coverage
early plies, AND deep-tail dispersed rows are all mandatory boundary coverage.

## 5. Abort conditions

1. The `gnn_axis_v1` sample does not reach the **>361-legal** regime → HOLD (leg 1's abort 1,
   which fired once already and is not allowed to fire silently twice).
2. `exported_mass > 1.0 + 1e-6` → HALT: double-count, the §2 invariant is wrong.
3. Not reproducible at a fixed seed → HALT.
4. **The off-window overflow the instrument feeds does not produce off-window CHILDREN** →
   HALT, do not report a zero. A zero drop reached because the tree still contains no
   off-window child would be leg 1's false-clear wearing leg 2's label — the exact failure
   R155's clause exists to prevent. The instrument asserts off-window children exist on at
   least one position before any zero may be reported as REFUTED.
