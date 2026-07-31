# PREREG — R153 characterization: dropped mass in the EXPORTED training target

**Written and frozen BEFORE any measurement is taken.** R153 requires the instrument and the
verdict rule pre-registered first, because n=1 carries no verdict (LAW-01/LAW-04) and because
a rule chosen after seeing the data is not a rule. Nothing below was authored with knowledge
of the outcome; the probe had not been run when this file was written.

Frozen at: **2026-07-31**, before `probe_r153_target_mass.rs` executed for the first time.

---

## 1. The authority being tested against

R153 states it and it is confirmed in-tree:

- The policy target IS the **raw visit distribution** (R34 recon).
- The training targets are documented as **deliberately NOT inheriting the off-window skip** —
  `crates/mantis-selfplay/src/runner/records.rs:481`.
- But `crates/mantis-search/src/mcts/policy.rs:166-168` documents the opposite in its own
  docstring: *"Off-window children with NO cluster coverage are dropped (today's `get_policy`
  behaviour)."*

**Two in-tree documents disagree about the same export.** That is the subject. The measurement
does not decide which is right — R153 already ruled the authority is the documented target
semantics. The measurement decides only whether the export actually diverges from it, and by
how much.

## 2. Quantity measured

For each sampled root position, after a completed search:

- `exported_mass` = `sum(dense) + sum(overflow)` from `get_policy_ls(temperature, n_actions)`.
- `dropped_mass` = `1.0 - exported_mass`.
  The raw visit distribution is normalised by construction (`v / total` over all children), so
  a correct no-drop export sums to exactly 1.0 and **any deficit IS the dropped mass**. This is
  why the instrument needs no second reimplementation of the target to compare against —
  the invariant is self-contained, which removes the "my reference is also wrong" failure mode.
- `dropped_children` = children whose visit mass reached neither `dense` nor `overflow`.
- `n_children`, `n_legal`, `ply`, and the child's in/off-window status, recorded per position.

Recorded as a DISTRIBUTION: fraction of positions affected, median / IQR / max of
`dropped_mass`, and `dropped_mass` as a function of ply. No single-number summary is a verdict.

## 3. Sample — representative, real-game, LAW-04 aware

- **Primary (mint-relevant): run5's geometry.** `gnn_axis_v1` — the encoding run5 mints.
- **Secondary (class boundary, R71):** `v6_live2_ls` (the ruled dense control arm) and `v6w25`.
  These establish whether the defect is geometry-specific or a property of the coverage gate.
- **Positions:** every root position of **complete games**, not hand-picked plies — ADJ-10's
  existing evidence is one position at ply 64 and that is exactly what R153 refuses to accept.
  Minimum **3 complete games per encoding**, distinct seeds (LAW-04: distinct games, not
  distinct positions of one game).
- **Tail probe:** additionally the dispersed regime (radius-6 style, >361 legal), because the
  coverage gate's exposure grows with dispersion and a game-only sample could miss the tail.
- `temperature = 1.0` for the exported-target path (the training export uses the temperature
  branch, not the argmax branch — both branches carry the same `is_covered` gate and both are
  measured).

## 4. VERDICT RULE — fixed here, before the data

R153: *"Any systematic drop confirms the class defect regardless of size — size sets urgency,
not guilt; a bug is not a semantics."* Encoded literally:

**CONFIRMED (class defect, fix rides this stack) iff BOTH:**
1. `dropped_mass > 1e-6` on **at least one** sampled position, AND
2. the drop is **systematic**, i.e. every dropped child is an off-window cell for which
   `is_covered` is false — the drop is a deterministic function of geometry, and reproduces
   on a re-run with the same seed and appears across independent seeds.

**NOT CONFIRMED iff** `dropped_mass <= 1e-6` on every sampled position across all encodings
and both regimes.

**Explicitly NOT verdict criteria** — recorded so they cannot be smuggled in afterwards:
- The SIZE of the drop. A 0.01% systematic drop confirms exactly as a 40% one does. Size is
  reported and sets urgency; it does not decide guilt.
- Whether run5's specific geometry is the worst case.
- Whether the drop is "small enough not to matter for learning" — that is a semantics
  argument, and R153 forbids settling a bug with one.

**Ambiguous outcome** (drop present but NOT attributable to `is_covered` — e.g. floating-point
renormalisation): does NOT confirm this class. It is a separate finding, reported and queued,
never folded into this verdict.

## 5. Pre-registered consequences

- **CONFIRMED** → fix rides this stack: R71 flip-set over the class boundary (every consumer of
  the coverage gate, not the one demo site) + an **exported-target parity oracle** asserting
  `export == raw visit distribution` on a frozen fixture. Mint stays blocked until it lands.
- **NOT CONFIRMED** → ADJ-10 closes as non-mint with the distribution as its measured grounds,
  and the `policy.rs:166-168` docstring is corrected under R73 name-truth, because it would
  then be describing a drop that does not occur.

## 6. Instrument

`crates/mantis-search/tests/` (or the selfplay crate, whichever owns a searchable fixture) —
Rust, because `get_policy_ls` has **no pyo3 binding** and adding one would be a production
change (R6) made to serve a measurement, which is the wrong direction. The probe is a test, so
it runs under `cargo test --workspace --locked` and cannot rot silently.

**Abort conditions for the measurement itself:**
1. The probe cannot construct a position in the >361-legal regime → HOLD; the tail is where the
   gate's exposure lives, and a sample that cannot reach it is not representative.
2. `exported_mass > 1.0 + 1e-6` → HALT and root-cause. That is not a drop, it is a
   double-count, and it would mean the invariant in §2 is not the invariant.
3. The probe's own numbers are not reproducible across two runs at a fixed seed → HALT; a
   non-deterministic instrument cannot characterise a deterministic gate.
