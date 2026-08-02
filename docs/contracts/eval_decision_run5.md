# Contract: the run5 eval decision

- version: v1
- owner: `mantis.eval`
- status: DECIDED — this is a mint input (R147 consequence 3) and it carries R151's
  control-arm honesty clause.

This document records what run5's promotion bar and external instrument **are**, as facts
about the shipped code. The durable definitions — what deploy-matched means, what a book
pin is, what eff_n counts — live once in `docs/contracts/eval_instrument.md` and are linked
from here, never restated. The values that are **not yet decided** (the random-floor count,
the fresh-seed question) are not here either: they are mint-prereg rows and they are the
operator's at mint (R119).

Every claim below is re-derived from `configs/run5.yaml`, from
`src/mantis/arena/books/manifest.toml` and from the shipped symbols by
`tests/eval/test_eval_decision_run5_doc.py`, which compares this document against those
sources rather than against a transcribed copy of them. A re-mint therefore reds this
document instead of silently agreeing with it, which is the correct direction for a drift
gate.

## The promotion bar (S-1)

Candidate versus the best snapshot, deploy-matched on both sides, screened then confirmed.
Both players are built by the same `_build_candidate_player` at the same simulation count
(`src/mantis/eval/worker.py:174-228`), which is what LAW-15 means by deploy-matched: the bar
measures the net that ships, at the compute it ships at.

- `eval.gate.stride` = 1 — every eval-eligible step is a candidate.
- `eval.gate.screen_games` = 80 — the screening block.
- `eval.gate.screen_confirm_lo` = 0.44 — escalate only if the screen reaches it.
- `eval.gate.confirm_games` = 128 — the confirmation block, on escalation only.
- `eval.gate.promotion_winrate` = 0.55 — the pooled, draw-aware threshold.
- `eval.gate.deploy_sims` = 150 — on BOTH sides (LAW-15).
- `eval.gate.bootstrap_resamples` = 1000 — percentile resamples.
- `eval.gate.min_distinct_per_pair` = 10 — the low-power floor.
- `eval.gate.seed_base` = 20260625 — the gate seed, and also the book-selection seed.

Promotion requires all three of: the pooled draw-aware win rate at or above the threshold,
the pooled distinct-game bootstrap lower bound strictly above 0.5, and not low-power
(`src/mantis/eval/aggregate.py:181-187`). There is exactly ONE deploy-decision site,
`src/mantis/eval/promote.py:37-73`, and it refuses unless the round's typed broken-reason is
absent.

## What the bar measures, and what it refuses (S-2, S-3, S-4, S-6)

- **The net that ships, over the full action set.** The graph decode consumes both halves the
  producer returns and expands through the same frame self-play uses
  (`src/mantis/eval/worker.py:137-152`). An unregistered encoding, or one whose declared
  pooling is unimplemented, is a **refusal** — never a fallthrough
  (`src/mantis/eval/worker.py:155-171`).
- **`elo_ci_lower_boot` is not Elo points.** It is the bootstrap win-rate lower bound minus
  one half (`src/mantis/eval/aggregate.py:217-227`), decision-equivalent for the
  above-zero test and for nothing else. Reading it as Elo overstates every margin.
- **eff_n counts DISTINCT games** (LAW-04), deduped on the trajectory hash
  (`src/mantis/eval/aggregate.py:56-84`); duplicates count once, and the low-power guard is a
  per-pair distinct count.
- **The CI is a bootstrap percentile**, not a normal approximation
  (`src/mantis/eval/aggregate.py:100-117`); an empty sample degenerates to an absent
  interval rather than raising.

## The gate is not evidence of external strength (S-7)

The gate is anchor-relative by construction and says nothing about strength outside the
family it compares within. `docs/registers/falsified.md` row F-30 records a promotion at step
45k while the external win rate collapsed to 2%, and row F-27 records a gate staying green
for 46k steps across a 33% → 5% external collapse. The external instrument is the ladder plus
the RandomBot floor, and it is separate from the gate on purpose.

## The ladder, as minted (S-8)

Six rungs are declared, in this order: `sealbot_d5`, `kraken_raw`, `sealbot_d6`,
`kraken_mcts200`, `strix_128`, `strix_256`. Every one is deploy-matched and capped at 32
games; the first is the only one active at round zero.

Four carry R139's operator-authorized skip grounds — verbatim, and per rung, so a reader of
the log can tell a ruled skip from a broken box: kraken, weights not cleanly accessible;
strix, actively changing. The two sealbot rungs resolve against the vendored engine pinned in
`vendor/pins.toml`; see the liveness clause below for what that does and does not claim.

**The Bradley-Terry fit therefore rests on two rungs of ONE engine family**, at two depths,
plus the floor. However many games are recorded, one opponent lineage is all the information
in the fit — that is a property of the instrument, not a defect to be worked around, and it
is stated here because a reader who does not know it will over-read every ladder-shaped
field in the run.

A second instrument property, for the same reason: a deterministic opponent at a fixed depth
facing a deterministic argmax head produces ONE trajectory per opening. eff_n on such a rung
is bounded by the number of distinct openings, not by the game count.

## Opening books (S-5)

Versioned, sha-pinned and paired. The gate's book is `book_v1_s20260625_p4`, sha256
`52943eabdfd911c36dcd9374da7d91f0a712b0d1eff47a99ab09a760b5088d06`, verified at load with a
hard failure on mismatch (`src/mantis/arena/books.py:39-57`). It holds 512 four-ply openings
minted at seed 20260625, and every opening is used exactly twice with the colours swapped.

## Per-side compute (S-14)

Stated per block, because the three blocks are not commensurable:

- Gate: candidate 150 simulations against best-snapshot 150 simulations — symmetric and
  deploy-matched.
- Sealbot ladder: `eval.sealbot_model_sims` = 128 candidate-side simulations against a
  FIXED SEARCH DEPTH of 5 and 6 respectively. Simulations and depth are different axes; there
  is no exchange rate between them and none is implied here.
- Random floor: `eval.random_model_sims` = 96 candidate-side simulations against a
  zero-compute uniform opponent.

The regime key stamped on every record carries the value actually used, never the gate's.

## The fixed-depth bar, and what it is comparable to (S-10, S-11)

The vendored engine is pinned by commit sha in `vendor/pins.toml`, its search depth is driven
rather than hoped for, its wall-clock cut is neutralised, and the depth it actually reached is
read back as a receipt after every move — a violation raises rather than reporting a shallower
opponent under the rung's name (`src/mantis/bots/sealbot.py`). Falsified-register row F-20 is
why: a bar that silently truncates is an opponent-instance artifact, and this repo has paid
for that once.

**The bar is reproducible within a run box; it is not certified across hosts.** The tracked
patch `vendor/patches/sealbot.patch` removes `-march=native`, which is the one
host-specific-by-construction term, but different compilers and standard libraries can still
differ. That is a weaker claim than "reproducible fixed-depth bar" sounds, which is exactly
why it is written down.

**No run5 number from this engine may be compared to a historical one.** Register rows F-01,
F-22, F-24, F-25, F-26, F-28 and F-29 all quote an external win rate taken from an UNPINNED
instance at an unrecorded depth. This is the first pinned bar; the historical series is not
its baseline.

## The RandomBot floor, and the dead asymmetry (S-12, S-13)

The floor is part of the bar (R147). Its opponent is uniform over the legal set, it uses the
gate's book, and it is aggregated with the same bootstrap knobs as the ladder.

- `eval.random_floor_games` = 0 as minted — the mint delta at the head of `configs/run5.yaml`
  reads `4 -> 0`, so the floor is presently allocated nothing. R147 supersedes that delta; the
  corrected value is a named mint-prereg row and it is the operator's at mint. A reader must
  not read the floor as part of the measured bar until it is armed.

The ladder asymmetry is dead at the head's own seat, measured rather than argued: the head
samples the full legal set, including moves outside the encoding window, against an opponent
that does the same (`tests/eval/test_eval_selfplay_child_parity.py`,
`tests/eval/test_rung_seat_off_window.py`).

## Control arm: wiring owed (R151)

The dense control arm is `v6_live2_ls`. That is a ruled fact (R148), not a measurement this
phase took.

**No dense eval result exists, and none can be produced at HEAD.** The encoding declares
`policy_pool='legal_set_scatter_max'`, and the decode entrance refuses it by name at exactly
one site, `src/mantis/eval/worker.py:87-98`, raising `EvalDecodeUnsupportedError` whose
message carries the encoding name, the declared pooling channel, the dropping behaviour it
will not perform, and ADJ-WP12R-4. The refusal is correct behaviour, not a defect; the gap is
the missing adapter, tracked as CARD-DENSE-EVAL-ADAPTER.

Coverage: `not_run` — refused by design, adapter owed. Nothing on the mint path consumes a
dense eval result: the promotion bar is graph candidate-versus-best and the external
instrument is the vendored engine plus the uniform floor. The first consumer is the post-mint
Stage 0 re-baseline, which cannot open until the card lands and the LAW-10 anchors are
re-measured on it.

## Ladder liveness: unverified in CI (R169)

2/6 resolve locally; liveness unverified in CI; verified at box preflight.

*Status: `not_run` — box preflight pending (WP12-R Phase D rider). No sealbot rung has been
observed playing.*

Resolution and execution are different properties and the split is the substance of the
claim. Resolution is a property of mantis code plus the local filesystem: it has a producer
that runs in CI, in both directions, and CI runs it. Execution is a property of a built
C++/pybind11 extension running, and it has no producer CI can run — `vendor/external/` is
gitignored, so the branch that would exercise it can never be taken there. The correct word
until the box rider returns its four measurements is therefore `not_run` — and `not_run` is a
RESULT, not an absence.

The box rider is specified in the WP12-R Phase A design and rides the Phase D box session. It
returns four measurements: rules agreement, the depth receipt, determinism, and one scored
round. All four passing upgrades the line above IN PLACE to the verified text; anything less
upgrades it to one of the two unverified texts, and the difference between "we did not look"
and "we looked and it was wrong" is preserved by using different words for them.

Those two unverified texts can both look applicable at once — a rider that stopped early
*because* something did not hold satisfies "fewer than four returned" and "measured, and it did
not hold" together. **The tiebreak is not a judgement call and it does not live here: the
WP12-R Phase A design, section 3.6, states it, and it governs.** Read it before editing the
line above; the marker recorded against the failing measurement decides, never the count of
measurements returned. The discriminator that separates this section from the control-arm
section is the dated status line below the heading, and only this section carries one.

## What is NOT decided here (S-16)

Three items are deliberately absent, and each is a mint-prereg row rather than a value chosen
by the porting work (R119):

1. the random-floor game count — see the floor section above;
2. whether run5 re-seeds, and to what — the run seed, the gate seed and the ladder bootstrap
   seed are three separate authorities, and the gate seed also selects which openings are
   played, so changing it changes the games and not merely the resample draw;
3. the external win-rate value the monitor thresholds read — it becomes a float the moment a
   sealbot ladder entry records at least one game, with no producer change; the thresholds
   themselves are already minted and are not touched here, and the hard abort that reads it
   stays disarmed by decision.

This document names the bar. The mint prereg arms it.
