# EVAL GATE MEMO (independent review, 2026-09-15) — sims, concurrency, the escalation rule, the eval ply cap, cadence

Status: a REVIEW for the operator's ruling, written by an independent agent against the run7 record; nothing here is minted. Its correction of `79d832cd`'s premise (run6 and the frontier ran at 256) is applied in place in `run_config_schema.md` v30 and `configs/run6.yaml`.

# MEMO — run7's eval round: gate sims, concurrency, escalation rule, ply cap, cadence

Independent review, 2026-09-15. No repo file edited, no box contact. Everything measured below was
read from the local mirror (`mantis-mirror/run7/logs/eval_spool.work/run7/r00000{1..5}_*`,
`events_run7_seg0001.jsonl`), the tree at `dev` (`79d832cd`), and the four measurement records the
task names. Simulations are in the scratchpad (`gate_sim.py`, `gate_grid.py`, `gate_sens.py`,
`book_power.py`, `cap128.py`, `rounds.py`, `trainer_rate2.py`).

**One fact the whole memo rests on, read from the tree:** a promotion moves ONLY the deploy tag
(`best_model.pt` + the gate's anchor). Self-play actors sync to the learner's latest weights every
`train.actor_sync_cadence_steps: 2` steps and "NEVER from a gate decision" (`src/mantis/run.py`,
the `ActorSync` seam; `src/mantis/eval/promote.py`'s docstring). run7 is therefore already
AlphaZero/KataGo-style "no gating" as a TRAINER; the gate is an INSTRUMENT whose only products
are (a) which checkpoint carries the deploy tag and (b) the anchor the next reading is taken
against. Its error rates cost nothing in training; its wall costs the trainer ~5 % of steps/h and
— when a round overruns the cadence — a skipped point and a killed rung. That reorders every
question below: the gate's job is to be cheap and readable, not to be a strict filter.

**A commit landed mid-review** (`79d832cd`, another session): `eval.max_plies: 128` as its own
minted row and `screen_confirm_lo 0.44 -> 0.5`, for run7's resume. Section 4 and 7 speak to it;
its "128 (run6's cap)" premise is corrected in §4.

---

## 0. The round arithmetic, measured

| round | anchor | screen (80) | confirm (128) | pooled / CI-lower(re-centred) | promoted | gate wall | s/ply | plies/game | rung (288) | rung wall | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 @3k | 0 | 54–26 = 0.675 | 95–33 = 0.742 | 0.716 / +0.151 | YES | 5 954 s | 0.50 | 57 | 211/288 = 0.733 | 2 142 s (0.19 s/ply, 38 plies) | 8 131 s |
| r2 @6k | 3k | 42–38 = 0.525 | 73–55 = 0.570 | 0.553 / −0.015 | no (CI) | 10 319 s* | 0.69* | 72 | 198/288 = 0.688 | 2 456 s (42 plies) | 12 816 s |
| r3 @9k | 3k | 51–29 = 0.637 | 70–58 = 0.547 | 0.582 / +0.015 | YES | 8 002 s | 0.58 | 66 | 193/288 = 0.670 | 4 432 s (0.25 s/ply, 60 plies) | 12 483 s |
| r4 @12k | 9k | 36–41–3 = 0.469 | 50–70–8 = 0.422 | 0.44 / — | no | 12 311 s | 0.64 | 92 (p90 215, 11 at cap 256) | killed in the rung | — | 14 400 (timeout); 15k kick skipped |
| r5 @18k | 9k | 35–41–4 = 0.463 | in progress | | | | | 85 (p90 163) | | | |

\* r2 shared the card with the TT A/B's cells.

- The promotion test as implemented (`src/mantis/eval/aggregate.py::gate_promotion_decision`) is
  NOT "confirm WR ≥ 0.55": it is pooled(screen+confirm) WR ≥ 0.55 AND the pair-bootstrap 95 %
  lower bound > 0.50. At 104 pairs with ~45 % seat-decided pairs that lower-bound test binds at
  pooled ≈ 0.567 — r2 read 0.553 and was refused by the CI, not by the bar.
- Seat-decided pairs (both games of an opening won by the same seat): gate 32–59 %, rung 38–47 %.
  Under the pair model μ = s/2 + (1−s)·q, a seat-decided share s caps the reading at 1 − s/2
  (0.775 at s = 0.45) and multiplies the SIGNAL of a skill gap by (1−s).
- Per-ply cost rose with game length inside the same regime: 0.50 → 0.58 → 0.64 s/ply as mean plies
  went 57 → 66 → 92 (r1, r3, r4). The radius-8 axis graph grows with the stone count, so the late
  plies of a long game are the expensive ones. Wall is games × plies × c(plies), superlinear in length.
- Trainer beside a round: hours inside a round averaged 821 steps/h vs 868 outside (events, hours
  2–22; noisy, ±150). "Second-order" is confirmed. What is NOT second-order: at 850 steps/h a
  3 000-step interval is 12 700 s, and a round now costs 12 500–16 700 s at r4's regime, so the
  eval child runs ~100 % of the time, every other kick is skipped as busy, and the timeout kills
  the rung (r4 lost its 288-game point AND the 15k point).
- The gate and the rung already disagree in sign over r1→r3: the gate promoted 9k over 3k (0.582,
  CI-lower barely > 0.5) while the rung fell 0.733 → 0.688 → 0.670 (CIs ±0.05, just overlapping).
  F-30 (promotion rewards anchor-exploit) is on the register; this is an observation for the
  operator, not a finding.

---

## 1. Gate sims: 512 vs 256 vs 128 (promotion gate only)

**Recommendation: 256 for the gate, 512 stays for the rung, done through a schema split
(`deploy.search.sims` 512 for the deploy tag and the rung, `eval.gate.sims` 256 for the gate), by
an amendment naming R351(c)'s "deploy and eval PUCT-512".** Not 128.

Reasoning.

1. *What the decision is for.* Relative: is the candidate better than the anchor. The absolute
   bar is the rung (sealbot_d5 at 512, strix at 15k). The contract's own definition of
   deploy-matched (`docs/contracts/eval_instrument.md`) is "both sides built by the SAME
   constructor at the SAME simulation count" — a 256-vs-256 gate satisfies its letter; what
   changes is that the gate's count no longer equals the deploy count, which is what R351(c)
   enacted and what the amendment must say.
2. *Does a lower-sims gate rank nets the same way?* The frontier's grid (PUCT vs sealbot_d5, 288
   games/cell, run6 3k/13k/18k/25k) is the only same-nets-three-sims data on record:

   | net | 128 | 256 | 512 | lift 128→256 | lift 256→512 |
   |---|---|---|---|---|---|
   | 3k | 0.431 | 0.542 | 0.628 | +0.111 | +0.086 |
   | 13k | 0.464 | 0.538 | 0.599 | +0.074 | +0.061 |
   | 18k | 0.568 | 0.646 | 0.774 | +0.078 | +0.128 |
   | 25k | 0.497 | 0.601 | 0.689 | +0.104 | +0.088 |

   Ranking 18k > 25k > {3k, 13k} holds at all three counts; the only swap (3k/13k) is a pair whose
   CIs overlap at every count. The sims lift is +6 to +13 pp per doubling on every net — within
   the ±5 pp per-cell resolution it is net-independent, which is the condition under which a
   256 reading ranks the same as a 512 reading. This is a sealbot-referenced check, not a
   net-vs-net one; the net-vs-net transfer is the bridging measurement below.
3. *The provenance of LAW-15 is about KIND, not count.* F-01 (static probes vs MCTS-matched eval),
   F-20 (wall-clock vs fixed-depth bars), F-30 (anchor-exploit), F-48/F-50/F-51 (the Gumbel head
   read the same net 24 pp low and INVERTED a head-to-head; sims made it worse). Under PUCT the
   count moved every net uniformly. The deploy-matched principle matters where the head could
   change the verdict — it protects the constructor, the argmax head, the leaf batch — and for
   the rung's absolute level. A gate at 256 keeps all of that.
4. *Noise vs games.* The gate's difficulty is the seat-decided share and the pair count, not the
   sims. At half the per-ply cost the same wall buys ~2× the pairs: 208 games at 256 ≈ 104 at
   512 in wall, and 208 pairs would shrink the CI half-width from ±0.067 to ±0.047 — a bigger gain
   in resolution than anything the sims give.
5. *Why not 128.* At 128 the frontier's readings compress toward 0.5 (0.43/0.46/0.57/0.50): the
   net-to-net spread is 14 pp vs 18 pp at 512, and the 3k/13k/25k trio is indistinguishable at
   288 games. 128 is also two doublings from the deploy count and one from the self-play fast arm
   (64) — a reading of a regime nothing plays. The operator's "a lower-sims net being weaker and
   improving is also data" is true of the RUNG (more headroom before saturation), not of the
   relative gate, whose reading is already bounded by 1 − s/2 regardless of sims.
6. *Precedent.* KataGo gated at 300–400 nodes while its self-play full searches were 600, with
   noise off, temperature 0.5 and resignation "to minimize noise" (Appendix E of Wu 2019); AGZ
   gated at self-play's 1 600 sims and argmax. Both gate at or below the search that generates data.

*Wall saved.* Gate block at r4's regime: 12 300 s at 512 → ≈ 6 800 s at 256 (0.55× per ply: the
search halves, the per-ply root build and pipeline latency do not) → ≈ 3 700 s at 128. The rung
(288 games, half its plies sealbot's on CPU) stays 2 100–4 400 s at 512.

*Cost in measurement.* A bridge: two run7 gate pairs already read at 512 (9k-vs-3k, pooled 0.582
promoted; 12k-vs-9k, 0.44 rejected) replayed at 256, 208 games each, on frozen checkpoints:
≈ 2 × 208 × 70 plies × 0.32 s ≈ 2.6 h of side-load at concurrency 8 beside the trainer — scheduled
BETWEEN rounds (the TT A/B's two cells beside r2 cost that round 4 700 s). Pre-registered reading:
the 256 verdicts agree in sign with the 512 verdicts and the paired deltas' CIs include 0; a
disagreement in sign on either pair keeps the gate at 512 and files the finding.

---

## 2. Gate concurrency 8 → 16

**Recommendation: hold at 8 until one A/B measures the two sides of the ledger. Do not change it
on the same commit as anything else.**

What the numbers say about where the child's bound is.

- Serial and idle (the frontier's PUCT-512 tail, concurrency 1): ≈ 40 s per rung game of ≈ 45
  plies, i.e. ≈ 1.3 s per candidate ply once sealbot's ≈ 0.5 s searches are taken out. Idle at
  concurrency 8 the rung reads 4.4–5.2 s/game (the frontier's "288 games in ≈ 25 min";
  CARD-SEALBOT-GIL-SERIAL's 1.7×) — 8–9× the serial rate, i.e. near-linear scaling: on an idle
  card the child is bound by per-stream round-trip latency (graph build + a ≤ 8-leaf forward), not
  by the GPU. Beside the trainer the same 8 streams get ≈ 60 % of that on the rung (7.4 s/game)
  and 28–59 s/game on the gate, whose plies are all net plies (2.0 plies/s ≈ 1 000 sims/s ≈ a
  quarter of the trainer's own 3 800 leaves/s).
- The trainer's server runs 3 800 leaves/s at 32 workers (PERF-A4) with the GPU at ~95 %. The eval
  child is a SECOND CUDA context on the same card; without MPS two contexts time-slice, and a
  small batch from the child pays a slice's latency for little work. The child's leaf batches are
  ≤ 8 per game × 8 games = 64 per call at most, and the gate is the one phase where every stream
  waits on inference all the time.
- CPU: 24 cores against 32 worker threads + the server thread + the trainer + the child's 8 game
  threads + its server + leaf builders. The Rust search releases the GIL (`py.detach` in
  `crates/mantis-bridge/src/mcts.rs`), so the child's threads can run in parallel; the frontier at
  4–12 cells read CPU 81 % / GPU 53 % on an idle box, which says the CPU side is not free either.
- Memory: the child's peak allocated was 1.14 GB in the gate block at concurrency 8 (r3's
  `device_memory`); 16 would be ≈ 2.3 GB. Headroom on the 16 GB card beside the trainer's 8.90 GiB
  allowance and the self-play server's reserve is not on the record.

So beside the trainer the child is most likely GPU-CONTENTION-bound (each call pays a slice's
latency under time-slicing between two CUDA contexts), not CPU-bound: concurrency 16 would double
the leaves per call and could raise the child's plies/s by up to ~2× if the idle near-linear
scaling extends past 8, but every extra slice is taken from the trainer's ~95 %. Whether the sum
is positive is exactly the thing no argument settles.

Measure before changing (one frozen pair, e.g. 9k-vs-3k, 80 games at 8 then 80 at 16, beside the
trainer, between rounds):

1. The child's aggregate plies/s (from `progress.txt` timestamps per phase) — the gain side.
2. The trainer's steps/h during each arm (from `trainer_step` events) — the cost side.
3. The child's `InferenceServer.batch_timing_snapshot()` (occupancy histogram, `queue_wait`,
   `collate`) — the instrument exists in `src/mantis/selfplay/inference_server.py` and is NOT
   emitted by the eval worker today; a LAW-18 gap to close first (the lever must log its own
   fire-rate), otherwise the A/B says "faster/slower" without saying why.
4. `nvidia-smi pmon -s um` per-process SM % and memory during each arm; the child's process CPU %
   (`ps -o %cpu`) — ≥ 600 % says CPU-bound, ~100–200 % says waiting on the GPU.
5. Round peak memory (`eval_round_device_memory`) at 16.

Decision rule to pre-register: adopt 16 iff plies/s ≥ 1.5× AND trainer steps/h ≥ 0.90× the
concurrency-8 arm. A bad outcome looks like the TT A/B beside r2: the child gains 20 %, the trainer
loses 15 % (1 400 → 470 steps/h was the extreme with two extra cells), total throughput falls, and
— worse — the child OOMs in the gate block, which ends the round as `eval_broken` and, if it is the
terminal round, rc 48. A second, quieter bad outcome: the games change (bf16 batch composition
already makes concurrency-8 games non-reproducible game-for-game, SEALBOT_TT_AB §D.3) — no new
loss, but any A/B that spans the change must be read at the level, never the game.

---

## 3. The escalation rule

**Current rule (as implemented):** screen 40 pairs; escalate iff WR_screen ≥ `screen_confirm_lo`
(0.44 at launch, 0.50 in the resume mint); confirm 64 pairs; promote iff pooled ≥ 0.55 AND
pair-bootstrap lower > 0.50. Every screen so far escalated at 0.44 (0.675, 0.525, 0.637, 0.469,
0.463); at 0.50 the last two would not have.

**Proposed rule: a GSPRT over pair outcomes** (the pentanomial-style test Fishtest and the
leela-chess match server use — Van den Bergh's normalized-t LLR, formula (4.14):
`LLR = (n/2)·log((1+(t̂−t0)²)/(1+(t̂−t1)²))`, `t̂ = (μ̂−½)/σ̂` over pairs), H0: μ = 0.52 vs
H1: μ = 0.62, α = 0.05, β = 0.10, checked every 8 pairs (one concurrency batch) from 16 to 104
pairs, and at the 104-pair maximum decided by the sign of the LLR. The pairs are drawn from one
104-pair window of the sha-pinned book exactly as today (screen window + confirm window), so the
book, the seat-swap and the trajectory dedupe are unchanged.

Simulated on the pair model calibrated to run7 (s = 0.45; 40 000 rounds per cell):

| true WR vs anchor | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 |
|---|---|---|---|---|---|
| **current (lo 0.44)** P(promote) | 0.00 | 0.03 | 0.29 | 0.81 | 0.99 |
| E[games] | 150 | 186 | 204 | 208 | 208 |
| P(escalate) | 0.54 | 0.83 | 0.97 | 1.00 | 1.00 |
| **current, lo 0.50 (the resume mint)** P(promote) | 0.00 | 0.03 | 0.28 | 0.80 | 0.99 |
| E[games] | 109 | 150 | 187 | 204 | 208 |
| **SPRT 0.52/0.62, α .05 β .10, 16–104 pairs, LLR sign at max** P(promote) | 0.00 | 0.03 | 0.29 | 0.79 | 0.98 |
| E[games] | 72 | 107 | 151 | 151 | 103 |
| **SPRT 0.50/0.60, α = β .05** (more permissive) P(promote) | 0.00 | 0.09 | 0.50 | 0.92 | 1.00 |
| E[games] | 101 | 148 | 173 | 140 | 89 |
| **fixed 80 games, no confirm, point ≥ 0.55** P(promote) | 0.05 | 0.23 | 0.54 | 0.84 | 0.98 |
| E[games] | 80 | 80 | 80 | 80 | 80 |

The first SPRT row reproduces the current rule's error curve to within 0.02 at every strength and
plays 26–50 % fewer games (mean over 0.50–0.65: 128 vs 202). Sensitivity: at s = 0.30 it reads
0.05/0.31/0.77 (games 126/164/165), at s = 0.60 0.02/0.25/0.80 (85/131/129) — the curve moves by
≤ 0.05 and the saving holds. With 10–20 % of games capped as draws (a 128 cap at r4's lengths) its
power at 0.60 falls to 0.71/0.60 while the current rule's falls to 0.77/0.72 — both lose; the
cap's compression is the cause, not the rule (§4).

What the table also says, and the operator should read before choosing:

- The current rule is effectively a fixed 208-game test with a bar at ≈ 0.567: escalation fires
  83 % of the time even for a candidate that is exactly as strong as the anchor, so the "screen"
  saves almost nothing in the regime the run is in. The 0.50 threshold in the resume mint saves
  20–40 games only when μ ≤ 0.55 and nothing above.
- 208 games at s = 0.45 cannot separate 0.55 from 0.50: power 0.29 at a true +5 pp. No sequential
  rule fixes an underpowered test; SPRT saves games when the truth is FAR from the band (μ ≤ 0.45
  or ≥ 0.62), which is where 3 of 5 rounds sat.
- The book is worth more than the rule. At fixed skill (q = 0.65 of player-decided pairs) the
  current rule promotes with P = 0.51 / 0.63 / 0.74 / 0.82 at s = 0.60 / 0.45 / 0.30 / 0.15. Halving
  the seat-decided share (the balanced-book card, book_v2) buys +11 pp of power at 208 games — as
  much as the rule change buys in games. Leela Zero's 2019 selection-process proposal argued the
  same (a panel of fair openings, the Nunn-positions precedent), and chess testing moved to
  measured-balance books for the same reason in reverse (too many draws; UHO).
- Since promotion feeds nothing but the deploy tag, the more permissive SPRT row (0.50/0.60,
  α = β = 0.05: promote a true +5 pp half the time, a true +10 pp 92 %) is arguably the better
  instrument: the anchor keeps up with the run, so each reading answers "better than the last
  credible net" rather than drifting to 0.46 against a stale anchor as r5 does. The Leela Zero
  simulation (Friday9i, issue #1524) found a 50 % threshold "almost equivalent to no gating" and
  worth +25 % Elo/net over 55 % — but that was a gate that FED self-play; here the only downside of
  a permissive bar is a deploy tag that moves on noise once in ten rounds.

Governance cost: the gate pair statistic is in LAW's protected set ("gate pair statistics",
R346 §1(d)), so the SPRT needs a ruling that names it, the schema keys (`mu0`, `mu1`, `alpha`,
`beta`, `min_pairs`, `max_pairs`, `check_every_pairs`) replacing `screen_games` / `confirm_games`
/ `screen_confirm_lo` / `promotion_winrate` with live consumers (LAW-08), a producer test and a
mutation self-test (LAW-07), and the contract doc (gate 13). Until it lands the resume mint's 0.50
is a harmless partial.

---

## 4. A per-eval ply cap of 128, capped = draw (0.5)

**Recommendation: do not cap at 128 for the wall; if the row stays at 128 (it is minted as of
`79d832cd`), state the unit change on the dashboard and bridge it, and keep `terminal=ply_cap`
distinct from a draw in every record.**

Measured on run7's own gate games (what a 128 cap WOULD have done):

| round | phase | games > 128 | WR as played | WR with > 128 scored 0.5 | plies saved |
|---|---|---|---|---|---|
| r1 | screen / confirm | 4 / 8 (5–6 %) | 0.675 / 0.742 | 0.662 / 0.719 | 3 % / 5 % |
| r2 | screen / confirm | 9 / 14 (11 %) | 0.525 / 0.570 | 0.519 / 0.562 | 5 % / 5 % |
| r3 | screen / confirm | 4 / 12 (5–9 %) | 0.637 / 0.547 | 0.650 / 0.562 | 5 % / 5 % |
| r4 | screen / confirm | 17 / 28 (21–22 %) | 0.469 / 0.422 | 0.494 / 0.453 | 15 % / 18 % |
| r5 | screen | 13 (16 %) | 0.463 | 0.456 | 13 % |
| r1–r3 | rung | 0 / 1 / 6 of 288 | 0.733 / 0.688 / 0.670 | 0.733 / 0.686 / 0.674 | 0–1 % |

1. *Sound for a relative gate?* Symmetric censoring: unbiased only if long games favour neither
   side. They do not, round by round: the > 128 games read 0.75/0.88 for the candidate in r1 and
   0.25/0.33/0.36 in r3/r4. The cap moves the reading by −0.02 to +0.03 and compresses the signal
   by the capped fraction (μ → 0.5 + (1−d)(μ−0.5)); at r4's 20 % that is a fifth of the signal on
   a test that already has power 0.29 at +5 pp. It also shifts the rule's own error curve (the
   table in §3: power at 0.60 0.81 → 0.72 at d = 0.2).
2. *The saving is small.* A 200-ply game still costs its first 128 plies, and those are the ones
   the child plays anyway: 3–5 % of plies in r1–r3, 13–18 % in r4/r5. Because per-ply cost grows
   with stones, the wall saving is somewhat larger than the ply saving — perhaps 1.5× — still
   ≤ 25 % at r4 and ≤ 8 % in a normal round. At r4's regime the resume mint's gate block would still
   cost ≈ 10 000 s and, with a 4 400 s rung, would still brush the 14 400 s timeout.
3. *The rung is unaffected.* 0–6 of 288 rung games exceeded 128 in r1–r3; the shift is ≤ 0.004.
4. *Comparability with run6 — the premise needs correcting.* `configs/run6.yaml` minted
   `selfplay.max_game_moves: 256` (commit `44ffe43f`, "ply cap 256"), and at run6's tree
   (`0f20896e`) `run.py:758` passed `max_plies=config.selfplay.max_game_moves` to the eval child —
   run6's rounds and the frontier's cells were capped at 256, not 128. The mirror confirms it: the
   frontier's `ck13k_puct512` and `ck13k_puct128` cells carry 18 and 17 games over 128 plies
   (max 256). run6's own in-run eval games never exceeded 90 plies (Gumbel-head games; 3 701
   progress rows, max 90), so for run6's ROUNDS the cap is moot; for the frontier's PUCT columns
   — run7's rung comparators — 6 % of games ran past 128. The "128 = run6's cap" in `79d832cd`
   is the pre-mint hardcoded constant the `rounds.py` comment records, not what run6 played under.
   Net effect on comparability: ≤ 1 pp at the rung, and none if the cap is stated on the record.
5. *Pitfalls.* (i) A capped game is a NON-RESULT the adjudication module deliberately keeps apart
   from a draw (`TERMINAL_PLY_CAP` vs `"draw"`; `ply_cap_adjudication: null`); scoring it 0.5 in
   the WR is fine, collapsing it INTO `"draw"` in the record is the mistake that "once made the
   eval instrument's whole outcome channel a constant" (that module's docstring). (ii) The F-02 /
   F-22 / F-52 ply-cap attractor history: a rising cap rate in EVAL is a signal to keep visible
   (`train.ply_cap_abort` reads self-play only). (iii) Long gate games are the census's finding —
   two sides that block every four and win only by built double threats — so capping them throws
   away the games that most distinguish two competent nets. (iv) If a cap is wanted for the wall,
   the honest instrument is adjudication by the existing `longest_run_margin` criterion at the
   cap rather than a draw — a different instrument, to be measured, not minted on a resume.

---

## 5. Cadence: 3 000 vs 6 000 vs decoupling the rung

Arithmetic at 850 steps/h (the current regime; 1 500–1 750 alone in hour 1 has not recurred):

| cadence | interval | round must finish in (0.75× margin) | round options that fit |
|---|---|---|---|
| 3 000 | 12 700 s | ≤ 9 500 s | SPRT gate at 256 (≈ 150 games × 70 plies × 0.32 s ≈ 3 400 s) + rung 512 (2 100–4 400 s) = 5 500–7 800 s. The current 208-game gate at 512 (12 300 s at r4) does NOT fit even with the 128 cap. |
| 6 000 | 25 400 s | ≤ 19 000 s | anything up to the 14 400 s timeout; 4 points per 25k block — below R350(d)'s "≥ 8 points at 25k" for witness (iii). |
| 3 000, rung every 2nd round | — | gate rounds ≤ 9 500 s, rung rounds ≤ 14 000 s | gate at 512 with SPRT (≈ 8 900 s) fits the gate-only rounds; the rung rounds need the 256 gate or the timeout raised. Needs a new key (the ladder has no "rung every k rounds"; `calibration_every_k_rounds` is for SATURATED rungs). |
| decoupled rung (offline) | — | gate only in-run | the only GPU is the box; "offline" means between rounds on the box or a block-end replay: the frontier read 31 cells in 8.3 h on the idle card (≈ 16 min per 288-game cell at 4–12-way), so a block's 8 rung points replay in ≈ 2 h idle. On the operator's CPU machine a 512-sim rung is ~10 h per point (0.45 s/ply idle GPU ≈ 3–5 s on CPU) — not viable per round. |

Recommendation: keep 3 000 and make the round fit it (gate at 256 with the SPRT; rung at 512),
because the block's witness needs ≥ 8 points and because a skipped kick is a lost point at the
step that mattered (15k was the strix point's neighbour). The rung stays in-run: its consumers are
LAW-18's live view and the deferred `sealbot_wr_abort` (warn-only; going offline would make a
monitor input producer-less, R4/LAW-07, and `sealbot_wr_gate_skipped` would fire every step). The
block-end replay on frozen checkpoints is the DEFINITIVE series regardless (R353(a): frozen
checkpoints are re-readable), which is also where the sealbot rung at 256 — more headroom than
512's ≈ 0.78 ceiling — can be read for every checkpoint at once without breaking run7's in-run
series. If the gate cannot move to 256, then 6 000 with the full round is the fallback; do not
run 3 000 with a round that overruns — that is the r4 shape, a killed rung and a skipped point.

`round_timeout_sec`: keep 14 400, but it is the wrong guard for this failure — the timeout should
be < the cadence interval, or the kick should wait rather than skip; both are code changes with
their own cards. Minimum: the mint preflight should refuse a round whose EXPECTED wall (games ×
measured plies × measured s/ply) exceeds 0.75 × cadence interval at the launch-time trainer rate.

---

## 6. How others gate and size evaluation (research digest)

| system | promotion rule | games / sims | book / balance | notes |
|---|---|---|---|---|
| AlphaGo Zero (Silver 2017, Methods) | candidate vs current best, "> 55 % (to avoid selecting on noise alone)" | 400 games, 1 600 sims, τ → 0 (argmax); a checkpoint every 1 000 steps | no book; self-play τ = 1 for 30 moves supplied diversity | gate at the SELF-PLAY search count; promoted net generates the next 25 000 games |
| AlphaZero (Silver 2017/2018) | none: "self-play games are generated by using the latest parameters ... omitting the evaluation step and the selection of best player" | — | — | the regime run7's actors are already in |
| Leela Zero (Go) | candidate vs best, 55 % at 400 games, later an SPRT-shaped early pass (dynamic 57 % → 55 % from 300 to 400 games) | 400, same visits as self-play | no book; a 2019 proposal (#2143) argued for a panel of fair openings (Nunn precedent) | a 2018 simulation (#1524) found a 50 % threshold ≈ "no gating" and +25 % Elo/net over 55 % |
| leela-chess (2018, pre-Lc0) | SPRT on match games, bounds [−20, 20] Elo, α = β = 0.05 | ≈ 250–457 games average, vs the earlier fixed 400 | | "promoting a slightly worse network is not a disaster and adds diversity" |
| Lc0 (later runs) | AlphaZero-style; match games at 800 playouts "measure progress" | | | FAQ text; no gating in the promotion sense |
| KataGo (Wu 2019, App. E) | "win at least 100 out of 200 games" — a 50 % bar | 200 games at 300–400 nodes vs 600-visit self-play full searches; noise/forced playouts/cap oscillation OFF, τ 0.5, resignation, komi fixed | rules/board size still randomized | "fairly lightweight"; 2 of 28 V100s on gating; the docs call gating OPTIONAL — "faster and will save compute power ... works perfectly fine without it" |
| ELF OpenGo (Tian 2019) | none (AZ-style, immediate deployment) | progress measured vs a prototype model at 1 600 rollouts | | found "significant variance in the model's strength as training progresses" |
| MiniGo | none ("the AZ approach, with no evaluation of the models"); post-hoc rating matches, auto-paired by rating uncertainty | | | "filter bad models" (best of 4) used in analysis only |
| OpenSpiel AlphaZero | none; evaluators "continually play games against a standard MCTS+Solver" scaled by sims | | | a fixed-search external bar, like sealbot |
| Stockfish Fishtest | GSPRT, pentanomial (paired, colour-swapped) model; bounds in normalized Elo so duration is book-independent; α = β = 0.05 | expected duration ≈ 10⁶ / (Δ normalized Elo)² games (Van den Bergh) | UHO unbalanced openings chosen to cut draws; books ranked by normalized-Elo sensitivity | the paired-game variance is ~15 % below the trinomial's |

Fixed-depth minimax as a bar: OpenSpiel's MCTS+solver at fixed sims, AlphaGo's Pachi/Fuego/GnuGo
at fixed playouts, KataGo's Leela/ELF nets at fixed 1 600 visits — the fixed-search external
reference is the norm; a classical fixed-depth alpha-beta bar is unusual in Go/chess RL but is
exactly what a k-in-a-row game affords (Connect6's own fairness work tested ~1 000 opening
templates by self-play; I-Chen Wu). Nobody on this list uses a uniformly random 4-ply book;
where a book is used it is measured for balance or deliberately unbalanced with the
pentanomial model absorbing the seat effect — which is the LAW-04-style "count distinct games"
posture with the seat effect priced in, and what `book_v2` should be measured against.

---

## 7. Recommended minted values for the next block

| key | value | grounds |
|---|---|---|
| `deploy.search.sims` (NEW, split from the gate as R351(c) split `kind`) | 512 | the deploy tag and the rung stay at the count the ladder plays; the frontier's 512 column and run7's r1–r3 series are the comparators |
| `eval.gate.sims` (renamed from `eval.gate.deploy_sims`) | **256** | §1: ranks preserved across 128/256/512 on the frontier's 4 nets (one tied swap); lift +6–13 pp uniform; gate block 64–85 % of the round wall halves; KataGo precedent; amendment names R351(c) |
| `eval.sealbot_model_sims` | 512 | series continuity (0.733/0.688/0.670) and the frontier's 512 column; saturation watch: two consecutive rounds with CI-lower ≥ 0.72 → d6 rung or a bridged 256 series |
| gate rule | SPRT μ0 0.52 / μ1 0.62, α 0.05, β 0.10, check every 8 pairs, 16–104 pairs, LLR sign at max | §3: same error curve as today (0.03/0.29/0.79 at 0.50/0.55/0.60), 27–50 % fewer games; a ruling naming "gate pair statistics"; or the permissive 0.50/0.60 α = β 0.05 if the operator wants the anchor to track the run |
| `eval.gate.screen_confirm_lo` (until the SPRT lands) | 0.50 | the resume mint's value; same error curve, saves 20–40 games only below 0.55; harmless |
| `eval.max_plies` | 256 preferred; if 128 stays, stated as a unit change and bridged | §4: 3–18 % of plies, ±0.03 at the gate, ≤ 1 pp at the rung; run6/frontier ran at 256 |
| `eval.concurrency` / `rung_concurrency` | 8 / 8 | §2: hold until the pre-registered A/B (plies/s ≥ 1.5× AND trainer ≥ 0.90×) |
| `train.eval_interval` / `checkpoint_interval` | 3 000 | §5: a round designed to ≤ 9 500 s fits; ≥ 8 points per 25k block for witness (iii) |
| `eval.round_timeout_sec` | 14 400, plus a preflight refusal when expected wall > 0.75 × interval | the timeout is not the guard for an overrunning cadence |
| `eval.ladder.round_games` / `games_max` | 288 | ± 5 pp; unchanged |
| `eval.gate.opening_book` | `book_v1_s20260625_p4` until `book_v2` is measured | comparability; §3: halving the seat-decided share is worth +11 pp of power — the best single lever after the wall |
| eval-child LAW-18 counters (code, not a key) | emit `batch_timing_snapshot()` per phase | required before the concurrency lever is touched |

---

## 8. What is NOT known and would need measuring

1. **Net-vs-net transfer 512 → 256.** The frontier's ranking evidence is sealbot-referenced. The
   bridge (§1): two run7 gate pairs at 256, ≈ 2.6 h of side-load between rounds.
2. **Where the eval child's bound is** (GPU time-slicing vs CPU vs the child's server thread): the
   §2 A/B with per-process SM %, child CPU %, and the server's occupancy/queue-wait counters,
   which the eval worker does not emit today.
3. **Card memory headroom** beside the trainer for a 16-way child (~2.3 GB peak expected).
4. **Per-ply cost along a game** (does c(plies) really grow with stones, and by how much): per-move
   timing in the game record; `move_sims` exists, time does not.
5. **The seat-decided share as a function of strength gap and of the book** — it ranged 32–59 %
   in the gate; `book_v2`'s design needs the per-opening seat advantage measured (288 games per
   opening set at one net pair; the frontier's 144-opening window already has 33 cells to read
   it from, no box time).
6. **Whether the gate's verdicts and the rung's direction agree** over r1→r5 (§0's observation):
   a block-end replay of run7's 3k/6k/9k/12k/18k against ONE fixed anchor and against sealbot at
   256 and 512 would say whether the moving anchor or the rung is the one to trust.
7. **The trainer's rate beside a round** at concurrency 8 vs 16 (item 2's cost side) and whether
   the 850 steps/h regime persists (hour 1's 1 720 was buffer-fill).
8. **The SPRT's behaviour on capped draws** if `max_plies` 128 stays: the pentanomial outcomes
   0.25/0.75 enter σ̂ — simulated above, not measured on real pairs.

---

## Sources

- AlphaGo Zero, Methods "Evaluator" (400 games, 1 600 sims, τ → 0, > 55 %): https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
- AlphaZero (no evaluator; latest parameters): https://ar5iv.labs.arxiv.org/html/1712.01815
- KataGo, Wu 2019, Appendix E "Gating" (100 of 200, 300–400 nodes, noise off): https://arxiv.org/pdf/1902.10565 ; gating OPTIONAL: https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md ; g170 GPU allocation: https://github.com/lightvector/KataGo/blob/master/TrainingHistory.md
- ELF OpenGo (AZ-style, strength variance, 1 600-rollout progress measure): https://ar5iv.labs.arxiv.org/html/1902.04522
- MiniGo RESULTS (no evaluation of models; rating matches): https://github.com/tensorflow/minigo/blob/master/RESULTS.md
- OpenSpiel AlphaZero (evaluators vs MCTS+Solver at scaled sims): https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- Leela Zero: gating simulation, 50 % vs 55 %: https://github.com/leela-zero/leela-zero/issues/1524 ; match-resource discussion: https://github.com/leela-zero/leela-zero/issues/545 ; selection process / fair openings proposal: https://github.com/leela-zero/leela-zero/issues/2143 ; low-playout prescreen proposal: https://github.com/leela-zero/leela-zero/issues/667 ; 400-game / 55 % rule: https://github.com/leela-zero/leela-zero/issues/504
- leela-chess SPRT stopping condition ([−20, 20], α = β = 0.05, ≈ 250–457 games): https://github.com/glinscott/leela-chess/pull/174 ; Lc0 FAQ (match games at 800 playouts measure progress): https://lczero.org/dev/wiki/faq/
- Fishtest mathematics (GSPRT, pentanomial, normalized Elo): https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html ; Van den Bergh, "Comments on normalized Elo" (formulas (4.14), (3.4)): https://cantate.be/Fishtest/normalized_elo_practical.pdf ; pentanomial simulator: https://github.com/vdbergh/pentanomial ; paired-game correlation issue: https://github.com/official-stockfish/fishtest/issues/348 ; chessprogramming SPRT: https://chessprogramming.org/Sequential_Probability_Ratio_Test ; Match statistics: https://www.chessprogramming.org/Match_Statistics
- Wald SPRT / ASN: https://encyclopediaofmath.org/wiki/Sequential_probability_ratio_test
- Opening books: Stockfish book tests (normalized-Elo sensitivity per book): https://github.com/official-stockfish/Stockfish/issues/3323 ; UHO 2024: https://www.sp-cc.de/uho_2024.htm
- Connect6 fairness (opening templates tested by self-play): https://en.wikipedia.org/wiki/Connect_6 ; https://link.springer.com/chapter/10.1007/11922155_14
- cutechess-cli (SPRT, paired openings, adjudication flags): https://github.com/cutechess/cutechess/blob/master/docs/cutechess-cli.6
- In-repo: `docs/governance/LAWS.md` (LAW-04, -07, -08, -15, -18; the protected set), `docs/contracts/eval_instrument.md`, `docs/governance/RULINGS.md` R350–R353, `docs/governance/falsified.md` F-01/F-20/F-30/F-48/F-50/F-51/F-52, `docs/design/measurements/{STRENGTH_FRONTIER_1_2026-09-13, GAME_QUALITY_CENSUS_2026-09-14, STRIX_RUNG_2026-09-14, SEALBOT_TT_AB_2026-09-14}.md`, `src/mantis/eval/{aggregate,worker,promote,pipeline}.py`, `src/mantis/run.py`, `src/mantis/arena/adjudicate.py`, `configs/run7.yaml` at `79d832cd`, `configs/run6.yaml` at `44ffe43f` / `0f20896e`.
