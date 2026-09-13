# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase — RUN6 STOPPED under R350; STRENGTH-FRONTIER-1 RUNNING on the box; WARMSTART-2 / BC-3 LANDED on dev

**run6 is stopped.** One SIGTERM to the supervisor at 2026-09-13 08:15:12 UTC: `shutdown_save` at
step **35 084** (+0.5 s), the bundle `run6_00035084_e1563d16` complete, both processes gone in 32 s,
the 35 000 and 35 084 bundles receipted by one manual puller cycle (the user service
`mantis-puller-run6` was stopped first and stays stopped; the dashboard timer is idle). The run
had gone 10 083 steps past the block at the same regime. The stop was not orderly past the save:
`close_out` waited on the in-flight round-35 eval child and the supervisor's 30 s grace SIGKILLed
it — `CARD-STOP-DRAIN-VS-GRACE`, fixed on this line (a resumable stop now ABANDONS the round,
`eval_broken(abandoned)`). The last hour's two game shards are mirrored but unindexed.

**R350 is landed verbatim** (`5743cff6`) with its landing note, and ANNOTATED at its foot: (a)'s
control — "same BC net, ALL heads … 53 % → 79 % vs sealbot" — is not what the record says. Read
from the box before the frontier ran: the R342-era bursts boot `loaded_keys=46` (value head FRESH,
the same seam run6 used); the deploy head until `6ee52ca7` (2026-09-09) was a since-deleted
PUCT-tree/transformed-Q-root hybrid; and 72.5 % = 58/80 and 78.8 % = 63/80 are GATE-SCREEN
fractions (the candidate vs its own step-0 anchor), 53.1 % = 17/32 the one sealbot reading.
falsified.md **F-48, F-49**. (a)'s MECHANISM stands and the trough analysis supports it
(`docs/design/measurements/INVESTIGATION1_TROUGH_2026-09-13.md`: H(target) 0.10–0.14 nats all
block — the completed-Q target IS an argmax — and KL(target ‖ prior) rising 2.08 → 2.75 nats into
the trough with the targets no softer).

**STRENGTH-FRONTIER-1 is running on the box** (`tools/strength_frontier.py`, driver sha
`1e586293055d` on tree `0f20896e` — the block's own eval worker, head, book and sealbot; record
`docs/design/measurements/STRENGTH_FRONTIER_1_2026-09-13.md`; cells and results under
`/workspace/frontier/{phase1,phase2}/`, `summary.jsonl` per phase; nothing on the box depends on
a session). Landed so far (288 paired games, one shared opening window, pair-level 95 % CI):

| cell | WR | CI |
|---|---|---|
| `bc_full` (every head) PUCT-150 | **0.413** | [0.358, 0.469] |
| `bc_full` Gumbel-160/m16 | **0.177** | [0.135, 0.219] |
| `bc_tp` (run6's seam: value head fresh) PUCT-150 | **0.038** | [0.017, 0.062] |
| `bc_tp` Gumbel-160/m16 | **0.014** | [0.003, 0.028] |
| `ck3k` / `ck13k` / `ck18k` / `ck25k` Gumbel-128 | 0.205 / **0.031** / 0.233 / 0.167 | [0.160, 0.250] / [0.014, 0.052] / [0.191, 0.278] / [0.128, 0.208] |
| `ck3k` / `ck13k` / `ck18k` / `ck25k` PUCT-128 | 0.431 / **0.464** / **0.568** / 0.497 | [0.372, 0.483] / [0.408, 0.519] / [0.510, 0.625] / [0.444, 0.549] |
| `ck35k` Gumbel-128 | 0.149 | [0.108, 0.191] |
| `ck18k` Gumbel-256 | **0.111** | [0.080, 0.146] |
| `ck25k` vs `bc_full`, Gumbel-160/m16 (25k as candidate) | **0.438** | [0.389, 0.486] |

Two answers already: the KIND — the Gumbel deploy head reads the SAME net 24 pp below PUCT; and
the HEAD SET — the BC checkpoint's own value head (trained on the corpus outcomes, thrown away by
the seam) is worth 0.413 vs 0.038 at PUCT-150, and a BC prior searched over a random value head
is WORSE than the prior. And the NET: under PUCT-128 the block's nets sit at or ABOVE the BC net
(0.431 → 0.568 at 18k → 0.497), so the block trained a stronger net than it started with, read
by the wrong head; the 13k "trough" is a 3 % Gumbel reading of a net PUCT reads at 0.464. Under the Gumbel head
more sims read WORSE (18k: 0.233 at 128 → 0.111 at 256), and the run's own promotion instrument
scores the 25k net BELOW its prior (0.438 vs `bc_full` at Gumbel-160) while PUCT-vs-sealbot has
it above. Phase 2's 27 cells (3k/13k/18k/25k × {128, 256, 512} × {gumbel, puct},
`ck35k` Gumbel-128, `ck25k` vs `bc_full` at Gumbel-160 and PUCT-150) finish over the next hours;
the record's §B/§C/§E fill as they land. The verdict on kind/sims for run7's mint is the
record's, not this file's.

**Landed on dev this leg (R350(b), (e), the stop card), one commit line each — see the
Exit facts:** (b)(i) the seam copies EVERY tensor and `identity.warm_start.reinit` names what
stays fresh (REQUIRED; `configs/run6.yaml` re-minted with `[value_head]`, the truth of what it
booted; run7 mints `[]`), with the step-0 witness `net_param_hash(live) == net_hash` refusing a
dropped head; (b)(ii) BC-3's held-out VALUE line rides every BC pass (the strip's value head was
already trained on the corpus outcomes — R350's landing note); (b)(iii) `train.policy_loss_weight_schedule.warmup_steps`
(0 everywhere; run7 proposes 2 000) leaves the policy term OUT of the loss so AdamW cannot move
the prior, the BC route refuses a non-zero value; (b)(iv) `trainer_step.policy_kl_target_vs_prior`
and `policy_target_entropy` from step 0, and the trough halt `train.policy_loss_trough_abort`
(`null` everywhere; run7 proposes `{0.2, 3, 5000}`; manifest row DEFERRED, exit 49, bounded
cadence); (e) α = 1.0 rows leave the policy loss and its denominator unconditionally, counted on
the step event. **The loader now reads a stamp that predates a schema leaf as provenance** (logged,
not refused) — before that fix the three REQUIRED additions orphaned every run6 bundle and the BC
artifact (a review finding; the re-strip it prompted was reverted).

**Dispatcher state for a fresh session.** Open, in order: (1) read the frontier's `summary.jsonl`
files off the box into the record's §B/§C and write §E (kind, sims, the head set); (2) the run7
re-mint from `docs/design/measurements/RUN7_PREREG_2026-09-13.md` once the operator forwards the
filled rows (the kind and sims rows are the frontier's); (3) preflight stamp on the box, the 4 h
shakedown twin, START on the operator's word; (4) BC-3's re-run is OPTIONAL — the existing strip
carries a trained value head; the operator may pin it with `reinit: []`; (5) the owed items in
`CARDS.md`: the three-row α reconstruction, the self-play search-stats sample
(`CARD-SELFPLAY-SEARCH-STATS`), the poller race (`CARD-DRAIN-POLLER-RACE`), the supervisor grace
as a relation. The box: `/workspace/hexo-mantis` at `0f20896e` (the frontier's tree; carry this
line's commits over by bundle before any run7 work there), run6's record under
`/workspace/runs/run6/`, the R342-era logs under `/workspace/r342/`.

## PERF-A4 — the serving levers, merged at `41a5fea8` (branch `perf-a4`, red-teamed)

Record: `docs/design/measurements/PERF_A4_2026-09-11.md`. Four levers landed: check 14 with the GIL
released (+17 %), pinned `non_blocking` H2D + `output_size=`, `inference.compile_trunk` (+17–21 %),
the two-thread software pipeline (+52 %). Combined at 32 workers: leaves/s 1,831 → 3,804
(+108 %). The block measured **1 163 steps/h whole** at 32 workers on the box
(`RUN6_BLOCK_2026-09-12.md`). The edge codebook is NOT landed (device-dependent numerics).

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

`compose_run(burst_stop_step=)` is the eighth census parameter; a production config stamps at
`sync_lag` from a ≈ 100-step burst (≈ 12 min on the box). `CARD-STAMP-FLOOR` carries the rejected
options. Phase W is NOT re-run before a start (`CARD-PHASE-W-AT-GUMBEL`).

## Hold 2 — the completed-Q target in decided positions: RULED by R350(e)

All 25 α = 1.0 rows of the START-path burst sat at `moves_remaining == 1` in LOST positions
(F-45); the block read 4.98 per 1 000, flat. R350(e) EXCLUDES them from the policy loss from this
tree on (`exclude_alpha_full_rows`, before the denominator; `trainer_step.policy_rows_excluded_alpha_full`).
The three-row reconstruction R349(c) ordered is still OWED.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` on the operator's machine, receipts beside each artifact on the box, the
preflight's rc 16 reading them. run6's mirror is `~/Work/HeXO/mantis-mirror/run6` (36 bundles,
153 shards, the final bundle receipted at the stop). The puller service is STOPPED; the run is.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 (`tests/train/test_law06_cpu_carveout.py`).
The dev box runs the CPU wheel; the tier's authority is the box.

## Minted values — `configs/run6.yaml`

Re-minted through its own header on this line (the 32 deltas replay; `tools/config_diff.py
--from-header` MATCH): `identity.warm_start.reinit: [value_head]` (what run6 booted — the seam of
the time copied trunk + policy), `train.policy_loss_weight_schedule.warmup_steps: 0`,
`train.policy_loss_trough_abort: null` — the last two at their template values, the OFF postures
run6 ran. The warm-start checkpoint stays `checkpoints/bc/run6_00006500_ca1afb71.ckpt` (net hash
`2e72abd4…`, 50 tensors). Gate 12 is GREEN at HEAD with `policy_loss_trough` printed DEFERRED.

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws (LAW-06 amended by R349(a));
no gate number added (still 17). Cards (`docs/governance/CARDS.md`): opened by R350 —
`CARD-STOP-DRAIN-VS-GRACE` (fixed in code; the grace relation is a run7 mint row),
`CARD-WARMSTART-CONTROL` (closes on the frontier's `bc_tp`/`bc_full` pair — landed: 0.038 vs
0.413), `CARD-DRAIN-POLLER-RACE`, `CARD-SELFPLAY-SEARCH-STATS`; closed — `CARD-EVAL-CADENCE`
(R350(d)), `CARD-ALPHA-TARGET-FORM` (R350(e)); folded — `CARD-SEALBOT-HORIZON` into the frontier.
STRENGTH-FRONTIER-1 is RUNNING; INVESTIGATION-1's trough item is measured (record above); TEST-1
is not dispatched.

## Exit facts — the block-verdict packet, 2026-09-13

- Ruling: R350 at `5743cff6`, verbatim, ANNOTATED at its foot on this line; numbering continues
  at R351.
- Commits on this line: `bd94dd96` (the seam, `reinit`, the loader's provenance tolerance),
  `fc83ccf1` (BC-3's held-out value line), `25073aed` (the warm-up, the KL line, the trough halt,
  the α exclusion), `49b7740d` (the resumable stop's abandon), `f161719d` (the frontier driver +
  floors), then this record. One line each, empty body, zero trailers.
- Stop witnessed: `shutdown_save` step 35 084 at 08:15:12 UTC; bundle `run6_00035084_e1563d16`;
  receipts on 35 000 and 35 084 by `tools/mirror_pull.py --once`.
- Frontier launched 08:27 UTC (phase 1, 4 cells, parallel 4) and 08:30 UTC (phase 2, 27 cells,
  parallel 8); box CPU ≈ 81 % busy, GPU ≈ 53 %; ≈ 13 s/game per cell under that load.
- Reviewed by a reviewer subagent before commit: 12 findings, all acted on or carded; the
  material ones — the loader orphaning every pre-R350 envelope (fixed: tolerance for a stamp
  that predates a leaf, the re-strip REVERTED), `load_from_bc` loading a reinit head's shape
  (fixed), `abandon_pending` dropping its escalation (fixed: `EvalBrokenReason.ABANDONED`),
  the trough window ingesting refused steps (fixed), the warm-up bypassing the non-finite guard
  (fixed), the α predicate in two precisions (fixed: `is_alpha_full`), the driver's readout
  without LAW-04 dedupe and its hand-copied rung/postures (fixed).
- **Collected tests: 4 614 → 4 656**; the gate 3c floor ratchets to 4 656. Comment ratchet:
  3 546 / 23 / 13 527 → **3 545 / 23 / 13 527** (the floor file carries 3 545).
- Contract: `docs/contracts/run_config_schema.md` v23–v25 (158 leaf key-paths, gate 13 green);
  `docs/contracts/event_manifest.md` carries the five new instrument rows.
- Gates on the committed tree (`tools/ci_gates/run_all.sh`, dev box, CPU wheel): 2 (cargo test
  + clippy) and the default tier ran through; the integration tier **33/34 in 26 min**, the one
  red (`tests/test_run_launcher.py::test_a_periodic_cadence_burst_streams_periodic_checkpoint_save`)
  stopped by an external SIGINT at step 4 under the detached shell and **green on the foreground
  re-run** (130 s) — the detached-shell class this file already records; a concurrent integration
  tier from another checkout (`/tmp/mantis-dash`) was on the machine at the time. The default
  tier alone on the committed tree: **4 596 passed, 8 skipped**. Gates 3c, 6–17 GREEN
  individually; 4/5 not run (no wasm, no bench surface touched); the slow tier not run on the
  dev box (CARD-OC7-OVERRUN).

## Provenance

Derived 2026-09-13 on `dev` at the exit commit of this leg, from `configs/run6.yaml`, the box's
`/workspace/frontier/*/summary.jsonl` (read at 11:15 UTC), the run6 mirror's events and rings,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
