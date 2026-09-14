# MEASUREMENT — run7's 3 h shakedown twin (`shakedown7g`, 2026-09-14 11:28 → 14:28 UTC)

Instrument: `/workspace/run_shakedown.sh /workspace/oc7/shakedown7g.yaml /workspace/runs/shakedown7g 10800`
on the box (RTX 5080, 24 cores), tree `de536a2d` — R352's re-mint: Gumbel-320/64 self-play with
every head loaded, warm-up 0, `train.ply_cap_abort {0.5, 600, 3000}`, `deploy.search.kind: puct`,
eval 512, `eval.rung_concurrency 8`, `eval.round_timeout_sec 10 800`. The twin's only delta from
`configs/run7.yaml` is `run_id` (`config_diff --expect run_id` MATCH); its own stamp
(`preflight_shakedown7g_…`, pass, MIRRORED; rung 185/288 = 0.642 at step 101). Ended by `timeout`
at 10 800 s, rc 124. The record is mirrored to `mantis-mirror/shakedown7g/` (events, games,
the step-3 000 bundle); card peak 10.99 GiB; no F-816-37 dump, no stray process, no
`training_alert`, no `hard_abort`.

## A. The trainer (the R352(a) question: is run6's regime back?)

| | shakedown7g (Gumbel-320/64, every head) | shakedown7 (PUCT 320/64) | run6 block |
|---|---|---|---|
| steps in 3 h | **3 938** | 2 363 in 4 h | — |
| steps/h (before the round / with the round beside it) | **1 630 / 1 310** | 600 | 1 163 whole |
| games/h | 1 700 → 1 350 | 606 | 1 170 |
| avg game length (cumulative) | 22 → **37 plies** | 35 → 120 | 23 → 42 |
| draw rate (cumulative) | 0.000 → **0.007** | 0.00 → 0.31 | 0.002–0.005 |
| ply-cap window (600 games), max / last | **0.015 / 0.008** (28 cap games of 4 033) | 0.84 in the last hour | — |
| `ply_cap_attractor` gate at boundaries 1k / 2k / 3k | 0.003 / 0.012 / 0.007 (3 000 checks, 505 skips = the window filling) | not armed | not armed |
| step 3 000 reached at | **+1.98 h** | never | — |

Run6's trainer is back and faster than the block (the block had the eval rounds and the actor
sync beside it; this run's first two hours had neither): decisive 22–37-ply games, 0.7 % draws,
the cap window two orders of magnitude under the halt. Policy loss 2.45 → 2.86 over 3 900 steps
(the trough line's early rise, a warning by R351(d)); value loss 0.58 → 0.51; KL(target ‖ prior)
2.3 → 2.7 nats. The halt's LAW-18 line reads in-run: `monitor_gates` carries the window at every
boundary.

## B. The contended round — the reading the mint's rows were for

Round `r000001_3000` kicked at step 3 000 (13:27:10 UTC) with training running beside it
(steps/h 1 630 → 1 310 during the round):

| phase | games | result | wall | the idle stamp (`RUN7_STAMP2`) |
|---|---|---|---|---|
| floor probe | 4 | 4/4 | 6 s | — |
| gate screen vs the step-0 anchor | 80 | **56 (0.70)** → escalates | **34.6 min** | 15 min |
| gate confirm | 128 | KILLED with the run at 14:28 (61 min in, unfinished) | > 61 min | 21 min |
| sealbot_d5 rung, 288 | not reached | — | (idle: 48 min) | 48 min |

The step-3 000 net beats its step-0 anchor 0.70 on the screen — the R352(a) comparison's first
point (run6's frozen 3k checkpoint read 0.63 vs sealbot at 512; this round never reached the
rung). **At the contention ratio measured on the screen (2.3×), the escalated round projects to
≈ 3.2 h** — over `eval.round_timeout_sec` 10 800 and over the 2.2 h cadence — and the
non-escalated round to ≈ 2.4 h. Escalation is the NORMAL case (a net at parity with its anchor
screens ≥ 0.44 half the time), so run7 as minted would have most rounds killed and the rest
skipping the next kick busy. The 48-min idle rung is the term that does it, and its cause is on
record: the vendored sealbot holds the GIL through its search (CARD-SEALBOT-GIL-SERIAL), so the
eight games in flight overlap only on our side.

## C. The LAW-16 save under `timeout` — the CARD-SHAKEDOWN-TIMEOUT-STOP shape again

The run's events end at 14:28:34 with ordinary rows; no `shutdown_requested`, no
`resume_bundle_published` beyond the periodic step-3 000 bundle, no `eval_round_abandoned`; rc
124 at 14:28:37. shakedown7 (2026-09-14 01:50 UTC, the same launcher) DID save under `timeout`;
this one did not — with the eval child in the process group and a round in flight. The card's
"not root-caused" stands with one more instance; START's path is the supervisor's direct SIGTERM,
which is the falsified-clean path, and a twin's lost 938 steps cost nothing.

## D. What the shakedown decides

1. The TRAINER rows are proven: Gumbel self-play with every head loaded is run6's regime at
   ≥ run6's rate, no attractor, the halt armed and reading.
2. The EVAL rows as minted do not fit their own bound under contention. The operator's decision
   (2026-09-14 14:35 UTC, "do the cleanest thing"): the GIL-release hunk lands in
   `vendor/patches/sealbot.patch` (a third hunk; no move, score or depth receipt changes — pinned
   by `tests/bots/test_sealbot_vendored.py::test_the_search_releases_the_gil_and_eight_concurrent_searches_agree_with_serial`),
   the extension is rebuilt here and on the box, run7 is RE-STAMPED on that tree, and START
   follows on the stamp's own terminal round reading the rung's new wall.

## Sources

`mantis-mirror/shakedown7g/logs/events_shakedown7g_seg0001.jsonl` (the mirrored record),
`/workspace/oc7/shakedown7g.launch.log`, `/workspace/runs/shakedown7g/shakedown.log`,
`.../eval_spool.work/shakedown7g/r000001_3000_progress.txt`; `RUN7_STAMP2_2026-09-14.md` for the
idle round; `SHAKEDOWN7_2026-09-14.md` and `RUN6_BLOCK_2026-09-12.md` for the comparison columns.
