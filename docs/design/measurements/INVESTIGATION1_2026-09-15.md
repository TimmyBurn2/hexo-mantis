# INVESTIGATION-1 — the repo scan at `83667687`: surviving ledger, killed list, REPAIR-A4 order, delete list (2026-09-15/16)

Launched by R354(f): six read-only domain agents (Rust; Python training path; inference/eval/
arena/GSPRT/bots; config/tests/gates/tooling; style under the R346(f) rule; docs and governance),
each in its own worktree at `83667687`, then three red teams (A = Rust + eval, B = training +
config/tests/gates, C = style + docs) that reproduced or refuted every finding at HEAD, killed
duplicates and restatements of open cards, and ranked by impact × confidence ÷ cost. Nothing
touched the run. The FORCED-MOVE CENSUS of R354(b) ran beside it
(`FORCED_MOVE_CENSUS_2026-09-15.md`) and one red-team item (A-2) is that census's residue read
through the search code. Impact: P0 wrong results / lost runs; P1 measurable strength, wall-clock,
or a false number a session acts on; P2 latent; P3 hygiene. Confidence is the RED TEAM's own
reproduction (high = reproduced on the production path; static = read only). Cost S < 30 lines,
M < 300, L. Coordinates are at `83667687` and go stale (derive at point of use).

## The three facts that bear on the 30k decision (R354(e))

1. **R354(b) is REFUTED** by the census: the forced block is inside the Gumbel-16 target's explicit
   support on 99.995 % of forced rows in every run7 ring and every run6 ring, flat with step
   (threshold to refute: 2 %). Fix (c)(i) — injecting forced moves into the candidate set — cannot
   address a miss that does not occur. The record cannot answer the σ-pair arm (the ring stores
   masses + α only).
2. **The residue is a search verdict, not a sampling reach:** on 2.5–4.4 % of forced-BLOCK rows the
   block was searched and the completed-Q one-hot went to a losing counter-threat (a mover four or
   five), at 64 and 320 sims alike. Red team A reproduced a mechanism on the real backup path
   (A-2): `apply_quiescence` counts ONE-stone completions in a two-stone-turn game, so it is blind
   to an opponent four-with-two-empties (a next-turn win; 27.9 % of ring rows) and to the mover's
   own two-stone win, and on 20/20 sampled residue rows it scores the counter-threat child equal to
   (0.000) or above (+0.300) the block child. Those values enter the completed-Q policy TARGET. It
   acts on run7 in-run, both arms, and on the PUCT gate head. Confidence that it is A mechanism of
   the residue: high; that it is THE load-bearing one: medium — the test is the fix on one ring.
   This is a fourth fix candidate for the architect, pre-registered here and NOT armed:
   (iv) the quiescence threat unit becomes "completable within the side's remaining stones", both
   sides, with a test on census row 3664 (counter-threat child ≤ −0.9, mover win-in-1 root = +1).
3. **The Gumbel deploy head was a defective instrument on every Gumbel reading on the record** (A-1):
   `_drive_gumbel` stops at its first transposition and spends 0.19–0.42 of a 512-sim budget. run7
   in-run is NOT affected (deploy is PUCT; self-play's `select_leaves_forced` spends exactly). F-48,
   F-50, F-51, CARD-DEPLOY-HEAD-BUDGET's numbers and GAME_QUALITY's "3 of 4 fours left standing" by
   the Gumbel head were read through it. What those falsified rows still conclude is a LAW-02
   question for the architect; it is stated, not re-litigated.

## Surviving ledger (40 lines; id | origin | statement | impact | confidence | cost | cheapest falsifier)

| # | id | statement | impact | conf | cost | falsifier |
|---|---|---|---|---|---|---|
| 1 | A-2 (D1-2 + census) | Quiescence threat unit is one-stone in a two-stone game; blind to fours and in-turn wins, wrong-signed on 0.68 % of rows; feeds the policy target in-run | P1 in-run | high (rule) / med (residue) | S–M | fix on one ring: does the 2.5–4.4 % fall |
| 2 | A-1 (D1-1) | Gumbel deploy head stops at the first TT hit; served budget 0.19–0.42 at 512 sims; every Gumbel-head reading on the record was through it | P1 instrument | high | S | one frontier cell (18k vs sealbot_d5, 512, 288 games) with the fix |
| 3 | A-3 (D3-1) | A round killed at the bound or abandoned by a stop discards a COMPLETED gate verdict (one sidecar at the end; broken → `gate_result=None`) | P2 → P0 when it fires | high | M | fake worker writes `promoted: true` then sleeps past the timeout |
| 4 | B-2 (D2-4) | `steps_per_hour` after a resume = all steps / hours since boot: 4.79e9 measured; the box dashboard's hero line prints it now | P1 live | high | S | a rates test constructed at step 23 829 |
| 5 | B-1 (D2-1) | The server serves `trainer.model` itself; ActorSync is a self-copy; `actor_lag_abort` (armed on run7) cannot fire; `ema.enabled: true` would load the shadow into the learner | P2 latent / P0 the day EMA is minted | high | M (or S: schema-refuse ema) | `server.model is not trainer.model` |
| 6 | B-3 (D2-2) | A resume at an exact `eval_interval` multiple never kicks that round; every run7 periodic bundle sits on one (23 829 was not) | P2 | high (measured) | S | producer test: resume at 3000 kicks r3000 |
| 7 | B-4 + B-11 (D4-1, D4-2, D2-10) | Mode-collapse WARN rules read `policy_entropy`, which no producer emits (20 test fakes do); the LAW-08 witness compares key SETS to free-text strings and cannot red on a dead key | P2 | high | S + M | a write site for `policy_entropy` in src/ |
| 8 | B-5 (D4-18, widened) | Every `uv run` outside `run_all.sh` (Makefile `test`, the test-count gate's test) re-syncs to the CPU torch group | P1 on the box | high (static) | S | `stat` torch before/after `make test` on a cuda venv |
| 9 | C-3 (red team C) | `tests/tools/test_hardcode_scanner_sees_the_graph_era.py:19` mutates `sys.path`; the LAW-17 witness scans src+tools only | P1 (a hard rule broken, gate green) | high | S | the line not being a `sys.path` write |
| 10 | B-6 (D4-12) | Gate 6 diffs merge-base..HEAD: `--base HEAD` inspects nothing (after a push, so does `make gates`); only added files are sized | P2 | high | S | a 1.1 MB modification against its parent |
| 11 | B-7 (D2-5) | Resume restores step/ring/round/p_hat/RNG but not the abort windows, guard counters or the EMA shadow; run7's draw-rate abort moved from 25k to 26k | P2 | high (static) | M | plant a 2-entry history, stop, resume, third fires |
| 12 | A-4 (D3-2) | Eval child's batcher threshold assumes one stream (`max_in_flight = leaf_batch_size` = 8) under `rung_concurrency 8` | P2 unmeasured wall | med | S | occupancy histogram (needs the LAW-18 snapshot), then a rung at 64 vs 8 |
| 13 | A-5 (D1-3) | `GraphQueue::close()` notifies outside the waiter's lock: a lost wakeup hangs `stop()` under the GIL | P2 latent | med (static) | S | a park hook between check and wait |
| 14 | C-1 (D6-1, D6-2) | STATE.md "Minted values" block says 38 deltas / PUCT-512 / floor 4 811; the tree says 41 / 256 / 4 862 | P1 false numbers | high | S | yaml carrying 512 |
| 15 | C-2 (D6-3, D6-4) | STATE.md's 2026-09-14 exit block survived the rewrite: "dev NOT pushed", run7 on 15109ac3, seg0001 | P2 | high | S | origin/dev behind HEAD |
| 16 | C-6 (D6-23, D6-24) | CLAUDE.md names neither STATE.md, RULINGS.md nor CARDS.md, nor `make build.cuda` | P1 entry path | high | S | a CLAUDE.md line naming STATE.md |
| 17 | C-4 (D6-15) | preflight_report.md pins "a production config can never be preflighted in the short tier" — reversed by CARD-STAMP-FLOOR; no such test | P1 false pin | high | S | a test asserting the refusal |
| 18 | C-5 (D6-21) | The "42–47 reserved band" literal (contract + `preflight_mint_parent.py:87`) is false: the derived set is 42–48 (rc 48 = terminal eval broken) | P1 false literal | high | S | `RESERVED_CODES` without 48 |
| 19 | C-7 (D6-9/10/12) | CARDS.md rows frozen before the record moved: "run6 never started", frontier RUNNING, GAME-RECORD-1 ORDERED, `gumbel_mcts: false`, MAXPLIES derived, stride 3, `--resume-from` NOT BUILT, SEARCH-STATS "before run7" lapsed | P2 | high | S | any row reading otherwise |
| 20 | B-9 (D4-11) | Gate 3c's tier census sees decorator markers only; 17 body-level `pytest.skip` + 8 `importorskip`, 7 files undeclared | P2 | high | S | a default-tier run with 0 skips on a host without sealbot |
| 21 | B-8 (D2-3) | Shutdown save O3 has no latch: a signal in the pre-O3 poll window saves twice and `prune_bundles(keep=2)` drops the last periodic bundle | P3 (narrow) | high (measured) | S | one save asserted from inside `_poll_eval_results` |
| 22 | A-6 (D1-4) | `expand_and_backup_ls_at` skips the virtual-loss unwind on a short batch its siblings do; unreachable today | P2 latent | high | S | unit test, VL == 0 on dropped leaves |
| 23 | A-7 (D3-4) | `empty_cache()` after every deploy-head move from 8 threads under `expandable_segments` | P2 | low (no CUDA here) | S | one rung at G = 8, plies/s + reserved peak |
| 24 | C-8 (D6-13) | R351/R352 text carries "deploy and eval PUCT-512"; run7 mints 256 by operator direction — ANNOTATION under R351's Status, never an edit | P2 | high | S | — |
| 25 | C-11 (D6-7/8) | LAWS.md: the `target-cpu` half of LAW-13 has no check; LAW-17's text claims a wider ban than pyproject enforces — annotation or ruling only | P2 | high | S | a tool that greps target-cpu |
| 26 | C-9 + C-10 (D6-17/18/20) | Contracts stale: envelope says "schema-validated" and names `CnnArch` while the loader tolerates a stamp predating a leaf (no contract row); registry.md spells "grid"; graph_wire names `queues/dense.rs` | P2 | high | S | a contract row naming the tolerance |
| 27 | A-8 + A-9 (D1-5/6, D3-5) | 144 MB node pool per tree filled under the GIL per eval game (≈ 4.6 GB / 32 workers, 2.2 GB in the child); Gumbel interior selector 45.7 allocs/leaf (≈ 1 % of worker time) | P3 | high | S–M | `ps -o rss` of the child at G = 8 |
| 28 | A-10 (D3-6) | `eff_n` means distinct games in the rung aggregate and pairs in the gate aggregates; the frontier pairs by loop parity | P3 | high | S | bootstrap r1's rung both ways |
| 29 | B-10 (D4-19) | ≈ 70 s of the 326 s default tier re-proves a cheaper twin (5 groups ≈ 24 s reproduced) | P2 dev wall | high (5) / unverified (total) | M | `--durations=0` after module-scoped fixtures |
| 30 | B-12 + B-13 + B-18 (D4-5/6/7/10) | Second default authorities: `MonitorConfig` 30 vs minted 120, pyo3 −0.1 vs −0.5, `arch_kind` None → V1 while runs mint V2, `concurrency` 1 vs 8; gate 12's trough prose says run7 arms what run7 mints null | P3 | high | S | `SelfPlayRunnerConfig()` at −0.1 |
| 31 | B-14 + B-15 + B-16 (D2-6/7/8/9, D4-3/4/8/9) | ≈ 300 dead lines (grid loss family, `push_dense`, `save_buffer_if_enabled` while its counter rides a gate, the F1 defer path); two `sha256_file` (ring hash written with one, verified with the other); 73 leaves identical in all 5 YAMLs, every lever at its off value | P3 | high | M | the consumer census in red team B §vi |
| 32 | B-17 + B-19 (D4-13/14/15/16/17) | Gate blind spots: gate 8's audit sub-check defers to an absent doc and the audit exits 1 at HEAD; gate 11's encoding list hardcodes three deleted names; gate 13 checks one of three stated totals; gate 16 excludes src/ | P3 | high | S–M | — |
| 33 | B-20 (D4-20/21) | Offender scans with no scanned-count floor; `test_strix_adapter.py:123` proves a different arm per host | P3 | high | S | monkeypatch the root to an empty dir |
| 34 | A-11..A-15 (D3-7/8/9/10, D1-8) | Spool `.pt` never unlinked; dump context says rung concurrency 1; GSPRT labels the block `n_screen`; second book reader live via a diagnostic; action codec and q-sign flip re-typed 13× each | P3 | high | S | — |
| 35 | C-13 (D5 B2) | 63 files ≤ 300 lines carry a stale `>300 justify` header (gate 15's own 235 − 172); 39 are comment blocks (280 lines), 24 inside docstrings; 14 files at 285–300 will regrow | P3 | high | S/M | `wc -l` > 300 on a listed file |
| 36 | C-14 (D5 class 3) | 1 096 ruling tokens in comments + docstrings (705 in docstrings the lint never reads; bare `R\d{2,3}` never load-bearing on its own) | P2 | high | M | `comment_lint --measure` ≥ 1096 |
| 37 | C-15 + C-16 (D5 classes 1/5/6) | 1 264 comment runs > 2 lines (453 invariant-tagged), 831 multi-line docstrings on private symbols (2 142 lines; a POLICY call — R346(f) names public APIs), Rust `///` uncounted by the lint; the floor is tight (one line of slack) | P3 | high | L | — |
| 38 | C-17 (D6 §1) | ARCHIVE 7 docs (2 330 lines) + repo_design's contract-#5 narrative block (368); DELETE community_bot_api.md (14); gate 10 reds on two of the moves unless CARDS.md is edited in the same commit | P3 | high | M | gate 10 green after a bare `git mv` |
| 39 | C-12 + C-18/19/20 (D6-11/14/5/22/19/16) | CARDS says closed cards are not kept and keeps ~20; F-01..F-43 vs F-52; STATE's "OPEN" list names 3 of ~30; RUN6_BLOCK's (ii) lacks its F-48/F-49 pointer; repo_design row 5 "v13"; eval_instrument's `worker.py:349-356` coordinate stale (posture TRUE) | P3 | high | S | — |
| 40 | B-4 note / C-3 note | Two gates report green over a broken rule today (LAW-08 witness, LAW-17 witness): each fix above lands WITH the widened witness, or the class returns | — | — | — | — |

## Killed list (with the reason; full lists in the red-team reports)

- D3-5 ≡ D1-6 (the same 4M-node pool measured twice; folded into #27). D3-3 restates CARD-EVAL-ROUND-OVERRUN's open snapshot item (folded into the card; fix shape kept). D1-7's GIL census folds three ways (A-5, CARD-SERVER-SYNC, #27); the census itself is clean.
- D1-1's "SH answer disagrees with most-visited on 4–6 of 8 boards" — a tie-break artefact of the probe's `max_by_key`, not a wrong answer; the mechanism and spent/budget stand.
- D1-2's fix shape "reorder the checks" — superseded: it keeps the one-stone unit and stays blind to fours (18 of 20 residue rows); #1 carries the corrected shape.
- D2-11 (8 host syncs/step) — folds into CARD-TRAINER-CADENCE; F-44 measured the trainer asleep 92 %, so the sites buy no wall-clock. D2-12 (except census) — no live swallow; wording-level only.
- D2-10's entropy half ≡ D4-1; D2-9 ≡ D4-3 ≡ D4-4 (dead levers); D2-6 ≡ D4-9 (the dead defer path); D4-5 ≡ D4-10 (schema defaults); D4-13/14/16/17 (gate scope notes with no biting instance) — merged as shown in the ledger.
- D4-15's evasion half — REFUTED: `DEFAULT_ENC = "gnn_axis_r8"` IS refused by gate 11's assignment arm (domain 4 did not run the gate on its own evasion). D4-11's "none declared" — 5 of 12 files are declared (narrowed to #20). D4-19 as P1 — the tier is the developer's wall, not a run's (P2).
- D2-1(c) "eval-mode hazard on the shared module" — no race: the module is never flipped; the arch has no dropout/BatchNorm. D2-3 downgraded to P3 (O5's sleep is after O3).
- D6-16 as "contract posture wrong" — REFUTED: `SealBotDepthError` and `RungUnresolvable` are siblings; the worker does not catch the former, so the round DOES end, as the contract says; only the coordinate is stale (#39). D6-7's LAW-15 half — REFUTED: `None` never promotes. D6-22's PERF_INVESTIGATION half — a dated measurement at a named tree with F-46's repair annotation in falsified.md; not a disagreeing doc.
- D6-2/4/10/12/24/8/5/14 and D6 §3 (owed items restate CARDS' own section) — merged as shown. D6 §5 supersession table — a reading aid, not findings.
- D5 "gate 15 cannot see stale headers" — REFUTED: it counts them (235 vs 172) and declines to flag under-cap files by stated design; B2 survives as #35 with corrected sizing (14 at 285–300, not 9; 280 block-deletable lines, not 430). D5's KEEP of the CARD-STAMP-FLOOR citation — the card is closed, so it is a shorten. D5 B1 "no judgment" — re-scoped to policy (#37). D5 class 2 narrative (719) — a triage aid, 15 % FP, not gated. D5 class 4 "0 file-top blocks" — a confirmation.
- Nothing in any report contradicts `falsified.md`: #1 sits inside F-15's sanctioned leaf-value override; #12 is the eval child's collector, not F-47's lever; no F-17/F-19 structure is proposed.

## Proposed REPAIR-A4 order (a proposal for the packet; nothing here is armed or lands on the run)

1. **#1 quiescence threat unit** — lands in the tree WITH the census-row test, behind the minted `quiescence_enabled` (no config change); armed for run8 or a resumed run7 only by the 30k decision. Then the falsifier: re-run the census on one ring under the fixed backup (offline; the rings hold the boards).
2. **#2 Gumbel deploy head budget** — `_drive_gumbel` through `select_leaves_forced`, pin `spent == budget`; then ONE frontier cell on the box (18k vs sealbot_d5, 512, 288 games) to re-derive the Gumbel-head record; the LAW-02 reading of F-48/F-50/F-51 goes to the architect with that number.
3. **#4 steps_per_hour**, **#6 resume-at-interval kick**, **#21 O3 latch** — three S fixes with producer tests; the first is a live false number on the dashboard.
4. **#3 partial gate sidecar** (M) — the CARD-EVAL-ROUND-OVERRUN mechanism; reverses the pin in `test_eval_broken.py`.
5. **#7 entropy producer + LAW-08 witness that resolves its strings** (S + M) and **#9 the `sys.path` line + the LAW-17 witness widened to tests/** (S) — each fix with its widened witness (#40).
6. **#5 server owns a copy** (M) — or the S floor first: schema-refuse `ema.enabled: true` until it does, and delete `actor_lag_abort`'s armed row (LAW-07: a lever that cannot fire).
7. **#8 UV_NO_SYNC** on the Makefile targets and the test-count gate's test (S) — the box hazard.
8. **#10 gate 6 scope**, **#20 tier census body skips**, **#32 gate 11 registry-derived list** (S each).
9. **#11 resume state** (M): windows, guard counters, EMA shadow in the sidecar.
10. **Doc repairs, no gates:** #14/#15/#16/#19/#39 (STATE, CLAUDE.md map line, CARDS rows), #17/#18/#26 (contracts), #24/#25 as ANNOTATIONS only.
11. **Deletes and archives** (below): the tools trio; the 7 archived docs + the repo_design block with the CARDS edits gate 10 needs; the 280 stale-header lines; the 23 separator rules and the 234 parenthetical citations (zero-line edits).
12. **Measurements, box, after the LAW-18 snapshot item:** #12 (batcher threshold), #23 (`empty_cache` per move), #27 (RSS). Then #13, #22, #28–#31, #33, #34 as hygiene commits on contact.
13. **Style backlog** (#35–#37): the mechanical half by script (private docstrings are a policy call the architect makes), the run backlog site by site under the ratchet, and the three new lint measures (`stale_r8_headers` in gate 15, `private_docstring_excess_lines`, `rust_doc_excess_lines`) each with its floor line.

## Delete list, sized in lines (derived by `wc -l` at `83667687`)

| class | items | lines |
|---|---|---|
| DELETE now, gate-10-safe | `tools/profile_eval.sh` 110, `tools/profile_selfplay.sh` 121, `tools/perf_prereg_skeleton.md` 89 (a run5-era pair never run; one reads a nonexistent key, one cites `plan/`) | 320 |
| DELETE, doc | `docs/contracts/community_bot_api.md` + repo_design §4 row 8 | 15 |
| ARCHIVE (relocate) | AUDIT_2026-09-09, MEASUREMENT_PERF3B, MEASUREMENT_R153 + LEG2 + both PREREGs, RUN7_PREREG (7 files) + repo_design L197–564 | 2 330 + 368 |
| MERGE — HOLD to the observatory ruling | `tools/observatory/readers/{stats,hexlogic,ladder}.py` (two byte-identical, one drifted; tests enforce the copies) — or delete `tools/observatory/` (962) if the design is refused | 294 |
| Dead code (judgment M) | grid loss family, `fp16_backward_step`, `push_dense`, `save_buffer_if_enabled`, `_extract_state`, the flat owned-keys set + defer path | ≈ 300 |
| Mechanical style, no judgment | 280 stale-header block lines + 11 collapsible runs/citation lines; 257 zero-line edits (23 separator rules, 234 parenthetical citations) | 291 |
| Judgment style | private multi-line docstrings 2 142; R8 paragraphs 384; narrative `//!` 52; history runs 296; the > 2-line run backlog 3 464 beyond two | ≈ 6 300 |
| **Total** | true deletions ≈ 7 200; relocations 2 698 | |

## Sources

Reports (this session's scratchpad `inv1/`, untracked): `domain{1..6}_*.md` + summaries; `redteam_A.md`,
`redteam_B.md`, `redteam_C.md` with their probes (`redteamA/probe` — a cargo probe against the real
bridge; `redteamA/deploy_probe.py`, `census_rows.py`; `redteamB/probe_*.py`, `census.py`,
`gate11_evasion.py`; `redC_checks.py`); the style linter `inv1/style/style_census.py` (stdlib, 298
lines, reproduces exactly at HEAD: 1264 / 719 / 1096 / 831 / 3316 / 172). Every number above is the
red team's reproduction, not the domain agent's claim, except where the ledger says "unverified".
Environment: the main checkout's venv read-only at `83667687`, `UV_NO_SYNC=1`; cargo in the
worktrees; no CUDA on this host (every CUDA-side magnitude is unmeasured here); no box access.
