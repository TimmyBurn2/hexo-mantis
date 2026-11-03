# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register; where a register disagrees with the tree, the
tree wins and the line says so. Counts are derived at point of use from the file or command named,
never transcribed here. A reader who finds a line stale repairs it in place (R311(c)).

## Current phase

**R403 (2026-11-03): THE PROGRAMME CLOSES; WHAT SHIPS, WHAT IS CUT, WHAT STAYS ON THE RECORD. Training is CLOSED.**
The net of record is `run14avg_00372000_647f0497` (file sha256 `5825bc67…`, net_hash `558acc92…`), a 4×256 GINE, the
uniform average of run14's saves 351k–372k. It ships in `hexo-mantis` web-v0.1.0, tagged at `00845838`, whose GitHub
release the operator published on 2026-11-03 (CARD-DEPLOY-WEB). run15, the net of record deepened to 6×256, is CLOSED:
its futility line stopped it at 157 980 on 2026-10-27, and its end read found no winner (run15avg rung 16 0.654
[0.616, 0.689], S 0.870 [0.842, 0.898] against 0.632 / 0.868). A head-to-head on the desk on 2026-11-03 read run15avg
against run14avg 287-287-2, 0.500 [0.457, 0.543], over 576 games at 256 sims a stone each. At 1.5× the cost per search
run15 does not ship (CARD-RUN15). The lineage's gains per 100k steps fell +0.32 → +0.14 → +0.08 → flat.
Rulers of record (R403(a)): Six gen455 at 16 nodes 0.632, at 128 nodes 0.45 (288k-era), at 256 nodes 0.31; Strix at
equal work 0.868; Bubble 410k at equal time 0.67 (288k). The browser engine at 10 s a turn on WebGPU read 0.701 against
Six n128 beside the native head's 0.674 at the same sims, which is a parity pass. At equal time against Six's full
second, Six is stronger; no other equal-time claim against Six is on record. What is left is the release (R403(b)–(g)),
with no box: NATIVE-1 (the `mantis-cli` binary; CARD-NATIVE-1), LADDER-1 (the strength table and the named levels, the
VPS's cost not measured by the operator's word; CARD-LADDER-1), ARENA-1 (the arena's side in the operator's repository, the `tools/ladder` rewrite here, the bots on
the operator's VPS; CARD-WEB-ARENA-SIDE), CLEAN-1 (the trim; CARD-CLEAN-1) and DOCS-1 (the reader's documentation in the repository;
CARD-DOCS-1). Then `main` at the release commit and the `v0.1.0` tag, both the operator's. CARDS is frozen at R403,
and the register continues only for the release's own rulings. No box is in use: the operator destroyed box A on
2026-10-27, and the desk's mirror holds every run's artifacts.

## The runs

- **run15** (`configs/run15.yaml`, 6×256 from `deepen6_00000000_e4f4c52a`): launched 2026-10-24 01:18 CEST on box A
  from `d6a8cc9f`; futility stop at 157 980 on 2026-10-27 (stop save `run15_00157980_6eb4e986`); no winner; CLOSED.
- **run14** (`configs/run14.yaml`, run13's second cycle with decided tails): launched 2026-10-19 from `94bc5512`;
  stopped at 372 000 on 2026-10-22 (R400(a)); the average of its last eight saves is the net of record.
- **run13** (`configs/run13.yaml`, warm from run12's 240k with the future head): launched 2026-10-16 from `1d1e27a9`;
  CLOSED at 176 681 on 2026-10-19, window MISS (R396(a)).
- **run12** (`configs/run12.yaml`, run11's 264k widened to 4×256): started 2026-10-12 from `abbca6be`; stopped at
  301 259 on 2026-10-16 (R394(a)); the anneal lifts.
- **run11** (`configs/run11a2.yaml`, RUN11-PRE's arm 2): 2026-10-05 to its stop at 593 030 on 2026-10-11 (R391(b));
  RUN11-FRESH (`configs/run11fresh.yaml`) ran to 139 933 and the fresh line closed (R388(c)).
- **run10** (`configs/run10.yaml`) was minted by R366 and never started (R376(c)); run9 never started and its config is
  deleted (R365(a), R367); run8 stopped at 55 170 on 2026-09-21 (R365); run7 at 83 482 on 2026-09-18; run6 at
  35 084 by R350(a).
- **Boxes:** box A (RTX 4080 SUPER) ran run11 to run15 and was destroyed by the operator on 2026-10-27. Box B ran
  RUN11-FRESH and was released for destruction on 2026-10-08; that destruction is the operator's act and its date is
  not on record. The desk's mirror (`tools/mirror_pull.py`) holds the runs. Renting a box is the operator's act
  (CLAUDE.md's price law).

## Configs

`configs/` holds every minted run config, run10 to run15 with RUN11-PRE's four arms and RUN11-FRESH, beside
`configs/dev_example.yaml`, `configs/smoke_preflight_armed.yaml` and `configs/smoke_wiring.yaml`. Production is a
CENSUS, never a list: `mantis.config.census.production_configs` (every `configs/` file minus its `EXEMPT_CONFIGS` rows,
which carry their grounds). Read a config's values from the file itself and diff two with `tools/config_diff.py`.

## Where things live

- Rulings: `docs/governance/RULINGS.md`; the latest is R403.
- Cards: `docs/governance/CARDS.md`, frozen at R403. Its first section lists the release's open cards and gives every
  other card that was open its closing line.
- Laws and the protected set: `docs/governance/LAWS.md`, whose protected set names each invariant's pinning tests
  (`tests/test_protected_set_pins.py` fails if one is gone). Falsified work: `docs/governance/falsified.md`.
- Build, test tiers and when gates run: `CLAUDE.md`; the gate set: `tools/ci_gates/run_all.sh` (`make gates`,
  `make gates.exit`).
- The shipped engine: `crates/mantis-player` (the deploy driver, `RustForward`, the wasm32 surface), `web/` (the
  browser package, contract `docs/contracts/web_engine.md`), and `tools/export_web.py` (the export from the stamp);
  the design is `docs/design/DEPLOY_WEB_2026-10-20.md`.
- The instruments: `tools/strix_follower.py`, `mantis.diagnostics.ring_audit`, `tools/ci_gates/preflight_mint.py`,
  `tools/mint_config.py` with `tools/config_diff.py`, `tools/probe1.py`, and the display tools `CLAUDE.md` admits
  under "Deliberately absent".

## Dropped in the R403 rewrite

The ruling paragraphs from R376 to R402, the long run lines and their provenance chain are
`git show 856a08cf:docs/governance/STATE.md`; each ruling's text is its entry in `docs/governance/RULINGS.md`.

## Dropped in the R368(f) rewrite, and where each class lives

The pre-rewrite text is `git show 029adc0a:docs/governance/STATE.md`; history is not rewritten.

| Dropped class | Where it lives |
|---|---|
| The chained "Current phase" paragraph (R353 → R367) | RULINGS R353–R367, each entry's Status line |
| run7's 2026-09-15 checkup, eval-cost reading and resume re-mint | `docs/design/measurements/RUN7_EVAL_COST_2026-09-15.md`; `docs/design/eval_gate_memo_2026-09-15.md`; R353, R354; commits `ce0a8ff6` (the GSPRT re-mint), `ba51fd46` (the resume push) |
| The SEALBOT-TT A/B | `docs/design/measurements/SEALBOT_TT_AB_2026-09-14.md`; commit `992cd519`; the rung itself deleted by R362 |
| PERF-A4, the serving levers | `docs/design/measurements/PERF_A4_2026-09-11.md`; the branch's landed tip `41a5fea8` |
| Hold 1, the stamp trap and the burst floor | commit `652b9f02` |
| Hold 2, completed-Q in decided positions | R350(e); the owed α = 1.0 reconstruction is its CARDS row (R349(c) / R350(e)) |
| The mirror arm; the integration tier's CUDA carve-out | R349(b); R349(a) |
| run7's minted values and delta counts | R362(c), R364 (the re-mints); commit `ce0a8ff6`; the file's `# delta:` header in history |
| The protected-set / laws / cards line | `docs/governance/LAWS.md` (the protected set with its pins), `docs/governance/CARDS.md` |
| Dispatcher items 1–8: run7's rounds and strix cadence, the observatory decision, REPAIR-A4, run8's mint, the witness correction, run7's stop | R355, R356, R357; `docs/design/measurements/STRIX_RUN7_60K_2026-09-17.md`, `docs/design/measurements/RUN8_PREREG_2026-09-17.md` |
| Items 9–12: run8's re-mint, shakedown, witness, audit, START, the follower chain, PERF-3 steps 1–2, twin inheritance | R358, R359, R360; `docs/design/measurements/PERF3_2026-09-18.md`; commits `86308bc7` (the re-mint leg's exit), `be863f16` (the START, the shakedown's witness and the audit), `f06c3234` (the re-mint), `5e25e40c` (the follower fix), `ee6e7abe` (twin inheritance) |
| Items 13–15: the eval census, the sealbot rung's deletion, the 15k/30k/45k cells, the ladder unit, the CPU-head profile | R361, R362, R363; `docs/design/measurements/EVAL_COST_2026-09-19.md`, `docs/design/measurements/CPU_HEAD_PROFILE_2026-09-20.md`, `docs/design/measurements/RUN9_PREREG_2026-09-19.md`; commits `0d6ee0ee` (the follower's promotion switch), `94b8286a` (the rung's deletion) |
| Items 16–17: run9's mint, the parameter-distance test, run8's stop, PERF-3 step 3, E1, PROBE-1 | R364, R365; `docs/design/measurements/PARAM_DISTANCE_2026-09-21.md`, `docs/design/measurements/PROBE1_2026-09-21.md`, `docs/design/measurements/PERF3_2026-09-18.md`; commits `61bd2fd1` (run9's mint), `d4804e58` (the E1 ruler-r6 variant) |
| Items 18–19: run10's composition, REVIEW-1/REVIEW-2 and the fix leg | R366, R367; `docs/design/measurements/RUN10_PREREG_2026-09-21.md`; `docs/audits/REVIEW_2026-09-21.md`, `docs/audits/REVIEW2_2026-09-21.md`; commit `f088407e` |
| Per-leg exit facts (gate results, collected-test counts, comment floors, push state) | the STATE commit that recorded each leg's exit: `7e0a424c` (R353), `0c880a6e` (R357), `86308bc7` (R358), `7278476a` (R359), `8379073e` (R360), `671bf12c` (R361), `07699fe6` (R362), `ea67af89` (R363), `6bbbfac9` (R364), `51f40a18` (R365), `19e8351d` (R366), `69e15329` (R367); SLIM-FIX's in `docs/slim/PROGRESS.md` |
| Box-local specifics (box branches, ops-dir logs, launch scripts, unit names, tunnels) | out of the tree (R368(f), R281(b)); they left the tree, and history is not rewritten |
| Per-item provenance chains | `git log -- docs/governance/STATE.md` |

## Provenance

Rewritten 2026-11-03 at R403 (CLOSE-1) on branch `close-1` over `origin/dev` `856a08cf`. The current phase comes from
R403's text, checked against the tree: the net of record's sha256 and net_hash from its mirrored file and `swa.json`,
and the head-to-head from its local record (`mantis-records/run15/endread/`). The run lines are compressed from the
`856a08cf` text, the configs line from `configs/`, and every path named above was checked at that commit. The
provenance of every earlier line is that commit's file and `git log -- docs/governance/STATE.md`.
