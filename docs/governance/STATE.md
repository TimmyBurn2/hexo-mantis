# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register; where a register disagrees with the tree, the
tree wins and the line says so. Counts are derived at point of use from the file or command named,
never transcribed here. A reader who finds a line stale repairs it in place (R311(c)).

## Current phase

**PERF-ADA (R369) EXITED 2026-09-25**, fast-forwarded into `dev`: the rented 4080S box is the run box. Serving
at B 64 reads 3 984 leaves/s IDLE (admission PASS, 1.62x the reference box's 2 466; it read 0.74x before the
packet), the real self-play loop 1 710 games/h (2.46x), the trainer step at the cap −25 % at 4.69 GiB peak. The
legs, benches, halts and decisions are the R369 ledger in
`docs/design/measurements/PERF_ADA_PROFILE_2026-09-24.md`; the reviews and the exit record are local records
outside the tree (R369(f)). Not landed, and why, is in the ledger and carded: the torch/CUDA bump (L3, not
faster), the exact eval cache (L4, on branch `perf-ada-l4`: the served path is not batch-invariant, R369(d)).

## The run

- **No run is live.** run7 stopped at step 83 482 on 2026-09-18 (the stop is recorded by commit
  `8cb5ca6a`; `docs/design/measurements/EVAL_COST_2026-09-19.md` reads its rounds) and run8 at 55 170 on
  2026-09-21 (R365); run9 was never started and its config is deleted (R365(a), R367). The strength
  series and every cell on record are in the measurement records named below.
- **run10 is ARMED, not started.** `configs/run10.yaml` is minted (R366, ratified by R367(c)); its
  order, witnesses and pre-registered reading are
  `docs/design/measurements/RUN10_PREREG_2026-09-21.md`, the box sequence its §6. run10 launches
  from PERF-ADA's exit tip, the `dev` commit that merged it (R369(a), amending R368(i)): the full gate set
  incl. the slow tier is green there, its resolved config matches its mint by value except the retired
  `train.value_target` (`config_diff --expect` MATCH), admission was re-read IDLE (3 984), and the parent's
  strix @ r8 re-read was WAIVED by the operator on 2026-09-25 (no reading exists for this tip). A re-read
  outside the parent's recorded CI would send the bar to the operator. SEAM-2's implementation
  merges only after run10 STARTs (R368(j)). Read a config's values from the file itself and diff
  two with `tools/config_diff.py`; STATE does not restate minted rows.
- **A box is rented** (2026-09-24: an RTX 4080 SUPER host, the operator's, R11; PERF-ADA made it the run box); the previous instance was
  destroyed on 2026-09-21 (R365, annotated by R367(d)); the operator's mirror (`tools/mirror_pull.py`) holds run7's and run8's
  artifacts, run10's parent and the ring its held-out slice reads. The run10 box criterion and its
  admission bench are R367(e), run with `tools/bench_server.py` per the prereg's §6; any admission
  reading taken since is not a tracked record. Renting, stopping or re-speccing a box is the
  OPERATOR'S act (R367(d), CLAUDE.md R11).

## Configs

The committed configs are `configs/run10.yaml`, `configs/dev_example.yaml` and
`configs/smoke_preflight_armed.yaml`. Production is a CENSUS, never a
list: `mantis.config.census.production_configs` (every `configs/` file minus its `EXEMPT_CONFIGS`
rows, which carry their grounds). run7's and run8's configs were deleted at `8b00b4dd` (R368(e)),
run6's by R369's packet (W0).

## Where things live

- Open work: `docs/governance/CARDS.md` (swept in W6; derived there, never enumerated here).
- Rulings: `docs/governance/RULINGS.md`; the latest is R368.
- Laws and the protected set: `docs/governance/LAWS.md`; R368(b) reads it as invariants held by
  symbols and pinning tests, mapped in `docs/slim/00_MAP.md` §3. Falsified work:
  `docs/governance/falsified.md`.
- Build, test tiers, the gate set and when it runs: `CLAUDE.md`. The test-count floor is
  `tools/ci_gates/test_count_floor.txt`; the comment ratchet's floors are
  `tools/ci_gates/comment_length_floor.txt`.
- The instruments at HEAD: `tools/strix_follower.py` (the strength series, strix @ r8, R366),
  `mantis.diagnostics.ring_audit` (the ring bands a prereg declares), `tools/ci_gates/preflight_mint.py`
  (the MANUAL mint preflight), `tools/mint_config.py` + `tools/config_diff.py` (mint and diff),
  `tools/probe1.py`, and the display tools `CLAUDE.md` admits under "Deliberately absent".

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
| The protected-set / laws / cards line | `docs/governance/LAWS.md`, `docs/slim/00_MAP.md` §3, `docs/governance/CARDS.md` |
| Dispatcher items 1–8: run7's rounds and strix cadence, the observatory decision, REPAIR-A4, run8's mint, the witness correction, run7's stop | R355, R356, R357; `docs/design/measurements/STRIX_RUN7_60K_2026-09-17.md`, `docs/design/measurements/RUN8_PREREG_2026-09-17.md` |
| Items 9–12: run8's re-mint, shakedown, witness, audit, START, the follower chain, PERF-3 steps 1–2, twin inheritance | R358, R359, R360; `docs/design/measurements/PERF3_2026-09-18.md`; commits `86308bc7` (the re-mint leg's exit), `be863f16` (the START, the shakedown's witness and the audit), `f06c3234` (the re-mint), `5e25e40c` (the follower fix), `ee6e7abe` (twin inheritance) |
| Items 13–15: the eval census, the sealbot rung's deletion, the 15k/30k/45k cells, the ladder unit, the CPU-head profile | R361, R362, R363; `docs/design/measurements/EVAL_COST_2026-09-19.md`, `docs/design/measurements/CPU_HEAD_PROFILE_2026-09-20.md`, `docs/design/measurements/RUN9_PREREG_2026-09-19.md`; commits `0d6ee0ee` (the follower's promotion switch), `94b8286a` (the rung's deletion) |
| Items 16–17: run9's mint, the parameter-distance test, run8's stop, PERF-3 step 3, E1, PROBE-1 | R364, R365; `docs/design/measurements/PARAM_DISTANCE_2026-09-21.md`, `docs/design/measurements/PROBE1_2026-09-21.md`, `docs/design/measurements/PERF3_2026-09-18.md`; commits `61bd2fd1` (run9's mint), `d4804e58` (the E1 ruler-r6 variant) |
| Items 18–19: run10's composition, REVIEW-1/REVIEW-2 and the fix leg | R366, R367; `docs/design/measurements/RUN10_PREREG_2026-09-21.md`; `docs/audits/REVIEW_2026-09-21.md`, `docs/audits/REVIEW2_2026-09-21.md`; commit `f088407e` |
| Per-leg exit facts (gate results, collected-test counts, comment floors, push state) | the STATE commit that recorded each leg's exit: `7e0a424c` (R353), `0c880a6e` (R357), `86308bc7` (R358), `7278476a` (R359), `8379073e` (R360), `671bf12c` (R361), `07699fe6` (R362), `ea67af89` (R363), `6bbbfac9` (R364), `51f40a18` (R365), `19e8351d` (R366), `69e15329` (R367); SLIM-FIX's in `docs/slim/PROGRESS.md` |
| Box-local specifics (box branches, ops-dir logs, launch scripts, unit names, tunnels) | out of the tree (R368(f), R281(b)); they left the tree, and history is not rewritten |
| Per-item provenance chains | `git log -- docs/governance/STATE.md` |

## Provenance

Derived 2026-09-25 at the PERF-ADA exit from the tree and the R369 ledger in
`docs/design/measurements/PERF_ADA_PROFILE_2026-09-24.md` (every reading there names its benched sha).
The configs and "where things live" sections below the run were carried from the SLIM-FIX rewrite and
re-checked against `configs/` and `census.py` at this tip.
