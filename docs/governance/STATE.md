# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register; where a register disagrees with the tree, the
tree wins and the line says so. Counts are derived at point of use from the file or command named,
never transcribed here. A reader who finds a line stale repairs it in place (R311(c)).

## Current phase

**RUN10-REPICK HALTED AT C (2026-09-26): the aux start is NOT recovered, so no re-mint; run10 is NOT ready
for START and its aux start goes to a design packet (R372(c)).** strix @ r8, 288 paired games each, on the
launch tree `451d23f9`: the parent's second cell B1 0.191 [0.146, 0.236] is a near-replay of S1 (fixed
`seed_base`: 272 of 288 games byte-identical), so the pooled parent is 0.191 on 576 games, 0.194 [0.150,
0.244] on its 304 distinct ones; the control, the preflight's 101-step net, 0.0625 [0.035, 0.094] fails the
test as it must; the twin's last save (step 12 000) 0.118 [0.083, 0.153] does not reach the pooled point.
Leg A (the preflight child's logging, the bench window, the OC-7 re-aim) is pushed at `c9fe9fcd`, gates.exit
green. No operator line came with the packet, so only A and B ran. Records are local.

**RUN10-PRESTART HALTED AT S6 (2026-09-26); run10 is NOT ready for START.** FINISH is merged and pushed
with R371 (`451d23f9`, run10's launch tree, R371(d)). On that tree, on the box: admission IDLE 3 955 leaves/s at
B 64 (PASS, 1.60x); the parent's strix @ r8 cell IDLE **0.191 [0.146, 0.236]** — outside [0.104, 0.181], the
operator ordered the sequence on; preflight PASS (stamp `3df6dd3b…`, MIRRORED; the terminal regression guard
rejected, the 101-step burst net 0.234 vs its anchor at 16 pairs); the EMA cell 0.149 [0.108, 0.194] → EMA waits;
the 4 h twin (`shakedown10`, inherited stamp) → F5 READ. The witness HALTS on two rows: the aux-weight ratio
settled at 2.92 (band [0.5, 2] → the §1a re-pick is a re-mint) and the step-3 000 ring's one-hot shares
(pooled 0.3151 vs < 0.30, mr1 0.4002 vs < 0.40; the 9k and 12k rings pass). Every other witness passes; the
cards, the operator's rulings owed and the numbers are CARDS' RUN10-PRESTART section and the local records.

**DEV-SPEED EXITED beside it (2026-09-26).** The integration tier's real boots were compute, not waits:
three preflight rows now read one module-scoped boot and the four in-process wiring rows boot
`configs/smoke_wiring.yaml` (census-exempt, ruled by R372(a)). On the desktop CUDA venv,
back to back: the integration tier 2 922 -> 466 s, `make gates.exit` 3 076 -> 627 s, ALL GREEN at the
tip (the base's one red was the pre-existing bench-server window, carded). Records are local.

## The run

- **No run is live.** run7 stopped at step 83 482 on 2026-09-18 (the stop is recorded by commit
  `8cb5ca6a`; `docs/design/measurements/EVAL_COST_2026-09-19.md` reads its rounds) and run8 at 55 170 on
  2026-09-21 (R365); run9 was never started and its config is deleted (R365(a), R367). The strength
  series and every cell on record are in the measurement records named below.
- **run10 is ARMED, not started.** `configs/run10.yaml` is minted (R366, ratified by R367(c)); its
  order, witnesses and pre-registered reading are
  `docs/design/measurements/RUN10_PREREG_2026-09-21.md`, the box sequence its §6. run10 launches
  from FINISH's exit tip (R370(i), moving R369(a) off PERF-ADA's tip): the full gate set incl. the
  slow tier green there, its resolved config matching its mint by value except retired leaves,
  admission re-read IDLE there under R367(e), and the parent's strix @ r8 cell re-read there (the
  operator waived that re-read for PERF-ADA's tip on 2026-09-25). A re-read
  outside the parent's recorded CI would send the bar to the operator. SEAM-2's implementation
  merges only after run10 STARTs (R368(j)). Read a config's values from the file itself and diff
  two with `tools/config_diff.py`; STATE does not restate minted rows.
- **A box is rented** (2026-09-24: an RTX 4080 SUPER host, the operator's, R11; PERF-ADA made it the run box); the previous instance was
  destroyed on 2026-09-21 (R365, annotated by R367(d)); the operator's mirror (`tools/mirror_pull.py`) holds run7's and run8's
  artifacts, run10's parent and the ring its held-out slice reads. The run10 box criterion and its
  admission bench are R367(e), run with `tools/bench_server.py` per the prereg's §6; any admission
  reading taken since is not a tracked record. Renting, stopping or re-speccing a box is the
  OPERATOR'S act (R367(d); CLAUDE.md's price law).

## Configs

The committed configs are `configs/run10.yaml`, `configs/dev_example.yaml`,
`configs/smoke_preflight_armed.yaml` and `configs/smoke_wiring.yaml`. Production is a CENSUS, never a
list: `mantis.config.census.production_configs` (every `configs/` file minus its `EXEMPT_CONFIGS`
rows, which carry their grounds). run7's and run8's configs were deleted at `8b00b4dd` (R368(e)),
run6's by R369's packet (W0).

## Where things live

- Open work: `docs/governance/CARDS.md` (swept in W6; derived there, never enumerated here).
- Rulings: `docs/governance/RULINGS.md`; the latest is R371.
- Laws and the protected set: `docs/governance/LAWS.md`, whose protected set names each
  invariant's pinning tests (R370(f)); `tests/test_protected_set_pins.py` fails if one is gone.
  Falsified work: `docs/governance/falsified.md`.
- Build, test tiers and when gates run: `CLAUDE.md`; the gate set itself: `tools/ci_gates/run_all.sh`
  (`make gates`, `make gates.exit`). The test-count floor is
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

Derived 2026-09-25 at the PERF-ADA exit from the tree and the R369 ledger in
`docs/design/measurements/PERF_ADA_PROFILE_2026-09-24.md` (every reading there names its benched sha).
The configs and "where things live" sections below the run were carried from the SLIM-FIX rewrite and
re-checked against `configs/` and `census.py` at this tip.
