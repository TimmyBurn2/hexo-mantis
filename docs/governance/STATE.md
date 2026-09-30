# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register; where a register disagrees with the tree, the
tree wins and the line says so. Counts are derived at point of use from the file or command named,
never transcribed here. A reader who finds a line stale repairs it in place (R311(c)).

## Current phase

**R379 (2026-09-30): THE SECOND TWIN. TACTICS-SELFPLAY's screen stands and the lever is not adopted from it; the
feed is ruled, and TACTICS-SELFPLAY-2 is ordered (CARD-TACTICS-SELFPLAY-2). CENSUS-1 is accepted.** The tactics
lane still goes first and run10 will not START (R376(c)). `configs/run10.yaml` stays the production config the
instruments read until run11's mint replaces it. run11 arms the deploy block at its mint by the operator's word
(R378(a)), and in self-play only on TACTICS-SELFPLAY-2's pass (R379(c), CARD-RUN11-DESIGN).

- **R379's order.**
  - The readings (R379(a)): the T4 and defence exams are the starvation instrument; the deploy-matched head is the
    strength reading for a tactics-era net (LAW-15, R378(b)); the net alone is a report-only diagnostic pooled over
    at least two saves.
  - The feed (R379(b)), at deploy and in self-play alike: a root whose every searched move is vetoed is re-searched
    over the non-vetoed set at the same budget, and that search's improved policy is the row's target and its winner
    the played move. A proven root's target mixes the proof at α = 0.5 only where the searched mass on it reads below
    0.5, row by row, replacing R378(e)'s median rule.
  - TACTICS-SELFPLAY-2 (R379(c)): one arm with the feed against the recorded arms; it passes when the deploy-matched
    reading's lower bound is above zero.
  - CENSUS-1 (R379(d)): growth, the regret and restart family, the next-ply aux head and learning from Six's
    positions are parked (CARD-NET-EXPAND held, CARD-JK-LAST carded for throughput); six items are carried to run11's
    design; MCGS earns a deploy A/B later.
  - PERF-2 (R379(e), CARD-PERF-2): the copies first, then the edge table, then CUDA graphs.

- **The tactics lane (R376(d), CARD-TACTICS-LANE).** ONE exact tactics module in Rust on the search path, used
  identically at deploy and in self-play (LAW-15). Deploy lands first, read as an A/B against the plain parent
  on both rulers. Self-play follows only with a pre-registered starvation witness, and with F-15, F-39, F-53
  and R239 re-validated under LAW-02 first. A solver terminal counts as a simulation, GPU evaluations are
  their own LAW-18 row, and served-sims exactness pins descents (R376(e)); a descent that ends at a proven
  terminal is one (R377(c)). A descent that backs up a value is a simulation, whatever backed it: the net, the
  table or the solver (R378(c)). The plain head's early end is a defect, fixed before the twin
  (CARD-SIMS-ACCOUNTING).
- **TACTICS-DESIGN exited 2026-09-28; R377(a) accepts it.** Its design is
  `docs/design/TACTICS_DESIGN_2026-09-28.md`, and its records are local.
  - On 8 966 quiet positions, Six's kind (a turn-level, strictly forcing threat-space search) proves 2 119–3 506
    and our per-stone `TacticalSolver` proves 0; none of the 7 473 WIN claims is refuted.
  - R377 makes Six's kind the core and retires our solver when the module lands. v1 proves wins and can't-cover
    losses. The quiescence override and its blend stay behind the shared analysis function.
  - TACTICS-DEPLOY reads the full module and an audit-off arm against the plain parent on both rulers, X's floor
    first. The defence audit lands only if its arm earns its cost: of 20 of D1's "safe" alternatives played out
    with Six on both seats (T3), 16 were still lost.
- **TACTICS-DEPLOY exited 2026-09-29; R378(a) ratifies it, and it merged (`dev` is its tip `0204edf9`).** It built
  the module and wired it at deploy with `search.tactics` null in both homes (the self-play path unchanged). It read
  the A/B on the box in 9.78 of the 12 box-h granted.
  - Lever X: +0.957 logit [0.752, 1.162]; lever S: +0.973 [0.663, 1.284].
  - The audit: +0.254 [0.049, 0.459] over audit-off on X.
  - The known-bad reads 0.052; no found proof's game was lost.
  - The wiring alone reads +0.027 over plain on X.
  - The read's block, leaf 256/3, root 20 000/8 and audit 2 000/8, is the deploy block of record. Every config still
    mints `tactics: null`.
  - The in-run gate is deploy-matched: candidate and anchor play the same block (R378(d)). The tree still arms the
    candidate alone (CARD-GATE-DEPLOY-MATCHED).
- **TACTICS-SELFPLAY exited 2026-09-29: NOT PASS-TO-RUN11 (CARD-TACTICS-SELFPLAY); R379(a) lets its screen stand.**
  It put the same module in the loop: a 4 h tactics twin (A) against a 4 h plain twin (B) from run8@45k, in 11.86
  box-h. Its code is on branch `tactics-selfplay` (gates.exit green), which `dev` contains (this line said "not yet
  in `dev`" until R379's record).
  - P0 closed CARD-SIMS-ACCOUNTING. P1 picked the SEARCHED proven-root target; P2 and P3 set the defence and ring
    bands. P4's cache hit reads 33 % at sync cadence 50, against 24.5 % at 2.
  - The screen fails: the net alone at 12k reads A − B −0.209 logit against a bar of > −0.17. Every band held, and
    throughput passed at 1.058× in positions/h.
  - Beside it, report-only: the shipped head at 12k reads A − B +0.676 [+0.342, +1.010], and the net alone pooled over
    9k and 12k reads +0.010.
  - CARD-TACTICS-TARGET-FEED (the module's moves reach the policy weakly) is ruled by R379(b);
    CARD-THROUGHPUT-IN-POSITIONS stays carded.
- **The strength series changes unit at the merge (R378(b)).** Every later cell plays the shipped head, earlier
  points are the plain unit, and both are named where they meet. A read that needs the net alone plays
  `--arm plain` and says so.
- **CENSUS-1 is accepted (R379(d), CARD-CENSUS-1); PERF-2 follows it (R378(g)).** Its record is
  `docs/design/measurements/CENSUS1_2026-09-29.md`. LEVERS_RESEARCH is its background (local, in
  `mantis-records/research/`), accepted with its §0 corrections. The in-run cache hit rate is measured, though:
  24.2 %.
- **The first stone is the origin (R377(g), CARD-ORIGIN-RULE).** The empty board's legal set becomes {origin},
  records and books are canonicalised by translation on load, and the frozen fixtures are re-pinned under grant.
  It lands in RUN11-PREP, and the LAWS bullet for arena legality moves with it (R378(h)). The ruler's book stays
  for the series; a random-opening share in self-play is a run11 arm (R377(h)).
- **Two rulers (R376(a)).** Six gen 30 @16, cache off, is X; strix @ r8 is S. Recipe decisions read X;
  milestones read X and S. SIX-RUNG landed X by pin and hash (`mantis.bots.six`, the follower's `six30_16`
  unit); it re-reads the parent 0.351 [0.295, 0.406], SIX-SCOUT's reading exactly.
- **DECIDE-1 is accepted (R376(b)).** Over run8's saves 36k–54k the process level is ≈ 0.11 on S and ≈ 0.29
  on X; the parent's 0.191 on S is an outlier, and every bar prices from a panel. In 106 of S1's 233 losses
  (0.455) one of our turns allowed the opponent a proven forced win where a safe turn existed, 100 of them our
  last turn before that run; Six's threat solver is worth ≈ 1 logit against the parent. That is the tactics
  lane's evidence. The damage-vs-luck noise nets were void (the
  KL-1.0 known-bad read 0.12 on X), and cooldown and EMA are neither shown nor excluded: they enter run11's
  design as screened levers with power lines (R376(g)). Records are local.
- **Six's outputs (R376(f)).** Learning from them is permitted as a means, never the end: probes first, a run
  only on a pass, and a net that learned from Six carries it in its lineage. The standing goal is to surpass
  Six by self-play with exact tactics.

The earlier phase paragraphs (DECIDE-1's order, RUN10-CONTROLS, SIX-SCOUT, RESEARCH-FORGE) are
`git show 3e3fd263:docs/governance/STATE.md`; the ones before them are `git show f24d3ea8:docs/governance/STATE.md`.

## The run

- **No run is live.** run7 stopped at step 83 482 on 2026-09-18 (the stop is recorded by commit
  `8cb5ca6a`; `docs/design/measurements/EVAL_COST_2026-09-19.md` reads its rounds) and run8 at 55 170 on
  2026-09-21 (R365); run9 was never started and its config is deleted (R365(a), R367). The strength
  series and every cell on record are in the measurement records named below.
- **run10 will not START (R376(c)).** `configs/run10.yaml` stays the production config the instruments
  read until run11's mint replaces it. It was minted by R366 and ratified by R367(c); its order, witnesses
  and pre-registered reading are `docs/design/measurements/RUN10_PREREG_2026-09-21.md`, and its launch
  conditions were R370(i)'s and R371(c)'s. R376(b) prices every bar from a panel. SEAM-2 may merge before
  any run starts (R375(e) lifts R368(j)'s hold). Read a config's values from the file itself and diff two
  with `tools/config_diff.py`; STATE does not restate minted rows.
- **A box is still rented** (2026-09-24: an RTX 4080 SUPER host, the operator's, R11; PERF-ADA made it the run box; DECIDE-1,
  SIX-RUNG's witness and TACTICS-DEPLOY's A/B ran there, and TACTICS-SELFPLAY's P4 and twin hold a grant of ≤ 12
  box-h on it; this line said SIX-RUNG left it idle and clean until R378's record); the previous instance was
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
- Rulings: `docs/governance/RULINGS.md`; the latest is R379.
- Laws and the protected set: `docs/governance/LAWS.md`, whose protected set names each
  invariant's pinning tests (R370(f)); `tests/test_protected_set_pins.py` fails if one is gone.
  Falsified work: `docs/governance/falsified.md`.
- Build, test tiers and when gates run: `CLAUDE.md`; the gate set itself: `tools/ci_gates/run_all.sh`
  (`make gates`, `make gates.exit`). The test-count floor is
  `tools/ci_gates/test_count_floor.txt`; the comment ratchet's floors are
  `tools/ci_gates/comment_length_floor.txt`.
- The instruments at HEAD: `tools/strix_follower.py` (the strength series: strix @ r8, R366, and Six gen 30
  @16 by its `six30_16` unit, R376(a)),
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
The current phase was rewritten 2026-09-28 at R376 from the DECIDE-1 and SIX-RUNG exit records (local); the
run, box and instrument lines were re-checked at `3e3fd263`. The tactics lines were updated at R377 from the
TACTICS-DESIGN design doc (`d17dfdcd`), and at the TACTICS-DEPLOY packet's first commit. The current phase was
rewritten 2026-09-29 at R378 from the TACTICS-DEPLOY exit record (local) and the TACTICS-SELFPLAY packet, with
`dev` = `origin/dev` = `0204edf9` read before the edit. The TACTICS-SELFPLAY lines were updated 2026-09-29 at its
exit from its exit record (local), on branch `tactics-selfplay` at `5db3280b`. The current phase was rewritten
2026-09-30 at R379, with `dev` = `origin/dev` = `2d107c0f` read before the edit.
