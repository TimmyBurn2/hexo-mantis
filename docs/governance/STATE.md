# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register; where a register disagrees with the tree, the
tree wins and the line says so. Counts are derived at point of use from the file or command named,
never transcribed here. A reader who finds a line stale repairs it in place (R311(c)).

## Current phase

**R384 (2026-10-04): THE RUN STARTS FROM THE READ. REG-1 is accepted: run11's value loss takes the re-drawn per-step
mask at 1/8, the ring's root value stores Σπ′·completedQ with the proof override, and the calibrated floors are the
halting rows. PERF-2 is ratified. The Bubble study is accepted as evidence; `book_v2` is CLOSED. RUN11-PRE is run11's
first leg: four arms from run8@45k, the picked arm continuing as run11. `reg-1` rebases onto `dev`, gates and pushes.**
run10 will not START (R376(c)). `configs/run10.yaml` stays the production config the instruments read until run11's
mint replaces it. run11 arms the deploy block at its mint by the operator's word (R378(a)), and in self-play arm A's
design, carried as a screen (R381(c) as re-stated by R384(c), CARD-RUN11-DESIGN).

- **PERF-3 EXITED 2026-10-04 (branch `perf-3`, unpushed; CARD-PERF-3-PACKET):** L2 (the cache key) and L3 (one trainer
  read per step) LANDED, L1 (a second server thread) CARDED; the production loop with the trainer 156 340 -> 170 940
  positions/h (+9.3 %). R370(c)'s cache-key wording is annotated (A1) on the operator's word to merge. Landing into run11 follows its prereg's
  resume rule, on the operator's word.
- **R384's order.**
  - REG-1 accepted (R384(a), CARD-REG-1 CLOSED):
    - run11's value loss uses a re-drawn per-step mask at p = 1/8 (CARD-VALUE-MASK-REDRAWN): effect of record −0.018
      nats at 8 draws, policy unchanged, calibrated exams held;
    - the ring's root-value field stores Σπ′·completedQ with the proof override (CARD-RING-V3-SEMANTICS);
    - the calibrated floors T4 V 0.154 / DEF V_att 0.100 are the halting rows of record;
    - the λ key is renamed to weight the search (0 = z) (CARD-RING-V3-KEY);
    - killed for run11: shrinkage (a fractional target needs a warmed head, CARD-FRACTIONAL-VALUE-TARGET-WARMUP), the
      head-LR factor, the fixed mask. The anchor stays unresolved.
  - PERF-2 ratified (R384(b)); trainer contention is a reason to read the step rate live, not to wait.
  - The Bubble study accepted as evidence on R383(f)'s terms (R384(c), CARD-BUBBLE-STUDY):
    - `book_v2` CLOSED: no power case; the natural book stays deferred;
    - R381(c)'s carry re-stated as A − parent ≈ +0.27, plain − parent ≈ −0.39, a screen;
    - the production run is the default state from run11 on; the censuses' serial sequencing is the architect's error.
  - RUN11-PRE (R384(d), CARD-RUN11-PRE) is run11's first leg, not a twin. Four arms from run8@45k under one
    pre-registration: 2.4 steps/game mask off (control); 2.4 mask 1/8; 1.2 mask 1/8; 1.2 mask 1/8 with a 32-sim quick
    arm at m 8. Matched wall-clock on matched hardware; the pick rule and halts are hashed before any arm starts; the
    picked arm continues as run11 from its last save. Arming the configs is the operator's word.
  - Bubble 185k against S and X, 288 pairs each, is recommended; running its weights is the operator's act. At or above
    our shipped head it becomes the third ruler (R384(e), CARD-BUBBLE-RULER).
  - Carded, not built before the start (R384(f)): within-turn tree reuse, head-to-head lever cells, the pair diagnostic,
    the Six-trainer read, ruler hygiene, the calibrated search-value teacher, the all-vetoed target fix. ORIGIN-1 is
    split: the deploy head plays the origin on an empty board now; the engine rule and canonicalisation land after the
    start by pinned resume.
  - Merge and boxes (R384(g)): `reg-1` rebased onto `dev` `9da49bdd` and pushed on 2026-10-04 (`origin/dev` =
    `ad3682d7`, local gates 24/24 green on it, CARD-REG-1-PUSH). Two matched boxes for the arms; renting is the
    operator's.
  - **RUN11-PRE: L1 LANDED on branch `run11-pre` (unpushed), gates.exit ALL GREEN on `6a1156b7`; L2 awaits the
    operator's word and the second box** (CARD-RUN11-PRE; local records `mantis-records/run11-pre/`). Landed: the
    re-drawn value mask (`train.value_mask_redraw_p`), the fill ramp (`train.training_steps_fill_ramp`), the ring's
    search value Σπ′·completedQ with the key renamed `train.value_target_search_weight`, the origin at deploy, the quick
    arm's own m (`selfplay.gumbel_m_quick`, its width in `search_levers`), the run monitor (`tools/run_monitor`).
    Contract v56, event manifest v12. Two fresh reviews, findings fixed. The prereg
    (`docs/design/RUN11_PREREG_2026-10-04.md`) is drafted, unarmed and unhashed; the four arm configs are drafted
    outside `configs/`. run10's ring bands would halt arm A's regime at its first save, so the prereg re-derives them
    on every arm-A-family ring on record. No arm is armed or started.

- **REG-1 EXITED 2026-10-03, ACCEPTED by R384(a)** (CARD-REG-1; branch `reg-1`; local records
  `mantis-records/reg-1/` EXIT.md + READINGS.md).
  - GEN: run8@45k frozen under arm A's config, 6 514 games. In-distribution held-out split: 1 800 games of the same
    actor.
  - Arms: 5 seeds at 8 draws per row.
  - The known-bad passes: +0.069 against a 2·SD of 0.009.
  - The pick by the hashed rule is the re-drawn per-step value mask at 1/8: −0.025 [−0.032, −0.018] calibration-free
    held-out value, effect of record −0.018. Policy is unchanged and the calibrated exams hold.
  - The direction holds at 4 draws: −0.015.
  - R1 v R2 reads "distinct boards", resolved but under 0.012.
  - Two-hot shrinkage kills the parent's value head. The value-head lr factor shows no effect of 0.012 or more, and the
    parent anchor is unresolved.
  - 0a: Σπ′·completedQ beats root W/N against z by 0.011.
  - The calibrated floors (T4 V 0.154, DEF V_att 0.100) are of record on GEN's held-out.
  - RUN11-PRE follows the ruling, with the mask on and a mask-off control.

- **R383's order.**
  - The clarification (R383(a)):
    - power guards nulls: a null is a bound only at power ≥ 0.8 for the pre-stated minimal effect;
    - the false-pass rate guards detections: a detection stands when its CI clears zero by the line, and its effect of
      record is the CI's bound nearest zero;
    - "screen, not adopt" binds marginal passes, not detections many SDs out.
  - CENSUS-3's readings under it (R383(a); CARD-CENSUS-3 CLOSED):
    - run11's planned 8 draws per row over-fits the value head on held-out games. D2 − D8 is −0.031 calibration-free,
      95 % CI [−0.039, −0.022], so the effect of record is ≥ 0.022.
    - The 13-draw warm start is the worst point.
    - Head shape is bounded at ≈ 0.009.
    - Weight decay and the 64-sim W/N target are null at this regime.
    - `tools/value_instrument` is the value reading of record (CARD-VALUE-INSTRUMENT LANDED, on branch `census-3`).
    - T4 V and DEF V_att are read on calibrated values, floors re-derived calibrated on run8's panel
      (CARD-EXAMS-CALIBRATED).
  - The problem and its levers (R383(b)):
    - measured as exposures per outcome bit;
    - the levers are the value label's reuse and the head's regularisation;
    - throughput (independent games per step) is a value lever; PERF-2 LANDED 2026-10-03 at +43 % positions/h
      in the production loop (CARD-PERF-2).
  - The warm start (R383(c)): training steps per game scale with ring fill (tspg × rows/capacity), holding draws per
    row at the steady state from the first row. R381(e)'s `min_buf_size` 100k is withdrawn.
  - Before the mint (R383(d)), the last two reads:
    - REG-1 (CARD-REG-1): the regularisers at run11's reuse on fresh parent games under run11's regime, an
      in-distribution held-out split, 5 seeds, a fixed-subset known-bad;
    - then RUN11-PRE (CARD-RUN11-PRE): the step rate read live at matched throughput on the instrument. Its packet
      follows REG-1's exit.
  - RING-V3 (R383(e)) is ratified.
    - The key is renamed to weight the search (0 = z) before any mint.
    - The field's semantics (root W/N v Σπ′·completedQ) are read on `search_stats` at 0 box-h, and the producer follows
      the read.
    - run11 trains at search weight 0; the field is recorded.
  - FORGE-2 (R383(f)) is accepted.
    - Derived numbers from archived scripts over our own records rank and card, never adopt.
    - Its record corrections land as annotations (CARD-FORGE2-ANNOTATIONS, LANDED 2026-10-02 at REG-1's first commit).
    - Five items are carded; the rest is killed as run11 levers.
    - ORIGIN-1 gains: the forced ply-0 row does not train.
  - Merge and box (R383(g)):
    - `dev` fast-forwards to `ring-v3`, and `census-3` is rebased onto it (done 2026-10-02: `ring-v3` + the instrument
      commit + this record).
    - The gates run on the combined tip before the push.
    - REG-1 needs the box within a day (≤ 40 box-h); keep or stop is the operator's.

- **R382's order.**
  - The void (R382(a)): every value reading ruled on through raw held-out value CE is void as evidence: CENSUS-1 C5's
    reuse reading (the audit's F1 counter-evidence) and any HL-Gauss line read on it. CENSUS-1's strength-based lines
    stand as screens.
  - The instrument (R382(b), CARD-VALUE-INSTRUMENT): calibration-free held-out CE with the temperature fitted on
    disjoint games, AUC, the temperature and the train/held-out gap, by ply band (0–10, 11–40, > 40); T4 V and DEF
    V_att co-primary; rings the parent never trained on; ≥ 3 seeds; the known-bad worse by ≥ 2 seed SDs before any
    arm runs. No instrument ranks a lever until it has read its known-bad. The same rows read at every run11 save.
  - The problem (R382(c)): one outcome bit per game shown ≈ 600 times on unique boards, so the head memorises game
    identity (0.16-nat gap, 2.7× overconfident, ≈ 0.08 nats of held-out skill, none before ply 10). The levers in
    order: a position-specific value target (z mixed with the search's root value), the label's reuse per game as a
    curve, the head's shape and activation re-initialised and warmed, weight decay. The dead opening head is a
    capacity defect with no shown cost.
  - The order (R382(d)), CENSUS-3 EXITED 2026-10-02 and ruled by R383(a): CENSUS-3 (CARD-CENSUS-3) reads those levers with 3 seeds and its picks set run11's value
    rows (R381(e) re-pointed). RING-V3 (CARD-RING-V3) lands the root value field with default-fill before the mint.
  - **RING-V3 EXITED 2026-10-01, RATIFIED by R383(e)** (CARD-RING-V3) on branch `ring-v3` from origin/dev `80b54f5a`, awaiting the operator's
    fast-forward: HEXG v3 with v2 default-fill, self-play writing the root value, `train.value_target_lambda` minted 0.0
    (the trainer bit-equal to v2 there; the eval step reads z whatever λ); gates.exit green. A λ > 0 run's held-out gap
    is not z against z until CARD-HELDOUT-GAP-Z lands.
    RESEARCH-FORGE-2 runs beside, read-only; nothing it proposes is adopted without a census.
  - HYGIENE-1 (R382(e)) is ratified with the operator-granted protected-set edits; the gate record keeps one shape
    across eras; the un-squashed pair stands. `dev` is fast-forwarded to `7978cf1b`.
  - The box (R382(f)): CENSUS-3 needs it within a day (≤ 32 box-h). Keep or stop is the operator's.

- **R381's order.**
  - The exit (R381(a)): F2 is killed by screen (R375(d)): A″ − A on the shipped head −0.48 [−0.80, −0.16] at one seed,
    with F-53's mechanism against it. The feed of record is arm A's: no re-search, no mixing. F2's code leaves the
    tree (CARD-HYGIENE-1); CARD-TACTICS-TARGET-FEED closes.
  - The instrument (R381(b)): a one-seed 12k-step twin reads starvation (the exams and bands) and nothing else, so
    R379(c)'s pass line was one it could not meet. Strength readings of self-play levers come from run panels (R375)
    or multi-seed twins with the count pre-stated.
  - run11's self-play (R381(c)): arm A's design is carried, not adopted, on two positive screens over plain (+0.65,
    +0.17) with the exams held at every save. run11's exams and ring bands halt it; its process level on the shipped
    head against the parent's anchor reads it. The fallbacks, in order: plain self-play under the tactics deploy
    (B's plain net took the full deploy lift, ≈ +1.1 logit; R376(d) annotated), then the audit's label-only design.
  - Before the mint (R381(d)): the training-path audit (`reports/training_path_audit.md`, local) is accepted, its F8
    evidence corrected by (c). On the desktop, HYGIENE-1 (CARD-HYGIENE-1) runs beside CENSUS-2 (CARD-CENSUS-2), and
    PERF-2 follows HYGIENE-1; ORIGIN-1 lands in RUN11-PREP. The natural book is deferred. Reuse stays at 2.4
    steps/game, read by run11's held-out gap. Carded, not built: the quick arm's noise, EMA as a shadow copy, the
    search-value target. Decided tails are parked; JK-last is dead for run11.
  - The mint rows (R381(e)): aux weight 2.0, `draw_reward` gone with reason-3 rows masked, `min_buf_size` 100k rows
    (WITHDRAWN by R383(c): steps per game scale with ring fill),
    sync cadence 50, and the value head and weight decay as CENSUS-2 reads them. A scalar re-mint inside a config's
    pre-registered envelope is a STATE line, not a ruling; the envelope is a prereg row (R381(f)). The value rows are
    re-pointed to CENSUS-3's picks (R382(d)).
  - **CENSUS-2 HALTED 2026-10-01 by its own known-bad** (CARD-CENSUS-2): raw held-out value CE read calibration, not
    skill. R382 rules it; its records are local.
  - The box (R381(g)): TACTICS-SELFPLAY-3 closes at 6.75 box-h. Nothing needs the box for two days or more; stopping
    it is the operator's act.
  - **HYGIENE-1 EXITED 2026-10-01** (CARD-HYGIENE-1) on branch `hygiene-1` from origin/dev `b1e34aa4`, ratified and
    fast-forwarded into `dev` by R382(e): gates.exit green; every item landed, the five touching protected pins under the operator's
    grant of 2026-10-01. PERF-2 follows it.

- **R380's order.**
  - The halt stands (R380(a)): no reading of A′ is a strength reading, and none of its saves is a parent candidate.
  - F1 is retracted (R380(b)): a root whose every audited move is vetoed records no policy target and plays the
    audit's best hold, as arm A did. The deploy block of record is unchanged. The mechanism of record, post-hoc and
    untested (R380(c)): F1's move rule ended each forced endgame at its first link (proven turn starts 282 against
    1,191), taking the 2–5-turn chain off the ring.
  - TACTICS-SELFPLAY-3 (R380(d)): one arm, A″ = arm A's feed with F2, 12k steps, same parent and seed, against the
    recorded A and B saves. A floor miss at any save halts and kills F2; T4 prior at 12k beside A@12k is F2's
    pre-registered direction; the pass line is R379(c)'s, the shipped head against B@12k.
  - The rules (R380(e)): T4's "all" floors halt; the per-length rows and every ceiling report. Throughput screens at
    0.9× arm A's positions/h over the same steps.
  - The branch (R380(f)): `tactics-selfplay-2` is not merged; TACTICS-SELFPLAY-3 starts from its tip, lands the
    retraction, and merges as one branch. The box stays through TACTICS-SELFPLAY-3, within 7 box-h (R380(g)).
  - **TACTICS-SELFPLAY-3 EXITED 2026-10-01 NOT PASS** (CARD-TACTICS-SELFPLAY-3), on branch `tactics-selfplay-3`,
    fast-forwarded into `dev` at `9e962173`. Every exam and ring band held at every save and throughput read 0.979× A's, but the shipped-head line
    missed: A″ − B@12k +0.174 [−0.158, +0.505]. R381 rules it.

- **R379's order.**
  - The readings (R379(a)): the T4 and defence exams are the starvation instrument; the deploy-matched head is the
    strength reading for a tactics-era net (LAW-15, R378(b)); the net alone is a report-only diagnostic pooled over
    at least two saves.
  - The feed (R379(b)), at deploy and in self-play alike: a root whose every searched move is vetoed is re-searched
    over the non-vetoed set at the same budget, and that search's improved policy is the row's target and its winner
    the played move (RETRACTED by R380(b)). A proven root's target mixes the proof at α = 0.5 only where the searched
    mass on it reads below 0.5, row by row, replacing R378(e)'s median rule (under test, R380(d)).
  - TACTICS-SELFPLAY-2 (R379(c)): one arm with the feed against the recorded arms; it passes when the deploy-matched
    reading's lower bound is above zero. **HALTED 2026-09-30 at arm A′'s 3k save** (CARD-TACTICS-SELFPLAY-2): T4 floor
    misses at every proof length (overall P 0.127, V 0.133 against A's 0.249 / 0.528) and the near-terminal band
    outside; no cell ran. The halt stands (R380(a)). The feed was built on branch `tactics-selfplay-2`, which reached `dev` through TACTICS-SELFPLAY-3 with F1 retracted (`3a7e4cd1`).
  - CENSUS-1 (R379(d)): growth, the regret and restart family, the next-ply aux head and learning from Six's
    positions are parked (CARD-NET-EXPAND held, CARD-JK-LAST carded for throughput); six items are carried to run11's
    design; MCGS earns a deploy A/B later. Five of them are opened as cards (CARD-VALUE-HEAD-DEAD-OPENING,
    CARD-QUICK-ARM-NOISE, CARD-SEARCH-VALUE-AUX, CARD-DECIDED-TAILS, CARD-MCGS-DEPLOY).
  - PERF-2 (R379(e), CARD-PERF-2): the copies first, then the edge table, then CUDA graphs; LANDED 2026-10-03.

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
- Rulings: `docs/governance/RULINGS.md`; the latest is R382.
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
2026-09-30 at R379, with `dev` = `origin/dev` = `2d107c0f` read before the edit. The TACTICS-SELFPLAY-2 line was
updated 2026-09-30 at its halt from its exit record (local), on branch `tactics-selfplay-2` at `2dde57aa`. The
current phase was rewritten 2026-09-30 at R380, on that branch at `7f351b54`. The TACTICS-SELFPLAY-3 line was
added 2026-09-30 at its packet's first commit, on branch `tactics-selfplay-3` at `aa0c5304`, and updated 2026-10-01
at its exit from its exit record (local), on that branch at `a3899487`. The current phase was rewritten 2026-10-01 at
R381, on that branch at `9e962173`. The HYGIENE-1 line was added 2026-10-01 at its packet's first commit, on
branch `hygiene-1` from origin/dev `b1e34aa4`, which also repaired the two stale "unpushed" branch lines, and updated
2026-10-01 at its exit from its exit record (local), on that branch. The current phase was rewritten 2026-10-02 at R383 from the CENSUS-3 exit record
(local), on branch `census-3` rebased onto `ring-v3` (`5b607454`). The REG-1 line was added 2026-10-02 at its packet's first commit, on
branch `reg-1` from `7b8d6a6e`, and updated 2026-10-03 at its exit from its exit record (local), on that branch.
