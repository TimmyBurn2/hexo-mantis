# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where the frozen register disagreed with
the tree, the tree won and the disagreement is recorded in the last section.

## Current phase

**run6 is MINTED and HELD. It has never started.** R345 accepted AUDIT-2 as evidence and held
run6 for REPAIR-A2, a seven-leg box packet with a planted break and a mutation self-test per leg.
Seven of eight legs landed; **leg 5 (the MCTS root child cap) halted to the architect with
numbers** and is what run6 is held on: the cap discards a mean 88% of the policy's prior mass on
99.97% of expansions, the tree-memory delta is exactly zero (the pool is preallocated at
MAX_NODES), and what a larger K costs is the armed-sims ceiling — which gates configs.

**`R319(d)` is LIVE and MINT-BLOCKING, and it is the heaviest of these.** R339(b) adjudicated it by
measurement and R340(a) closed it by re-reading the failing round as a weak-net artifact; **R341(b)
WITHDREW that closure and refuted it by measurement.** Run 2's step-1000 gate round ran 3600.11 s,
was SIGTERM'd (`eval_broken`, `reason: round_timeout`, exit -15) and returned `wr_sealbot: null`,
`promoted: false` after 139 games. R340(a)'s 2.3x headroom came from STANDALONE rounds on an idle
box; under contention with 16 self-play workers and the trainer the same round is ~7x slower per
game (4.0 -> 25.9 s/game) and does not fit. The candidate was the warm-started BC net at a 31-ply
median — R340(a)'s own regime — so "weak-net artifact" is not the explanation, and R319(d)'s
original failing round agrees with this one to within 11%, which lifts it above n = 1. The archive
states the consequence outright: **run6 as minted CANNOT EVALUATE DURING A RUN.** The withdrawal is
HOST-INDEPENDENT, and **no clause in R342, R343, R344 or R345 touches R319** — nothing re-closes it.

A third row is mint-blocking with no close recorded: **`F-816-24`** — `monitor/supervise.py`
constructs a bare `MonitorConfig()`, so every minted `supervisor_*` value reaches no process. A fix
packet was ordered at R291(b); no merge or close appears in the frozen record.

Owed, named so it is not mistaken for done:

- the leg-5 cap value — the thing the hold is on;
- the `AUDIT_2026-09-09.md` analysis text, never forwarded, so the label AUDIT-2 points at an
  absent document;
- the `supervisor_kill_grace_sec: 600.0` reading. It was armed on the interpretation that
  "only if the supervisor is used" describes when the value takes effect rather than
  conditioning the change. A one-line re-mint if the operator reads it the other way.

Riding the run rather than holding it: **`F-816-37` is open and not root-caused.** Every firing on
record is on the host R341 condemned and R342 downgraded to suspect, and the work moved to a
different box — so the halt is spent, the class is not. Its 1-in-1 eval-path instrument with
dump-on-fire is in the protected set. See `docs/governance/CARDS.md`.

## Minted values — `configs/run6.yaml`, verified at HEAD

The config carries 19 `# delta:` lines and all 19 are applied in the body.

| key | minted value |
|---|---|
| `run_id` / `seed` | `run6` / `20260718` |
| `identity.encoding` | `gnn_axis_r8` |
| `identity.representation` | `graph` |
| `identity.arch_kind` | `GnnArchV2` |
| `identity.warm_start.checkpoint` | `checkpoints/bc/run6_00006500_5191bd09.ckpt` |
| `identity.warm_start.net_hash` | `2e72abd44ff13d47ec7de6ecf8824f99045c5b992b014f5063442dc7cded3f65` |
| `allocator_posture` | `expandable_segments` |
| `selfplay.n_workers` | `16` |
| `selfplay.max_game_moves` | `256` |
| `selfplay.mcts.n_simulations` | `50` (PUCT; 96 was refused on projection) |
| `selfplay.gumbel_mcts` | `false` (`gumbel_m 16`, `gumbel_explore_moves 10`, `gumbel_variant legacy`) |
| `selfplay.playout_cap` | disarmed — `fast_prob 0.0`, `full_search_prob 0.0` |
| `train.checkpoint_interval` | `1000` |
| `train.eval_interval` | `1000` |
| `train.min_buf_size` | `4096` |
| `train.batch_size` / `replay_capacity` | `256` / `100000` |
| `train.microbatch_caps` | `max_edges 4500000`, `max_nodes 170000` |
| `train.draw_rate_abort` | `threshold 0.25`, `min_step 25000`, `N_pool_min 50`, `consec 3` |
| `train.amp_dtype` | `fp16` — INERT on this run; see the note below |
| `inference.fused_graph_caps` | `max_fused_edges 1373143`, `max_fused_nodes 56645` |
| `eval.gate.stride` | `3` |
| `eval.gate` geometry | `screen_games 80`, `confirm_games 128`, `promotion_winrate 0.55`, `screen_confirm_lo 0.44`, `deploy_sims 150`, `min_distinct_per_pair 10`, `bootstrap_resamples 1000`, `seed_base 20260625`, book `book_v1_s20260625_p4` |
| `eval.concurrency` | `8` |
| `eval.strength_floor` | `probe_games 4`, `min_decisive_rate 0.25`, `min_winrate 0.0` |
| `eval.random_floor_games` | `20` |
| eval sims | `random 96`, `sealbot 128`, `kraken 128`, `strix 128` |
| `monitor.gate_interval` | `1000` |
| `monitor.supervisor_kill_grace_sec` | `600.0` |

**Radius is 8 on this lineage.** `crates/mantis-encoding/src/registry.toml` `[encodings.gnn_axis_r8]`
sets `graph_radius = 8` and `legal_move_radius = 8`. `gnn_axis_v1` stays at radius 6 and is run5's
identity, unmutated.

**`train.amp_dtype: fp16` does not reach run6's forward.** `mantis.model.amp.amp_dtype_for`
returns `torch.bfloat16` for the graph representation and ignores the declared value (LAW-06 pin);
the declared key is live only for the grid representation. A reader taking `fp16` from this config
as run6's autocast dtype would be wrong.

## Armed rows — `src/mantis/config/armed_aborts.py`, verified at HEAD

Eight rows: six REQUIRED, two DEFERRED. **All six required rows are armed in `configs/run6.yaml`.**
The dataclass enforces that a deferred row carries an owner and a source pin, and that a required
row carries no owner.

| row | status | arming key | armed in run6 |
|---|---|---|---|
| `actor_lag` | REQUIRED | `monitor.actor_lag_abort_enabled` | yes — `true` |
| `draw_rate_collapse` | REQUIRED | `train.draw_rate_abort.threshold` | yes — `0.25` |
| `disk_space_exhausted` | REQUIRED | `monitor.disk_guard.fail_gb` (+ live producer probe) | yes — `5.0` |
| `terminal_eval_broken` | REQUIRED | `train.terminal_eval_enabled` | yes — `true` |
| `fused_graph_caps_calibrated` | REQUIRED | `inference.fused_graph_caps.max_fused_edges` | yes — `1373143` |
| `allocator_posture_minted` | REQUIRED | `allocator_posture` | yes — `expandable_segments` |
| `grad_norm_hard_abort` | DEFERRED | `train.hard_gn_threshold` under ceiling `monitor.alert_grad_norm_max` | no — `1e9` is above the ceiling, so the predicate is false |
| `sealbot_wr_abort` | DEFERRED | `monitor.wr_hard_abort_enabled` | no — `false` |

Config partition holds: `PRODUCTION_CONFIGS` = run5, run6, shakedown_20260807; the other five
`configs/` files are exempt. 3 + 5 = 8 = the file count.

## Protected set

The eleven protected items are listed in `docs/governance/LAWS.md`. That list is the whole of what
ruling-protected code means; nothing on it moves without a ruling that names it.

## Open cards

`docs/governance/CARDS.md`.

## Where the frozen register disagreed with the tree

The tree won every one of these. They are recorded so a reader of
`docs/governance/archive/` knows which of its claims not to carry forward.

- `eval.concurrency`: archive carries "= 1 IS THE RUN6 VALUE" and, in a live R336(d) row, "eval
  concurrency has no config key at HEAD, run6 runs SERIAL, G=1". The key exists in the schema and
  run6 mints `8`.
- `eval.gate.stride`: R344's landing text says `1`. run6 mints `3` (R345(c) moved it).
- `train.eval_interval` / `train.checkpoint_interval`: R343's body says `750` for both. Both are
  `1000`. R343's foot annotation corrects this; the body text was never repaired, by design.
- Run length: R341(e) and R343(f) are carried as LIVE at a 12 h block. R344 §0.5 superseded that
  with a 25 001-step minimum, and the live rows were never annotated.
- Mint size: the archive says run6 is minted at 15 deltas. It carries 19.
- Radius: the archive's §2 LOCK still reads "R26/R238 — radius = 6, never 8" with no amendment on
  the lock line itself. run6 is radius 8 by R328(b); §3 annotates the inversion, §2 does not.
- `eval.random_floor_games`: the archive's R147 row says "every other committed config carries 4".
  run6 carries `20`, so the claim is false of exactly one config.
- R345 — the ruling that holds run6 — has NO row in the archive's §5 live-force section. Its rows
  stop at R337. R345 exists there only in a header stamp and the curation log.
- Leg count: two consecutive curation entries say "seven severable legs" and "seven of eight legs
  landed".
- Every `configs/run6.yaml` line citation in the archive is off by 3 or more; two of them give
  different line numbers for the same key. Derive coordinates at point of use, never carry them.
- Gate/test figures: the archive's last entry records `dev` at `7561169` with a 5 008-test default
  tier. The floor file at HEAD reads `5081`, and `dev` has moved past that commit.

## Exit facts — R346 (CLEANUP ERA), 2026-09-09

- Governance moved to `docs/governance/`: `LAWS.md`, `STATE.md`, `RULINGS.md`, `CARDS.md`,
  `falsified.md`. `docs/registers/` is dissolved.
- The old register, the ACTIVE index and the pre-R346 `laws.md` are frozen under
  `docs/governance/archive/` behind one README, with no tooling. `git mv` was used, so history
  follows all four files.
- Census, stamp, mirror and sync tooling: **nothing to delete.** None of it was ever in this
  repository — it lived in the migration workspace. The only trace is a comment in
  `tools/ci_gates/tier_census.py` naming two tools that do not exist here (carded).
- Sitting records and `plan/`: never tracked here. Nothing moved, nothing is missing.
- `falsified.md` de-duplication: no repeated entry existed, so no row was removed.
- `RULINGS.md` carries 322 entries over 321 numbers, R23 to R345, plus 20 register annotations.
  R227 and R228 have no entry, both on operator direction.
- The CLEANUP ERA ruling is **R346**, derived as max + 1 from the archive's own census
  (`R23-R345, 316 sections, 316 distinct numbers`) and confirmed by a whole-tree grep: `R346`
  appeared nowhere before this branch.
- Gate set: **unchanged.** No gate was added, renumbered or repurposed. Gate 17 in this repo is
  `tools/ci_gates/rule7_gate.py` (rule-7 host content), not a governance gate, and it stays.
- Gates run on this branch: 6 (artifact) green, 17 (rule 7) green, **10 (tracked refs) RED** on
  five CLAUDE.md lines that still name `docs/registers/`. See CARDS.md — this is the one
  blocking item and only the operator can clear it.

## Provenance

Derived 2026-09-09 on branch `gov-1`, from `configs/run6.yaml`,
`src/mantis/config/armed_aborts.py`, `src/mantis/config/schema/`, `src/mantis/model/amp.py`,
`crates/mantis-encoding/src/registry.toml` and `tools/ci_gates/test_count_floor.txt`. Ruling
texts: `docs/governance/RULINGS.md`; frozen sources: `docs/governance/archive/`.
