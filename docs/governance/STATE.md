# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase

**run6 is MINTED and HELD. It has never started.** CLEANUP WAVE 2 (R347) is complete: three
agents merged, the grid/dense path is gone, the Gumbel row is sparse, and the comment rule is
enforced by a ratchet. What run6 is still held on is unchanged in kind — it is now held on the
**wave-3 re-mint**, not on a code defect.

**The config still mints `search.kind: puct` at 50 simulations.** R347(b) prereg'd the Gumbel
regime — full 320 / m 16 at p = 0.25, fast 64 / m 16 at p = 0.75, mean 128 — and R346(h) puts the
arming at the wave-3 re-mint, so the kind row and the sims regime are OWED, not armed. What R347
did move is minted and verified: `selfplay.c_scale 1.0` (was 0.1), `selfplay.gumbel_m 16`,
`train.fast_policy_weight 0.0`, `MAX_CHILDREN_PER_NODE 1024` in a `MAX_NODES 4_000_000` pool.
A reader taking `puct`/`50` from this config as run6's search would be reading the pre-re-mint
state correctly; a reader taking it as R347's intended regime would be wrong.

**`F-816-24` remains mint-blocking with no close recorded** — `monitor/supervise.py` constructs a
bare `MonitorConfig()`, so every minted `supervisor_*` value reaches no process. Ordered fixed at
R291(b); no merge or close appears in the record. Wave 2 did not touch it.

**`R341(b)` / `R319(d)` stays DISCHARGED AT G=8 ONLY (R343(a)).** run6 mints
`eval.concurrency = 8`, so it is not a hold. It stays LIVE at lower concurrency: G=1 is 53.33
s/game and G=4 is 13.99 s/game, both consuming the full 3600 s `round_timeout_sec` and returning
`wr_sealbot: null`, against G=8 at 7.09 s/game. The row is discharged **by the armed value, not by
the geometry becoming safe** — a session that lowers `eval.concurrency` walks back into a timeout,
and a null is not a slow reading: witness (iii) fits an Elo slope over at least 5 rounds and
cannot fit nulls.

Riding the run rather than holding it: **`F-816-37` is open and not root-caused**; its 1-in-1
eval-path instrument with dump-on-fire is in the protected set.

## The one thing wave 2 could not clear

**`CARD-OC7-OVERRUN` — BLOCKING the integration tier, and therefore `make gates.exit`.**
`tests/train/test_clean_stop_save.py::test_a_clean_run_at_the_minted_bound_leaves_one_stamped_checkpoint`
exceeds its own stated 300 s ceiling without bound (killed at a 300 s cap; past 900 s uncapped),
and the tier **with that test deselected** also failed to finish inside 3 000 s. Pre-existing on
`dev` — it reproduces identically at `MAX_NODES` 1 M and 4 M, so wave 2 did not cause it. Its
bound was fixed by a host-relative rule and measured at **240.2 s on 2026-08-01**; the measurement
no longer holds on the same box. **Not separated: host drift versus a slowdown landing after that
date.** Until it is, the bound must not be re-aimed — the test's own docstring records that the
measurements were taken before the row existed precisely so the bound could not be lowered to make
a red go away.

**The consequence, stated plainly: no `make gates.exit` run in this repository can complete.**
CLAUDE.md says remote CI is suspended and local green is the gate; while this stands, that gate is
one that never finishes, which is the false-clean class in the time dimension. Wave 2 ran **17 of
17 other gates green at every merge**, on operator direction, and claims nothing about the
integration tier. Separating drift from regression belongs to AUDIT-3 and is the first thing
wave 3 should do.

## Minted values — `configs/run6.yaml`, verified at HEAD

**127 leaf keys** (190 before wave 2: 40 deleted by ruling, 23 moved to schema defaults).

| key | minted value |
|---|---|
| `run_id` / `seed` | `run6` / `20260718` |
| `identity.encoding` / `representation` / `arch_kind` | `gnn_axis_r8` / `graph` / `GnnArchV2` |
| `identity.warm_start.checkpoint` | `checkpoints/bc/run6_00006500_5191bd09.ckpt` |
| `identity.warm_start.net_hash` | `2e72abd44ff13d47ec7de6ecf8824f99045c5b992b014f5063442dc7cded3f65` |
| `search.kind` | `puct` — **OWED: R347(b) prereg's `gumbel`, armed at the wave-3 re-mint** |
| `selfplay.mcts.n_simulations` | `50` — **OWED: 320 full / 64 fast at m 16, p 0.25 / 0.75** |
| `selfplay.c_scale` (Mctx `value_scale`) | `1.0` — moved from `0.1` by R347(b) |
| `selfplay.c_visit` (Mctx `maxvisit_init`) | `50.0` |
| `selfplay.gumbel_m` | `16` — both arms, R347(b) |
| `train.fast_policy_weight` | `0.0` — new key, R347(b); ablation, not a default |
| `selfplay.n_workers` | `16` |
| `selfplay.playout_cap` | disarmed — `fast_prob 0.0`, `full_search_prob 0.0` |
| `train.batch_size` | `256` |
| `train.eval_interval` / `checkpoint_interval` | `1000` / `1000` |
| `eval.concurrency` | `8` |
| `allocator_posture` | `expandable_segments` |

**Search-engine constants, `crates/mantis-search` (R347(c)).** `MAX_NODES = 4_000_000`,
`MAX_CHILDREN_PER_NODE = 1024`, `MAX_ROOT_CHILDREN = u16::MAX`. Both ceilings **re-derived, not
transcribed**: `MAX_ARMED_SIMS = 976` and `MAX_ARMED_SIMS_GUMBEL = 960`, each ≥ the 320 the prereg
regime needs. The ruling's own "976" reproduces exactly.

**HEXG is at v2.** The graph row is stored SPARSE — m explicit `(action, target)` entries plus one
tail mass α, bounded by the minted `gumbel_m`; a v1 file refuses **by name** rather than
mis-parsing a probability out of a stone coordinate. The trainer rebuilds the tail from its own
**detached** current prior renormalized over the unstored legal set.

**Radius is 8 on this lineage** — `crates/mantis-encoding/src/registry.toml`
`[encodings.gnn_axis_r8]` sets `graph_radius = 8` and `legal_move_radius = 8`.

**The resolved complete config is now written to the run dir before the first collaborator** and
re-validated from its own bytes, so a later default revision cannot rewrite what an old run used.

## Armed rows — `src/mantis/config/armed_aborts.py`

Gate 12 (armed-abort manifest audit) is GREEN at HEAD, which is the proof that every `required`
row is armed in every production config. `PRODUCTION_CONFIGS` is now **run6 + the armed preflight
smoke**; `configs/` holds **three** files, not the two R346(f) named — see the deviations below.

**Two START pre-flight HALTs are new (R347(d))**, keyed on what the config declares rather than on
sniffing the host: **rc 16** for a run dir that is not on a persistent volume, **rc 17** for a
cuda-declaring config on a CPU torch (a real matmul, not `is_available()`).

## Protected set

Listed in `docs/governance/LAWS.md`. Proven after every deletion in wave 2, not once at the end:
the 7 graph goldens are **byte-identical** across the whole wave (sha256 diff empty), and
`served_sims_exact` passed 6/6 at every checkpoint with no arm's asserted value moving.

**LAW-10 is DELETED** (R347(d)) — grid-era, no producer, gating nothing. LAWS.md carries a
tombstone; the number is retired, not reused. **Seventeen laws stand.**

## Open cards

`docs/governance/CARDS.md`. Two opened by R347: `CARD-OC7-OVERRUN` (above) and
`CARD-GATE17-LOCAL-COUPLING` (a tracked vacuity test binds its verdict to gate 17's *untracked*
local supplement, so a tracked test's outcome depends on local state). Four R346 cards closed.

**`CARD-CLAUDEMD-REPOINT` is discharged in fact** — gate 10 is green and `CLAUDE.md` no longer
names the dissolved `docs/registers/`. It is owed a closing line in a ruling entry, since CARDS.md
requires the closing reasoning to live there.

## Deviations from R346(f), unratified

Three, all reasoned in place rather than drifted. **A ruling corrects only by annotation, so these
stand as deviations until the operator rules on them.**

1. **`configs/` keeps THREE files, not "run6 and one smoke profile".** `dev_example.yaml` survives
   because ADJ-13 N-3 makes it gate 12's only disarmed real-tree red-capability demonstration:
   delete it and gate 12 can no longer go red on `configs/` at all. Grounds are written into
   `EXEMPT_CONFIGS`.
2. **`tools/mint_config.py` was NOT simplified.** Nothing in it became dead, and the
   `--set` / `--mint-row` split earns its keep more after CONFIG-1, not less — overriding an
   operational constant is now a minted row with a stamped header line.
3. **R347(d)'s CUDA index pin is NOT discharged.** The gate half is armed (rc 17). The pyproject
   pin is not: `[tool.uv.sources]` conditions on environment markers, which describe interpreter
   and platform and never GPU presence, so "host-conditional" needs a *chosen mechanism* — an
   extra, a sync-time env var, or a second lock — each of which changes what `uv sync` means, and
   `uv sync` is CLAUDE.md's one bootstrap contract. That is a design decision, so the CPU pin and
   its stated WP9 MKL/AVX512 parity grounds are untouched.

## Exit facts — R347 (CLEANUP WAVE 2), 2026-09-11

- Merge order held: GUMBEL-3 → DELETE-1/CONFIG-1 → COMMENT-1. Every merge `--no-ff`, gates run in
  the MAIN checkout rather than trusted from a worktree.
- **Tree: 448,148 → 388,478 tracked lines (−59,670), 995 → 872 files.** Measured across the whole
  wave, `fc951dc` (the ratified wave-1 exit) to `24c9dab0`, by counting blobs at each ref rather
  than by transcribing a figure. The intermediate points are `8fc3a94` 450,414 / 999 after
  GUMBEL-3 (which ADDED 2,266 lines) and `b2e72de4` 417,321 / 870 after the deletion leg. `archive/grid-path` was tagged before the first deletion and is pushed.
- **Collected tests: 5,096 → 4,530.** The fall is the grid-path suites; the rise back from 4,481
  is the comment-lint producer suite. Gate 3c grew a *self-expiring sanctioned ratchet-down
  record* so a ruling-ordered mass deletion must state its own decrease instead of the floor being
  edited down silently; that record was spent and deleted in the same wave.
- **Comment and docstring lines: 62,365 → 32,839 (−47.3%)**, banner lines 1,276 → 23 (−98.2%),
  with **zero non-comment tokens changed** — proven by a net hash, and independently re-verified
  by the coordinator over all 604 pre-existing `.py` files via AST comparison with docstrings
  blanked. Held by gate 14's comment ratchet, wired as its FIRST arm so a pyright refusal (a host
  condition, rc 2) cannot take the comment measures down with it. **No new gate number: still 17.**
- **The two deferred characterization tests were RUN at wave exit on merged `dev` and PASS** —
  `r153_characterize_exported_target_dropped_mass` 894.91 s and
  `r153_leg2_run5_exposure_through_production_expand` 561.55 s. They are 86% of the cargo gate,
  so they were deferred from each leg's pass and run once here; they were never `#[ignore]`d and
  the deferral is a schedule, not an exemption. Rust has no `slow`-tier equivalent to pytest's,
  which is why this had to be done by hand.
- **Two latent false-cleans closed by contact, not by search.** (1) Fifteen `RunnerStats` fields
  whose engine getters left with the grid path — one, `gridls_zero_policy_rows`, was still riding
  the LAW-18 `target_integrity` channel and would have published a permanent `0` as "measured,
  none found". (2) A tail-mass test planted 1 row in 8 and sampled 8 **with replacement**, missing
  its own subject (7/8)^8 ≈ 34% of the time; any other with-replacement assertion of that shape
  has the same exposure.
- **Gate 10 was scanning a dissolved directory.** It globbed `docs/registers/` under a *combined*
  `MIN_GLOB_FILES = 5` that `docs/contracts/` met alone, so it reported green over a scope nobody
  chose. Now per-directory floors plus a `DISSOLVED_PATHS` check; scan 15 → 18 files.
- **Rule 7's local supplement now exists on BOTH machines** (R347(d)), git-ignored, 12 live terms,
  and gate 17's operator-term arm runs. Adopting it found one real leak the tracked floor cannot
  catch: a partially-redacted git email in the frozen archive, where an earlier pass had redacted
  the `user@host` half and left the numeric account id beside it. Redacted in the established
  `[REDACTED:<pattern>:<sha256(match)[:8]>]` format. **It is in pushed history and was not
  rewritten** — the account is already named publicly in `LICENSE` and the remote URL, so a
  history rewrite of 634 commits was judged not worth it. That is a decision, not an oversight.
- **rustfmt: HEAD is not rustfmt-clean and this wave did not make it so.** 39 of the 84 touched
  `.rs` files were rustfmt-dirty at HEAD; 22 are now. No file that was clean became dirty.
  Formatting strictly improved; the rest was not COMMENT-1's to change under a
  zero-non-comment-token constraint.
- **Shell / TOML / YAML / Makefile comments are untouched** — ~1,190 lines. Shell heredocs make
  whole-line `#` deletion unprovable, so that class was left alone rather than asserted. The
  tree's longest narrative headers are now in `tools/ci_gates/*.sh`.
- `CLAUDE.md` was amended to record the ONE sanctioned exception to R316(e)'s "applied on contact,
  never as a cleanup pass" — R346(f) ordered a tree-wide pass, and the tension is now on the
  record rather than resolved silently.
- **PERF-3b did not run.** It is box-only and the coordinator has no box measurement leg; the
  protocol is written and the numbers are OWED — games/h at the mean-128 regime against R347(b)'s
  predicted 450–640, α distribution, KL witness reading, the checker-thread A/B, and eval s/game
  at G=8. R347(b)'s "~2–2.5 days per 25k-step block" is a PREDICTION with no measurement behind it.
- Two `wip:` commits (`3dd20b49`, `d20dcff2`) reached `dev`. One line, empty body, no trailers, so
  conformant in form, but "wip" states no reason. Not squashed: interactive rebase is unavailable
  in this environment and rewriting a 473-file branch to reword two subjects was judged the worse
  risk. Recorded rather than hidden.
- Every commit on the wave: one line, empty body, **zero trailers**, verified mechanically.

## Provenance

Derived 2026-09-11 on `dev` at `24c9dab0`, from `configs/run6.yaml`,
`src/mantis/config/armed_aborts.py`, `src/mantis/config/schema/`,
`crates/mantis-search/src/mcts/{node,mod}.rs`, `crates/mantis-selfplay/src/replay/hexg/`,
`crates/mantis-encoding/src/registry.toml`, `docs/governance/LAWS.md` and
`tools/ci_gates/test_count_floor.txt`. Gate readings are from the wave's own runs in the main
checkout. Ruling texts: `docs/governance/RULINGS.md`.
