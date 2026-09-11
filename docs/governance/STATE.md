# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase

**run6 is MINTED and HELD. It has never started.** WAVE 3 (R348) is in progress on branch
`wave3`; `dev` is at the wave-2 exit until the full gate set is green on the box. What run6 is
held on is unchanged in kind — the **wave-3 re-mint** — and one new host fact stands in front of
it (next section).

**The config still mints `search.kind: puct` at 50 simulations.** The kind row and the sims regime
are OWED to the re-mint. A reader taking `puct`/`50` from this config as run6's search would be
reading the pre-re-mint state correctly. **New for the re-mint, found by minting the PERF-3b
throwaway:** the schema pairs `train.policy_target` with `search.kind`, so the re-mint moves
`raw_visit_distribution → completed_improved_policy` in the same act or the mint is refused.

**Wave 2 is RATIFIED (R348 §0) and its three deviations are ruled:** `configs/` keeps three
files, `mint_config` keeps its `--set`/`--mint-row` split, and the CUDA index pin is a uv EXTRA —
landed at `669b6008`: a bare `uv sync` installs the CPU wheel through a default group,
`make build.cuda` (`uv sync --extra cuda --no-group cpu`, the one shape uv accepts) installs
`+cu128`, and the box's `box_setup.sh` syncs that way and HALTS on a `+cpu` venv.

## The halt in front of the re-mint: the box has no persistent volume

`/` and `/workspace` on the box are `overlay`. R347(d)'s START halt (rc 16) therefore refuses
**every** preflight there, no stamp can be written, and `mantis.run` (R348(c)) refuses to launch.
That chain is working exactly as ruled; it means the shakedown and START cannot happen on this
instance until a persistent volume is attached (or the mirror arm is proven off-box). This is the
first operator action of the re-mint, not a code item. PERF-3b's measurement burst drives
`launch_run` in-process for that reason, stated in its driver, and produces no stamp.

## CARD-OC7-OVERRUN — discriminated: HOST, not code

Record: `docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`. The row **passes in 178.4 s on
the box** at HEAD and the minted 14 workers. On the dev box (Ryzen 7 3700X, AVX2, no AVX-512
BF16) LAW-06's bf16 autocast on the CPU trainer takes ATen's generic path — one GEMM at the
trainer's edge shape measures **72× slower than fp32** — so a step is ~40–60 s and the docstring's
rule yields no bound. The 2026-08-01 tree is equally slow there today; no bisect. The bound is
NOT re-aimed: which of (the tier runs on the box / the row is `slow` / a LAW-06 CPU carve-out) is
the operator's.

**The integration tier now runs on the box in ~20 min.** Running it found what the dev box
could not: DELETE-1 had shrunk the Rust `GameResultRow` 10→8 and left the Python drain unpacking
10, so the real self-play seam died on its first game while every default-tier fake stayed green
(fixed `92643671`, arity parity test added); CONFIG-1 had moved `supervisor_*` keys to schema
defaults and the supervisor witnesses still `--set` them; the eval-round oracle still passed the
deleted grid arm's `inference_batching=None`; and R347(d)'s durability halt refused every
preflight-driving test's tmp dir (tmpfs/overlay). All four repaired at `9137ab1a`; the preflight
reads a planted mount table only through `MANTIS_PREFLIGHT_MOUNTS_TABLE`, the reading names the
table it used, and the stamp REFUSES any table but `/proc/mounts`, so the seam launches nothing.
The tier's box reading after the repair is in the exit facts.

## R348(c) — the preflight stamp, landed at `e1e06843`

`tools/ci_gates/preflight_mint.py` writes `$XDG_STATE_HOME/mantis/preflight/<config_sha>.json`
(else `~/.local/state/...`) after `_verdict_exit`, carrying config hash, tree SHA + dirty flag,
timestamp, the two START halts' readings, the booted (burst) hash and the report path; it clears
any earlier stamp for that identity at entry. `mantis.run.main` refuses by NAME — missing,
malformed, planted table, or a stamp from another HEAD — before `launch_run`. The store is host
state and dies with the host, which is the R302(c) semantics ("instance recreation voids them").

## `F-816-24` — FIXED IN CODE since `c8bd7190` (2026-08-21); the close is a ruling line

Read at contact for AUDIT-3's lead: `supervise.main` loads the minted config and resolves every
`monitor.supervisor_*` through `resolve_monitor_config`; the witnesses are in
`tests/monitor/test_supervisor_config_witness.py`. The record said LIVE because no ruling closed
it. Still riding the run, not holding it: **`F-816-37`** (open, not root-caused; its 1-in-1
eval-path instrument with dump-on-fire is protected). **`R341(b)` / `R319(d)`** stay DISCHARGED
AT G=8 ONLY (R343(a)) — unchanged.

## Minted values — `configs/run6.yaml`, byte-unchanged since `24c9dab0`

The table in the wave-2 STATE stands verbatim; nothing under `configs/`,
`src/mantis/config/armed_aborts.py`, `crates/mantis-search/src/mcts/`,
`crates/mantis-encoding/src/registry.toml` or `docs/governance/LAWS.md` has changed since the
wave-2 exit (`git diff --stat 24c9dab0 -- <those paths>` is empty). Search-engine constants,
HEXG v2, radius 8, the resolved-config write: as recorded there. Gate 12 is GREEN at HEAD.

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws; no gate number added (still
17). Open cards: `docs/governance/CARDS.md` — `CARD-OC7-OVERRUN` carries its discrimination in
place; `CARD-GATE17-LOCAL-COUPLING` unchanged; `CARD-CLAUDEMD-REPOINT` closed by R348.

## Exit facts — WAVE 3 leg 1 (OC-7 → box rebuild → traps), 2026-09-11

- Ruling: R348 at `63671f08`, verbatim; `RULINGS.md` numbering continues at R349.
- Commits on `wave3` since `b117e657`: `92643671` drain fix, `63671f08` R348, `e1e06843` stamp
  trap, `669b6008` cuda extra, `5e529a44` gate-17 lock carve-out, `ad747400` OC-7 record,
  `9137ab1a` tier repairs. One line each, empty body, zero trailers.
- **Collected tests: 4,530 → 4,559** (gate 3c floor file still 4,530; ratchets at the merge).
- **Tree: 878 tracked files, 389,706 lines** (`git ls-files | xargs cat | wc -l`).
- Comment ratchet: floor lowered 13561 → 13543 docstring lines across the leg; comment and
  banner measures held at 3551 / 23.
- Gate 17 gained a carve-out: in `uv.lock` an IPv4 counts only as a URL host, because the cuda
  extra's four-part CUDA library versions pass the octet ranges; proven both ways in the
  self-test.
- Box: rebuilt through `box_setup.sh wave3` with the extra — `torch 2.11.0+cu128`, RTX 5080,
  extension, vendored sealbot and the warm-start checkpoint all present; host identity lines are
  in the box's own setup log. The instance is on the same physical host the archive records.
- Local gates on the dev box at `ad747400`: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 GREEN;
  3a not re-run since the leg's targeted suites; **3b cannot run on the dev box** (above);
  2a/2b/4/5 not run here (no Rust changed this leg). The box carries the full set at the exit.
- PERF-3b: **NOT YET MEASURED at this handoff** — the throwaway config is minted (run6's deltas
  + `search.kind gumbel`, `n_simulations 320`, PCR `full_search_prob 0.25 / n_sims_full 320 /
  n_sims_quick 64`, `deploy_sims 160`, `policy_target completed_improved_policy`, bounded to 200
  steps with one gate round at 150) and the burst is queued behind the tier on the box. The
  checker-thread lever of R347(e) does not exist in code (`verify_edge_geometry` still runs inline
  in `_check_semantic`), so its A/B is OWED behind the lever itself.
- Two harness defects of this leg's own making are recorded so they are not re-learned: a
  py-spy wrapper without a `__main__` guard re-ran pytest inside the eval worker's spawn child
  (fixed before any box number was taken), and the comment ratchet counts TRACKED files only, so
  a green measured before `git add` is not a measurement.

## Provenance

Derived 2026-09-11 on `wave3` at `9137ab1a`, from `configs/run6.yaml`, the box's
`/workspace/oc7/{box_head,tier_box.log}` readings, the dev box's scratchpad `oc7/` artefacts,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
