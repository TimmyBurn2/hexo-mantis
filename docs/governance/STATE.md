# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase

**run6 is MINTED and HELD. It has never started.** WAVE 3 (R348) is complete through AUDIT-3 and
REPAIR-A3 on branch `wave3` (OC-7 → box rebuild → traps → PERF-3b → AUDIT-3 → REPAIR-A3); `dev`
is at the wave-2 exit until the operator rules on the calls below. What run6 is held on is
unchanged in kind — the **wave-3 re-mint** — and two facts stand in front of it (next two
sections). The re-mint itself is STAGED as two branches the operator merges or discards:
`remint-warmstart` (the one forced row) and `remint-gumbel` (R348(e)'s rows as a proposal).

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

## run6's warm-start artifact no longer loads on this tree — the one red in the default tier

Wave 2's schema shrink (DELETE-1 / CONFIG-1) invalidated the stamped config inside
`checkpoints/bc/run6_00006500_5191bd09.ckpt` (49 `extra_forbidden` errors on read), so every
warm start of run6 refuses at `load_checkpoint`. Nobody saw it because the artifact is
gitignored and lived only on the box, and the box never ran the gate set after CONFIG-1. The
LAW-12 route is done: `strip_and_restamp` produced `checkpoints/bc/run6_00006500_ca1afb71.ckpt`
(weights-only, GnnArchV2, **net_param_hash unchanged** `2e72abd4…`), on the box and off-box.
The re-mint must move `identity.warm_start.checkpoint` to it — an identity row, so the
operator's — and until then
`tests/train/test_f32_launch_pin_wiring.py::test_the_minted_warm_start_row_names_an_artifact_whose_hashes_AGREE`
is a TRUE red wherever the artifact exists (the dev box now holds a copy). It is the only red in
the default tier (4,502 pass); it is not hidden by removing the copy.

## PERF-3b — measured (R348(d)); record `docs/design/measurements/MEASUREMENT_PERF3B_2026-09-11.md`

Gumbel arm on the rebuilt box, warm-started from the strip: **positions/h ≈ 43.4k, leaves/h ≈
5.6M at the mean-128 regime** (R347(b)'s ~4.4M leaves/h holds), games/h 1,125 steady at 38.6
plies/game (2× the prediction only because games are half the assumed length — games/h is not
the wall-clock line). The block is **trainer-bound**: 678 steps/h at 2.85M edges/step → **25k
steps ≈ 37 h**. Eval at deploy 160/m16, G=8: **4.10 s/game** (264 games, 1,083 s, uncontended)
against 7.09 at PUCT-150. α: 98.9% of rows carry a tail, mean 0.0006, **max 1.0 rows exist** —
the architect's reading. **Host term for K/MAX_NODES at 16 workers: run RSS 3.84 GiB at boot,
4.55 median, 4.70 max; GPU 7.66 GB max** without an eval round. **The checker-thread A/B is
taken** (lever at `8443d0e5`): collate per part −57% (5.85 → 2.51 ms), end-to-end ≤ +5%
positions/h, because fill stays at 55% and the pop wait absorbs the freed time — the server was
not the bottleneck; arming it is a mint row, nothing forces it. `train.policy_target` moves with
`search.kind` at the re-mint (schema pairs them); both are on `remint-gumbel`.

## CARD-OC7-OVERRUN — discriminated: HOST, not code

Record: `docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`. The row **passes in 178.4 s on
the box** at HEAD and the minted 14 workers. On the dev box (Ryzen 7 3700X, AVX2, no AVX-512
BF16) LAW-06's bf16 autocast on the CPU trainer takes ATen's generic path — one GEMM at the
trainer's edge shape measures **72× slower than fp32** — so a step is ~40–60 s and the docstring's
rule yields no bound. The 2026-08-01 tree is equally slow there today; no bisect. The bound is
NOT re-aimed: which of (the tier runs on the box / the row is `slow` / a LAW-06 CPU carve-out) is
the operator's.

**The integration tier now runs on the box in ~20–25 min and is green there.** Running it —
and then the Gumbel regime at scale — found what the dev box could not, all pre-existing on
`dev` from wave 2: (1) DELETE-1 shrank the Rust `GameResultRow` 10→8 and left the Python drain
unpacking 10, so the real self-play seam died on its first game while every default-tier fake
stayed green (`92643671`, arity parity test added); (2) GUMBEL-3's drain dropped the sparse row's
tail mass and the push defaulted it to 0, so R347(a)'s α never reached the ring and the first row
with a real tail was refused at insert (`79b2b2c3`); (3) CONFIG-1 moved `supervisor_*` keys to
schema defaults and the supervisor witnesses still `--set` them; (4) the eval-round oracle still
passed the deleted grid arm's `None` specs; (5) R347(d)'s durability halt refused every
preflight-driving test's tmp dir (tmpfs on a workstation, overlay in a container) — the preflight
now reads a planted table only through `MANTIS_PREFLIGHT_MOUNTS_TABLE`, the reading names the
table it used, and the stamp REFUSES any table but `/proc/mounts`, so the seam launches nothing;
(6) the drain golden's manifest row and the Makefile's declared target set followed.

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

## AUDIT-3 and REPAIR-A3 — `docs/audits/AUDIT_2026-09-11.md`

AUDIT-2's four P0s re-read at HEAD: three REPAIRED by REPAIR-A2 and verified, P0-3 HALF
(identity drift refused; the flat `RESUME_CHECKPOINT_OWNED_KEYS` filtered nothing on the nested
shape). Four new P0s from running the tree (the two drain breaks, the warm-start stamp, the gate
runner's silent CPU re-sync). REPAIR-A3 landed at `8443d0e5`: `RESUME_CHECKPOINT_OWNED_PATHS`
(nested, leaf-wise, with a real-checkpoint producer test — a launch `train.eta_min` no longer
overrides the baked one and the ignored value is published); the ten CUDA-venv rows state
`device="cpu"` (all 150 rows of the affected suites pass on `+cu128`); R347(e)'s lever behind
`inference.edge_geometry_check` with LAW-18 counters and the F-816-37 dump-on-fire halt.

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws; no gate number added (still
17). Open cards: `docs/governance/CARDS.md` — `CARD-OC7-OVERRUN` carries its discrimination in
place; `CARD-GATE17-LOCAL-COUPLING` unchanged; `CARD-CLAUDEMD-REPOINT` closed by R348.

## Exit facts — WAVE 3 leg 1 (OC-7 → box rebuild → traps), 2026-09-11

- Ruling: R348 at `63671f08`, verbatim; `RULINGS.md` numbering continues at R349.
- Commits on `wave3` since `b117e657`: `92643671` drain arity, `63671f08` R348, `e1e06843`
  stamp trap, `669b6008` cuda extra, `5e529a44` gate-17 lock carve-out, `ad747400` OC-7 record,
  `9137ab1a` / `eb1a9b85` / `408af297` / `ff26f3a0` tier and default-tier repairs, `79b2b2c3`
  tail mass, `97711e65` floor ratchet. One line each, empty body, zero trailers.
- **Collected tests: 4,530 → 4,576**; the gate 3c floor ratcheted to 4,576.
- Commits after `6941328a`: `7487d111`/`4c1b91e1` (REPAIR-A3 P1-1/P1-2, fork 1), `cb467637`/
  `e0df50be` (the lever, fork 2), merge `8443d0e5`; proposals `9bdcebeb` (remint-warmstart),
  `d472917d`+`c20297af` (remint-gumbel).
- **Tree: 878 tracked files, 389,706 lines** (`git ls-files | xargs cat | wc -l`).
- Comment ratchet: floor lowered 13561 → 13543 docstring lines across the leg; comment and
  banner measures held at 3551 / 23.
- Gate 17 gained a carve-out: in `uv.lock` an IPv4 counts only as a URL host, because the cuda
  extra's four-part CUDA library versions pass the octet ranges; proven both ways in the
  self-test.
- Box: rebuilt through `box_setup.sh wave3` with the extra — `torch 2.11.0+cu128`, RTX 5080,
  extension, vendored sealbot and the warm-start checkpoint all present; host identity lines are
  in the box's own setup log. The instance is on the same physical host the archive records.
- Gates, dev box: 3a — the two forks each ran the full default tier on their worktrees
  (4,509 and 4,516 pass; the warm-start row loud-skips where the artifact is absent and is the
  **1 true red** where it is present); 6–17 GREEN at every commit; **3b cannot run here** (bf16
  emulation); 2a/2b/4/5 ran on the box.
- Gates, box, `make gates.exit` at `ff26f3a0` (48 min): 2a, 2b, 4, 5, slow tier, 3c, 6–17 all
  GREEN; 3a and 3b RED on three rows — the warm-start row (true), the signal-posture row (an
  artefact of a detached job's SIG_IGN; passes in a foreground session), and the non-CUDA-host
  row, whose failure exposed that **`run_all.sh`'s bare `uv run` re-synced the venv to the CPU
  wheel mid-set** — no box gate run in history has ever gated a CUDA venv. Fixed at `8dfa8b5f`
  (`UV_NO_SYNC=1`; the runner prints the torch build it gates). Gated AS BUILT on `+cu128`, 3a
  then shows the rows that assume a CPU torch (CARD-GATES-ON-CUDA-VENV, 10 model/slow rows) —
  a leg of their own. **So "gates.exit green on the box" is NOT claimed at this handoff**: green
  on a CPU-wheel box was never the measurement it appeared to be, and on the CUDA venv the
  remaining reds are named, carded and not this leg's code.
- Three harness defects of this leg's own making are recorded so they are not re-learned: a
  py-spy wrapper without a `__main__` guard re-ran pytest inside the eval worker's spawn child
  (fixed before any box number was taken); the comment ratchet counts TRACKED files only, so a
  green measured before `git add` is not a measurement; and a burst whose producer died left its
  parent alive for 27 min, overlapping the next burst — the rider's own warning — so the
  PERF-3b self-play numbers are from a third burst on a box verified idle by PID first.

## Provenance

Derived 2026-09-11 on `wave3` at `8f89545c`, from `configs/run6.yaml`, the box's
`oc7/{box_head,tier_box*.log}` and `perf3b_*` readings, the dev box's scratchpad `oc7/` artefacts,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
