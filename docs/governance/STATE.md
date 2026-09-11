# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase

**run6 is RE-MINTED at the Gumbel regime; the two holds the START path found are DECIDED by
matrix under operator delegation and the mint procedure is next.** R349 (the START-path packet, forwarded 2026-09-12) is landed on this
line through its order (e) up to and including (d): the two re-mint merges, the LAW-06 carve-out,
the puller and receipts in place of the deleted overlay halt, the α = 1.0 rows reconstructed and
their run-fatal half fixed, the trainer step profiled. What stands in front of the mint now is in
the next two sections. **`dev` and `origin/dev` are one line at `e078b929`**: the wave3 → dev
fast-forward was pushed after the box's `make gates.exit` read ALL GREEN on the decision commit.

**`configs/run6.yaml` mints `search.kind: gumbel`** (`completed_improved_policy`, 320 full / 64
quick at p 0.25, deploy 160/m16, grace 30 s, sealbot-only rungs) and
`identity.warm_start.checkpoint: checkpoints/bc/run6_00006500_ca1afb71.ckpt` (the LAW-12 strip,
net hash `2e72abd4…` unchanged). The default tier's one true red (the warm-start hash row) is
green wherever the strip exists.

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

R348(c)'s trap met the schema's reachability rule: run6 mints `train.draw_rate_abort.min_step
25000`, so the old `_apply_burst_override` refused every burst below **25 001 steps** and the
only stamp-writing preflight of run6 was the block itself (≈ 22.6 h); the shakedown twin carried
the same floor. The operator delegated the decision to a matrix (2026-09-11) and it selected
**the burst as a stop-step bound over the MINTED config**: `compose_run(burst_stop_step=)` is
the eighth census parameter (a prefix of the run, refused outside it, no launcher route); the
preflight child boots the minted identity with the bound, so the child's published hash IS the
stamp's; the refusing floors are the two actor-sync rows; the draw-rate row decides the TIER; and
the stamp records the tier it proved. A production config therefore stamps at `sync_lag` from a
≈ 100-step burst (≈ 12 min on the box); tier `full` keeps its meaning. The rejected options and
their grounds are in `CARD-STAMP-FLOOR`. Two companion decisions by the same matrix: Phase W (the
n_workers sweep) is NOT re-run before START (F-47 measured 16 → 32 at +5 %, inside the sweep's
own noise; `CARD-PHASE-W-AT-GUMBEL` carries the falsifier), and the completed-Q target form is
KEPT for the start and decided on the block's count (`CARD-ALPHA-TARGET-FORM`).

## Hold 2 — the completed-Q target in decided positions: KEPT for the start, decided on data

R349(c)'s question is answered in `docs/design/measurements/MEASUREMENT_STARTPATH_2026-09-11.md`
§B and falsified.md F-45: all 25 α = 1.0 rows sit at `moves_remaining == 1` in LOST positions;
the perspective flip is correct (root `−0.95`, children `+0.98` from the opponent's side); the
cause is value-saturation asymmetry lifting `v_mix` above the clustered visited Q's, after which
Mctx's min-max rescale × `c_scale 1.0` underflows every explicit mass. **The run-fatal half is
fixed** (`cc5bf14f`): the recorder dropped zero-mass sampled cells, an all-underflow row was
refused at the ring as `EmptyTarget`, and the START-path burst died of it at step 496 — run6
would have died inside its first hour. Every sampled cell is now stored, at zero too, with parity
vectors on both sides of the FFI. **The target-quality half is owed**: ~1 row per 1 000 (rising
with saturation) trains "none of the 16 searched moves". `CARD-ALPHA-TARGET-FORM` carries the
three options; none is a hold on run6 by itself, and `iteration_complete.gumbel_alpha_full` plus
the dashboard's "alpha = 1.0 rows per 1,000" panel count it from step 0.

## What R349(d) found: the trainer is not the bound

py-spy over the whole burst: the trainer thread is asleep **92 %** of the run — 75 % in O5 "no
new games", 17 % in the ring-fill warm-up. Its step costs **0.32 s in-run, 0.244 s standalone**
(GPU 0.156 s: `index_add_` 22 %, GEMMs 16 %, 31 stream syncs; the 1-in-1 check 53 ms; the Rust
rebuild 57 ms). The cadence is `train.training_steps_per_game: 1.0` / `max_train_burst: 1` —
one step per completed game, **0.986 measured**, 6.6 draws per row — so steps/h ≡ games/h:
**1 105 steps/h in the stepping window, the 25 001-step block ≈ 22.6 h**, not PERF-3b's 37 h
(a from-boot average over a 13-min burst; repaired in place, falsified.md F-44). No trainer code
lever is S and ≥ ×2. The serving side from the same record: the inference-server thread spends
32 % of its wall in `_node_offsets_to_batch_vec`'s implicit device→host sync and 19 % in check 14
inline; the 16 workers wait on it 99.6 % of theirs (`CARD-SERVER-SYNC`, INVESTIGATION-1 item 3).

## The performance investigation (operator-ordered, two read-only agents, same day)

Record: `docs/design/measurements/PERF_INVESTIGATION_2026-09-11.md` (the full report and its
one-page summary; scripts and raw harvests in the dispatcher's scratchpad and under the box's
`/workspace/oc7/pi`, `/workspace/runs/pi_*`). What it settled: **the inference-server thread is
the block's price — 90.6 % busy at 16 workers, 99.0 % at 32**, cost linear in leaves (2.7 ms per
pop + 0.515 ms per leaf), GPU busy 52.8 % of wall because the chain's CPU and GPU halves never
overlap; **the landed `checker_thread` lever is a net loss as built (−12.6 % at 32 workers, the
Rust verifier holds the GIL; F-46 — do not arm it)**; more workers buys +5 %, batch 128 −3 %
(F-47); **eval rounds run self-play at 0.79× while alive** (`CARD-EVAL-CONTENTION`, ≈ 5–10 % of
the block). Block ≈ 20.6 h + eval ≈ 21.6–22.7 h. Ten levers ranked with pre-registered gains and
falsifiers (a two-stage server pipeline, check 14 with the GIL released, `torch.compile` of the
trunk, an edge codebook, pinned H2D, a fused message pass …); stacked, an ESTIMATE of ≈ 7 h per
block. All are LAW-09 proposals; nothing landed.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` runs on the operator's machine against an rsync spec (`<alias>:/path`),
verifies complete bundles against their manifests and closed shards against the index's sizes,
writes `<artifact>.receipt.json` beside each verified copy and syncs the receipts up; the
preflight's rc 16 is now `PreflightMirrorReceiptsError` (decided AFTER the boot, waits
`--receipt-wait-sec`, requires receipts on the burst's bundle and first shard); `mantis.run`
refuses a stamp without the `MIRRORED` verdict; `resume_state_persisted.unreceipted_bundles` is
the run's lag reading and the dashboard warns at two. R347(d)'s overlay halt, `mantis.util.mounts`
and the planted-table seam are DELETED. Proven over the alias on the burst: one cycle, 11.1 s,
the step-200 bundle receipted on the box. **The run dir on the box is `/workspace/runs/<run_id>`;
the mirror on the operator's machine is the operator's to place** (this session used its
scratchpad for the proof); loss-on-recycle is RECORDED as accepted.

## Two more defects the burst surfaced, both fixed on this line

- The launch-time anchor (`best_model.pt` at fresh init) was a BARE state dict; the resilient
  loader rebuilt the encoding's incumbent kind and hit a size mismatch on V2 weights — a run6
  resume before its first promotion died at the anchor. Every anchor is stamped (`901139d9`).
- `test_the_graph_buffer_is_composed_with_the_derived_visit_capacity` could no longer tell a
  derivation from a constant once run6 minted Gumbel (a sparse row's capacity is m whatever the
  sims); it drives the full-vector kind for its two shapes and the minted config as a third.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 (`9491b4d0`; the autocast context,
never the dtype pin; `tests/train/test_law06_cpu_carveout.py` pins both halves; LAW-06's text
carries the clause). On the dev box the tier still runs the CPU wheel — `make build.cuda` on the
3070 is the operator's install step and was not taken this session — so the tier's authority
this exit is the box.

## Minted values — `configs/run6.yaml`

Changed since `24c9dab0` by the two re-mint merges only (`1793fee1`, `a37d2e5e`): the seven rows
above plus `monitor.supervisor_kill_grace_sec 600 → 30`. Nothing under
`src/mantis/config/armed_aborts.py`, `crates/mantis-search/src/mcts/`,
`crates/mantis-encoding/src/registry.toml` has changed since the wave-2 exit; LAW-06's clause is
the one edit to `docs/governance/LAWS.md`. Gate 12 is GREEN at HEAD.

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws (LAW-06 amended by R349(a));
no gate number added (still 17). Cards: `docs/governance/CARDS.md` — opened by R349 and the investigation:
`CARD-ALPHA-TARGET-FORM`, `CARD-TRAINER-CADENCE`, `CARD-SERVER-SYNC`, `CARD-DEPLOY-HEAD-BUDGET`,
`CARD-STAMP-FLOOR`, `CARD-CHECKER-THREAD-GIL`, `CARD-EVAL-CONTENTION`;
closed: `CARD-BOX-VOLUME`, `CARD-WARMSTART-STAMP-SCHEMA`, `CARD-GATES-ON-CUDA-VENV`,
`CARD-TIER-HOST` (ruled), `F-816-24`; `CARD-ALPHA-MAX-ROWS` discriminated. TEST-1 and
INVESTIGATION-1 are **NOT dispatched**: both run "during the block" and the block has not started;
INVESTIGATION-1's items 2 and 3 should open from `CARD-TRAINER-CADENCE` and `CARD-SERVER-SYNC`
rather than from "why is the step 5 s".

## Exit facts — the START-path packet, 2026-09-11

- Ruling: R349 at `c5687452`, verbatim; `RULINGS.md` numbering continues at R350.
- Commits on this line since `f1139c54`: `1793fee1` / `a37d2e5e` (the merges), `9491b4d0`
  (carve-out), `c5687452` (R349 + closes), `4c4a83d4` (α counter, `root_raw_value`), `8e307af1`
  (R349(b)), `901139d9` (anchor), `cc5bf14f` (recorder), `d9c57e92` (buffer-route row),
  `7a97fa80` (records). One line each, empty body, zero trailers.
- **Collected tests: 4 576 → 4 592**; the gate 3c floor ratchets to 4 592. Comment ratchet:
  3 550 / 23 / 13 532 → **3 549 / 23 / 13 531**.
- **Tree: 892 tracked files, 392 090 lines** (`git ls-files | xargs cat | wc -l`).
- Gates, dev box (CPU wheel): 3a default tier **4 532 pass, 4 skipped**, the four reds repaired
  in `d9c57e92` and `7a97fa80` (a gate-10 untracked reference, gate 15 on an accidental crate-wide
  rustfmt sweep that was reverted before commit, the buffer-route row); 6–17 GREEN at HEAD; 2a/2b
  for the touched crates GREEN, `records.rs` rustfmt-clean (72 pre-existing fmt diffs elsewhere
  in `mantis-selfplay` are NOT this session's — R336(e)); 3b/4/5 not run here.
- Gates, box: `make gates.exit` **ALL GREEN twice** — on `dd01cb49` (15:14–16:03 UTC) and on
  `652b9f02`, the stamp-floor decision (16:07–17:00 UTC; `/workspace/oc7/gates_exit_startpath.log`):
  2a, 2b, 4, 5, 3a (4 536 pass), **3b on CUDA (46 pass, 32 min)**, slow tier, 3c, 6–17. The first
  fully green packet-exit set on a CUDA venv in the repo's history. An earlier run on `7a97fa80`
  had 3b red on three preflight rows (the mirror halt demanded a bundle a clean completion never
  writes; fixed at `2302aa40`/`7b2bf053`), and a first attempt on `f253e65e` lost gates 2a/2b/4/5
  to a detached shell without `~/.cargo/bin` and was killed (PID-verified) before the relaunch.
- **`origin/dev` is at `e078b929`** — the wave3 → dev fast-forward was pushed on the green set
  (the operator authorised pushes mid-session after the tool's classifier had refused them).
- Box artefacts: `/workspace/runs/startpath_cd/` (the burst: events, `pyspy_idle_native.raw`,
  bundles at 200 and 400 with rings, receipts on 200), `/workspace/oc7/{alpha_probe.py,
  alpha_probe_warmstart.json, trainer_profile.py, trainer_profile/, startpath_cd.yaml,
  drive_burst.py, burst.sh}`; the dev box's scratchpad `startpath/` mirrors them.
- Two harness facts recorded so they are not re-learned: a job detached from an ssh command
  runs without `~/.cargo/bin` on PATH (export it in the script); and `cargo fmt -p <crate> --
  <file>` formats the WHOLE crate — use `rustfmt <file>` for a touched file.

## Provenance

Derived 2026-09-11 on `dev` at `7a97fa80`, from `configs/run6.yaml`, the box's
`/workspace/runs/startpath_cd/` readings, `tools/ci_gates/test_count_floor.txt`,
`tools/ci_gates/comment_length_floor.txt` and `docs/governance/LAWS.md`. Ruling texts:
`docs/governance/RULINGS.md`.
