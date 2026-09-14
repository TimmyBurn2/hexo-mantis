# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase — R352 LANDED; run7 RE-MINTED as run6's Gumbel trainer with every head loaded and the ply-cap halt armed; the first stamp PASSED (0.507 vs sealbot at step 101) and exposed the round's wall; the SECOND stamp (round timeout 10 800) is RUNNING; VIEWER-1 and RUNG-2 LANDED

**R352 is landed verbatim** (`60b752ad`), the operator's forward of the run7-kind packet (dated
2026-09-14): shakedown7 falsified PUCT self-play at the minted regime (**F-52**), run7 is run6's
Gumbel-320/64 trainer with exactly one change — every head loaded — plus the instrument fixes
(PUCT-512 reader, 3 000/288 cadence, `eval.rung_concurrency 8` RATIFIED), the value warm-up dropped
to 0, the ply-cap HALT ordered, RUNG-2 (strix) and VIEWER-1 ordered. R351's status line marks (c)'s
PUCT arm and (d)'s warm-up superseded for run7. Cards opened: **CARD-PUCT-ATTRACTOR** (the four
hypotheses, to be read on the games in the viewer), **CARD-GUMBEL-HEAD-RESIDUE** (F-51's residue),
and this leg's **CARD-SEALBOT-GIL-SERIAL** (below).

**Landed on `dev` this leg, one line each (`origin/dev` is 148261c8; `dev` is 9 ahead, unpushed):**

- `60b752ad` — R352 verbatim, F-52, the two cards, R351's status.
- `d13bc7c3` — **the ply-cap halt** (R352(c)): `train.ply_cap_abort {rate, window_games,
  min_step}` (`PlyCapAbortConfig | None`, `default=...`, contract **v29**), the pool's ring of
  per-game cap flags (`util.constants.PLY_CAP_RING_GAMES` 4 096, `PoolInstrumentation._ply_cap_ring`,
  `WorkerPool.ply_cap_window_counts`), read EVERY training step by `step.py::_run_ply_cap_gate`
  (`min_step` gates the fire, a window still filling is a skip), `check_ply_cap_attractor`
  strictly above the rate, exit **50** (`PLY_CAP_ATTRACTOR_EXIT_CODE`, the fifth cooperative
  member), manifest row `ply_cap_attractor` DEFERRED (run6.yaml mints null truthfully) with the
  new `Cadence.TRAIN_STEP_FLOOR`; `monitor_gates` carries `ply_cap_abort_rate`,
  `ply_cap_window_games`, `ply_cap_rate`. Producer test + planted break
  (`tests/train/test_ply_cap_gate.py`). The dashboard's NINTH health input (`ply-cap attractor`,
  terms off `monitor_gates` or R352(c)'s minted {0.5, 600}) and a fifth quality multiple, the
  windowed cap share from game 0 with the halt rate drawn. **run7 RE-MINTED** through its own
  header with four row changes (`selfplay.search.kind gumbel`, `train.policy_target
  completed_improved_policy`, warm-up 0, the halt {0.5, 600, 3000}); shakedown7's record re-read at
  those terms: the 600-game window first exceeds 0.5 at game 2 229 (+3.4 h, step ≈ 2 170) — the
  packet's "+1.9 h" was an estimate off the 300-game table (`SHAKEDOWN7_2026-09-14.md` §E).
- `2d227ed6` — **VIEWER-1** (R352(g)): `tools/game_viewer.py` + `tools/viewer/` (`make viewer
  RUNS="id=dir …" OUT=…`), a static page over GAME-RECORD-1 shards — hex SVG board, ply stepping
  by key and swipe, the last two stones ringed, the six-in-a-row outlined, the visit heatmap where
  stats exist, run/channel/result/plies/termination filters, `?g=run/id&ply=N&heat=1` links,
  per-shard data loaded on demand. The owner-of-a-ply and win-line facts are derived in Python and
  checked against every record (0 findings over run6's 38 988 games and shakedown7's 2 421).
  `repo_design.md` amended (the viewer admitted on the dashboard's terms). Built into the mirror:
  `mantis-mirror/viewer/index.html` (7.2 MB index, 82 shard files), served by any file server over
  that directory. HAND-CHECKED ON A DESKTOP (headless Chromium captures of the shakedown7 page: the
  board, the ring, the outline, the heatmap on a run6 promotion game); the PHONE check is the
  operator's — headless Chromium does not paint the 7 MB two-run page, the DOM dump is its check.
- `56b51004` — **RUNG-2** (R352(e)): `SootyOwl/hexo-strix` pinned at `5a771e57` (upstream `main`
  on 2026-09-13) with `checkpoint_00237000.pt`'s sha256 `351ed562…` and the sha256 of its embedded
  config triple (the operator supplied NO `config.toml` and NO commit; the pin discloses it);
  `make vendor.strix` builds strix's own venv (CPU torch, `hexo_rs` via maturin, `torch_geometric`);
  `tools/strix_driver.py` plays the checkpoint as a bot PROCESS (noise-off argmax of the Gumbel
  improved policy — strix's own eval acting policy; the position rebuilt by `GameState.from_state`
  and translated so a p1 stone sits at strix's fixed origin); `mantis.bots.strix` reads the fence
  at contact (legal sets compared every move, disagreements counted as findings, an out-of-fence
  move forfeited by the arena); `tools/strength_frontier.py` plays strix cells (`opponent: strix`,
  `strix_sims`; ours on `RoundSpec.strix_model_sims`). WITNESS: 20 legal games end to end at the
  pinned commit, zero fence disagreements, on this machine AND on the box; 40 real run6 positions
  read identical legal sets at radius 8 on both engines (the R257 fence). Cost line measured on
  this machine's CPU: strix 0.9 s/move at 128 sims, 1.9 s at 256 (4 threads).
- `4e0fd430` — floors: collected tests 4 777 → **4 804**, docstring ratchet 13 512 → **13 510**,
  the strix witness declared `integration`, `tools/strix_driver.py` excluded from pyright with
  grounds. `9719f586` — the stamp record. `de536a2d` — **`eval.round_timeout_sec` 10 800 MINTED**
  (38 deltas, header replays; prereg §7). `53cd235c` — the GIL card.

**The first stamp of the re-mint (`d13bc7c3`) PASSED** (`preflight_run7_20260914T052717Z.json`,
config identity `1676730c…`, MIRRORED to `mantis-mirror/run7-preflight.d13bc7c3`): 101 steps in
≈ 6 min, 195 games at 23.8 plies, 0 draws — run6's Gumbel character from the first minute. **Its
terminal round is the leg's reading:** `sealbot_d5` **0.507 [146/288] at PUCT-512 at step 101**
(the `ccfaf699` PUCT-mint net: 0.271; the BC net 0.413 at PUCT-150), the gate 0.49 (screen) /
0.53 (confirm, not promoted) against the step-0 anchor. **And the round's wall:** 5 782 s —
screen 15 min, confirm 21 (escalated at 0.49 ≥ 0.44), the 288-game rung **48 min** — because the
vendored sealbot holds the GIL through its search, so `rung_concurrency 8` overlaps only the
candidate's side (**CARD-SEALBOT-GIL-SERIAL**; the GIL-release hunk is proposed in the leg's
scratch record, R145-class, the operator's to apply and rebuild — the classifier refused the
rebuild in-session; nothing is ever pushed to the SealBot repository). The terminal round runs
under the drain caps, but every IN-RUN round is bound by `eval.round_timeout_sec`, whose schema
default 3 600 would have killed every run7 round → minted 10 800 (`de536a2d`), a ceiling the twin
shakedown measures. The rung block's progress rows land in ONE batch at block end — a rung that
shows no row for an hour is not stalled (the dispatcher misread this once; the record says so).

**NOW RUNNING on the box: the SECOND stamp**, tree `de536a2d` (branch `r352b`), launched
07:18:54 UTC, `/workspace/oc7/preflight_run7_r352b.log`, the puller cycling into
`mantis-mirror/run7-preflight`. Then, in order: the twin's stamp (`/workspace/oc7/shakedown7g.yaml`,
only delta `run_id`, `config_diff --expect run_id` MATCH), the **3 h shakedown twin**
(`/workspace/run_shakedown.sh /workspace/oc7/shakedown7g.yaml /workspace/runs/shakedown7g 10800`
— reaches one contended round at step 3 000 under the 10 800 s timeout), then **START** on the
route's word (`/workspace/oc7/box_start_run6.sh`'s shape with run7; the puller under tmux/systemd
on the operator's machine). After START: the strix step-0 cells (A as-shipped 128/16 vs PUCT-512,
B equal-work 256/256; `/workspace/oc7/cells_strix_step0.json`, 288 games each, concurrency 8, via
`/workspace/oc7/run_cells.sh`) and the rung-cost cells (`cells_rungcost.json`) — the box shares
its card with run7 then, and the s/game line is measured under that contention and says so.

**Dispatcher state for a fresh session.** Box `/workspace/hexo-mantis` on `r352b` at `de536a2d`
(`make build.cuda`, never a bare `uv sync`); strix vendored there (`vendor/external/hexo-strix`,
`.venv`, the checkpoint under `vendor/external/strix_models/`, witness green). Open, in order:
(1) read the second stamp's verdict and round wall; (2) the twin stamp; (3) the 3 h shakedown —
the contended round's wall vs 10 800 and the ply-cap window from `monitor_gates`; (4) START;
(5) strix cells at step 0 and `git push origin dev` (9 commits; the classifier refusal is not
governance — the operator approves the push in-session). Owed (`CARDS.md`): the α = 1.0 three-row
reconstruction, `CARD-SELFPLAY-SEARCH-STATS`, `CARD-DRAIN-POLLER-RACE`, `CARD-PUCT-ATTRACTOR`
(read shakedown7's cap games in the viewer), `CARD-GUMBEL-HEAD-RESIDUE`, `CARD-SEALBOT-GIL-SERIAL`.

## PERF-A4 — the serving levers, merged at `41a5fea8` (branch `perf-a4`, red-teamed)

Record: `docs/design/measurements/PERF_A4_2026-09-11.md`. Four levers landed: check 14 with the GIL
released (+17 %), pinned `non_blocking` H2D + `output_size=`, `inference.compile_trunk` (+17–21 %),
the two-thread software pipeline (+52 %). Combined at 32 workers: leaves/s 1,831 → 3,804
(+108 %). The block measured **1 163 steps/h whole** at 32 workers on the box
(`RUN6_BLOCK_2026-09-12.md`). The edge codebook is NOT landed (device-dependent numerics).

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

`compose_run(burst_stop_step=)` is the eighth census parameter; a production config stamps at
`sync_lag` from a ≈ 100-step burst (≈ 6 min on the box under the Gumbel mint, 12 under PUCT).
`CARD-STAMP-FLOOR` carries the rejected options. Phase W is NOT re-run before a start
(`CARD-PHASE-W-AT-GUMBEL`).

## Hold 2 — the completed-Q target in decided positions: RULED by R350(e)

All 25 α = 1.0 rows of the START-path burst sat at `moves_remaining == 1` in LOST positions
(F-45); the block read 4.98 per 1 000, flat. R350(e) EXCLUDES them from the policy loss from this
tree on (`exclude_alpha_full_rows`, before the denominator; `trainer_step.policy_rows_excluded_alpha_full`).
The three-row reconstruction R349(c) ordered is still OWED. The re-mint's burst emitted 19
`alpha_full_row` events in 101 steps.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` on the operator's machine, receipts beside each artifact on the box, the
preflight's rc 16 reading them. run6's mirror is `~/Work/HeXO/mantis-mirror/run6` (36 bundles,
153 shards); shakedown7's is beside it; the game shards of both are mirrored (the viewer's
subjects). The run7 puller runs under a session for the stamps; for the run it is the operator's
tmux/systemd unit.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 (`tests/train/test_law06_cpu_carveout.py`).
The dev box runs the CPU wheel; the tier's authority is the box.

## Minted values — `configs/run7.yaml` (and `configs/run6.yaml`, a finished run's record)

`run7.yaml`: **38 deltas** from the `dev` template, replayable (`tools/config_diff.py --from-header`
MATCH). `selfplay.search.kind: gumbel` (320/64 at p 0.25, `c_visit` 50, `c_scale` 1.0, `q_rescale`
true, `gumbel_m` 16, `fast_policy_weight` 0.0, `temp_min` 0.5), `deploy.search.kind: puct`,
`train.policy_target: completed_improved_policy`, `identity.warm_start: {checkpoints/bc/run6_00006500_ca1afb71.ckpt,
2e72abd4…, reinit: []}`, `policy_loss_weight_schedule.warmup_steps` 0, **`train.ply_cap_abort {rate:
0.5, window_games: 600, min_step: 3000}`**, `train.eval_interval` 3000 = `checkpoint_interval`,
`eval.gate.deploy_sims` 512, `eval.sealbot_model_sims` 512, the sealbot rung at `games_max` 288 with
`round_games` 288 and calibration 288 every round, `random_floor_games` 20, `gate.stride` 1,
`eval.concurrency` 8, `eval.rung_concurrency` 8, **`eval.round_timeout_sec` 10 800**,
`policy_loss_trough_abort: null`, `draw_rate_abort {0.25, 25000, 50, 3}`, `supervisor_kill_grace_sec`
120, `n_workers` 32, the PERF-A4 serving rows, `actor_lag_abort_enabled` true, `seed` 20260914.
Gate 12 is GREEN with `policy_loss_trough` and `ply_cap_attractor` printed DEFERRED. `run6.yaml`
gained only the template's `ply_cap_abort: null` row (no delta; it stays the truth of what run6 ran).

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws (LAW-06 amended by R349(a));
no gate number added (still 17). Cards (`docs/governance/CARDS.md`): opened by R352 —
`CARD-PUCT-ATTRACTOR`, `CARD-GUMBEL-HEAD-RESIDUE`, `CARD-SEALBOT-GIL-SERIAL`; opened by R350 and
still open — `CARD-DRAIN-POLLER-RACE`, `CARD-SELFPLAY-SEARCH-STATS`; `CARD-STOP-DRAIN-VS-GRACE`
fixed in code (the grace relation is minted); `CARD-WARMSTART-CONTROL` closed on the frontier.
STRENGTH-FRONTIER-1 is COMPLETE; INVESTIGATION-1's trough item is measured; TEST-1 is not
dispatched.

## Exit facts — the R352 packet, 2026-09-14 (a checkpoint; the leg is still on the box)

- Ruling: R352 at `60b752ad`, verbatim; the packet's `R{next}` → R352 resolved in the entry;
  numbering continues at R353. R351's status line marks its superseded clauses.
- Full local gate set (`tools/ci_gates/run_all.sh --with-slow`, dev box, CPU wheel) on
  `d13bc7c3`: **ALL GREEN, 20 gates**, the slow tier ran; gate 3a 4 716 passed. The viewer and
  strix legs were validated in a worktree and then in the main checkout (bots/eval/tools/config
  with the integration marker on: 2 011 passed; gate 14 GREEN at 224 files; gates 15/16/10/17
  green) — the FULL set has not been re-run after `2d227ed6..53cd235c`; it is owed at the leg's
  exit before the push.
- Collected tests: 4 729 → **4 804**. Comment ratchet: 3 534 / 23 / **13 510**.
- Contract: `docs/contracts/run_config_schema.md` v29 (164 leaf key-paths, gate 13 green);
  `event_manifest.md` carries the three `monitor_gates` ply-cap keys; `eval_instrument.md`
  carries the strix rung; `repo_design.md` carries the viewer's amendment and exit code 50.
- The box: `r352b` at `de536a2d`; the second stamp RUNNING since 07:18:54 UTC.

## Provenance

Derived 2026-09-14 on `dev` at `53cd235c`, from `configs/run7.yaml`, `configs/run6.yaml`, the
box's `/workspace/runs/run7-preflight/preflight_run7_20260914T052717Z.json` and its mirror
(`run7-preflight.d13bc7c3`), the round's `r000001_101_terminal_progress.txt`,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
