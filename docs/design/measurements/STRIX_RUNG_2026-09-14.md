# MEASUREMENT — RUNG-2, the strix rung at step 0 (2026-09-14, box, tree `de536a2d`)

Instrument: `tools/strength_frontier.py --config configs/run7.yaml --cells /workspace/oc7/cells_strix_step0.json`
(one cell at a time, 8 games in flight per cell), `vendor/pins.toml [pins.hexo-strix]` at
`5a771e57` with `checkpoint_00237000.pt` (sha256 `351ed562…`, 237 000 strix training steps),
strix's own venv on the box (CPU torch 2.11, `hexo_rs` built by maturin), eight
`tools/strix_driver.py` processes (4 torch threads each) speaking JSON lines to
`mantis.bots.strix.StrixBot`; the candidate is `bc_full` — every head of the BC checkpoint
`run6_00006500_ca1afb71` — i.e. run7's step-0 net, on the card at PUCT. Book
`book_v1_s20260625_p4`, 144 openings × both colours, `round_index` 0, a pair-level bootstrap CI over
distinct games (LAW-04; every cell 288 distinct trajectories). Between the twin's stamp and the
3 h shakedown, the card otherwise idle. Raw cells mirrored to `mantis-mirror/strix_step0/`
(`cell.json`, `games/`, `child.log` per cell); viewers `mantis-mirror/viewer-strix-A/`, `-B/`.

## A. The two cells (R352(e))

| cell | ours | strix | WR | 95 % CI (pairs) | W–L–D | pairs: player / seat / split | p1 wins | median plies | s/game | cell wall |
|---|---|---|---|---|---|---|---|---|---|---|
| **A — as-shipped** | PUCT-512 | 128 sims, m 16, noise off | **0.083** | [0.052, 0.122] | 24–264–0 | 126 / 18 / 0 | 136/288 | 39 | 6.96 | 2 006 s |
| **B — equal-work** | PUCT-256 | 256 sims, m 16, noise off | **0.073** | [0.045, 0.104] | 21–267–0 | 125 / 19 / 0 | 137/288 | 40 | 7.81 | 2 251 s |

Fence findings (legal sets compared on every strix move, R257): **0** in 576 games; forfeits 0.
"Player / seat / split" pairs: both games of an opening won by the same PLAYER / by the same
SEAT / one each.

## B. What it says

1. **The instrument has range where sealbot has none.** The same step-0 net reads 0.41 vs
   `sealbot_d5` at PUCT-150 (`STRENGTH_FRONTIER_1`), 0.507 / 0.597 / 0.642 at PUCT-512 after
   101 Gumbel steps (the three stamps) — and **0.08 vs strix**. R352(e)'s ground for the rung
   ("without a harder rung the run has no external instrument past ~20k") is measured true at
   step 0: run7 has ≈ 90 pp of headroom against strix and ≈ 35 against sealbot.
2. **The equal-work cell reads the same as the as-shipped cell** (0.073 vs 0.083, CIs overlapping):
   doubling strix's sims from 128 to 256 while halving ours from 512 to 256 moves the reading by
   one point. At this gap the sims are not the lever; the net is.
3. **Seat-decided pairs are 12–13 % here** against 37–57 % in every sealbot cell on the same
   book (`GAME_QUALITY_2026-09-14.md` §D.1), and p1 wins 47 % (sealbot cells: the p2 seat 62 %).
   The book's lopsidedness against sealbot is an INTERACTION with the fixed-depth bar (it
   converts a good seat and cannot hold a bad one), not a property of the openings alone.
4. **Cost line:** 7–8 s/game at 8 in flight on the box's CPU for strix and its card for us —
   ≈ 35 min per 288-game cell, ≈ 70 min for the point. At the 15 000-step cadence R352(e) sets
   (≈ every 13 h of run7 at run6's rate) the point costs < 10 % of the card's time if played
   beside the run, and nothing if played on the mirrored checkpoints elsewhere; the cadence is
   CONFIRMED as set. Strix is CPU-only by the pin's design (`--group cpu`; after START the card
   is run7's); its per-move cost is its Python graph building, not its 284k-parameter net.

## C. What it does NOT say

- Nothing about strix's absolute strength beyond "well above our step-0 net"; the one other
  reading is `GAME_QUALITY` §D.8, strix-128 beating sealbot_d5 30–2 (n 32).
- 288 games of a 0.08 reading carry ± 3.5 pp; the discriminating pairs are the 24 + 21 wins.
- The opening book is v1 (uniformly random 4-ply); a balanced book is a separate instrument
  change, ruled before it replaces this one (the readings stay on v1 for comparability).

## Sources

`/workspace/oc7/strix_step0.log`, `/workspace/oc7/strix_step0/*/cell.json` (mirrored:
`mantis-mirror/strix_step0/`), `tests/bots/test_strix_adapter.py` (the witness, green on the box
at this tree), `vendor/pins.toml`, `configs/run7.yaml`, the pair counts from the cells' `games/`
records paired by `game_index` (the frontier tool's own convention).
