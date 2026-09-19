# LADDER-1 SHAKEDOWN — the bot adapter's first 20 games on the operator's HeXO server (2026-09-19)

The LADDER-1 packet (CARD-LADDER-RUNG) ordered a design + shakedown: `tools/ladder_bot.py` on the
`tools/ladder/` package, one process per registered bot, both backends behind one seam, witnesses
BEFORE the first reading, 20 games parent (42k) vs strix on the server, read beside the follower's
parent cell and LABELLED a different unit. This is that record. **Nothing here is a series point,
a promotion input or a prereg row**; the rung becomes an instrument by R363 or later.

## 0. What was played, by what

| | mantis | strix |
|---|---|---|
| bot name (receipt) / display name | `mantis:a9a46c55` / Mantis | `strix:351ed562` / Strix |
| net | run7 42k parent, `run7_00042000_46fdb931.ckpt`, net_hash `a9a46c55…` (run8's `identity.warm_start.net_hash`) | the pin: `checkpoint_00237000.pt`, sha256 `351ed562…`, commit `5a771e57` |
| search | the deploy head, `configs/run8.yaml`'s `deploy.search.kind = puct`, `eval.gate.deploy_sims = 256` per stone, σ/batching the config's | `tools/strix_driver.py`, Gumbel, 256 sims per stone, m 16, root VCF solver ON, noise off (the unit on record) |
| host | the dev host CPU (8 cores / 16 threads), torch 2.11 cpu, 8 threads each, the two processes side by side | same |
| pairing | Mantis challenges Strix, one challenge at a time, `firstPlayer` alternating challenger/challenged; time control `unlimited`; every bot game is unrated server-side | Strix holds `?open=1`, accepts Mantis only |

The deployed server is `TimmyBurn2/HeXO` branch `deploy` at `8166053` (verified against the live
server, not the spec file; the four places the spec file lags are on the card).

## 1. The witnesses (packet §1.3) — held BEFORE the reading

- **Determinism.** Every live receipt replayed through its own backend (`--replay`, the positions
  rebuilt from the server's move record, the head seeded from the game id): the two smoke games,
  both sides — **4 of 4 PASS**, 0 move mismatches over 7 + 12 (mantis) and 8 + 12 (strix) turns.
  In-process, the same game id gives the same turn on both backends (`tests/tools/test_ladder_backends.py`).
- **Budget, read off the heads' own counters** (`DeployHeadPlayer.last_sims`, `StrixBot.last_sims`,
  both added for this): every UNDECIDED position spends exactly 2 × 256 = 512 leaves per compound
  turn on both backends. On a DECIDED position neither does, by construction of each search:
  strix's root VCF solver answers in **3** visits (its last three turns of every won game), and
  mantis's PUCT tree stops at **427** of 512 once every path hits a terminal (`select_leaves`
  returns nothing on an exhausted tree; R355(b)'s "exactly n_sims" test is a mid-game position).
  The replay reproduces both counts exactly, so `below_budget` is a REPORTED fact, never a verdict.
- **Receipt** per game beside the tool's work dir, keyed by net hash
  (`receipts/<net_hash[:8]>/<gameId>.json`): both bot names + display names + profile ids, net
  hashes, sims configured and spent per turn, the two placements per turn, the result and reason,
  plies (from the server's record) and plies seen, wall, think total, the server's own clock
  (`startedAt`/`finishedAt` and the `Date` header of the first and last move answer). Refused
  before it is written when it cannot be read back; LAW-07 planted break + mutation self-test in
  `tests/tools/test_ladder_receipt.py`.

## 2. CPU cost (packet §1.5) — what sizes the VPS

Measured on the dev host (8 cores / 16 threads, 8 torch threads per process, the other bot idle or
thinking in turn): **mantis 9–15 s per compound turn** at PUCT-256 (8.7 s at 3 stones rising to
≈ 13.5 s at 25; replay means 11.7 and 14.8 s/turn), **strix 2.5–5.4 s per turn** at 256
(≈ 30 ms when its solver decides). A game of 31–49 plies takes ≈ 100–190 s wall, ≈ 80 % of it
mantis's think. A 4-vCPU VPS reads roughly twice that; `unlimited` or `turn:60000`+ keeps both
inside the clock. Fixed budgets are the instrument: a `time_limit` shorter than the think is
logged, never adapted to (none occurred: `unlimited`).

## 3. The reading (packet §1.4) — a sanity band, not a series point

20 games, 2026-09-19 16:56–17:59 UTC, 10 with Mantis as x (challenger first) and 10 as o, one
challenge at a time; **0 rejections, 0 stream drops, 0 reconnects, 20 receipts on each side**, every
receipt's wall within 0.1 s of the server's own `finishedAt − startedAt`.

| colour | games | result | plies | wall / game | mantis think (turns) | strix think (turns) | trajectory (sha of the move list, `moveNumber` order) |
|---|---|---|---|---|---|---|---|
| Mantis x | 10 | 10 losses, six-in-a-row | 31 every time | 95–109 s | 74.5–85.8 s (7) | 18.6–22.1 s (8) | `27ec9a3d7a74`, ONE trajectory in 10 |
| Mantis o | 10 | 10 losses, six-in-a-row | 49 every time | 215–251 s | 161.7–182.5 s (12) | 49.0–65.3 s (12) | `e594e1c9eb1e`, ONE trajectory in 10 |

**Mantis 0 of 20 — which is 0 of 2 DISTINCT games (LAW-04: eff_n 2).** Budget over the 20:
mantis 180 of 190 turns at exactly 512 leaves, the other 10 being the LAST turn of every x-game at
427 (the decided position, identical every time); strix 140 of 200 turns at 512, the other 60
being the last three turns of every game at 3 (its solver deciding). Mean s/turn: mantis 13.1,
strix 3.4.

**Beside the follower's parent cell** — `run7_00042000_46fdb931.ckpt.strix256.json`: 0.111
[0.073, 0.149] at 256/256 over 288 paired games on `book_v1_s20260625_p4`, eff_n 288, median
43 plies, CONTENDED on the box. **A DIFFERENT UNIT**: no opening book (the server auto-places the
origin and both bots are deterministic, so every game with the same colour assignment is the
SAME game), the server's clock and legality, the dev host's CPU. The band [0.02, 0.30] is a
bug-hunt threshold; 0 of 2 distinct games is CONSISTENT with the parent cell's 0.111 (P(0 of 2 | 0.111) = 0.79)
and is not a reading of strength at all. Nor is it a bug hunt on the tool: the witnesses hold, no
move was rejected, and the arena's own loop from the auto-placed origin reproduces the server's two
games MOVE-FOR-MOVE (`mantis.arena.match._play_one_game` from `[(0, 0)]`, 31 and 49 plies, both colours). What it IS is finding 1 below.

## 4. What the shakedown found (the server side is the operator's to change)

1. **No opening diversity between two deterministic bots.** LAW-04's dedupe makes the 20 games
   2 distinct trajectories (one per colour assignment). A series on this server needs openings from somewhere: the
   contract has no opening field, the challenger could seed the first turns, or the bots draw
   per-game noise (mantis's PUCT draws none; strix's driver searches with `seed=0`).
2. **The finished-game record's `moves[]` is racy within a compound turn** (`appendMove` is
   fire-and-forget; game 3 carried moveNumber 4 before 3) while `moveNumber` is right. The receipt
   sorts by `moveNumber` (fixed during the run; the first 5 receipts written before the fix carry
   the array order in `moves_full` — the replay witness is set-based and unaffected). `moveNumber`
   starts at 2.
3. **The spec file lags the deployed server** in four places (card).
4. **House bots refuse bot challenges** (engine-driven, "played from the lobby dialog"), so the
   ladder needs two stream bots; SealBot on the roster is not reachable by this adapter.
5. Stability over the 20 games: 20 challenge/accept cycles, 20 games, 0 rejections, 0 drops, 0
   reconnects, 0 unknown events; the two streams held for the whole hour.

## 5. What remains (the card)

Install on the VPS (`tools/ladder/vps/`); the witnesses on the VPS; one 288-game cell read beside
a follower cell on the same checkpoint — which needs item 1 answered first; then R363 or later.
