# Contract: game record (GAME-RECORD-1)

- version: `game-record-v1`
- owner: `mantis.monitor.game_record` (writer, reader and the two record builders — one module,
  because a record shape defined apart from the writer that stores it drifts from it)
- status: v1 — landed by R344(b). Self-play and all three eval channels are LIVE. Per-position
  search stats are live on the eval channels and have **no producer** on the self-play channel
  (`CARD-GAME-RECORD-SELFPLAY-STATS`).

## Summary

Every game a run plays is written to a per-run, append-only record store under
`<out_dir>/logs/games/`, one JSON object per line. This is the contract any viewer, replayer or
miner builds against; nothing reads the trainer's memory and nothing re-derives a game from the
event stream.

**It is not a second event stream.** `game_complete` already carries a self-play game's move
list into the JSONL event channel and keeps doing so. That stream is keyed by TIME and mixes
forty event kinds; this store is keyed by GAME and carries an index.

## Format

JSONL. `msgpack` is not a dependency of this project and R344(b) forbids adding one for this, so
its own fallback clause selects JSON.

### Files

| File | Shape |
|---|---|
| `games_<run_id>_seg<NNNN>_<YYYYMMDDHH>.jsonl` | one shard: a `shard_opened` header line, then one game per line |
| `games_<run_id>_index.jsonl` | one `shard_closed` row per CLOSED shard |

**The key is (run, SEGMENT, hour), not (run, hour).** The segment names the WRITER and the hour
names the window. A run writes from two kinds of process — the trainer continuously, and each
eval round's child for its own lifetime — and a resume landing in the same wall-clock hour would,
under an hour-only key, append into the stopped process's file. That is the TOCTOU
`monitor/sink.py` earned its `O_CREAT|O_EXCL` claim over, and this store claims the same way. A
writer KEEPS its segment across an hour rotation; only construction scans for a free one.

### The record

Common to every channel:

| field | meaning |
|---|---|
| `contract` | `game-record-v1` |
| `game_id` | unique within the run. Self-play: uuid4. Eval: `<round_id>_<phase>_<NNNNN>`, deterministic so a reader holding a round's logs can name the game those logs are about |
| `run_id` | the run |
| `channel` | `selfplay` \| `promotion` \| `external` \| `random_floor` |
| `step` | the training step this game is attributed to |
| `step_kind` | **`actor`** on self-play (the step whose WEIGHTS played it, forwarded by `ActorSync`) \| **`round`** on eval (the learner step the candidate snapshot was cut at). The two channels measure "which step" differently and the field says which, so a reader cannot plot them on one axis by accident (LAW-03) |
| `seed` | the seed that determined this game's opening/regime |
| `served_sims` | sims per move this game was played at |
| `plies` | ply count; equals `len(moves)` |
| `result` | `p1` \| `p2` \| `draw` \| `unknown`. SEAT-relative — p1 is the side that moved first. `unknown` is a winner code the engine could not decode and is NEVER folded into `draw` (AUDIT-1 F-28/C04) |
| `termination` | `six_in_a_row` \| `colony` \| `ply_cap` \| `other_draw` \| `unknown` (self-play) or `mantis.arena.adjudicate.TERMINAL_REASONS` (eval) |
| `moves` | `[[q, r], …]`, axial, **one entry per PLY**. A turn places two stones and the first turn one, so the turn grouping is DERIVED by the reader from its own index rather than stored |

Self-play only: `worker_id`, `game_id_byte_hash` (LAW-04's dedupe input, carried so effective-n
is counted off the record rather than recomputed).

Eval only: `rung`, `phase`, `game_index`, `colors` (`{candidate, opponent}` seats),
`trajectory_hash`, and `search_stats` when the candidate's search exposed its root.

**`game_index` JOINS a record to its progress row.** The round's progress writer and this one
are fed from ONE fan-out in loop order, so their per-round counters advance in lockstep and a
`<round_id>_progress.txt` row and a game record carrying the same index are the same game.
`play_paired_match` calls its sink in loop order under every concurrency, which is what makes
that true at `concurrency > 1` as well.

`search_stats` is a list of
`{"ply": int, "by": "candidate"|"opponent", "root_value": float, "visits": [[q, r, n], …]}`, one
entry per ply whose MOVER exposed a search root.

**`by` is load-bearing, not decoration.** On the PROMOTION channel both players are deploy heads
(candidate net vs anchor net), so the list covers every ply from BOTH sides; on a rung or the
random floor the opponent is a plain bot with no root and only the candidate's plies appear. The
two are otherwise indistinguishable, and a reader would take a two-sided list for a one-sided
one — halving every per-move statistic it computed.

**An EMPTY `visits` is a real state and is KEPT.** At a very low simulation budget the root is
expanded and nothing is backed up to a child, so there is no visited child to record: a local
boot at `eval.gate.deploy_sims: 1` produced 127 roots, every one with an empty support. The
entry stays because `root_value` is still information and because `len(search_stats)` must keep
counting the plies that were searched. Empty is DIFFERENT from the field being absent — absent
means no producer, empty means a search whose support was empty.

`visits` carries the SUPPORT only — visited children — because
a zero-visit child is part of the distribution, carries none of its information, and at radius 8
would be most of the bytes. A reader takes absence as zero, which is what a visit distribution
means.

### Absent is not zero, and it applies to fields

* `colors` is **absent** on self-play: both seats are the same net, so a `{"p1": 1, "p2": -1}`
  dict would read like information.
* `search_stats` is **absent** on self-play. It is a GAP, not a nothing: the visit distribution
  exists in the engine and reaches the replay ring, but every row is pushed `game_id=-1` by
  construction, so no position can be attributed to a game without an engine change on the hot
  drain path (LAW-09). A viewer draws this as a stated gap, never as an empty heatmap.
* On the eval channels `search_stats` may be `None` (no player exposed a root) or `[]` (the
  candidate never moved). They are different facts and neither is a measurement.

## Reading

`read_shard(path) -> (records, skipped)` and `iter_run_games(dir, run_id)`.

**A partial trailing line is normal and is not fatal.** A shard being read while it is being
written has no closing guarantee on its last line, and a shard whose process was killed mid-write
keeps that partial line forever. `read_shard` SKIPS it and COUNTS it — counting is the half that
matters, because a bare skip cannot tell "one torn tail" from "this file is garbage".

## Failure posture

Construction RAISES (`GameRecordError` / `RunIdError`): a store that cannot open at boot is a
loud startup error, which is the `JsonlEventSink` posture. After construction a write failure
increments `persist_errors_total`, logs an ERROR and DISABLES the store — the eval progress
writer's posture under R319(e)(ii), for its reason: losing the record of a game must not kill an
otherwise healthy run, and the counter is what makes the loss visible rather than silent. A
disabled store stays disabled: retrying once per game into a full volume turns a disk failure
into a log flood.

## Who asserts what where

| fact | pinning test |
|---|---|
| a closed shard re-opens byte-exact | `tests/monitor/test_game_record.py::test_a_closed_shard_reopens_byte_exact` |
| a planted truncated shard is skipped and counted, not fatal | `::test_a_planted_truncated_shard_is_SKIPPED_not_fatal` |
| two writers in one run and one hour claim different shards | `::test_two_writers_in_one_run_and_one_hour_claim_DIFFERENT_shards` |
| an hour rotation keeps the writer's own segment | `::test_an_hour_rotation_KEEPS_the_writers_own_segment` |
| the index names only shards whose bytes are down | `::test_the_index_names_only_shards_whose_bytes_are_down` |
| a write failure disables, counts, and never raises | `::test_a_write_failure_DISABLES_and_COUNTS_and_never_raises` |
| an undecodable winner code is `unknown`, never `draw` | `::test_an_undecodable_winner_code_is_unknown_and_NEVER_a_draw` |
| the production pool is built with a real recorder | `::test_the_production_pool_is_BUILT_with_a_real_recorder` |
| a real eval round writes every game it played | `tests/eval/test_game_record_eval_channel.py::test_a_real_round_writes_every_game_it_played` |
| eval games carry per-position search stats | `::test_the_gate_block_carries_PER_POSITION_SEARCH_STATS` |
| production rounds always get a record target | `::test_the_pipeline_ALWAYS_gives_its_rounds_a_record_target` |
