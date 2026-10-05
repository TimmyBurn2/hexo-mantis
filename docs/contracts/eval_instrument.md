# Contract: eval instrument

- version: v6
- owner: mantis.arena
- status: LIVE. <!-- AUDIT-1 F-52: this read "SKELETON — contract text lands with the
  subsystem port" over a filled contract, beside a shipped eval subsystem. The label is
  decision-adjacent: a reader deciding whether the deploy-matched bar is specified would
  have concluded it is not. -->

## Summary
deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games

## Who asserts what where

This section is the DURABLE instrument: what each term means, and which module is the single
authority for it. The run-specific choices — which numbers — live in the minted config and
are read from it, never restated here. **R362(c) (2026-09-19) DELETED the sealbot rung** from
the production round (`eval.ladder`, `eval.sealbot_model_sims`, `eval.rung_concurrency`, the
ladder state, the Bradley-Terry fit, `eval_channel_health`, `wr_sealbot` and the coordinator's
WR gate): the round is the strength-floor probe, the gate block and the random floor, and the
EXTERNAL scale is the strix cell (`tools/strength_frontier.py`, `tools/strix_follower.py`). The
rung machinery (`RoundSpec.rung_jobs`, `worker._play_rung_block`) survives for that cell alone.
v2: the sealbot adapter, its fixed-depth receipt and its refusal classes are DELETED (R368(e)).
v3: the sealbot vendor side — build script, patch, pin, `vendor.sealbot` target and both pin
tests — is DELETED (R368(e)).
v4: the strix cell's regime is the load of the host that PLAYS it, not the run's mirrored heartbeat
(PERF-ADA H7): off the box that heartbeat is always live and labelled every cell CONTENDED. The
receipt's `schema_version` moves 1 → 2 with it, so a v1 receipt's regime reads under the heartbeat rule.
v5: the SIX ruler (R374(b), SIX-RUNG) lands beside strix: a pin may name release assets by url and
sha256 (`make vendor.six`).
v6: the cell tools' tactics arms (TACTICS-DEPLOY): `--arm plain|full|audit-off|known-bad` over a
`search.tactics` block (`--tactics-block`, JSON) arms the CANDIDATE side of a ruler cell (and, since the gate became
deploy-matched under R378(d), both heads of a snapshot-opponent cell, stamped `sides: both` in its receipt), through
`mantis.config.resolve.tactics.arm_from_file` (the one reader); with no arm the config's own
`deploy.search.tactics` plays, and a cells file naming a block is refused. An arm's cells are its own: the
frontier labels them `<label>_<arm>` (so their games' directory), and the follower's receipt is
`<ckpt>.<unit>.<arm>.json`, beside and never shadowing the config's; a re-read of that receipt under another block
is refused (another block is another arm). A record's `tactics` section (present iff an arm was named or the block
was non-null) carries the arm, the resolved block, `module_sha256` (the TREE's tactics sources,
`crates/mantis-search/src/tactics/**` and `mcts/tactics_*.rs`) and `engine_sha256` (the BYTES of the `mantis._engine`
the child loaded), both read before the child plays; then the candidate's rows summed over the cell's games, and
`proof_games_lost` / `proof_games_drawn`: each game in which the candidate found a root proof and did not win, with
its termination. Such a game is a refuted claim, a continuation the search lost after the proof's turn, or a ply-cap
draw, and is read game by game. The engine hash names what played; the module hash names the tree it was meant to
be built from (a build-time hash is `CARD-TACTICS-BUILD-HASH`). The dash reads each arm, and each block under it
(its hash's first 8 hex), as its own ruler.

The run5 decision document carried that run's choices and is DELETED with its config
(R346(f)): a decision document whose subject config is not in the tree
has nothing to be checked against, and its drift gate derived every expectation from that
file. The tag `archive/grid-path` carries both. Two of its properties were NOT run-specific
and are folded in here, because a reader of any ladder reading needs them:

- **The Bradley-Terry fit was only as wide as the opponent set** — with one bot family on the
  ladder, ONE opponent lineage was all the information in it; read any pre-R362 record's `bt`
  field that way. The fit is deleted with the ladder (R362(c)).
- **eff_n on a deterministic rung is bounded by the openings, not by the game count.** A
  fixed-depth opponent facing a deterministic argmax head produces ONE trajectory per opening,
  which is why LAW-04 counts trajectory-hash-distinct games and why the opening book is
  sha-pinned rather than merely named.

- **Deploy-matched** means both sides of a comparison are built by the SAME player
  constructor at the SAME simulation count, with the SAME tactics block (R378(d)). `mantis.eval.worker` builds the candidate and the
  best snapshot through one `_build_candidate_player` call site at `eval.gate.deploy_sims`,
  and the `RegimeKey` stamped on each record carries the value that was actually used. A
  rung job (a strix cell's) is deploy-matched to its own per-kind simulation count, never to
  the gate's — playing at one value while stamping another is a mislabelled record, not a
  rounding difference.
- **The strix rung (RUNG-2, R352(e)) is a FIXED external reference of the other kind: a
  net.** `SootyOwl/hexo-strix` publishes no model, so the rung IS the operator's
  `checkpoint_00237000.pt`, pinned by sha256 in `vendor/pins.toml` beside the commit it is
  played through (the checkpoint is not in the tree — R7 — and is placed under
  `vendor/external/strix_models/`, verified at every load). It plays as a bot PROCESS in the
  vendored tree's own venv (`tools/strix_driver.py`, `mantis.bots.strix`), at strix's own
  eval/SPRT acting policy — argmax of the Gumbel-MCTS improved policy with Gumbel noise off —
  so it is deterministic by construction; the position is rebuilt from the board's stones on
  every call (`hexo_rs.GameState.from_state`, translated so a p1 stone sits at strix's origin,
  which the game's translation symmetry makes exact) and the opening single is answered by the
  adapter (every opening cell is the same position). THE FENCE IS READ AT CONTACT (R257): every
  reply carries strix's legal set, compared with the board's; a disagreement is counted as a
  finding, and a reply outside our fence is returned unchanged for the arena to FORFEIT, never
  substituted. It is NOT a ladder rung: `tools/strength_frontier.py` plays it as a cell in one
  of two units — AS-SHIPPED (strix at 128 sims / m 16, its `play_vs_shrimp` defaults, vs ours at
  PUCT-512) and EQUAL-WORK (both at 256 NN evaluations per move) — 288 paired games, both
  colours, outside the promotion gate. Its regime key carries its own sims
  (`strix:checkpoint_00237000@128`), so the two units are two instruments, not one. SINCE
  R356(a) the CADENCE cell is the equal-work unit, played by `tools/strix_follower.py` on every
  15 000-step checkpoint AND on every promotion, both read off the run's event stream
  (`periodic_checkpoint_save`, `eval_round_complete.promoted`), never off filenames; its receipt
  is a sidecar beside the checkpoint (`<ckpt>.strix256.json`: the checkpoint's sha256 and the
  net's `net_param_hash`, strix's pinned commit and checkpoint sha256, the unit's two sims, the
  trigger, the regime — CONTENDED when the host playing the cell is busy at cell start (any GPU at
  or above `GPU_BUSY_PCT` by nvidia-smi, or its 1-minute load per logical CPU at or above
  `LOAD_BUSY_PER_CPU`, over the CPUs the process may use), IDLE otherwise, with that host's load (`regime_evidence.host`: `load_1m`,
  `cpu_count`, `load_per_cpu`, `gpu_util_pct`, `null` without nvidia-smi) and every run's heartbeat
  age under the runs root (`live` names a live twin or parent) as evidence — and the pair-level readout: games, eff_n, wins, losses,
  draws, wr and its CI). The sidecar is the receipt: an existing one is never re-read, the stamp
  is never touched (LAW-12), and a failed cell writes `<ckpt>.strix256.failed.json`, which is
  not a receipt. The as-shipped cell reads at block ends only (`--once --unit as_shipped`,
  `<ckpt>.strix512.json`). The 256/256 reading of a run's parent, taken once, is both the bridge
  from the as-shipped series and that run's baseline.
- **The Six ruler (R374(b)) is a second FIXED external reference: `CixMango/Six`'s `sixengine`** —
  the v1.2.0 Linux release, built from engine sources identical to the pinned commit — playing a
  pinned network, gen 30 at 16 nodes, tactics as shipped, search cache off (R375(a)). Every asset is
  pinned by url and sha256 in `vendor/pins.toml`, and the engine, its ONNX Runtime (the library it
  loads and the CUDA provider's) and the network are re-hashed at every engine start (`mantis.bots.six`). ONE engine
  process per concurrent game (the rung block's first opponent resolves the rung before any game and,
  under concurrency, is closed at once; every opponent is closed when the block ends) speaks the Six protocol: `position radius R moves …` with R the
  board's `legal_move_radius`, `go nodes N`, `bestmove`; `setoption cacheEntries 0`, because Six's
  expansion cache survives `newgame` and would make a game depend on earlier ones. Six answers a
  whole compound turn; the second stone is played only on exactly the board it was chosen for. It
  reads the stone ORDER, which the arena reports (`observe_move`); a log that is not the board's
  stones is refused. The execution provider is read off the engine's stderr and logged: a cuda
  worker requires CUDA (the engine's CUDA libraries are the venv's NVIDIA wheels) and any other
  provider refuses the rung by name, as does an engine that fails to start or answers without a
  network (`id name HexBot Net`). A failed answer (`bestmove none`, an `error` line, a failed
  search's fallback, a malformed `bestmove`) returns an occupied cell (off the board on an empty one)
  and an illegal stone returns itself, both for the arena to FORFEIT, each counted and logged under
  `six_forfeit_finding`; a bot logs its searches and stale second stones under `six_engine_closed`
  when it closes. Its regime key reads `six:gen0030@16`. `tools/strix_follower.py` plays it as the `six30_16` unit (ours PUCT-256; `--once`,
  or `--follow` on the same triggers as the equal-work unit, which with it are the two ruler units),
  through `tools/strength_frontier.py`'s `six` cell (`six_net`, `six_nodes`). Its receipt is
  `<ckpt>.six30_16.json`, the strix receipt's fields with a `six` block in place of `strix` — the
  pin's commit, the engine, network and runtime sha256s the engines re-verified at start, the generation, the
  nodes, `cache_entries`, the provider, the engine starts, the searches and the stale second stones,
  all read off the child's log — and `six_findings`, the forfeits. A cell whose engines played any
  other bytes than the pin's, or more than one engine, runtime or network, writes `.failed.json`, not a receipt. The dash reads it as its own ruler beside strix's (the DASH-2 amendment).
- **Vendoring.** External engines are pinned by commit sha in `vendor/pins.toml` and fetched
  by `make vendor`, which CLONES and does not build. A pin's release assets (each a url and a
  sha256; an archive with `unpack`, a member of one with `from`) are fetched only by
  `tools/vendor_fetch.sh <pin>` — `make vendor.six` — and each is verified on arrival: a download
  becomes its file only once its sha256 is the pin's, an archive is extracted beside its directory and
  renamed into place whole, and a warm file that no longer verifies is refused, never re-fetched over. The one build step is the strix rung's
  venv: `make vendor.strix` (`tools/vendor_build_strix.sh`) syncs strix's OWN venv inside
  the fetched tree — its torch is the CPU wheel and its `hexo_rs` engine builds by maturin,
  apart from the mantis environment — and refuses before building on a missing tree or a
  drifted sha.
- **Books** are versioned, sha-pinned and paired: `mantis.arena.books` verifies the sha256 at
  load and raises on mismatch, and every opening is played exactly twice with the colours
  swapped, so a colour advantage cancels within the pair rather than across the sample.
- **eff_n is distinct games** (LAW-04): `mantis.eval.aggregate` dedupes on the trajectory
  hash before any interval is computed, and the low-power guard counts distinct games PER
  PAIR. Games are not evidence; distinct trajectories are.
- **The interval is a bootstrap percentile** over those distinct outcomes, seeded from the
  gate seed (gate blocks) or the rung job's own bootstrap seed (a cell's, `RungJob.bootstrap_*`,
  the frontier tool's pinned terms since R362(c)). An empty sample degenerates to an absent
  interval rather than raising, so a zero-game block cannot manufacture a bound; the random
  floor reports a point win rate and no interval, since nothing reads one.
- **A rung job that cannot resolve is RECORDED, never fatal**: the child's `skipped_rungs`
  list names it, and the frontier tool reads a skipped job as a FAILED cell, never a 0-game
  reading. (The production round's four skip channels — `eval_rung_skipped`, the ERROR line,
  the skip list, the `eval_rung_skip_class` counter — left with the sealbot rung, R362(c).)
- **A terminal round that yields no promotion decision is rc 48** (WP12-R Phase O / R152,
  discharging R133's "rc 0 does not certify eval health"). LAW-15's "no promotion decision =
  deliverable incomplete" is enforced at the PROCESS boundary: the terminal round's typed
  `eval_broken_reason` (`mantis.eval.errors.EvalBrokenReason`, seven members) is latched
  set-once by `drain.run_terminal_eval` and resolved to
  `monitor.heartbeat.TERMINAL_EVAL_BROKEN_EXIT_CODE` through the ONE resolver,
  `mantis.config.armed_aborts.exit_code_for_abort`. A MID-RUN broken round is deliberately
  NOT covered — rounds recur, and persistent breakage stays the heartbeat watchdog's
  jurisdiction (R133's split). Which of the seven broke is read in the ONE channel, on the
  `eval_broken` event's `reason`; the rc says only that the deliverable is incomplete.
  Pinned by `tests/train/test_terminal_eval_rc.py`.

## Pinning tests

Each row names the test that would go RED if the claim above stopped being true. A claim with
no runnable producer is not listed as covered here; where the producer cannot run in CI, the
row says so and names what does run.

| claim | pinning test | runs in CI? |
|---|---|---|
| deploy-matched gate, both sides at the same constructor and count | `tests/eval/test_gate_parity.py` | yes |
| the rung job plays at its per-kind simulation count, not the gate's | `tests/eval/test_rung_seat_off_window.py` | yes |
| the head answers outside the encoding window at both seats | `tests/eval/test_eval_selfplay_child_parity.py`, `tests/eval/test_rung_seat_off_window.py` | yes |
| an unimplemented declared pooling is REFUSED, never a fallthrough | `tests/eval/test_value_pool_guard.py`, `tests/eval/test_eval_decode_guard_ordering.py`, `tests/eval/test_graph_round_encoding.py` | yes |
| eff_n is trajectory-hash-distinct; the low-power guard is per pair; an empty sample degenerates rather than raising | `tests/eval/test_aggregate_regime.py` | yes |
| a pin is a commit sha | `tests/tools/test_vendor_pins_strix.py` | yes |
| a release asset becomes its file only once its sha256 is the pin's; a planted wrong sha256, a stale member, a directory that is not a clone and an unpack away from its archive's directory each refuse; an interrupted unpack is redone, never trusted; a bare `make vendor` fetches no asset | `tests/tools/test_vendor_fetch_assets.py` | yes (offline, `file://` assets) |
| the Six pin names the repo at its commit and the v1.2.0 Linux release, its engine, the ONNX Runtime it loads and the gen 30 and gen 455 networks, each by url and sha256 | `tests/tools/test_vendor_pins_six.py` | yes |
| a player that reads the move order hears every applied stone in order, the opening's included, and never a forfeited one | `tests/arena/test_move_observer.py` | yes |
| the refusal reasons name exactly their own missing step, and no environment key | `tests/bots/test_strix_adapter.py`, `tests/bots/test_protocol.py` | yes |
| the gate's rule fields ride `eval_round_complete.gate`, `null` when no gate ran; the A-3 partial carries them on a broken route | `tests/eval/test_gate_fields_ride_the_round_complete_row.py` | yes |
| a strength-floor refusal is a third thing on the routed mapping and the stream | `tests/eval/test_strength_floor_verdict_on_the_routed_mapping.py` | yes |
| the strix pin names the commit, the checkpoint and both sha256s, and discloses the unsupplied config | `tests/tools/test_vendor_pins_strix.py` | yes |
| the strix adapter sends the position, counts fence disagreements, returns an out-of-fence move for the forfeit, verifies the pinned sha, and answers the opening single itself | `tests/bots/test_strix_adapter.py` | yes (against a recording double) |
| the six adapter parses the protocol, sends the observed order at the board's radius, plays the turn's second stone only on the board it was chosen for, forfeits a failed, malformed or illegal answer through the arena and counts it, refuses a planted wrong network or runtime hash, an engine that cannot start or loads no network and a cuda rung whose engine fell back, puts the release's own runtime first on the library path, and logs its counters when it closes | `tests/bots/test_six_adapter.py` | yes (a recording double and a fake engine process) |
| the vendored Six plays two games on CUDA | `tests/bots/test_six_adapter.py::test_the_vendored_engine_plays_two_games_on_cuda` | where CUDA is; LOUD SKIP without it or without `make vendor.six` |
| a six job reads the candidate's sims on `rung_model_sims` and resolves at its own nodes on the round's worker device; the rung block closes its probe opponent before concurrent games and every opponent it made at the end | `tests/eval/test_strix_rung_sims.py` | yes |
| the follower fires ONE equal-work cell per cadence checkpoint and per promotion read off the event stream, a planted duplicate fires nothing, a promotion waits for its checkpoint, a failed cell leaves no receipt, the sidecar carries unit + regime + the net's hash and the checkpoint bytes are untouched | `tests/tools/test_strix_follower.py` | yes (the cell runner is a recording double; the unit's RoundSpec is composed through the real frontier) |
| a strix cell composes the rung at the pinned checkpoint with its own sims and the candidate's on `rung_model_sims`; a production round refuses a strix job by name | `tests/tools/test_strength_frontier.py`, `tests/eval/test_strix_rung_sims.py` | yes |
| each tactics arm resolves the one block or is refused by name; an arm's ruler cell arms the candidate only and writes its own receipt, whose `tactics` section carries the arm, the block, the sides it armed, the module hash and the summed rows with the proofs a game did not bear out; the dash reads an arm as its own unit, never on another's line | `tests/config/test_search_tactics.py`, `tests/tools/test_strength_frontier.py`, `tests/tools/test_strix_follower.py`, `tests/tools/test_dash_sidecars.py`; the gate pair's deploy-matched arming through the real worker, `tests/eval/test_game_record_eval_channel.py::test_an_armed_round_arms_both_gate_sides_and_every_record_carries_the_candidates_rows` | yes (the cell runner is a recording double; the worker round is real) |
| the six30_16 unit composes the six cell; its receipt carries the generation, the nodes, the provider, the counters, the forfeits and the engine, network and runtime bytes the engines verified, and a cell that played other bytes is a failed cell; the real producer's log lines read back through the frontier; `--follow` reads the two ruler units only; the dash reads the rung as its own series | `tests/tools/test_six_ruler_cell.py` | yes (the cell runner is a double; the producer round trip runs the real `mantis.bots.six` against a fake engine) |
| the REAL vendored strix plays 20 legal games end to end at the pinned commit | `tests/bots/test_strix_adapter.py::test_the_live_driver_plays_twenty_legal_games_end_to_end` | **no** — `@pytest.mark.integration`; LOUD SKIP naming the missing step without the vendored venv and checkpoint |
