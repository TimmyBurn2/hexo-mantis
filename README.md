# mantis

AlphaZero-style self-play bot for Hex Tac Toe — hex grid, 6-in-a-row to win, compound
2-stone turns, unbounded board. Rust engine (cargo workspace) + Python training/eval
(uv, src-layout), PyO3 bridge.

**STATUS: the port has landed and has trained production runs.** The engine, the training
loop, the self-play runner and the eval ladder are all here and tested — `python -m
mantis.run` starts a real run against a minted config. Multiple production runs have run and
stopped; `docs/governance/STATE.md` records where the current run stands.

## Quickstart

```
uv sync          # builds the compiled extension (mantis._engine) via maturin
make test        # pytest default tier + cargo test --workspace
make check.wasm  # mantis-graph stays wasm32-clean
```

## The native engine: `mantis-cli`

One binary, the browser engine's search (the deploy block of record, the tactics solver, the per-game cache, tree
reuse, the early stop and the clock budget) with the net's forward on the CPU, speaking Six's text protocol on stdio.

**Run it.** Unpack `mantis-cli-<version>-linux-x86_64.tar.gz` and start `./mantis-cli`: it finds the
`*.manifest.json` beside it (or `--net PATH`), refuses to start unless the weights file is the one the manifest binds by
sha256, plays one warm-up turn to measure its cost a leaf, and then reads commands. `--threads N` sets its threads
(default 4, at most the machine's; 0 is refused). To build it: `cargo build --release -p mantis-player --bin mantis-cli`;
`tools/release_cli.sh NET_DIR OUT_DIR` packs the tarball on a clean commit.

**Drive it.** One command a line; answers on stdout, one counters line a turn on stderr:

```
six                                   -> id name mantis-cli 0.1.0 / id net <stem> manifest <sha256> / sixok
newgame                               -> (the next go starts a fresh game)
position radius 8 moves 0 0 1 0 0 1   -> (the game's stones in play order, x first)
go nodes 800                          -> bestmove q1 r1 q2 r2   (800 sims for the turn: 400 a stone)
go movetime 5000                      -> bestmove q1 r1 q2 r2   (a 5 s turn, the budget rule's sims)
quit
```

`go nodes N` is the turn's budget, as Six reads it, split evenly over the stones due. The `id net … manifest <sha256>`
line names the net a driver is playing. A refused position or search answers `error <why>`, and a
refused `go` then `bestmove none`. Keep sending the whole game: a position that extends the last one keeps the tree and
cache. The protocol, the refusals and the counters are in `docs/contracts/web_engine.md` (#13).

**The arena adapter.** It is witnessed through the tree's Six-protocol driver: `src/mantis/bots/six.py`'s `SixBot`,
over a small engine adapter (`new_game` / `search` / `close`), played NATIVE-1's 288-game cell against Six. The arena's
own example bot drives a Six-protocol engine on stdio, but it has not been run against `mantis-cli`. Offer clocks at or
above the `min_turn_ms` the binary prints at start.

## Layout

| path | what |
|---|---|
| crates/ | seven-crate cargo workspace: core, graph, encoding, search, selfplay, player (the deploy engine and `mantis-cli`), bridge (the only PyO3 crate) |
| src/mantis/ | the one Python package (src-layout) |
| tests/ | single test-collection root + fixtures manifest |
| configs/ | complete, schema-validated configs, minted via tools/mint_config.py |
| docs/ | design contract, seam contracts, governance |
| tools/ | dev-only tooling + locally runnable CI gate scripts |
| vendor/ | pins.toml + `make vendor` fetcher (no submodules) |

Pointers: docs/design/repo_design.md (structural contract), CLAUDE.md (operating
rules), docs/governance/ (LAWS, STATE, RULINGS, CARDS, falsified register; R346 froze
the old registers into docs/governance/archive/). Every CI gate is locally runnable
(tools/ci_gates/).
