# mantis

AlphaZero-style self-play bot for Hex Tac Toe — hex grid, 6-in-a-row to win, compound
2-stone turns, unbounded board. Rust engine (cargo workspace) + Python training/eval
(uv, src-layout), PyO3 bridge.

**STATUS: the port has landed; the first full run has not.** The engine, the training
loop, the self-play runner and the eval ladder are all here and tested — `python -m
mantis.run` starts a real run against a minted config. What remains is the run itself: the
run6 identity is being minted and no production training run has completed.

<!-- AUDIT-1 F-52: this said "pre-migration skeleton … the game/training code is not here
     yet" over a tree with a live launcher, a five-crate Rust workspace and a four-figure
     test count. It had been wrong for many work packages. -->

## Quickstart

```
uv sync          # builds the compiled extension (mantis._engine) via maturin
make test        # pytest default tier + cargo test --workspace
make check.wasm  # mantis-graph stays wasm32-clean
```

## Layout

| path | what |
|---|---|
| crates/ | six-crate cargo workspace: core, graph, encoding, search, selfplay, bridge (the only PyO3 crate) |
| src/mantis/ | the one Python package (src-layout) |
| tests/ | single test-collection root + fixtures manifest |
| configs/ | complete, schema-validated configs, minted via tools/mint_config.py |
| docs/ | design contract, seam contracts, registers |
| tools/ | dev-only tooling + locally runnable CI gate scripts |
| vendor/ | pins.toml + `make vendor` fetcher (no submodules) |

Pointers: docs/design/repo_design.md (structural contract), CLAUDE.md (operating
rules), docs/registers/ (laws + falsified register). Every CI gate is locally runnable
(tools/ci_gates/).
