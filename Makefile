.PHONY: build build.native test test.integration lint lint.rust gates bench bench.baseline check.wasm vendor vendor.sealbot clean

UV ?= uv

build:
	$(UV) sync

# Local perf builds only. target-cpu never appears in committed build config;
# artifacts built this way are host-specific and must not be distributed.
build.native:
	RUSTFLAGS="-C target-cpu=native" $(UV) sync --reinstall-package mantis-engine

test:
	$(UV) run pytest -m "not integration and not slow"
	cargo test --workspace --locked

test.integration:
	$(UV) run pytest -m integration

# CI gate 14 (R98): curated lint/type gate — zero-error baseline, self-tested trigger.
lint:
	bash tools/ci_gates/lint_gate.sh --self-test

# CI gate 2b. `--all-targets` is the load-bearing flag: without it the seven non-smoke
# `[[bench]]` targets are compiled by NO local command, so the 28 floors in
# tools/bench_floors.toml stand behind code nothing builds (AUDIT-1 F-09).
lint.rust:
	cargo clippy --workspace --all-targets --locked -- -D clippy::all

# THE LOCAL GATE SET (AUDIT-1 F-09). R311(b) made local green the gate; this is what
# "local green" means. Gate 1 is opt-in (`--with-fresh-sync`) — see the script's header.
gates:
	bash tools/ci_gates/run_all.sh

bench:
	cargo bench -p mantis-core --bench smoke_bench --locked -- --warm-up-time 0.5 --measurement-time 1

bench.baseline:
	cargo bench -p mantis-core --bench smoke_bench --locked -- --save-baseline local --warm-up-time 0.5 --measurement-time 1

check.wasm:
	cargo check -p mantis-graph --target wasm32-unknown-unknown --locked

vendor:
	bash tools/vendor_fetch.sh

vendor.sealbot:
	bash tools/vendor_build_sealbot.sh

clean:
	cargo clean
	rm -rf dist
