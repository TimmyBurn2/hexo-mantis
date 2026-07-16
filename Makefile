.PHONY: build build.native test test.integration bench bench.baseline check.wasm vendor clean

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

bench:
	cargo bench -p mantis-core --bench smoke_bench --locked -- --warm-up-time 0.5 --measurement-time 1

bench.baseline:
	cargo bench -p mantis-core --bench smoke_bench --locked -- --save-baseline local --warm-up-time 0.5 --measurement-time 1

check.wasm:
	cargo check -p mantis-graph --target wasm32-unknown-unknown --locked

vendor:
	bash tools/vendor_fetch.sh

clean:
	cargo clean
	rm -rf dist
