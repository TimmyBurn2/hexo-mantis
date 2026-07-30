"""tools/bench_floors.toml — schema, provenance and liveness validation (WPBOX CB-3).

The floors file is the R29/R18 production baseline: criterion mid estimates measured
serialized on the production box. Its RATE consumer is LAW-09's bench discipline (a
hot-path change re-runs the named bench and compares against the floor, IQR-gated);
THIS test is the structural consumer that keeps the file from rotting: every floor row
names a bench target that exists in the workspace, provenance is complete and carries
no host identifiers beyond the four permitted fields, and the numbers are sane
(low <= mid <= high, all positive). A floors file no gate or test reads would be the
phantom-input class (LAW-07) — this is the producer-side pin.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOORS_PATH = REPO_ROOT / "tools" / "bench_floors.toml"

REQUIRED_PROVENANCE = ("interpreter", "numpy", "rustc", "cpu_model")


def _load() -> dict:
    return tomllib.loads(FLOORS_PATH.read_text())


def test_floors_file_exists_and_parses() -> None:
    assert FLOORS_PATH.is_file(), "tools/bench_floors.toml is the R18 baseline — missing"
    data = _load()
    assert data.get("floor"), "an empty floors table floors nothing"


def test_provenance_is_complete_and_bounded() -> None:
    """Exactly the four permitted fields (R112: interpreter/numpy/rustc/CPU-model ONLY —
    a host identifier here would leak the box into the repo)."""
    prov = _load().get("provenance", {})
    assert sorted(prov) == sorted(REQUIRED_PROVENANCE), (
        f"provenance must carry exactly {REQUIRED_PROVENANCE}, got {sorted(prov)}"
    )
    for key in REQUIRED_PROVENANCE:
        value = prov[key]
        assert isinstance(value, str) and value and value != "UNKNOWN", (
            f"provenance.{key} must be a real recorded value, got {value!r}"
        )


def test_every_floor_row_names_a_live_bench_and_sane_numbers() -> None:
    """Liveness: the named bench source exists under crates/*/benches; sanity:
    0 < low <= mid <= high. A floor for a deleted bench is a stale declaration —
    the same class the config-declaration partition refuses (`stale`)."""
    bench_sources = {p.stem for p in REPO_ROOT.glob("crates/*/benches/*.rs")}
    data = _load()
    for key, row in data["floor"].items():
        assert row["bench"] in bench_sources, (
            f"floor.{key} names bench {row['bench']!r} with no source under "
            f"crates/*/benches ({sorted(bench_sources)})"
        )
        low, mid, high = row["low_s"], row["mid_s"], row["high_s"]
        assert 0 < low <= mid <= high, (
            f"floor.{key} carries non-sane bracket low={low} mid={mid} high={high}"
        )
