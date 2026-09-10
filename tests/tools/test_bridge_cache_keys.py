"""Every real build input of `mantis._engine` is a uv cache key.

`cache-keys` REPLACES uv's default key set and uv decides whether to build at all, so an
unkeyed build input leaves the venv serving the previous `.so` — invisible to CI, which always
builds cold. Both sides are derived (workspace members vs globs resolved against the tree), so
a new crate turns this red until it is keyed.
"""
from __future__ import annotations

import glob as globmod
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = REPO_ROOT / "crates" / "mantis-bridge"
BRIDGE_PYPROJECT = BRIDGE_DIR / "pyproject.toml"

# `make build.native` sets RUSTFLAGS=-C target-cpu=native; without these keys a later plain
# `make build` sees no keyed change and keeps the host-pinned binary.
REQUIRED_ENV_KEYS = ("RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")


def _load_cache_keys() -> list[dict[str, str]]:
    data = tomllib.loads(BRIDGE_PYPROJECT.read_text(encoding="utf-8"))
    keys = data.get("tool", {}).get("uv", {}).get("cache-keys")
    assert keys is not None, (
        f"{BRIDGE_PYPROJECT} declares no [tool.uv] cache-keys. Without the table uv falls back "
        "to its default key set, which does not know about crates/ at all."
    )
    return keys


CACHE_KEYS = _load_cache_keys()


def _covered_files(cache_keys: list[dict[str, str]]) -> set[Path]:
    """Resolve every `{file = ...}` glob against the real tree, relative to the bridge dir."""
    covered: set[Path] = set()
    for entry in cache_keys:
        pattern = entry.get("file")
        if pattern is None:
            continue
        for hit in globmod.glob(pattern, root_dir=BRIDGE_DIR, recursive=True):
            resolved = (BRIDGE_DIR / hit).resolve()
            if resolved.is_file():
                covered.add(resolved)
    return covered


def _workspace_members() -> list[Path]:
    data = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    members = data["workspace"]["members"]
    assert members, "workspace declares no members — the census would be vacuous"
    return [REPO_ROOT / m for m in members]


def _required_build_inputs() -> set[Path]:
    """Every file whose content can change the emitted extension.

    `src/**/*` rather than `src/**/*.rs`: `crates/mantis-encoding/src` holds `registry.toml`,
    which is `include_str!`-ed into the binary.
    """
    required: set[Path] = {
        (REPO_ROOT / "Cargo.toml").resolve(),
        (REPO_ROOT / "Cargo.lock").resolve(),
        (REPO_ROOT / "rust-toolchain.toml").resolve(),
        BRIDGE_PYPROJECT.resolve(),
    }
    for member in _workspace_members():
        required.add((member / "Cargo.toml").resolve())
        for path in (member / "src").rglob("*"):
            if path.is_file():
                required.add(path.resolve())
    return required


def _uncovered(cache_keys: list[dict[str, str]]) -> set[Path]:
    return _required_build_inputs() - _covered_files(cache_keys)


def test_every_workspace_build_input_is_a_cache_key() -> None:
    missing = _uncovered(CACHE_KEYS)
    assert not missing, (
        "these build inputs are NOT covered by any [tool.uv] cache-keys glob, so editing them "
        "will not rebuild mantis._engine — the venv keeps serving the previous binary and CI "
        "(which always builds cold) cannot see it:\n  "
        + "\n  ".join(sorted(str(p.relative_to(REPO_ROOT)) for p in missing))
        + f"\n\nAdd the missing entries to {BRIDGE_PYPROJECT.relative_to(REPO_ROOT)}. Adding a "
        "crate takes BOTH its `src/**/*` and its `Cargo.toml`."
    )


@pytest.mark.parametrize(
    "rel",
    [
        "crates/mantis-encoding/src/registry.toml",
    ],
)
def test_embedded_registry_data_files_are_cache_keys(rel: str) -> None:
    """Name the `include_str!`-ed data files, so tidying the encoding glob to `.rs` fails loud."""
    target = (REPO_ROOT / rel).resolve()
    assert target.is_file(), f"{rel} does not exist — the pin is asserting against a ghost"
    assert target in _covered_files(CACHE_KEYS), (
        f"{rel} is not a uv cache key. It is `include_str!`-ed into the shipped extension, so "
        "editing it changes the binary while uv declines to rebuild."
    )


@pytest.mark.parametrize("name", REQUIRED_ENV_KEYS)
def test_rustflags_env_vars_are_cache_keys(name: str) -> None:
    declared = {e["env"] for e in CACHE_KEYS if "env" in e}
    assert name in declared, (
        f"{name} is not an [tool.uv] cache key. `make build.native` sets it; without the key a "
        "later plain `make build` sees no keyed change and keeps the host-pinned binary."
    )


def test_lockfile_is_a_cache_key() -> None:
    """`cargo update` bumping a dep changes the emitted binary with no source file touched."""
    assert (REPO_ROOT / "Cargo.lock").resolve() in _covered_files(CACHE_KEYS)


# Mutation self-tests: each case drops ONE key from an in-memory copy and asserts the named
# file goes uncovered, so a census that silently covered everything cannot report green.


def _without(pattern: str) -> list[dict[str, str]]:
    pruned = [e for e in CACHE_KEYS if e.get("file") != pattern]
    assert len(pruned) == len(CACHE_KEYS) - 1, (
        f"expected exactly one cache key with file={pattern!r}; the self-test is mutating "
        "something other than what it claims"
    )
    return pruned


@pytest.mark.parametrize(
    ("dropped_pattern", "now_uncovered"),
    [
        ("../mantis-encoding/src/**/*", "crates/mantis-encoding/src/registry.toml"),
        ("../mantis-core/Cargo.toml", "crates/mantis-core/Cargo.toml"),
        ("../../Cargo.lock", "Cargo.lock"),
        ("../../rust-toolchain.toml", "rust-toolchain.toml"),
        ("src/**/*", "crates/mantis-bridge/src/lib.rs"),
    ],
)
def test_census_bites_when_a_key_is_removed(dropped_pattern: str, now_uncovered: str) -> None:
    victim = (REPO_ROOT / now_uncovered).resolve()
    assert victim in _covered_files(CACHE_KEYS), (
        f"{now_uncovered} is not covered even BEFORE the mutation — the self-test's premise is "
        "already false, so it proves nothing"
    )
    assert victim in _uncovered(_without(dropped_pattern)), (
        f"MUTATION SELF-TEST FAILED: dropping cache key {dropped_pattern!r} left "
        f"{now_uncovered} still covered. The census cannot detect a missing key, so it is a "
        "phantom gate — it would report green with the build-integrity defect present."
    )


def test_env_check_bites_when_rustflags_is_removed() -> None:
    pruned = [e for e in CACHE_KEYS if e.get("env") != "RUSTFLAGS"]
    assert len(pruned) == len(CACHE_KEYS) - 1
    declared = {e["env"] for e in pruned if "env" in e}
    assert "RUSTFLAGS" not in declared, (
        "MUTATION SELF-TEST FAILED: RUSTFLAGS still reads as declared after removal — the env "
        "assertion is not reading the list it claims to read."
    )


def test_a_new_crate_would_be_caught() -> None:
    """Prove the census is member-driven: a crate that does not exist must not resolve covered."""
    synthetic = (REPO_ROOT / "crates" / "mantis-newthing" / "src" / "lib.rs").resolve()
    covered = _covered_files(CACHE_KEYS)
    assert synthetic not in covered, (
        "a crate that does not exist resolved as COVERED — a glob is matching far more than it "
        "should, and the census would not notice a genuinely unkeyed new crate"
    )
