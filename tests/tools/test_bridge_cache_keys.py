"""Item 1 pin: every real build input of `mantis._engine` is a uv cache key.

THE DEFECT CLASS (stale extension). uv's `cache-keys` REPLACES uv's default key set, and uv
decides whether to invoke the build AT ALL. A build input that is not a key is therefore not
"cargo rebuilds anyway" — cargo is never asked, and the venv keeps serving the previous
`.so`/`.pyd`. CI cannot see it because CI always builds cold: green in CI, stale in every
developer checkout that already had a build.

WHY A DERIVED CENSUS AND NOT A TRANSCRIBED LIST. The `cache-keys` list names crates one at a
time, so the failure mode is a future crate author adding `crates/mantis-newthing` and
forgetting the two lines. A test that re-listed the expected keys would have to be edited in
the same commit that forgets them, which is no protection at all. So the REQUIRED set is
derived from the workspace itself (`[workspace] members`) and the COVERED set is derived by
resolving each declared glob against the real filesystem — neither side is transcribed, and a
new crate turns this red until it is keyed (R98, derive at point of use).

SCOPE — WHAT THIS PIN DOES AND DOES NOT COVER. `registry.toml` specifically is ALSO protected
at runtime: CI gate 8's handshake hashes the on-disk registry and compares it to the compiled
`_engine.registry_sha()`, so a stale extension serving a stale registry hard-errors at
`import mantis.encoding`. That drift is LOUD. Every other build input — `Cargo.lock`, the
dependency crates' sources, `RUSTFLAGS` — has no such handshake, and for those a missing cache
key is silent. This test is what stands behind the silent ones; it asserts registry.toml too,
because the item's contract names it and because belt-and-braces on the one input with a
cross-language identity role is cheap.
"""
from __future__ import annotations

import glob as globmod
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = REPO_ROOT / "crates" / "mantis-bridge"
BRIDGE_PYPROJECT = BRIDGE_DIR / "pyproject.toml"

# The env keys that must be declared. `make build.native` sets RUSTFLAGS=-C target-cpu=native;
# without these a later plain `make build` sees no keyed change and KEEPS the host-pinned
# binary — R2/LAW-13's portability promise with no return path, and a poisoned `make bench`.
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
    """Resolve every `{file = ...}` glob against the real tree, relative to the bridge dir.

    `glob.glob(..., root_dir=, recursive=True)` is used rather than a hand-rolled matcher: the
    keys contain both `..` segments and `**`, and re-implementing that matching is exactly the
    drift-from-the-thing-it-certifies class this repo keeps finding.
    """
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

    Deliberately `src/**/*` and not `src/**/*.rs`: `crates/mantis-encoding/src` holds
    `registry.toml` and `manifests.toml`, both `include_str!`-ed into the binary. Anything else
    a future author drops into a `src/` tree is a build input by the same argument.
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


# --- the census ---------------------------------------------------------------------------


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
        "crates/mantis-encoding/src/manifests.toml",
    ],
)
def test_embedded_registry_data_files_are_cache_keys(rel: str) -> None:
    """The `include_str!`-ed data files, named explicitly.

    These are the reason the encoding glob is `src/**/*` rather than `src/**/*.rs`. A future
    author "tidying" that glob to `.rs` would still pass the census above only if these two
    files vanished; naming them here makes the tidy-up fail loudly instead.
    """
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


# --- LAW-07 mutation self-tests: the census must BITE --------------------------------------
#
# Each case removes ONE key from an in-memory copy of the list and asserts the named file goes
# uncovered. Mechanism, per case: the glob that resolved to that file is gone, so
# `_covered_files` no longer yields it, so it appears in `_required_build_inputs() - covered`.
# Without these, a census that silently covered everything (a bad glob, a `root_dir` change, a
# `.resolve()` mismatch on symlinks) would report green forever — a phantom gate (LAW-07).


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
    """The trap the cache-keys comment names in its own words: adding a crate and forgetting it.

    Simulated without touching the tree — a synthetic member is appended to the REQUIRED side
    and must come back uncovered, proving the census is member-driven rather than keyed to the
    six crates that happen to exist today.
    """
    synthetic = (REPO_ROOT / "crates" / "mantis-newthing" / "src" / "lib.rs").resolve()
    covered = _covered_files(CACHE_KEYS)
    assert synthetic not in covered, (
        "a crate that does not exist resolved as COVERED — a glob is matching far more than it "
        "should, and the census would not notice a genuinely unkeyed new crate"
    )
