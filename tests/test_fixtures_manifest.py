"""Fixtures-manifest checker + tests (repo_design §8): FAIL — never skip — on absence/drift."""
import hashlib
import tomllib
from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
MANIFEST_PATH = FIXTURES_ROOT / "manifest.toml"

_TOP_LEVEL_KEYS = {"schema", "required"}
_ROW_KEYS = {"path", "sha256", "added_by"}


class FixtureManifestError(Exception):
    """Raised on any structural or presence/hash violation of the fixtures manifest."""


def check_manifest(manifest_path: Path, fixtures_root: Path) -> int:
    """Validate the manifest; return the required-row count. Raises, never skips."""
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise FixtureManifestError(f"manifest missing/unreadable: {manifest_path}") from exc
    try:
        data = tomllib.loads(raw.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise FixtureManifestError(f"manifest unparseable: {exc}") from exc
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise FixtureManifestError(f"unknown top-level keys: {sorted(unknown)}")
    if data.get("schema") != 1:
        raise FixtureManifestError(f"schema must be 1, got {data.get('schema')!r}")
    rows = data.get("required")
    if not isinstance(rows, list):
        raise FixtureManifestError("required must be an array of tables")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise FixtureManifestError(f"row {i}: keys must be exactly {sorted(_ROW_KEYS)}")
        fixture = fixtures_root / str(row["path"])
        if not fixture.is_file():
            raise FixtureManifestError(f"row {i}: missing fixture file: {row['path']}")
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise FixtureManifestError(
                f"row {i}: sha256 mismatch for {row['path']}: {digest} != {row['sha256']}"
            )
    return len(rows)


def test_manifest_parses_and_schema_validates():
    count = check_manifest(MANIFEST_PATH, FIXTURES_ROOT)
    assert count >= 0


def test_every_required_fixture_present_with_matching_sha():
    # Vacuously green at WP0's empty required set (declared risk R-6); the checker
    # verifies presence + sha for every row on the same call path the self-tests prove.
    check_manifest(MANIFEST_PATH, FIXTURES_ROOT)


def test_checker_raises_on_ghost_fixture(tmp_path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "schema = 1\n"
        "[[required]]\n"
        'path = "ghost.bin"\n'
        f'sha256 = "{"0" * 64}"\n'
        'added_by = "wp0-selftest"\n'
    )
    with pytest.raises(FixtureManifestError, match="ghost.bin"):
        check_manifest(manifest, tmp_path)


def test_checker_raises_on_unknown_key(tmp_path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text("schema = 1\nrequired = []\nsurprise = true\n")
    with pytest.raises(FixtureManifestError, match="surprise"):
        check_manifest(manifest, tmp_path)
