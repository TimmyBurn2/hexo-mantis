"""Registry-sha handshake — the stale-`.so`/stale-registry guard (LOCKED #8).

[WRITTEN-FIRST] NEW-BUILD contract. At `import mantis.encoding` the on-disk
`registry.toml` is hashed and compared to the compiled `_engine.registry_sha()`;
a drift HARD-ERRORS. LAW-07 mutation self-test: mutate the on-disk TOML (a tmp
copy) → the handshake raises; the unmutated file passes; a truly-absent TOML
(installed-wheel layout) SKIPs with a logged reason — never a silent pass.
"""
from __future__ import annotations

import hashlib
import shutil

import pytest

from mantis import _engine
from mantis.encoding import EncodingRegistryError, _registry_sha_handshake, _resolve_registry_toml


def test_clean_import_passed_the_handshake() -> None:
    """`import mantis.encoding` above succeeded ⇒ the module-level handshake
    fired and passed against the real on-disk registry.toml."""
    toml = _resolve_registry_toml()
    assert toml is not None and toml.is_file(), (
        "in-repo test must resolve the on-disk registry.toml"
    )
    disk = hashlib.sha256(toml.read_bytes()).digest()
    assert disk == _engine.registry_sha(), (
        "on-disk registry.toml sha must match the compiled _engine.registry_sha()"
    )


def test_handshake_passes_on_unmutated_copy(tmp_path) -> None:
    src = _resolve_registry_toml()
    assert src is not None
    copy = tmp_path / "registry.toml"
    shutil.copyfile(src, copy)
    # No raise — byte-identical copy hashes to the compiled sha.
    _registry_sha_handshake(copy)


def test_handshake_hard_errors_on_mutation(tmp_path) -> None:
    """LAW-07 — the guard BITES: a mutated on-disk registry raises."""
    src = _resolve_registry_toml()
    assert src is not None
    mutated = tmp_path / "registry.toml"
    shutil.copyfile(src, mutated)
    with mutated.open("ab") as fh:
        fh.write(b"\n# drift injected by the mutation self-test\n")
    with pytest.raises(EncodingRegistryError, match="drifted"):
        _registry_sha_handshake(mutated)


def test_missing_toml_skips_not_silently_passes(tmp_path) -> None:
    """A truly-absent on-disk TOML (installed-wheel layout) SKIPs — it must NOT
    raise (that would break installed use) and must NOT masquerade as a match."""
    missing = tmp_path / "does_not_exist" / "registry.toml"
    assert not missing.exists()
    # No raise; the skip is logged (asserting the log is covered by caplog below).
    _registry_sha_handshake(missing)


def test_missing_toml_logs_a_reason(tmp_path, caplog) -> None:
    missing = tmp_path / "absent.toml"
    with caplog.at_level("INFO", logger="mantis.encoding"):
        _registry_sha_handshake(missing)
    assert any("SKIPPED" in rec.message for rec in caplog.records), (
        "the skip must be logged, never silent"
    )


def test_handshake_uses_the_compiled_sha(tmp_path) -> None:
    """A stubbed engine whose registry_sha disagrees must trip the guard even on
    the real, unmutated on-disk file (proves the compiled sha is the authority)."""
    src = _resolve_registry_toml()
    assert src is not None

    class _StubEngine:
        @staticmethod
        def registry_sha() -> bytes:
            return b"\x00" * 32

    with pytest.raises(EncodingRegistryError, match="drifted"):
        _registry_sha_handshake(src, engine=_StubEngine())
