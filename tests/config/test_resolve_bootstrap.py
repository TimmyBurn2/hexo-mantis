"""O5 — bootstrap path resolver (resolve/bootstrap.resolve_bootstrap).

CLI arg, not a config-file key -> no schema field. Validates at launch (exists-or-raise);
existence probe is injectable and called exactly once.
"""
import pytest

from mantis.config.resolve.bootstrap import (
    BootstrapNotFoundError,
    ResolvedBootstrap,
    resolve_bootstrap,
)


def test_none_is_fresh_run_no_raise():
    res = resolve_bootstrap(None)
    assert res == ResolvedBootstrap(path=None, source="none")


def test_existing_path_is_cli_source():
    res = resolve_bootstrap("exists.pt", exists=lambda _: True)
    assert res == ResolvedBootstrap("exists.pt", "cli")


def test_missing_path_raises_naming_path_and_knob():
    with pytest.raises(BootstrapNotFoundError) as exc:
        resolve_bootstrap("missing.pt", exists=lambda _: False)
    msg = str(exc.value)
    assert "missing.pt" in msg and "BOOTSTRAP" in msg


def test_error_is_file_not_found_error():
    assert issubclass(BootstrapNotFoundError, FileNotFoundError)


def test_existence_checked_exactly_once():
    calls = []

    def _exists(p):
        calls.append(p)
        return True

    resolve_bootstrap("exists.pt", exists=_exists)
    assert calls == ["exists.pt"]
