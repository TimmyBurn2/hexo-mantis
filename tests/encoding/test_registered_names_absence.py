"""Census/grep — the hardcoded name tuple is GONE (LOCKED #7 KILL).

`_REGISTERED_NAMES` must not survive anywhere in the encoding shim: the
registered-name set comes from ONE `_engine.all_specs()` call, and the filename-
beats-shape detector order is killed (no `_filename_match` dispatch). This is the
grep that catches a regression re-introducing either killed construct.
"""
from __future__ import annotations

from pathlib import Path

import mantis.encoding.registry as registry
from mantis import _engine
from mantis.encoding import all_specs

_PKG_DIR = Path(registry.__file__).resolve().parent


def _pkg_sources() -> list[Path]:
    return sorted(p for p in _PKG_DIR.glob("*.py"))


def test_registered_names_tuple_is_absent_from_source() -> None:
    offenders = [
        p.name for p in _pkg_sources() if "_REGISTERED_NAMES" in p.read_text()
    ]
    assert not offenders, (
        f"the killed `_REGISTERED_NAMES` tuple reappeared in {offenders}; the "
        f"registered set must come from _engine.all_specs()"
    )


def test_registry_module_exposes_no_name_tuple() -> None:
    assert not hasattr(registry, "_REGISTERED_NAMES"), (
        "registry._REGISTERED_NAMES must not exist (names derive from all_specs())"
    )


def test_name_set_is_the_compiled_all_specs_set() -> None:
    shim_names = {s.name for s in all_specs()}
    engine_names = {s.name for s in _engine.all_specs()}
    assert shim_names == engine_names, (
        f"shim name set {sorted(shim_names)} must equal the compiled all_specs() "
        f"set {sorted(engine_names)}"
    )


def test_filename_first_detector_is_gone() -> None:
    """The filename-beats-shape order is killed: no `_filename_match` dispatch
    helper survives in the shim (compat delegates to the unified detector)."""
    offenders = [
        p.name for p in _pkg_sources() if "_filename_match" in p.read_text()
    ]
    assert not offenders, (
        f"the killed filename-first `_filename_match` helper reappeared in "
        f"{offenders}; detection is marker/stamp-first with no filename dispatch"
    )
