"""SC-A4 oracle — radius shape (ii): NO config field, registry alone is the radius
authority (DESIGN_P2.md §5 / PREREG_P2.md suite #7, REV1 MUST-FIX #1).

Mixed suite (not uniformly RED-at-import): the two `import mantis.config[.resolve]`
regression-pin tests are GREEN at HEAD today (nothing is broken yet) and MUST STAY green
through SC-A4's `__init__.py` edits — the producer test for those edits (LAW-07): if either
package's `__init__.py` still re-exports a deleted radius symbol after SC-A4, THIS is where
it `ImportError`s, instead of silently at some unrelated call site. The remaining tests
assert the POST-SC-A4 absence of `RadiusStage` / `legal_move_radius_schedule` /
`mantis.config.resolve.radius` and are RED at HEAD today (all three still exist).
"""
from __future__ import annotations

import importlib

import pytest

_RADIUS_SYMBOLS = (
    "OfflineRadiusUnresolvableError",
    "RadiusStage",
    "require_offline_radius",
    "resolve_radius_from_schedule",
)


def test_radius_stage_not_importable_from_schema():
    with pytest.raises(ImportError):
        from mantis.config.schema import RadiusStage  # noqa: F401


def test_selfplay_config_has_no_radius_field():
    from mantis.config.schema import SelfplayConfig

    leaves = set(SelfplayConfig.model_fields)
    assert "legal_move_radius_schedule" not in leaves
    assert "legal_move_radius" not in leaves


def test_resolve_radius_module_does_not_exist():
    with pytest.raises(ImportError):
        importlib.import_module("mantis.config.resolve.radius")


def test_import_mantis_config_succeeds_cleanly():
    # regression pin: `import mantis.config` (triggered transitively by nearly every test
    # and by tools/mint_config.py, checkpoints.py, eval/ladder.py, config/loader.py,
    # config/emit.py) must not ImportError once RadiusStage is deleted from schema.py.
    importlib.import_module("mantis.config")


def test_import_mantis_config_resolve_succeeds_cleanly():
    importlib.import_module("mantis.config.resolve")


def test_radius_symbols_absent_from_mantis_config_all():
    import mantis.config as config_pkg

    for name in _RADIUS_SYMBOLS:
        assert name not in config_pkg.__all__, f"{name} still in mantis.config.__all__"
        # exec() replicates the real `from X import Y` bytecode (IMPORT_FROM), which
        # raises ImportError when Y is neither an attribute nor a submodule of X — a
        # plain getattr() would raise the wrong exception type (AttributeError).
        with pytest.raises(ImportError):
            exec(f"from mantis.config import {name}")  # noqa: S102


def test_radius_symbols_absent_from_mantis_config_resolve_all():
    import mantis.config.resolve as resolve_pkg

    for name in _RADIUS_SYMBOLS:
        assert name not in resolve_pkg.__all__, (
            f"{name} still in mantis.config.resolve.__all__"
        )
        with pytest.raises(ImportError):
            exec(f"from mantis.config.resolve import {name}")  # noqa: S102
