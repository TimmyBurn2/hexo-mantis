"""INV22 — `mantis.encoding.EncodingSpec` IS `_engine.RegistrySpec` (type alias).

Pins:
  1. `mantis.encoding.EncodingSpec is _engine.RegistrySpec` — alias byte-identity
     (no parallel Python @dataclass mirror).
  2. Every registered encoding exposes the 19 schema fields + 6 derived accessors
     as Python attributes (not method calls), values mirroring the Rust spec.
  3. Consumer-side: import via `mantis.encoding`, construct via `from_registry`,
     field reads identical to a direct `_engine.RegistrySpec.from_registry` read.

The registered-name set is DERIVED from `all_specs()` (no hardcoded tuple).
"""
from __future__ import annotations

import pytest

from mantis import _engine
from mantis.encoding import EncodingSpec, all_specs

# Registered names DERIVED from all_specs() — no hand-synced tuple (the killed
# `_REGISTERED_NAMES`). Parametrising over the live set surfaces accessor gaps.
_REGISTERED_NAMES: tuple[str, ...] = tuple(sorted(s.name for s in all_specs()))


_REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "board_size",
    "trunk_size",
    "cluster_window_size",
    "cluster_threshold",
    "legal_move_radius",
    "n_planes",
    "plane_layout",
    "policy_logit_count",
    "has_pass_slot",
    "is_multi_window",
    "value_pool",
    "policy_pool",
    "sym_table_id",
    "schema_version",
    "notes",
    "kept_plane_indices",
    "n_source_planes",
    "k_max",
)
_REQUIRED_DERIVED: tuple[str, ...] = (
    "n_actions",
    "n_cells",
    "state_stride",
    "chain_stride",
    "aux_stride",
    "policy_stride",
)


def test_registered_set_is_nonempty_and_derived() -> None:
    assert _REGISTERED_NAMES, "all_specs() must expose at least one encoding"
    # Byte-parity with the compiled binding.
    assert _REGISTERED_NAMES == tuple(sorted(s.name for s in _engine.all_specs()))


def test_inv22_encoding_spec_is_engine_registry_spec_alias() -> None:
    assert EncodingSpec is _engine.RegistrySpec, (
        f"mantis.encoding.EncodingSpec must be the _engine.RegistrySpec type "
        f"alias; got {EncodingSpec!r} vs {_engine.RegistrySpec!r}"
    )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_inv22_required_attribute_surface(name: str) -> None:
    spec = _engine.RegistrySpec.from_registry(name)
    for field in _REQUIRED_FIELDS:
        assert hasattr(spec, field), f"{name}: missing schema field {field!r}"
        value = getattr(spec, field)
        assert not callable(value), (
            f"{name}.{field} is callable; must be a field/property, not a method"
        )
    for derived in _REQUIRED_DERIVED:
        assert hasattr(spec, derived), f"{name}: missing derived accessor {derived!r}"
        value = getattr(spec, derived)
        assert not callable(value), (
            f"{name}.{derived} is callable; must be a property, not a method"
        )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_inv22_alias_read_equals_direct_read(name: str) -> None:
    direct = _engine.RegistrySpec.from_registry(name)
    via_alias = EncodingSpec.from_registry(name)
    for field in _REQUIRED_FIELDS + _REQUIRED_DERIVED:
        d_val = getattr(direct, field)
        a_val = getattr(via_alias, field)
        if isinstance(d_val, list):
            d_val = tuple(d_val)
            a_val = tuple(a_val)
        assert d_val == a_val, (
            f"{name}.{field}: alias-read {a_val!r} != direct-read {d_val!r}"
        )
