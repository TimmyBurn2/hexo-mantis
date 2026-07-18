"""inv17 (O2): the legacy `EncodingSpec` pyclass is RETIRED; `RegistrySpec` is
the single spec type. `RegistrySpec.from_registry(name).name == name` for every
registered encoding.
"""
import pytest

from mantis import _engine

REGISTERED = ["v6", "v6w25", "v6_live2_ls", "gnn_axis_v1"]


def test_encoding_spec_absent():
    assert not hasattr(_engine, "EncodingSpec")
    with pytest.raises(AttributeError):
        _ = _engine.EncodingSpec  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", REGISTERED)
def test_from_registry_name_round_trips(name):
    assert _engine.RegistrySpec.from_registry(name).name == name


def test_from_registry_unknown_raises():
    with pytest.raises(ValueError):
        _engine.RegistrySpec.from_registry("__not_a_real_encoding__")
