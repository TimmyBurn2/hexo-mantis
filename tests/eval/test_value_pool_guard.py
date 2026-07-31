"""ADJ-WP12R-6 producer: the eval decode refuses a value_pool it does not implement.

`value_pool` had ZERO Python consumers. `LocalInferenceEngine.infer_batch` hardcodes the
cluster reduction (`v = float(board_values.min())`) and the graph arm reduces nothing, so a
registry row later declaring `value_pool="mean"` would pass every existing check and be
SILENTLY min-pooled. This is the value-channel half of the class R138 named on the policy
channel, and these oracles are what make the silent case loud.

LAW-07: each assertion below names the producer it fires against. LAW-08: the guard IS the
live consumer that `value_pool` previously lacked, so this file is also that citation.
"""
from __future__ import annotations

import dataclasses

import pytest

from mantis.encoding import lookup
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.eval.worker import (
    _DECODE_IMPLEMENTED_VALUE_POOLS,
    _assert_decode_implements_declared_pooling,
)


def _spec_with(name: str, **overrides):
    """A registered spec with fields replaced — the ONLY way to reach an unimplemented
    value_pool, because no registered row declares one (that is the point: the defect is
    latent until a new row lands, so the oracle must synthesise the future row)."""
    spec = lookup(name)
    if dataclasses.is_dataclass(spec):
        return dataclasses.replace(spec, **overrides)

    class _Shim:
        def __init__(self, inner, over):
            self._inner, self._over = inner, over

        def __getattr__(self, item):
            if item in self._over:
                return self._over[item]
            return getattr(self._inner, item)

    return _Shim(spec, overrides)


def test_every_registered_encoding_declares_an_implemented_value_pool() -> None:
    """The census: all four registered rows pass the guard today, so arming it changes
    NOTHING for any shipped encoding. A guard that refused a live encoding would be a
    regression, not a fix — this is the assertion that distinguishes the two."""
    for name in ("v6", "v6w25", "v6_live2_ls", "gnn_axis_v1"):
        spec = lookup(name)
        assert spec.value_pool in _DECODE_IMPLEMENTED_VALUE_POOLS, (
            f"{name} declares value_pool={spec.value_pool!r}, outside the implemented set"
        )


def test_run5_encoding_passes_the_value_channel_guard() -> None:
    """run5 mints `gnn_axis_v1`. Its round must not be refused by the new guard."""
    _assert_decode_implements_declared_pooling(lookup("gnn_axis_v1"))


@pytest.mark.parametrize("pool", ["mean", "max"])
def test_unimplemented_value_pool_is_refused_by_name(pool: str) -> None:
    """THE MUTATION THIS GUARD EXISTS FOR. `mean` and `max` are both registry-legal
    (`spec/mod.rs:83` parses none|min|max|mean) and both would be silently min-pooled.

    Note `max` is the sharper case: it is legal in the registry AND semantically opposite
    to the hardcoded `.min()`, so the silent version reports the BEST window as the board
    value where the encoding declared the worst.
    """
    spec = _spec_with("v6w25", value_pool=pool)
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        _assert_decode_implements_declared_pooling(spec)
    message = str(excinfo.value)
    assert f"value_pool={pool!r}" in message, "the raise must name the offending value"
    assert "min" in message, "the raise must name what the decode actually does instead"


def test_policy_channel_refusal_is_unchanged_by_the_new_guard() -> None:
    """REGRESSION GUARD on the shipped ordering. `v6_live2_ls` declares an unimplemented
    policy_pool and an IMPLEMENTED value_pool ('min'), so it must still fail with the
    POLICY message — the message `probe_quadrants.py` and the shipped oracles pin.
    """
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        _assert_decode_implements_declared_pooling(lookup("v6_live2_ls"))
    assert "policy_pool='legal_set_scatter_max'" in str(excinfo.value)


def test_value_guard_fires_even_when_the_policy_pool_is_fine() -> None:
    """Independence: the value check is not shadowed by the policy check passing."""
    spec = _spec_with("v6", value_pool="mean")  # policy_pool='none' — implemented
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        _assert_decode_implements_declared_pooling(spec)
    assert "value_pool" in str(excinfo.value)
