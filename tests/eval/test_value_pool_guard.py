"""The eval decode refuses a value_pool it does not implement.

`value_pool` had ZERO Python consumers. `LocalInferenceEngine.infer_batch` hardcodes the
cluster reduction (`v = float(board_values.min())`) and the graph arm reduces nothing, so a
registry row later declaring `value_pool="mean"` would pass every existing check and be
SILENTLY min-pooled. This is the value-channel half of the same class named on the policy
channel, and these oracles are what make the silent case loud.

Each assertion below names the producer it fires against. The guard IS the
live consumer that `value_pool` previously lacked, so this file is also that citation.
"""
from __future__ import annotations

import dataclasses

import pytest

from mantis.encoding import lookup
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.eval.worker import _assert_decode_implements_declared_pooling


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


@pytest.mark.parametrize("pool", ["mean", "max"])
def test_unimplemented_value_pool_is_refused_by_name(pool: str) -> None:
    """THE MUTATION THIS GUARD EXISTS FOR. `mean` and `max` are both registry-legal
    (`spec/mod.rs:83` parses none|min|max|mean) and both would be silently min-pooled.

    Note `max` is the sharper case: it is legal in the registry AND semantically opposite
    to the hardcoded `.min()`, so the silent version reports the BEST window as the board
    value where the encoding declared the worst.
    """
    spec = _spec_with("gnn_axis_v1", value_pool=pool)
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        _assert_decode_implements_declared_pooling(spec)
    message = str(excinfo.value)
    assert f"value_pool={pool!r}" in message, "the raise must name the offending value"
    assert "min" in message, "the raise must name what the decode actually does instead"


def test_value_guard_fires_even_when_the_policy_pool_is_fine() -> None:
    """Independence: the value check is not shadowed by the policy check passing."""
    spec = _spec_with("gnn_axis_v1", value_pool="mean")  # policy_pool='none' — implemented
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        _assert_decode_implements_declared_pooling(spec)
    assert "value_pool" in str(excinfo.value)
