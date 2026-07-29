"""WPBRIDGE Phase T — TD-4 / CARD-POOL-ENCODING-BRIDGE at its own seam, plus the
mutation self-test that proves this suite detects the census'd defect (LAW-07).

The subject is the exact call `WorkerPool.__init__` makes (`selfplay/pool.py:97`) on the
exact input a real boot hands it: `RunConfig.model_dump()`. TD-4 was measured at HEAD
(`ca237d2`) as parent rc 33 / child rc 1 in mode PREFLIGHT, ~1.4 s in, with a real Trainer
already built — the wall that made `preflight_mint.py --config configs/run5.yaml` unable to
run a burst at all.

The mutation self-test re-introduces the defect at the one line that carried it (the
resolver's flat-only read) and asserts this suite goes RED there and ONLY there — the
R81/R86 condition: not self-satisfying, no unrelated casualty.
"""
from __future__ import annotations

from typing import Any, Mapping

import pytest

from mantis.config import load_config
from mantis.encoding.resolvers import MissingEncodingError
from mantis.selfplay import hparams as hparams_mod
from mantis.selfplay.hparams import resolve_pool_encoding


def _run5_dump() -> dict[str, Any]:
    """The real production config, through the real loader — no hand-built stand-in.
    R64 posture: the oracle resolves what run5 resolves."""
    return load_config("configs/run5.yaml").model_dump()


# ── the seam TD-4 named ─────────────────────────────────────────────────────────────────


def test_pool_resolves_encoding_from_a_real_run_config_dump() -> None:
    """THE TD-4 oracle. RED at HEAD with `MissingEncodingError`."""
    resolved = resolve_pool_encoding(_run5_dump(), arch=None)
    assert resolved.encoding_name == "gnn_axis_v1"
    assert resolved.registry_spec.representation == "graph"
    assert resolved.board_size > 0
    assert resolved.trunk_size > 0


def test_pool_resolution_agrees_with_the_config_identity_key() -> None:
    """One authority: what the pool resolves IS what the operator declared. A bridge that
    guessed — or that defaulted — would pass the test above and fail this one."""
    dump = _run5_dump()
    assert resolve_pool_encoding(dump, arch=None).encoding_name == dump["identity"]["encoding"]


def test_pool_still_refuses_a_config_that_declares_no_encoding() -> None:
    """LAW-11 survives the widening: strip the declaration and the pool dies, loudly."""
    dump = _run5_dump()
    dump.pop("identity")
    with pytest.raises(MissingEncodingError):
        resolve_pool_encoding(dump, arch=None)


# ── mutation self-test (LAW-07): does this suite actually detect the defect? ─────────────


def _flat_only_resolve(cfg: Mapping[str, Any] | None) -> Any:
    """The pre-fix `resolve_from_config`, verbatim in behaviour: reads the flat `encoding`
    key and nothing else. This IS the census'd defect."""
    if cfg is None:
        raise MissingEncodingError("resolve_from_config(None)")
    section = cfg.get("encoding")
    if section is None:
        raise MissingEncodingError(
            "config has no 'encoding' key; an explicit `encoding: <name>` "
            "declaration is required (LAW-11, R28)"
        )
    from mantis.encoding.resolvers import lookup

    return lookup(section) if isinstance(section, str) else lookup(section["version"])


def test_mutation_reintroducing_the_defect_reds_the_td4_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flip-set: re-pointing the pool's resolver at the flat-only implementation must make
    the TD-4 oracle fail with the census'd error — the same `MissingEncodingError`, on the
    same input. An oracle that stayed green under this mutation would be pinning nothing."""
    monkeypatch.setattr(hparams_mod, "resolve_from_config", _flat_only_resolve)
    with pytest.raises(MissingEncodingError, match="no 'encoding' key"):
        resolve_pool_encoding(_run5_dump(), arch=None)


def test_mutation_leaves_the_flat_shape_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unrelated casualty (R86 'alone'): the mutation is confined to the nested shape.
    A legacy flat config resolves identically before and after it, so the oracle above is
    detecting the bridge specifically and not a broken resolver in general."""
    flat = {"encoding": "v6", "selfplay": {}, "mcts": {}}
    before = resolve_pool_encoding(flat, arch=None).encoding_name
    monkeypatch.setattr(hparams_mod, "resolve_from_config", _flat_only_resolve)
    assert resolve_pool_encoding(flat, arch=None).encoding_name == before == "v6"
