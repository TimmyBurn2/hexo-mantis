"""The identity primitive's VALUE survives its promotion out of `diagnostics/` (R321(d)).

A determinism gate that can change value by being moved is not a gate. R317(c)(i) compares
net hashes across control drives and across ladder rungs; if relocating the implementation
could shift the digest, every comparison spanning the move would read DIVERGED for a reason
that has nothing to do with the nets. The literal below was measured on the PRE-MOVE
implementation at `worker_sweep._net_param_hash` and committed in the same act as the move,
so the pin is a before/after measurement rather than a transcription of the new behaviour.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mantis.config.loader import load_config
from mantis.diagnostics import worker_sweep as ws
from mantis.model import net_param_hash
from mantis.model.arch import arch_from_spec_and_config
from mantis.selfplay.hparams import resolve_pool_encoding

_REPO = Path(__file__).resolve().parents[2]

#: A committed GRAPH config, read through the real loader — the same one the R81 determinism
#: oracle uses, so both tests denominate the same net.
_CONFIG = _REPO / "configs" / "smoke_gnn.yaml"

#: Measured on the PRE-MOVE `worker_sweep._net_param_hash` at `configs/smoke_gnn.yaml`
#: (`seed: 20260719`), twice, before the promotion landed.
_GOLDEN_PRE_MOVE = "1ab0f3cb5cd76a39bb95c4648ce1966242b5fb4bfa2294dfc1901b9509682787"


def _built_net() -> torch.nn.Module:
    config = load_config(_CONFIG)
    raw = config.model_dump()
    resolved = resolve_pool_encoding(raw, arch=None)
    arch = arch_from_spec_and_config(resolved.registry_spec, raw)
    return ws.build_sweep_net(config, arch, torch.device("cpu"))


def test_the_promoted_hash_reproduces_the_PRE_MOVE_VALUE() -> None:
    """The golden. A relocation that changed the digest would silently void R317(c)(i)."""
    assert net_param_hash(_built_net()) == _GOLDEN_PRE_MOVE, (
        "the parameter hash changed when the implementation moved out of diagnostics/ — "
        "R317(c)(i)'s gate compares hashes across drives and rungs, so a digest that depends "
        "on where the function lives makes every cross-move comparison read DIVERGED"
    )


def test_worker_sweep_calls_the_ONE_canonical_implementation() -> None:
    """R321(d) asks for one implementation with callers re-pointed, not a second copy."""
    assert ws.net_param_hash is net_param_hash
    assert not hasattr(ws, "_net_param_hash"), (
        "the pre-move private definition is still present; a second implementation is exactly "
        "what the promotion exists to remove"
    )


def test_the_hash_is_STABLE_across_two_builds_of_the_same_config() -> None:
    """Negative control for the golden: a digest that varied per build would match the literal
    only by luck, and this test is what tells those two cases apart."""
    assert net_param_hash(_built_net()) == net_param_hash(_built_net())
