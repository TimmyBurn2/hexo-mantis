"""The identity primitive's VALUE survives its promotion out of `diagnostics/` (R321(d)).

A determinism gate that can change value by being moved is not a gate. R317(c)(i) compares
net hashes across control drives and across ladder rungs; if relocating the implementation
could shift the digest, every comparison spanning the move would read DIVERGED for a reason
that has nothing to do with the nets. The literal below was measured on the PRE-MOVE
implementation at `worker_sweep._net_param_hash` and committed in the same act as the move,
so the pin is a before/after measurement rather than a transcription of the new behaviour.

The config it was measured at has since been deleted (R346(f)); see `_GOLDEN_SEED` below for
how the measured net is rebuilt without re-measuring the golden.
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
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"

#: THE SEED THE MEASUREMENT WAS TAKEN AT, restated because the file it was taken from is gone.
#: `build_sweep_net` seeds from `config.seed` and then builds `build_net(arch)`, so the digest is
#: a function of the seed and the arch alone. The pre-move measurement was taken at
#: `configs/smoke_gnn.yaml`, which minted `seed: 20260719` on the same `gnn_axis_v1` identity;
#: R346(f) deleted that file, and `smoke_preflight_armed.yaml` carries the identical identity at
#: a different seed. Overriding the seed IN MEMORY rebuilds the net the golden denominates —
#: verified: the literal below reproduces exactly. Re-measuring at the new file's own seed would
#: have replaced a before/after measurement with a transcription of current behaviour, which is
#: the one thing this file's docstring says the pin must not become.
_GOLDEN_SEED = 20260719

#: Measured on the PRE-MOVE `worker_sweep._net_param_hash`, twice, before the promotion landed.
_GOLDEN_PRE_MOVE = "1ab0f3cb5cd76a39bb95c4648ce1966242b5fb4bfa2294dfc1901b9509682787"


def _built_net() -> torch.nn.Module:
    raw = load_config(_CONFIG).model_dump()
    raw["seed"] = _GOLDEN_SEED
    config = load_config(_CONFIG).__class__.model_validate(raw)
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
