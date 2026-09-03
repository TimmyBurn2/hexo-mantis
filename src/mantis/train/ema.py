"""Exponential moving average of model weights (WP10 §a.4 PORT; old training/ema.py).

The EMA model is updated every ``update_every`` optimizer steps via a decayed running
mean of the trainer's raw parameters. Self-play inference / eval / best-model promotion
read EMA weights when EMA is enabled; the trainer's raw weights keep driving the next
gradient step. Anti-colony lever (kept — context-law run-safety smoothing, not a
falsified lever). Behaviour-exact; the only change is the docstring path references.

Hand-rolled state_dict-level EMA (not `torch.optim.swa_utils.AveragedModel`, which
deep-copies the model on construction — the net carries a PyO3 spec that has no
deep-copy protocol). A flat ``name -> tensor`` shadow keyed off ``state_dict()`` rides
all parameters + buffers (`use_buffers=True` semantics), updated in place.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

DEFAULT_DECAY = 0.999
DEFAULT_UPDATE_EVERY = 10

#: The members `train.ema` must carry. Read by key; absent is an error (R1).
_EMA_MEMBERS: tuple[str, ...] = ("enabled", "decay", "update_every")


class MissingEmaConfigError(ValueError):
    """`train.ema` is absent or incomplete (AUDIT-1 F-06 / R332(d))."""


def _base_of(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a `torch.compile` OptimizedModule (`_orig_mod`) if present so the EMA
    shadow is keyed off the raw module's names."""
    return getattr(model, "_orig_mod", model)


class EmaModel:
    """State-dict-level EMA of a wrapped model's parameters and buffers.

    Floating-point entries mix via ``avg = avg + (1 - decay) * (cur - avg)``;
    non-floating entries (int buffers, e.g. ``num_batches_tracked``) are copied
    verbatim so the EMA state stays `load_state_dict`-compatible.
    """

    def __init__(self, model: torch.nn.Module, decay: float = DEFAULT_DECAY) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be in [0, 1); got {decay}")
        self.decay: float = float(decay)
        base = _base_of(model)
        self._shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in base.state_dict().items()
        }

    def update_parameters(self, model: torch.nn.Module) -> None:
        """Apply one EMA mixing step from `model`'s current weights."""
        base = _base_of(model)
        with torch.no_grad():
            for name, cur in base.state_dict().items():
                shadow = self._shadow.get(name)
                if shadow is None:
                    self._shadow[name] = cur.detach().clone()  # arch change mid-run — re-seed
                    continue
                if cur.dtype.is_floating_point:
                    shadow.mul_(self.decay).add_(cur.detach(), alpha=1.0 - self.decay)
                else:
                    shadow.copy_(cur.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Shallow-copied view of the shadow state (tensors are the EMA's own storage
        — callers that mutate must clone first)."""
        return dict(self._shadow)

    @property
    def module(self) -> _EmaModuleView:
        """A module-like proxy over the shadow (for `state_dict`/`parameters` call
        sites); NOT a real `nn.Module` — it has no `forward`."""
        return _EmaModuleView(self)


class _EmaModuleView:
    """Module-like proxy exposing the EMA shadow via `state_dict` / `parameters`."""

    def __init__(self, owner: EmaModel) -> None:
        self._owner = owner

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._owner.state_dict()

    def parameters(self) -> Iterable[torch.Tensor]:
        for t in self._owner._shadow.values():
            if t.dtype.is_floating_point:
                yield t


def build_ema_model(model: torch.nn.Module, decay: float = DEFAULT_DECAY) -> EmaModel:
    """Construct an EMA wrapper (fresh shadow storage) around `model`."""
    return EmaModel(model, decay=decay)


def resolve_ema_config(config: Mapping[str, Any]) -> tuple[bool, float, int]:
    """Read the EMA lever's arming block from `train.ema`. Returns
    `(enabled, decay, update_every)`.

    AUDIT-1 F-06 / R332(d). THIS FUNCTION USED TO READ FOUR KEYS THAT DO NOT EXIST:
    `config.get("ema")`, `("ema_enabled", False)`, `("ema_decay", 0.999)` and
    `("ema_update_every", 10)`, against a `RunConfig` that is `extra="forbid"` and had no `ema`
    leaf anywhere. So the lever this module's docstring calls an "anti-colony lever (kept)" was
    OFF on every run, no config could turn it on, and nothing said so — a disabled lever and an
    absent one produce identical runs. `train.ema` is now a REQUIRED schema block and this
    reads it BY KEY: absent is an error, never a default (R1/LAW-08).

    Raises:
        MissingEmaConfigError: the config carries no `train.ema` block, or it is not a mapping,
            or a member is absent. A config that reaches here in that state did not come
            through `load_config`, and there is no code-side default to stand in for it.
    """
    train = config.get("train") if isinstance(config, Mapping) else None
    block = train.get("ema") if isinstance(train, Mapping) else None
    if block is None:
        raise MissingEmaConfigError(
            "train.ema is absent. It is a REQUIRED schema block (R332(d)); absence used to "
            "resolve to a code-side `False`, which is how the EMA lever stayed off on every "
            "run while reading as 'kept'. State the posture in the config."
        )
    if not isinstance(block, Mapping):
        raise MissingEmaConfigError(
            f"train.ema is {type(block).__name__}, expected a mapping with "
            f"{sorted(_EMA_MEMBERS)}."
        )
    missing = [m for m in _EMA_MEMBERS if m not in block]
    if missing:
        raise MissingEmaConfigError(
            f"train.ema is missing {missing}. Every member is REQUIRED by the schema, so a "
            "config reaching here without them did not come through the one loader."
        )
    update_every = int(block["update_every"])
    if update_every < 1:
        raise ValueError(f"ema.update_every must be >= 1; got {update_every}")
    return bool(block["enabled"]), float(block["decay"]), update_every
