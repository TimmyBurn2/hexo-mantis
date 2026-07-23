"""Pretrain checkpoint validation (WP10 §a.7 PORT of `bootstrap/pretrain_validate.py`).

`validate` — verify a pretrain checkpoint round-trips and runs a forward pass at the right policy
width. Rehomed onto `build_net(arch_from_spec_and_config(...))` (WP9 construction authority) — no
kwargs ctor, no shape-inference.

Two ratified reductions from the old validator (both killed/deferred, not reachable numeric paths):
  * the v8 / pma_global / gpool_bias_active skip branches are GONE — those encodings/pools are
    KILLED (v8 never crosses; F-04/F-05 falsified) so the special-case skips have no subject;
  * the play-100-greedy-vs-RandomBot smoke DEFERS — `mantis.bots.RandomBot` is not yet migrated
    (a separate WP). The round-trip load + forward-pass shape check (the checkpoint-format
    correctness guard) ports here; the win-rate smoke re-enters when the bot surface lands.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from mantis.encoding import lookup as _lookup_encoding
from mantis.model import arch_from_spec_and_config, build_net

_LOG = logging.getLogger(__name__)


def _config_encoding(cfg: dict) -> str:
    """Resolve the encoding name from a saved checkpoint config — the WP8 nested
    `identity.encoding`, a legacy `{version: …}` dict, or a flat string; default v6."""
    ident = cfg.get("identity")
    if isinstance(ident, dict) and isinstance(ident.get("encoding"), str):
        return ident["encoding"]
    enc = cfg.get("encoding")
    if isinstance(enc, str):
        return enc
    if isinstance(enc, dict):
        return str(enc.get("version", "v6"))
    return "v6"


def validate(ckpt_path: Path, device: torch.device) -> None:
    """Verify a pretrain checkpoint round-trips and runs a forward pass at the registry policy
    width. Rebuilds the net via the declared arch (never shape-inference); asserts the policy
    output shape matches `spec.policy_logit_count`."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt and "config" in ckpt):
        raise AssertionError(
            f"pretrain checkpoint missing model_state/config; got "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}"
        )
    cfg = ckpt["config"]
    encoding = _config_encoding(cfg)
    spec = _lookup_encoding(encoding)

    arch = arch_from_spec_and_config(spec, cfg if isinstance(cfg, dict) else {})
    model = build_net(arch)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    in_channels = int(spec.n_planes)
    board_size = int(spec.trunk_size)
    n_actions = int(spec.policy_logit_count)
    dummy = torch.zeros(1, in_channels, board_size, board_size, device=device)
    with torch.no_grad():
        out = model(dummy.float())
    log_pol = out[0]
    if tuple(log_pol.shape) != (1, n_actions):
        raise AssertionError(
            f"unexpected policy shape: {tuple(log_pol.shape)} (expected (1, {n_actions}))"
        )
    _LOG.info("checkpoint_forward_pass_ok encoding=%s policy_width=%s", encoding, n_actions)
