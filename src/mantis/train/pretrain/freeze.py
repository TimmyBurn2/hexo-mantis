"""Fine-tune freeze helper (WP10 §a.7 PORT of `bootstrap/pretrain_freeze.py`).

`_apply_finetune_freeze` — the §171 A4 staged-pretrain freeze pattern. Sets `requires_grad` on
`trunk.input_conv` / `trunk.input_gn` and per-block `trunk.tower` entries. Used by the pretrain
CLI when `--freeze-trunk-entry` or `--unfreeze-blocks` is passed. Behaviour-exact relocation.
"""
from __future__ import annotations

from typing import Dict, Optional


def _apply_finetune_freeze(
    base_model,
    *,
    freeze_trunk_entry: bool,
    unfreeze_blocks: Optional[set],
) -> Dict[str, int]:
    """Apply the §171 A4 fine-tune freeze pattern.

    - `freeze_trunk_entry=True`: `requires_grad=False` on `trunk.input_conv` + `trunk.input_gn`.
    - `unfreeze_blocks={i,...}`: `requires_grad=False` on every `trunk.tower[k]` where k not in
      the set. Heads (policy / opp_reply / value) are left at their construction default (never
      frozen here). Returns counts for logging.
    """
    trunk = base_model.trunk
    tower = trunk.tower

    if freeze_trunk_entry:
        for p in trunk.input_conv.parameters():
            p.requires_grad = False
        for p in trunk.input_gn.parameters():
            p.requires_grad = False

    if unfreeze_blocks is not None:
        n_blocks = len(tower)
        for idx in unfreeze_blocks:
            if not (0 <= idx < n_blocks):
                raise ValueError(
                    f"--unfreeze-blocks entry {idx} out of [0, {n_blocks}); "
                    f"trunk has {n_blocks} blocks"
                )
        for i, block in enumerate(tower):
            keep = i in unfreeze_blocks
            for p in block.parameters():
                p.requires_grad = keep

    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    return {
        "freeze_trunk_entry": int(bool(freeze_trunk_entry)),
        "unfreeze_blocks": sorted(unfreeze_blocks) if unfreeze_blocks else [],
        "total_params": int(total),
        "trainable_params": int(trainable),
    }
