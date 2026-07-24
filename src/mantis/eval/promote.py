"""apply_gate_decision — the EXACTLY-ONE gate-decision call site (design §a.3/§c.5).

Sequence (run3 parity, eval_drain.py:78-114): load the EVALUATED candidate snapshot (the
exact bytes the worker played — F-12/LAW-12; never the live trainer module) into the
resolved anchor's best-model slot via the injected guarded loader -> `save_anchor(...)` ->
iff `sync_inference`: `promotion_target.sync_inference_weights(sd)` +
`.update_checkpoint_step(step)` -> update the resolved anchor's recorded step -> return the
promoted step. NO read of the resolved anchor's recorded state anywhere in this module
except to WRITE the post-decision update (no actor-weight proxy reads).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PromotionHooks:
    """Constructed by the composition root (train side)."""

    promotion_target: Any            # the pool — PromotionTarget (pool_hooks.py:64-76)
    anchor_state: Any                # train.anchor.AnchorState (anchor.py:75-85)
    best_model_path: Path
    run_id: str
    encoding: str
    save_anchor: Callable[..., None]        # injected train.anchor.save_best_model_atomic
    guarded_load: Callable[[Any, dict], None]  # injected train.anchor._guarded_load_state_dict


def apply_gate_decision(
    hooks: PromotionHooks, result: Mapping[str, Any], *, sync_inference: bool
) -> "int | None":
    """No-op (`None`) unless `result["promoted"] is True` and the round was not broken.

    # WP-UNFREEZE seam: PromotionTarget (pool_hooks.py:64-76) will split into
    # ActorSyncTarget/DeployTag; this is the single gate-decision call site — do not add
    # another.
    """
    if not result.get("promoted") or result.get("eval_broken"):
        return None

    from mantis.eval.snapshot import load_model_snapshot

    step = int(result["step"])
    snapshot_path = result.get("candidate_snapshot_path")
    loaded = load_model_snapshot(snapshot_path, device="cpu") if snapshot_path else {}
    # `load_model_snapshot` returns a built nn.Module in production (snapshot.py); the
    # gate-parity oracle monkeypatches it to hand back a bare state_dict directly — both
    # shapes are handled here without a live-module state read.
    state_dict = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded

    resolved_anchor = hooks.anchor_state
    hooks.guarded_load(resolved_anchor.best_model, state_dict)
    hooks.save_anchor(
        resolved_anchor.best_model, hooks.best_model_path,
        step=step, run_id=hooks.run_id, encoding=hooks.encoding,
    )
    if sync_inference:
        hooks.promotion_target.sync_inference_weights(state_dict)
        hooks.promotion_target.update_checkpoint_step(step)
    resolved_anchor.best_model_step = step
    return step


__all__ = ["PromotionHooks", "apply_gate_decision"]
