"""apply_gate_decision — the EXACTLY-ONE gate-decision call site (deploy seam).

DEPLOY-TAG SEAM (WP-UNFREEZE, R49): gate decisions move ONLY the deploy tag (the
resolved anchor + best_model.pt). Actor weights sync continuously in
mantis.train.actor_sync — this module must never name, hold, or call an actor-side
surface, and `DeployTagHooks` carries no attribute through which one could be reached
(field set pinned by tests/train/test_actor_sync_isolation.py).

Sequence (F-12/LAW-12): load the EVALUATED candidate snapshot (the exact bytes the
worker played — never the live trainer module) into the resolved anchor's best-model
slot via the injected guarded loader -> `save_anchor(...)` -> update the resolved
anchor's recorded step -> return the promoted step. NO read of the resolved anchor's
recorded state anywhere in this module except to WRITE the post-decision update (no
actor-weight proxy reads).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeployTagHooks:
    """Constructed by the composition root (train side): the deploy-side collaborators,
    and nothing else — there is deliberately no actor-shaped field here (R49)."""

    anchor_state: Any                # train.anchor.AnchorState (anchor.py:75-85)
    best_model_path: Path
    run_id: str
    encoding: str
    save_anchor: Callable[..., None]        # injected train.anchor.save_best_model_atomic
    guarded_load: Callable[[Any, dict], None]  # injected train.anchor._guarded_load_state_dict


def apply_gate_decision(hooks: DeployTagHooks, result: Mapping[str, Any]) -> int | None:
    """No-op (`None`) unless `result["promoted"] is True` and the round was not broken.

    A gate pass advances the deploy tag and ONLY the deploy tag; the actor's weights
    are none of this function's business (WP-UNFREEZE, R49).
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
    resolved_anchor.best_model_step = step
    return step


__all__ = ["DeployTagHooks", "apply_gate_decision"]
