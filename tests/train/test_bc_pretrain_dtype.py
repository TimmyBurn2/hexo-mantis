"""AUDIT-1 F-30 — BC pretrain runs at the DECLARED autocast dtype, on both arms.

THE FINDING, AND THE HALF OF IT THAT IS FALSE AT CONTACT (REPAIR-2 §0 verifies before it
repairs, and this row is why the rule exists):

* **GRID — TRUE, repaired, and then DELETED.** `pretrain/trainer.py::BootstrapTrainer` read
  `config.get("fp16", True)` — a code-side default on a key the schema REQUIRED — and autocast
  at a LITERAL `torch.float16`, while `amp_dtype_for` is the ONE dtype authority (LAW-06). The
  repair routed both through `training_terms`; R346(f) then deleted the module, the key and the
  second dtype spelling outright, so the arm has no subject left to pin.
* **GRAPH — FALSE.** The finding reads *"`pretrain/graph_route.py` uses no autocast at all"*,
  and the module indeed contains none — but the step it drives is
  `Trainer.train_step_from_graph_batch`, which autocasts at `self.amp_dtype`, resolved by
  `amp_dtype_for` in `Trainer.__init__`. `run_graph_pretrain` builds the REAL `Trainer`
  (`graph_route.py:285`), not `BootstrapTrainer`. So the graph BC route was already running
  at the authority's dtype; the absence of an autocast in that file is correct delegation,
  not a missing one. Recorded rather than "repaired", so nobody adds a second autocast there.
"""
from __future__ import annotations

import inspect

import torch



def test_the_graph_bc_route_delegates_its_autocast_to_the_trainer() -> None:
    """The FALSE half, pinned as what it is. `graph_route` autocasts nowhere BECAUSE the step
    it drives does — and this row is what stops someone "fixing" it by adding a second one."""
    from mantis.train.pretrain import graph_route
    from mantis.train.trainer import core

    route_src = inspect.getsource(graph_route)
    assert "autocast" not in route_src, (
        "an autocast appeared in `graph_route`. The graph BC step runs through "
        "`Trainer.train_step_from_graph_batch`, which already autocasts at `self.amp_dtype`; "
        "a second one here would be the duplicate authority AUDIT-1 F-30 is about, added in "
        "the name of fixing it."
    )
    assert "Trainer(" in route_src, (
        "`run_graph_pretrain` no longer builds the real `Trainer` — the delegation this row "
        "depends on is gone, and the graph arm's dtype needs re-deriving"
    )
    step_src = inspect.getsource(core.Trainer.train_step_from_graph_batch)
    assert "dtype=self.amp_dtype" in step_src, (
        "the graph train step stopped autocasting at the resolved dtype, so the graph BC "
        "route now runs at torch's default — the defect F-30 reported, arriving for real"
    )
