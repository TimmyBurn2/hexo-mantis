"""AUDIT-1 F-30 — BC pretrain runs at the DECLARED autocast dtype, on both arms.

THE FINDING, AND THE HALF OF IT THAT IS FALSE AT CONTACT (REPAIR-2 §0 verifies before it
repairs, and this row is why the rule exists):

* **GRID — TRUE, and repaired.** `pretrain/trainer.py::BootstrapTrainer` read
  `config.get("fp16", True)` — a code-side default on a key the schema REQUIRES — and
  autocast at a LITERAL `torch.float16`, while `amp_dtype_for` is the ONE dtype authority
  (LAW-06) and the trainer BC warm-starts runs it. Both now come off the minted config
  through `training_terms`, the same one-read-path fix F-816-25 made for six sibling keys.
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
from typing import Any

import torch

from mantis.model.amp import amp_dtype_for


def _train_cfg() -> Any:
    from mantis.config.loader import load_config
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    return load_config(repo / "configs" / "smoke_gnn.yaml").train


def test_the_pretrain_read_path_carries_both_dtype_terms() -> None:
    """`training_terms` is THE read path for a bootstrap pretrain's training terms. Both dtype
    terms are in it, so neither can be re-defaulted at the trainer."""
    from mantis.train.pretrain.cli import training_terms

    terms = training_terms(_train_cfg())
    assert terms["fp16"] == _train_cfg().fp16
    assert terms["amp_dtype"] == _train_cfg().amp_dtype


def test_the_grid_bootstrap_trainer_has_no_dtype_literal_and_no_defaulted_fp16() -> None:
    """The two halves of the grid defect. STRUCTURE, not text — an AST walk, because the
    comment that RECORDS the removed code contains the removed code, and a substring search
    cannot tell the record from the thing."""
    import ast

    from mantis.train.pretrain import trainer as bootstrap

    tree = ast.parse(inspect.getsource(bootstrap))
    defaulted_gets, dtype_literals = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" and len(node.args) == 2 \
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "fp16":
            defaulted_gets.append(node.lineno)
        name = func.attr if isinstance(func, ast.Attribute) else \
            func.id if isinstance(func, ast.Name) else None
        if name == "autocast":
            for kw in node.keywords:
                v = kw.value
                if kw.arg == "dtype" and isinstance(v, ast.Attribute) \
                        and isinstance(v.value, ast.Name) and v.value.id == "torch":
                    dtype_literals.append(f"line {node.lineno}: torch.{v.attr}")
    assert not defaulted_gets, (
        f"a DEFAULTED `fp16` read is back at line(s) {defaulted_gets} — a code-side default on "
        "a REQUIRED schema key, which is R1's class and AUDIT-1 F-30's first half"
    )
    assert not dtype_literals, (
        f"an autocast dtype literal is back: {dtype_literals}; `amp_dtype_for` owns the "
        "decision (LAW-06)"
    )
    assert "amp_dtype_for" in inspect.getsource(bootstrap), (
        "the trainer no longer consults the dtype authority at all"
    )


def test_the_grid_trainer_resolves_its_dtype_through_the_authority() -> None:
    """Constructed, not just grepped: the attribute the autocast reads IS the authority's
    answer for the arch's representation."""
    from mantis.encoding import lookup
    from mantis.model import build_net, select_arch
    from mantis.train.pretrain.cli import training_terms
    from mantis.train.pretrain.trainer import BootstrapTrainer
    import tempfile
    from pathlib import Path

    train_cfg = _train_cfg()
    spec = lookup("v6_live2_ls")
    arch = select_arch(spec, {}, arch_kind="CnnArch")
    config = {"encoding": spec.name, "in_channels": int(spec.n_planes),
              **training_terms(train_cfg), "pretrain_total_steps": 1}
    with tempfile.TemporaryDirectory() as tmp:
        t = BootstrapTrainer(
            build_net(arch), config, torch.device("cpu"), Path(tmp), arch=arch, sink=None,
        )
    assert t.amp_dtype is amp_dtype_for(str(arch.representation), str(train_cfg.amp_dtype))


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
