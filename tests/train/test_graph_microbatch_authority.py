# >300 justify (R8): the four rows are ONE claim — "the cap block has exactly ONE authority, on
# exactly ONE route, and moves nothing else" — and three of them are structural censuses over the
# same parsed `src/` tree. Splitting them would separate the one-authority census from the
# single-tail census it shares a parse with.
"""One authority on one route for the micro-batch cap block.

The defect each row is the ONLY witness to:

- a minted block with no live consumer on the route its OWN config declares (R1/LAW-08).
- a SECOND reader of the caps appearing. Two authorities agree right up until they diverge, and no
  behavioural oracle can see the second one. An `ast.parse` census, not a grep, because a grep
  cannot tell a reader from a string; subscripts with a constant string index are censused beside
  attributes, since the resolver reads a config DICT.
- a tail statement migrating INTO the accumulation loop. The behavioural rows catch that only at
  the M they happen to run; the AST catches it at any M. The same leg pins the SINGLE TAIL, which
  makes `grad_norm`'s presence in the returned dict structural.
- "non-binding by construction" silently becoming false, with CI exercising a split by accident.

The behavioural rows take their config, keys, identity, resolver and route from `configs/*.yaml`;
the census rows are pure `ast` over the shipped source.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import _microbatch_harness as H
from mantis.monitor.config import MonitorConfig
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.arch_scope import ArchScopedKeyOutsideItsArchError
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.encoding import lookup
from mantis.selfplay.graph_wire_split import plan_microbatches
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.trainer.core import Trainer

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"
_CONFIGS = _REPO / "configs"

#: The three names a reader of this block must mention. Frozen with NO line numbers, so the
#: census cannot go stale against an unrelated edit.
_CAP_NAMES = frozenset({"microbatch_caps", "max_edges", "max_nodes"})


def test_of2_8_run5s_own_config_reaches_the_split_through_its_own_route(tmp_path) -> None:
    """The production config reaches the split through its OWN route.

    Real loader, representation read FROM the config, caps through the REAL resolver behind the
    REAL thunk, step through the REAL dispatcher. The caps are overridden IN MEMORY ONLY — the
    minted values are non-binding on any batch a test can build, so binding them is what makes the
    consumer LIVE, and editing the minted FILE to pass would be tuning to green.
    """
    cfg = load_config(_CONFIGS / "run6.yaml")
    assert cfg.identity.representation == "graph", (
        "run5 no longer declares the graph representation — this row's premise is gone")
    full_config = cfg.model_dump()
    assert "microbatch_caps" in full_config["train"], (
        "configs/run6.yaml carries no train.microbatch_caps block")

    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    bind_e, bind_n = H.caps_for_exactly(replay.wire, 2)
    full_config["train"]["microbatch_caps"] = {"max_edges": bind_e, "max_nodes": bind_n}

    coord = StepCoordinator(
        monitor_cfg=MonitorConfig(),
        trainer=None, buffer=None, pretrained_buffer=None, recent_buffer=None, pool=None,
        eval_pipeline=None, subsystems=None, anchor_state=None, shutdown=None,
        eval_model=None, bufs=None,
        config=SimpleNamespace(selfplay_stall_timeout_sec=1800.0),
        full_config=full_config)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=4, augment=False,
                            recency_weight=0.0, recent_buffer=None,
                            caps_provider=coord._microbatch_caps, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
    ev = sink.named("trainer_step")[0]
    assert ev["microbatches"] >= 2, (
        f"run5's own declared route produced {ev['microbatches']} micro-batches — the block "
        "has no live consumer on the route its own config declares (R1/LAW-08)")
    assert (ev["caps_max_edges"], ev["caps_max_nodes"]) == (bind_e, bind_n)


def test_of2_8_run5s_caps_are_typed_and_inside_the_schema_range(tmp_path) -> None:
    """The SHIPPED production values are read back and typed. Type and the schema's own `ge=1`
    range only — the numbers are the operator's, and the ARMING has its own row below."""
    cfg = load_config(_CONFIGS / "run6.yaml")
    caps = resolve_microbatch_caps(cfg.model_dump())
    assert isinstance(caps.max_edges, int) and caps.max_edges >= 1
    assert isinstance(caps.max_nodes, int) and caps.max_nodes >= 1


# ═══ N-1 — the fix's own ARMING, pinned WITHOUT pinning the operator's numbers ════════════
def _template_caps_of(config_name: str) -> dict:
    """The caps in the TEMPLATE `config_name` was minted from, read out of the config's own
    `# template:` header line, so this follows a re-template instead of assuming `dev`."""
    text = (_CONFIGS / config_name).read_text(encoding="utf-8")
    template = next(line.split(":", 1)[1].strip()
                    for line in text.splitlines() if line.startswith("# template:"))
    tmpl = yaml.safe_load(
        (_REPO / "tools" / "config_templates" / f"{template}.yaml").read_text(encoding="utf-8"))
    return tmpl["train"]["microbatch_caps"]


#: The production UNCAPPED batch, CENSUSED — the line the arming pin binds against. Two readings:
#: E = 18 735 930 / N = 699 533 (the census the card opened on) and E = 19 259 836 / N = 712 937
#: (independent re-census, +2.8% / +1.9%). The SMALLER pair is used, so the assertion is the
#: STRICTER of the two. A censused quantity rather than the sized cap, so this pins ARMING while
#: leaving the operator's number free anywhere below the line; "tighter than the template" was
#: measured to admit a 22x window on edges and 23x on nodes.
#:
#: THE FRONTIER CONSTRAINT — `a + b*max_edges + c*max_nodes <= budget`, all four MEASURED:
#:   a = 34 752 164 B (E-only fit intercept), b = 1 620.96 B/edge, c = 15 741.05 B/node,
#:   budget = 10 126 561 000 B (9.431 GiB, derived line-by-line).
#: Binding alone is not enough: `{18 735 929, 699 532}` binds by one edge, passes 852 tests, and
#: predicts 38.572 GiB per micro-batch, 4.09x over budget. THE PIN IS TIGHT — the shipped pair
#: predicts 9.318 GiB against 9.431 GiB, 1.20% headroom.
#:
#: A SIBLING BUDGET WAS RE-SIZED AND THIS CONSTANT DELIBERATELY DID NOT MOVE WITH IT:
#: `test_graph_microbatch_bound.py`'s `_SIZING_BUDGET_GIB` went 9.431 -> 8.40, re-derived as the
#: trainer's MEASURED peak (7.443 GiB) plus 12.9%. Same nominal quantity, two DENOMINATIONS —
#: that one bounds a measured `max_memory_allocated` delta, this one a MODELLED per-micro-batch
#: peak that runs 25.2% above the measurement. Moving this to 8.40 would red the row below on the
#: SHIPPED, measured-green pair and demand a cap re-fit, which is circular. Needs a RE-FIT.
_FIT_INTERCEPT_BYTES = 34_752_164
_FIT_BYTES_PER_EDGE = 1_620.96
_FIT_BYTES_PER_NODE = 15_741.05
_SIZING_BUDGET_BYTES = 10_126_561_000

_RUN5_CENSUSED_EDGES = 18_735_930
_RUN5_CENSUSED_NODES = 699_533
_CENSUS_BATCH_SIZE = 256


#: BOTH production configs, so the arming/sizing witness covers the config actually being
#: launched. The transfer is guarded: the censused (E, N) describe this batch verbatim only at
#: `_CENSUS_BATCH_SIZE`, and the staleness guard inside the test re-derives that premise.
_PRODUCTION_CAPPED = ("run6.yaml", "run6.yaml")


@pytest.mark.parametrize("name", _PRODUCTION_CAPPED)
def test_n1_run5_is_ARMED_with_a_sized_cap_not_the_templates_non_binding_default(
        name: str) -> None:
    """The arming of the fix itself. Nothing else in the repository pins this.

    A re-mint that loses the `# delta:` header line gives the production config the TEMPLATE's
    pair — a cap that is present, resolves, reports as present in the LAW-18 event, and does not
    bind at the measured E. The fix would be silently disarmed with every test green.

    VALUE-AGNOSTIC BY CONSTRUCTION: this row asserts NO number, only that the pair sits BELOW the
    CENSUSED uncapped batch and UNDER the measured frontier. Both are kept because they assert
    different properties — binding and sizing — and the failure messages say which one broke.
    """
    cfg = load_config(_CONFIGS / name)
    minted = resolve_microbatch_caps(cfg.model_dump())
    template = _template_caps_of(name)
    assert (minted.max_edges, minted.max_nodes) != (template["max_edges"],
                                                    template["max_nodes"]), (
        f"configs/{name} carries the TEMPLATE's non-binding caps — its sized delta has "
        "been lost. The cap resolves and reports as present while bounding nothing: the fix "
        "is disarmed (R4/LAW-07)")
    # The staleness guard on the censused constants: they were measured at that batch size, so a
    # re-mint to a different one means they no longer describe this batch.
    assert int(cfg.train.batch_size) == _CENSUS_BATCH_SIZE, (
        f"{name} mints batch_size={int(cfg.train.batch_size)}, but the censused (E, N) this "
        f"row binds against were measured at batch_size={_CENSUS_BATCH_SIZE}. Re-census "
        "before trusting the arming check — the constants no longer describe this batch")
    # THE ARMING PROPERTY. Tighter-than-the-template is its CONVERSE:
    # `run5 >= template => cannot bind` does not give `run5 < template => can bind`.
    assert minted.max_edges < _RUN5_CENSUSED_EDGES, (
        f"{name} max_edges {minted.max_edges} is NOT below the censused production edge count "
        f"{_RUN5_CENSUSED_EDGES}, so this config's own batch never reaches the cap and the "
        "step is never split. The cap resolves, types, round-trips its `# delta:` header and "
        "reports present in the LAW-18 event — and bounds nothing. That is the phantom-arming "
        "shape (R4/LAW-07) this row exists to kill")
    assert minted.max_nodes < _RUN5_CENSUSED_NODES, (
        f"{name} max_nodes {minted.max_nodes} is NOT below the censused production node count "
        f"{_RUN5_CENSUSED_NODES} — same defect on the node member")
    # THE SIZING PROPERTY. Binding is necessary and not sufficient: a cap can sit one edge below
    # the censused batch, split it in two, and still ask for 4x the card per micro-batch.
    predicted_peak = (_FIT_INTERCEPT_BYTES
                      + _FIT_BYTES_PER_EDGE * minted.max_edges
                      + _FIT_BYTES_PER_NODE * minted.max_nodes)
    assert predicted_peak <= _SIZING_BUDGET_BYTES, (
        f"{name}'s caps ({minted.max_edges}, {minted.max_nodes}) predict a per-micro-batch peak "
        f"of {predicted_peak / 1024 ** 3:.3f} GiB against the measured budget of "
        f"{_SIZING_BUDGET_BYTES / 1024 ** 3:.3f} GiB — "
        f"{predicted_peak / _SIZING_BUDGET_BYTES:.2f}x over. The cap BINDS but is not SIZED: "
        "every micro-batch would still exceed the card. Re-size against "
        "MEASUREMENT_SIZING §9.1's frontier, or if the budget itself was re-measured, move "
        "these constants with it")


@pytest.mark.parametrize("name", _PRODUCTION_CAPPED)
def test_n1_run5s_minted_header_records_the_microbatch_caps_delta(name: str) -> None:
    """The sized pair arrived by MINT, not by a hand-edit (R1).

    The value half above would still pass if someone hand-edited the body and left the header
    silent; that config would then fail to replay from its own provenance record. This asserts the
    header carries the delta line and that its NEW slot round-trips to the body's live value.
    """
    text = (_CONFIGS / name).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines()
             if line.startswith("# delta: train.microbatch_caps:")]
    assert len(lines) == 1, (
        f"{name}'s minted header records {len(lines)} `train.microbatch_caps` deltas, want "
        "exactly 1 — the sized cap must be a recorded mint act, replayable from the header "
        "(R1: configs are minted, never hand-varied)")
    _, _, rest = lines[0].partition("# delta: train.microbatch_caps:")
    _, sep, new_slot = rest.strip().partition(" -> ")
    assert sep, f"the delta line is not splittable into old -> new: {lines[0]}"
    minted = resolve_microbatch_caps(load_config(_CONFIGS / name).model_dump())
    recorded = yaml.safe_load(new_slot)
    assert recorded == {"max_edges": minted.max_edges, "max_nodes": minted.max_nodes}, (
        f"the header's recorded delta {recorded} disagrees with the config body "
        f"({minted.max_edges}, {minted.max_nodes}) — the provenance record is false")


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the function that lexically encloses it ("<module>" when none
    does). Built by walk rather than by parent pointers, which `ast` does not carry."""
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_name = (child.name
                          if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                          else name)
            owner[child] = child_name
            visit(child, child_name)

    owner[tree] = "<module>"
    visit(tree, "<module>")
    return owner


def _cap_reader_census() -> collections.Counter:
    """`Counter` over `(module, receiver_source, enclosing_function)` for every ATTRIBUTE and every
    SUBSCRIPT-with-a-constant-string-index naming one of the three cap names."""
    counter: collections.Counter = collections.Counter()
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        rel = path.relative_to(_REPO).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _CAP_NAMES:
                counter[(rel, ast.unparse(node.value), owner.get(node, "<module>"))] += 1
            elif (isinstance(node, ast.Subscript)
                  and isinstance(node.slice, ast.Constant)
                  and node.slice.value in _CAP_NAMES):
                counter[(rel, ast.unparse(node.value), owner.get(node, "<module>"))] += 1
    return counter


#: FROZEN, with NO line numbers. Three reads in the resolver and two in the graph arm;
#: `coordinator/step.py` names none of the three — it calls the resolver and stores the result.
#: Schema and dataclass field declarations are `AnnAssign` TARGETS, not loads, so the block's own
#: definition is not mistaken for a reader. A THIRD read anywhere adds a key and reds.
_EXPECTED_CENSUS = {
    ("src/mantis/config/resolve/microbatch.py", "train_section", "resolve_microbatch_caps"): 1,
    ("src/mantis/config/resolve/microbatch.py", "block", "resolve_microbatch_caps"): 2,
    ("src/mantis/train/coordinator/dispatch.py", "caps", "_build_graph_parts"): 2,
}


def test_of2_9_leg1_exactly_one_authority_reads_the_caps() -> None:
    """The frozen cap-reader census: a second read, or an alias bound at `__init__`, adds a key
    that no behavioural row here can see. DISCLOSED RESIDUAL: `getattr(spec, "max_" + "edges")`
    still defeats it — the census bounds the PLAUSIBLE homes of a second reader, not every one."""
    census = _cap_reader_census()
    assert dict(census) == _EXPECTED_CENSUS, (
        f"the cap-reader census moved.\n  got:  {dict(census)}\n  want: {_EXPECTED_CENSUS}\n"
        "A new key is a SECOND authority over one fact — re-derive before editing this "
        "expectation (PREREG_DFIX §4, OF2-9 leg 1).")


def test_of2_9_leg1_totals_match_the_pre_registered_three_and_two() -> None:
    """The pre-registered totals — 3 in the resolver, 2 in the arm — asserted per
    (module, function), so the registered number is checked directly and not only through the
    finer key above."""
    per_site: collections.Counter = collections.Counter()
    for (module, _receiver, function), n in _cap_reader_census().items():
        per_site[(module, function)] += n
    assert dict(per_site) == {
        ("src/mantis/config/resolve/microbatch.py", "resolve_microbatch_caps"): 3,
        ("src/mantis/train/coordinator/dispatch.py", "_build_graph_parts"): 2,
    }


_TAIL_KEYS = {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}
#: `update_parameters` is the EMA update, a TAIL statement exactly like `self.step += 1`. It is
#: named because the behavioural rows cannot always reach it — a trainer built without EMA never
#: executes that branch — so this token kills the mutation STRUCTURALLY, at any M.
_FORBIDDEN_IN_LOOP = {"_maybe_periodic_checkpoint", "emit_via", "step",
                      "update_parameters"}


def _graph_step_fn() -> ast.FunctionDef:
    tree = ast.parse((_SRC / "train" / "trainer" / "core.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "train_step_from_graph_batch"):
            return node
    raise AssertionError("train_step_from_graph_batch not found in trainer/core.py")


def test_of2_9_leg2_the_graph_step_has_exactly_one_tail_returning_five_keys() -> None:
    """Exactly ONE `return`, of a dict literal whose keys are exactly the five.

    This is what makes `grad_norm`'s presence structural: the grad-norm gate reads
    `loss_info.get("grad_norm", 0.0)`, so a path returning a dict WITHOUT the key would silently
    feed an armed abort a `0.0` that always passes its threshold.
    """
    fn = _graph_step_fn()
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1, f"{len(returns)} return statements, want exactly 1"
    assert returns[0] is fn.body[-1], "the return is not the last statement of the function"
    value = returns[0].value
    if isinstance(value, ast.Name):
        # The tail names a local bound ONCE from a dict literal — the shape the emit and the
        # periodic-checkpoint call require, since both consume the dict before it is returned.
        literals = [n.value for n in ast.walk(fn)
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets)]
        assert len(literals) == 1, (
            f"the returned name {value.id!r} is not bound exactly once from a dict literal")
        value = literals[0]
    assert isinstance(value, ast.Dict), "the tail does not return a dict literal"
    keys = {k.value for k in value.keys if isinstance(k, ast.Constant)}
    assert keys == _TAIL_KEYS, f"tail keys {sorted(keys)} != {sorted(_TAIL_KEYS)}"


def test_of2_9_leg2_no_tail_statement_lives_inside_the_accumulation_loop() -> None:
    """The accumulation loop body contains ZERO tail statements.

    A tail statement moved inside the loop reds the behavioural row only at M >= 2 and reds this
    at any M — which is the gap that would let it be a fold-in rather than a caught defect.
    """
    fn = _graph_step_fn()
    # The accumulation loop is identified by what it ITERATES, not by position: the function also
    # carries a ban loop, so "the only top-level loop" was never the right handle.
    loops = [n for n in fn.body
             if isinstance(n, ast.For) and isinstance(n.iter, ast.Name)
             and n.iter.id == "parts"]
    assert len(loops) == 1, (
        f"expected exactly one top-level `for ... in parts:` accumulation loop in the graph "
        f"step; found {len(loops)}")
    body = loops[0]
    for node in ast.walk(body):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else "")
            assert fname not in _FORBIDDEN_IN_LOOP, (
                f"{fname!r} is called inside the accumulation loop — it belongs to the tail "
                "and would fire once per MICRO-BATCH (MB-25)")
        if isinstance(node, ast.AugAssign):
            tgt = node.target
            assert not (isinstance(tgt, ast.Attribute) and tgt.attr == "step"), (
                "`self.step +=` inside the accumulation loop — the counter would advance M "
                "times per training step (MB-8)")


_NON_RUN5 = ("dev_example.yaml", "smoke_preflight_armed.yaml")


@pytest.mark.parametrize("name", _NON_RUN5)
def test_of2_14_the_smoke_configs_caps_do_not_bind(name: str) -> None:
    """The non-production configs' caps do not bind, which the design claims by construction.

    A smoke config whose cap BOUND would make CI exercise a split BY ACCIDENT with no count
    changing to say so; the split's coverage has to come from the oracles, where its M is asserted.
    """
    cfg = load_config(_CONFIGS / name)
    caps = resolve_microbatch_caps(cfg.model_dump())
    batch_size = int(cfg.train.batch_size)
    buf = H.uniform_graph_buffer(max(8, batch_size))
    wire, _targets = buf.sample_graph_batch(batch_size, augment=False, recent_frac=0.0)
    ec, nc = H.per_graph_counts(wire)
    parts = plan_microbatches(np.concatenate([[0], np.cumsum(ec)]).astype(np.int64),
                              np.concatenate([[0], np.cumsum(nc)]).astype(np.int64),
                              caps.max_edges, caps.max_nodes)
    assert len(parts) == 1, (
        f"{name}: the minted caps ({caps.max_edges}, {caps.max_nodes}) SPLIT its own "
        f"batch_size={batch_size} batch into {len(parts)} micro-batches (E={int(ec.sum())}, "
        f"N={int(nc.sum())}) — CI would be exercising a split by accident (MB-24)")
    assert cfg.identity.representation == "graph"   # the sweep's own premise, executed


def test_of2_14_run5_is_excluded_deliberately_and_the_set_is_the_whole_directory() -> None:
    """The two above plus the production config are ALL the configs.

    Enumeration is `discover_configs`, the ONE authority both gates 7 and 12 consume; a second flat
    glob here would let a subdirectory or `.yml` shape both gates make legal slip out silently. The
    production config is EXCLUDED because its caps are fitted against a measured partition.
    """
    live = sorted(p.relative_to(_CONFIGS).as_posix() for p in discover_configs(_CONFIGS))
    assert live == sorted((*_NON_RUN5, "run6.yaml"))
