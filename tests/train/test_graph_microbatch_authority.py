# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence. (This
# file's first version stated one and it was already false at submission.)
# The four rows are ONE claim — "the cap block has
# exactly ONE authority, on exactly ONE route, and moves nothing else" — and three of them are
# structural censuses over the same parsed `src/` tree. Splitting them would separate the
# one-authority census from the single-tail census it shares a parse with, and would put
# OF2-12's dense-invariance golden in a file with no other statement about what F2 must NOT
# move. Executable content is a minority: the rest is the per-row LAW-07 rationale.
"""⊕ WP12-R dispatch 6 phase F2 — one authority, one route, and the dense path untouched
(DESIGN_DFIX §5.2, PREREG_DFIX §4).

The defect each row is the ONLY witness to:

- **OF2-8** — a minted block with no live consumer on the route its OWN config declares
  (R1/LAW-08). `configs/run5.yaml` through the real loader, representation read FROM the
  config, the caps overridden IN MEMORY ONLY to bind, and the step driven by the real
  dispatcher over a real `HexgBuffer`.
- **OF2-9 leg 1** — a SECOND reader of the caps appearing. Two authorities agree right up
  until they diverge, and no behavioural oracle can see the second one. An `ast.parse` census,
  not a grep: a grep cannot tell a reader from a string (R93/DR-11). **Subscripts with a
  constant string index are censused beside attributes**, because the resolver reads a config
  DICT and an attribute-only census would be blind to a second reader added there.
- **OF2-9 leg 2** — a tail statement migrating INTO the accumulation loop. The behavioural
  rows catch that only at the M they happen to run; the AST catches it at any M (MB-25). The
  same leg pins the SINGLE TAIL, which is what makes `grad_norm`'s presence in the returned
  dict a structural property rather than a fact about one code path.
- **OF2-12** — the `fp16_backward_step` decomposition moving the DENSE update. The golden is
  captured BEFORE the `losses.py` edit, so it is a true before/after (MB-18).
- **OF2-14** — DESIGN_DFIX §7.4's "non-binding by construction" silently becoming false, with
  CI exercising a split by accident and no count changing to say so (MB-24).

**What is real and what is not.** OF2-8 and OF2-14 take their CONFIG, keys, identity, resolver
and route from `configs/*.yaml` — the network is small; the wiring is production's. OF2-9 is
pure `ast` over the shipped source. OF2-12 substitutes the ARCH and nothing else.
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
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.encoding import lookup
from mantis.model import CnnArch, build_net
from mantis.selfplay.graph_wire_split import plan_microbatches
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.trainer.core import Trainer

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"
_CONFIGS = _REPO / "configs"
_GOLDEN = _REPO / "tests" / "fixtures" / "wp12r_dfix" / "dense_backward_golden.json"

GRID_ENCODING = "v6_live2_ls"
_DSPEC = lookup(GRID_ENCODING)

#: The three names a reader of this block must mention. Frozen with NO line numbers, so the
#: census cannot go stale against an unrelated edit.
_CAP_NAMES = frozenset({"microbatch_caps", "max_edges", "max_nodes"})


# ═══ OF2-8 — LAW-08 on run5's OWN route ══════════════════════════════════════════════════
def test_of2_8_run5s_own_config_reaches_the_split_through_its_own_route(tmp_path) -> None:
    """OF2-8 — the card's reason to exist. `configs/run5.yaml` is loaded by the REAL loader,
    its representation is read FROM the config rather than asserted by this test, its caps go
    through the REAL resolver behind the REAL `StepCoordinator._microbatch_caps` thunk, and
    the step runs through the REAL `run_declared_train_step`.

    The caps are overridden IN MEMORY ONLY. The minted values are non-binding on any batch a
    test can build — that is the point of the sizing pass, not a defect — so binding them here
    is what makes the consumer LIVE rather than merely present. The FILE is not touched: a
    test that edited a minted config to make itself pass would be tuning to green (R61)."""
    cfg = load_config(_CONFIGS / "run5.yaml")
    assert cfg.identity.representation == "graph", (
        "run5 no longer declares the graph representation — this row's premise is gone")
    full_config = cfg.model_dump()
    assert "microbatch_caps" in full_config["train"], (
        "configs/run5.yaml carries no train.microbatch_caps block")

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
                            caps_provider=coord._microbatch_caps)
    ev = sink.named("trainer_step")[0]
    assert ev["microbatches"] >= 2, (
        f"run5's own declared route produced {ev['microbatches']} micro-batches — the block "
        "has no live consumer on the route its own config declares (R1/LAW-08)")
    assert (ev["caps_max_edges"], ev["caps_max_nodes"]) == (bind_e, bind_n)


def test_of2_8_run5s_caps_are_typed_and_inside_the_schema_range(tmp_path) -> None:
    """OF2-8 second limb — the SHIPPED run5 values are read back and typed.

    RENAMED (N-1): this row used to be called `..._are_the_prereg_rows`, which claimed a pin
    it does not make. It asserts type and the schema's own `ge=1` range and NOTHING about the
    numbers — the numbers are the operator's (R119/R193(c)). The property its old name implied
    is the ARMING, and that now has its own row below, expressed value-agnostically."""
    cfg = load_config(_CONFIGS / "run5.yaml")
    caps = resolve_microbatch_caps(cfg.model_dump())
    assert isinstance(caps.max_edges, int) and caps.max_edges >= 1
    assert isinstance(caps.max_nodes, int) and caps.max_nodes >= 1


# ═══ N-1 — the fix's own ARMING, pinned WITHOUT pinning the operator's numbers ════════════
def _template_caps_of(config_name: str) -> dict:
    """The caps in the TEMPLATE `config_name` was minted from — read out of the config's own
    `# template:` header line, so this follows a re-template instead of assuming `dev`."""
    text = (_CONFIGS / config_name).read_text(encoding="utf-8")
    template = next(line.split(":", 1)[1].strip()
                    for line in text.splitlines() if line.startswith("# template:"))
    tmpl = yaml.safe_load(
        (_REPO / "tools" / "config_templates" / f"{template}.yaml").read_text(encoding="utf-8"))
    return tmpl["train"]["microbatch_caps"]


#: **run5's UNCAPPED batch, CENSUSED — the line the arming pin binds against.**
#:
#: Transcribed ONCE, here, with its provenance and its independent replication beside it, in
#: the `_SIZING_BUDGET_GIB` style: a measured quantity may be written down if the measurement
#: travels with it.
#:
#:   MEASUREMENT_D  (the census the card was OPENED on): E = 18 735 930, N =   699 533
#:   MEASUREMENT_SIZING §4.1 (independent re-census):    E = 19 259 836, N =   712 937  (+2.8%/+1.9%)
#:
#: The SMALLER pair is used, which makes the assertion the STRICTER of the two readings: a cap
#: below 18 735 930 is below both censuses, so the row cannot pass on the strength of having
#: picked the friendlier measurement.
#:
#: WHY A CENSUSED QUANTITY AND NOT THE SIZED CAP: this pins ARMING while leaving the operator's
#: number free ANYWHERE below the line (R119/R193(c)). The previous form asserted only
#: "tighter than the template", which RED-TEAM measured to admit a **22x window on edges and
#: 23x on nodes** in which the cap resolves, types, round-trips its header, reports present —
#: and cannot bind at run5's real batch. It minted {99 999 999, 3 999 999} and the ENTIRE run5
#: detector surface returned 850 passed, 0 failed: the fix disarmed, the repository saying it
#: was armed.
#: **THE FRONTIER CONSTRAINT — `a + b*max_edges + c*max_nodes <= budget`.**
#: `PREREG_DFIX` §7.1 states it as the sizing rule; `MEASUREMENT_SIZING` §9.1 supplies every
#: coefficient, and all four are MEASURED, none is a policy choice:
#:
#:   a      =     34 752 164 B      intercept of the real E-only fit        (§5.1)
#:   b      =      1 620.96 B/edge  two-variable fit                        (§6.4)
#:   c      =     15 741.05 B/node  two-variable fit                        (§6.4)
#:   budget = 10 126 561 000 B      9.431 GiB, derived line-by-line         (§8)
#:
#: **WHY THIS IS VALUE-AGNOSTIC and does NOT constrain the operator (R119/R193(c)).** It admits
#: the ENTIRE feasible region under the frontier and rejects only pairs whose predicted peak
#: exceeds the measured budget — which is not a preference, it is the definition of a SIZED
#: cap. The operator may mint anything below the line.
#:
#: **What it closes.** The censused check below pins that the cap can BIND; it does not pin
#: that the cap was SIZED. RED-TEAM measured the residue: `{18 735 929, 699 532}` binds by one
#: edge, passes 852 tests, and predicts **38.572 GiB per micro-batch — 4.09x over budget**,
#: i.e. essentially the uncapped batch. Reproduced here: this constraint rejects it at 4.09x,
#: rejects the template pair at 22.23x, and passes the shipped pair.
#:
#: **DISCLOSED — this pin is TIGHT.** The shipped pair predicts 9.318 GiB against a 9.431 GiB
#: budget: **1.20 % headroom**. A re-mint materially above the frontier reds this row, which is
#: correct (the model says it would exceed the budget) but leaves little room, and it means a
#: re-sized BUDGET must move these constants in step. They are transcribed together for that
#: reason.
_FIT_INTERCEPT_BYTES = 34_752_164
_FIT_BYTES_PER_EDGE = 1_620.96
_FIT_BYTES_PER_NODE = 15_741.05
_SIZING_BUDGET_BYTES = 10_126_561_000

_RUN5_CENSUSED_EDGES = 18_735_930
_RUN5_CENSUSED_NODES = 699_533
_CENSUS_BATCH_SIZE = 256


#: F-P2B (R259, review finding 1): BOTH production configs, so the arming/sizing witness
#: covers the config actually being launched. The transfer is legitimate and guarded:
#: `shakedown_20260807.yaml` mints run5's caps at run5's `batch_size: 256` on the graph arm,
#: so the censused (E, N) describe its batch verbatim — and the `_CENSUS_BATCH_SIZE`
#: staleness guard inside the test re-derives that premise per config rather than assuming it.
_PRODUCTION_CAPPED = ("run5.yaml", "shakedown_20260807.yaml")


@pytest.mark.parametrize("name", _PRODUCTION_CAPPED)
def test_n1_run5_is_ARMED_with_a_sized_cap_not_the_templates_non_binding_default(
        name: str) -> None:
    """**N-1 — the arming of the fix itself.** Nothing else in the repository pins this.

    THE DEFECT THIS IS THE ONLY WITNESS TO: a future re-mint that loses run5's
    `# delta: train.microbatch_caps:` header line gives run5 the TEMPLATE's pair
    (100 000 000 / 4 000 000) — a cap that is present, resolves, reports as present in the
    LAW-18 event, and **does not bind at run5's measured E = 18 735 930**. The fix would be
    silently disarmed with every test in this repository green. That is the phantom-gate shape
    (R4/LAW-07) applied to the arming of the fix — and it is not hypothetical: R178(a)
    re-minted six configs and R187 re-minted two in this dispatch alone, and Q-DFIX-3
    guarantees run5's delta set gets edited again.

    **VALUE-AGNOSTIC BY CONSTRUCTION, and that is the hard requirement (R119/R193(c)).** The
    operator's minted numbers must stay cheap to change, so this row asserts NO number. It
    asserts that run5's pair sits BELOW run5's own CENSUSED uncapped batch — a measured
    quantity, not the sizing answer — which leaves the operator free anywhere beneath that
    line.

    **THREE LAYERS, each closing the previous one's residue — and each residue was MEASURED,
    not imagined.**

    1. *Tighter than the template.* What this row first asserted, justified by *"the template
       pair is non-binding by construction, so run5 at or above it cannot bind."* True, and the
       **converse** of the row's claim: `run5 >= template => cannot bind` does not give
       `run5 < template => can bind`. RED-TEAM measured a **22x window** through it.
    2. *Below the CENSUSED production batch.* Closes that, and pins **BINDING**. Its own
       residue, also measured: `{18 735 929, 699 532}` binds by one edge, passes 852 tests, and
       predicts **38.572 GiB per micro-batch — 4.09x over budget**. A 4.16x window.
    3. *Under the measured FRONTIER.* Pins **SIZING**. This is what makes the pair a sized cap
       rather than merely a smaller number, and it rejects every known variant.

    All three are kept: they assert different properties, and the failure messages say which
    one broke."""
    cfg = load_config(_CONFIGS / name)
    minted = resolve_microbatch_caps(cfg.model_dump())
    template = _template_caps_of(name)
    assert (minted.max_edges, minted.max_nodes) != (template["max_edges"],
                                                    template["max_nodes"]), (
        f"configs/{name} carries the TEMPLATE's non-binding caps — its sized delta has "
        "been lost. The cap resolves and reports as present while bounding nothing: the fix "
        "is disarmed (R4/LAW-07)")
    # The staleness guard on the censused constants below. They were measured AT
    # `batch_size: 256`; if this config re-mints a different batch size they no longer
    # describe its batch and this row must be re-derived rather than quietly keep passing.
    assert int(cfg.train.batch_size) == _CENSUS_BATCH_SIZE, (
        f"{name} mints batch_size={int(cfg.train.batch_size)}, but the censused (E, N) this "
        f"row binds against were measured at batch_size={_CENSUS_BATCH_SIZE}. Re-census "
        "before trusting the arming check — the constants no longer describe this batch")
    # THE ARMING PROPERTY. Tighter-than-the-template is NOT this property, it is its converse:
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
    # THE SIZING PROPERTY. Binding is necessary and not sufficient: a cap can sit one edge
    # below the censused batch, split it into two, and still ask for 4x the card per
    # micro-batch. The frontier is what makes the pair a SIZED cap rather than merely a
    # smaller number.
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
    """N-1's provenance half — the sized pair arrived by MINT, not by a hand-edit (R1).

    The value half above would still pass if someone hand-edited the body and left the header
    silent; that config would then fail to replay from its own provenance record. This asserts
    the header carries the delta line, and that the line's NEW slot round-trips to the body's
    live value — again without pinning what that value is. Parametrized over both production
    configs (F-P2B, review finding 1)."""
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


# ═══ OF2-9 leg 1 — the one-authority census ══════════════════════════════════════════════
def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the function that lexically encloses it ("<module>" when
    none does). Built by walk rather than by parent pointers, which `ast` does not carry."""
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
    """`Counter` over `(module, receiver_source, enclosing_function)` for every ATTRIBUTE and
    every SUBSCRIPT-with-a-constant-string-index naming one of the three cap names."""
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


#: FROZEN, with NO line numbers. Three reads in the resolver (`microbatch_caps`, then the two
#: members off the resolved block) and two in the graph arm; `coordinator/step.py` names none
#: of the three — it calls the resolver and stores the result. Schema and dataclass field
#: declarations are `AnnAssign` TARGETS, not `Attribute`/`Subscript` loads, so they do not
#: appear and the block's own definition is not mistaken for a reader.
_EXPECTED_CENSUS = {
    ("src/mantis/config/resolve/microbatch.py", "train_section", "resolve_microbatch_caps"): 1,
    ("src/mantis/config/resolve/microbatch.py", "block", "resolve_microbatch_caps"): 2,
    ("src/mantis/train/coordinator/dispatch.py", "caps", "_graph_step"): 2,
}


def test_of2_9_leg1_exactly_one_authority_reads_the_caps() -> None:
    """OF2-9 leg 1 — the frozen census. MB-15 (a second read added in
    `coordinator/step.py`) and MB-16 (`self._caps = ...` aliased at `__init__`) both add a
    key; neither is visible to any behavioural row here.

    DISCLOSED RESIDUAL (MB-16): `getattr(spec, "max_" + "edges")` still defeats it. The
    census bounds the PLAUSIBLE homes of a second reader, not every conceivable one, and
    saying so is the honest form of the claim."""
    census = _cap_reader_census()
    assert dict(census) == _EXPECTED_CENSUS, (
        f"the cap-reader census moved.\n  got:  {dict(census)}\n  want: {_EXPECTED_CENSUS}\n"
        "A new key is a SECOND authority over one fact — re-derive before editing this "
        "expectation (PREREG_DFIX §4, OF2-9 leg 1).")


def test_of2_9_leg1_totals_match_the_pre_registered_three_and_two() -> None:
    """OF2-9 leg 1, the pre-registered form — PREREG states the totals as "3 in the resolver,
    2 in the arm". Asserted per (module, function) so the number the prereg registered is
    checked directly and not only through the finer key above."""
    per_site: collections.Counter = collections.Counter()
    for (module, _receiver, function), n in _cap_reader_census().items():
        per_site[(module, function)] += n
    assert dict(per_site) == {
        ("src/mantis/config/resolve/microbatch.py", "resolve_microbatch_caps"): 3,
        ("src/mantis/train/coordinator/dispatch.py", "_graph_step"): 2,
    }


# ═══ OF2-9 leg 2 — the SINGLE-TAIL structure ═════════════════════════════════════════════
_TAIL_KEYS = {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}
#: `update_parameters` is the EMA update. It is named here because it is a TAIL statement
#: exactly like `self.step += 1` — DESIGN §3.6 lists them in the same row — and because
#: the behavioural rows cannot always reach it: a trainer built without EMA has
#: `ema_model is None`, so `core.py`'s EMA branch never executes and an update moved into
#: the accumulation loop would red nothing. This token kills that mutation STRUCTURALLY,
#: at any M, whether or not the driving fixture has EMA enabled.
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
    """OF2-9 leg 2 — exactly ONE `return`, and what it returns is a dict literal whose keys
    are exactly the five.

    This is what makes `grad_norm`'s presence structural. `coordinator/step.py`'s grad-norm gate
    reads
    `loss_info.get("grad_norm", 0.0)`, so a path that returned a dict WITHOUT the key would
    silently feed an armed abort a `0.0` that always passes its threshold. The degenerate-mask
    cases return finite-or-`nan` through this same tail (HEAD behaviour, and the gate's own
    `math.isfinite` guard already handles `nan`); the two failure paths RAISE."""
    fn = _graph_step_fn()
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1, f"{len(returns)} return statements, want exactly 1"
    assert returns[0] is fn.body[-1], "the return is not the last statement of the function"
    value = returns[0].value
    if isinstance(value, ast.Name):
        # the tail names a local bound ONCE from a dict literal — the shape the emit and the
        # periodic-checkpoint call require, since both consume the dict before it is returned
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
    """OF2-9 leg 2 second half — the accumulation loop body contains ZERO calls to
    `_maybe_periodic_checkpoint` / `emit_via` / `scheduler.step` / `optimizer.step` and no
    `self.step` augmented assignment.

    MB-25 (a tail statement moved inside the loop) reds OF2-4 only at M >= 2 and reds THIS at
    any M — which is the gap that would let it be a fold-in rather than a caught defect."""
    fn = _graph_step_fn()
    # The accumulation loop is identified by what it ITERATES, not by position: the function
    # also carries the `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` ban loop (already there at HEAD, and
    # §3.7 says it does not move), so "the only top-level loop" was never the right handle.
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


# ═══ OF2-12 — the dense path does not move ═══════════════════════════════════════════════
def _fixed_dense_batch(n: int = 4):
    """A DETERMINISTIC dense batch in `train_step_from_tensors`' own argument order.

    Constructed, not sampled: `ReplayBuffer.sample_batch_with_pos` draws through the Rust RNG,
    which `torch.manual_seed` does not reach — measured, two same-process runs disagreed — so
    a sampled batch cannot carry a bit-identity golden."""
    rng = np.random.default_rng(H.SEED)
    s = int(_DSPEC.board_size)
    n_cells = s * s
    states = rng.standard_normal((n, int(_DSPEC.n_planes), s, s)).astype(np.float16)
    chain = rng.standard_normal((n, 6, s, s)).astype(np.float16)
    policies = np.zeros((n, int(_DSPEC.policy_stride)), dtype=np.float32)
    for i in range(n):
        policies[i, (i * 7) % n_cells] = 1.0
    outcomes = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)], dtype=np.float32)
    own = np.zeros((n, s, s), dtype=np.uint8)
    wl = np.zeros((n, s, s), dtype=np.uint8)
    ifs = np.ones(n, dtype=np.uint8)
    pos = np.arange(n, dtype=np.uint16)
    vv = np.ones(n, dtype=np.uint8)
    return states, chain, policies, outcomes, own, wl, ifs, pos, vv


def _dense_step_digests(tmp_path) -> tuple[dict[str, str], str]:
    torch.set_num_threads(1)
    torch.manual_seed(H.SEED)
    arch = CnnArch(board_size=int(_DSPEC.board_size), in_channels=int(_DSPEC.n_planes),
                   filters=8, res_blocks=1)
    net = build_net(arch)
    trainer = Trainer(net,
                      {"identity": {"encoding": GRID_ENCODING, "representation": "grid"},
                       "train": {"amp_dtype": "fp16"}},
                      arch=arch, checkpoint_dir=tmp_path / "ckpt",
                      device=torch.device("cpu"), train_hparams=H.graph_hparams())
    states, chain, policies, outcomes, own, wl, ifs, pos, vv = _fixed_dense_batch()
    trainer.train_step_from_tensors(
        states, policies, outcomes, chain_planes=chain, ownership_targets=own,
        threat_targets=wl, is_full_search=ifs, n_pretrain=0, n_recent=0,
        position_indices=pos, value_target_valid=vv)
    params = {name: p.detach().cpu().numpy().tobytes() for name, p in net.named_parameters()}
    per = {k: hashlib.sha256(v).hexdigest() for k, v in params.items()}
    whole = hashlib.sha256()
    for k in sorted(params):
        whole.update(k.encode())
        whole.update(params[k])
    return per, whole.hexdigest()


def test_of2_12_the_dense_step_is_bit_identical_to_the_pre_edit_golden(tmp_path) -> None:
    """OF2-12 — the dense path is not this phase's to move. `fp16_backward_step` is
    DECOMPOSED into `backward_accumulate` + `clip_and_step` and REDEFINED as their
    composition: the same five statements in the same order on the same objects. MB-18
    (recomposed with the clip before the backward) reds this bit-exactly.

    The golden is a BIT-IDENTITY golden — sha256 over each post-step parameter tensor's raw
    BYTES, so dtype and shape are covered too, and per-tensor digests localise a difference.
    It was captured at the FAMILY-A TIP, before the `losses.py` edit; family A touches only
    `gine.py`, which the dense CNN path does not use (`build_net` routes `CnnArch` to
    `HexTacToeNet`), so the golden is identical at `982da03` and at the capture commit. The
    capture commit is recorded in the golden itself.

    DISCLOSED: a bit-identity golden is environment-bound. The golden carries the torch
    version and platform it was taken on, and this row LOUD-SKIPS (grounds printed) rather
    than firing a false HALT when the running environment differs — a red on a different BLAS
    would say nothing about the decomposition. `torch.set_num_threads(1)` on both sides
    removes the one intra-environment source this rig controls."""
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    here = (torch.__version__, f"{sys.platform}/{platform.machine()}")
    there = (golden["torch"], golden["platform"])
    if here != there:
        pytest.skip(
            f"OF2-12 golden was captured on {there} and this environment is {here}; a "
            "bit-identity comparison across BLAS/torch builds tests the environment, not the "
            f"decomposition. Golden capture commit: {golden['capture_commit']}.")
    per, whole = _dense_step_digests(tmp_path)
    if whole != golden["whole_sha256"]:
        moved = [k for k, v in per.items() if golden["per_tensor_sha256"].get(k) != v]
        raise AssertionError(
            f"the dense post-step parameters moved: {len(moved)} of {len(per)} tensors "
            f"differ ({moved[:5]}...). The `fp16_backward_step` decomposition must execute "
            f"the same five statements in the same order. Golden capture commit "
            f"{golden['capture_commit']}.")
    assert set(per) == set(golden["per_tensor_sha256"])


# ═══ OF2-14 — the five smoke configs do not bind ═════════════════════════════════════════
_NON_RUN5 = ("dev_example.yaml", "smoke_gnn.yaml", "smoke_preflight_armed.yaml",
             "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")


@pytest.mark.parametrize("name", _NON_RUN5)
def test_of2_14_the_smoke_configs_caps_do_not_bind(name: str) -> None:
    """OF2-14 — DESIGN_DFIX §7.4 claims the five non-run5 configs are "non-binding by
    construction". Rev-1 asserted that and tested nothing; this row is its producer.

    A smoke config whose cap BOUND would make CI exercise a split BY ACCIDENT, with no count
    changing to say so — and the split's coverage has to come from the oracles, where it is
    deliberate and its M is asserted. The probe batch is built at the config's OWN
    `train.batch_size`.

    For the two GRID configs the caps have no live consumer at all — `_grid_step` is never
    given the provider — so the check there is an UPPER BOUND on a route those configs never
    take, and it is labelled as one rather than dressed up as a route drive."""
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
    if cfg.identity.representation != "graph":
        assert cfg.identity.representation == "grid"   # the caps are never read on this route


def test_of2_14_run5_is_excluded_deliberately_and_the_set_is_the_whole_directory() -> None:
    """OF2-14 premise — the five above plus the production configs are ALL the configs.
    Enumeration is `discover_configs` (R71/R75), the ONE discovery authority both gates 7
    and 12 consume — a second flat `configs/*.yaml` glob here would be exactly the
    divergence ADJ-13 F-1 was: a subdirectory/`.yml` shape both gates make legal would
    slip out of this sweep silently while staying invisible to nobody else (N4, F-P2B/N4).

    F-P2B (R259): `shakedown_20260807.yaml` joins run5 on the EXCLUDED side, on run5's own
    grounds — it mints run5's CARD-RUN5-GPU-OOM caps (4500000/170000) at run5's batch_size
    256, and those caps exist BECAUSE they bind on the production GPU. Putting it through
    the "caps do not bind" sweep would assert the opposite of the caps' purpose."""
    live = sorted(p.relative_to(_CONFIGS).as_posix() for p in discover_configs(_CONFIGS))
    assert live == sorted((*_NON_RUN5, "run5.yaml", "shakedown_20260807.yaml"))
