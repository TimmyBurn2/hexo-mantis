# >300 justify (R8). The stated LINE COUNT is DELETED, not updated (G-DFIX-4): R8 asks for a
# one-line justification, not a line tally, and a number that must be re-edited whenever a
# sibling row is added will eventually be wrong and then be read as evidence. This file had
# already gone stale on that number once. The justification itself is what R8 wants and it
# stands unchanged below.
# The nine rows are ONE claim — R173's leg-1 seam: `train.checkpoint_interval` is read in
# exactly ONE place, both training-step tails converge on it, and the write it triggers rides
# the ONE stamped writer. Splitting them would separate the two R173 mutation arms (OP-1/OP-2)
# from the dense convergence they are converged WITH (OP-3), and would fork the graph harness
# (tiny GnnNet + real HexgBuffer + the real declared dispatcher) into copies that drift while
# both stay green — the fork-and-drift failure `test_clean_stop_save.py:7-11` argues against.
# R5 bars cross-test imports, so the harness is re-created LOCALLY here rather than imported
# from `test_train_step_dispatch.py`. Executable content is a minority: the rest is the
# per-row "what defect is this the only witness to" rationale LAW-07 asks of each row and the
# reachability note R166 asks of each drive.
"""⊕ WP12-R CARD-CS2 / OP-1..OP-9 (R173) — the periodic checkpoint seam.

At `5a519e6` `train.checkpoint_interval` has exactly one reader in `src/`
(`trainer/core.py:487`, inside `_train_on_batch`) and `train_step_from_graph_batch`
(`core.py:493-545`) contains no interval read and no `save_checkpoint` call at all. So on
run5's DECLARED representation (`configs/run5.yaml` `identity.representation: graph`) the
minted knob has no consumer: a graph run cannot checkpoint on a cadence at any value.
R173 cards the fix as ONE resolver — `Trainer._maybe_periodic_checkpoint` — that both step
tails call, and orders the two mutation arms these oracles are: *interval N → checkpoints at
N and 2N* (a step SET, never a count: an off-by-one writes the same NUMBER of files) and
*interval 0 → none but the final*.

The defect each row is the ONLY witness to:

- **OP-1** — R173 arm 1 on the graph route: the card's whole claim, a graph run that never
  checkpoints. Reads the filesystem (count AND step set) and the event stream.
- **OP-2** — R173 arm 2: a cadence that fires when DISABLED, which is the posture every
  committed config ships (`checkpoint_interval: 0` in all six). It also pins that the
  explicit final save emits NO `periodic_checkpoint_save` — the pin that stops the emit
  migrating into `save_checkpoint`, where every leg's write would look periodic.
  **This row asserts an ABSENCE and therefore PASSES before the resolver exists** (PREREG
  §6 G-0b): its ORACLE-FIRST proof is its MUTATION (MP-3, its sole killer), not a pre-fix
  red. Manufacturing a red for it would mean asserting something it should not.
- **OP-3** — the dense leg regressing when its own read is deleted, and the FIRST producer
  test leg 1 has ever had on EITHER route (no test in the tree drives `core.py:487-489` to a
  write; every `checkpoint_interval` occurrence under `tests/` is the literal `0`, the
  COORDINATOR/buffer field, or prose).
- **OP-4** — a SECOND interval authority re-appearing, which no behavioural oracle can see
  (two readers agree right up until they diverge). An `ast.parse` census over `src/mantis/`
  — the composition root and `config/resolve/` included, because those are the two homes a
  second authority would look native in.
- **OP-5** — LAW-08 on run5's OWN wiring: the config, the key, the identity, the hparams
  resolver and the step route are run5's; only the net is tiny. The card's reason to exist.
- **OP-6** — LAW-14: a swallowed periodic-save failure, i.e. a run reporting a cadence it
  never wrote (~200 times over run5). SR-6: every post-condition sits AFTER the
  `pytest.raises` block, because a post-condition inside it is unreachable exactly when a
  mutation kills the raise.
- **OP-7** — rule 3 / LAW-12: an artefact written outside the ONE stamp path, or stamped
  with the wrong step/lineage.
- **OP-8** — the leg-1/leg-3 terminus coincidence (`stop_step % interval == 0`, true on run5
  at the recommended N): leg 1 silently breaking Phase CS's exactly-once, and a
  timestamp-RESOLUTION change collapsing the two same-step artefacts into one overwrite.
- **OP-9** — LAW-18 ordering: the event carries the WRITER's returned path, so the emit
  follows the write. `loop.py:122`'s pre-emit shape copied here would put a claim of a save
  in the stream on every failed write.

**What is real and what is not.** Real everywhere: the interval, the step counter, the
predicate, `Trainer.save_checkpoint`, `checkpoints.save_checkpoint`, the stamp, the
filesystem. Fake: the ARCH (tiny nets) and the SINK (a spy) in OP-1/2/3/6/7/9; OP-5 adds a
tiny arch to run5's own wiring; OP-8 additionally fakes the worker pool. **No row fakes the
writer or the filesystem.** OP-6's only rig is a monkeypatched `torch.save` raising
`OSError` — the seam `tests/train/test_lifecycle_contract.py:258` already uses.
"""
from __future__ import annotations

import ast
import collections
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis.monitor.config import MonitorConfig
from mantis._engine import HexgBuffer, ReplayBuffer
from mantis.config.loader import load_config
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.train import checkpoints
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.dispatch import resolve_step_spec, run_declared_train_step
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.train.trainer.core import Trainer, TrainHParams

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_CORE = _SRC / "mantis" / "train" / "trainer" / "core.py"

GRAPH_ENCODING = "gnn_axis_v1"
_GSPEC = lookup(GRAPH_ENCODING)
GRID_ENCODING = "v6_live2_ls"
_DSPEC = lookup(GRID_ENCODING)

#: The two step-function tails the ONE resolver converges.
_DENSE_STEP = "_train_on_batch"
_GRAPH_STEP = "train_step_from_graph_batch"
_RESOLVER = "_maybe_periodic_checkpoint"
_EVENT = "periodic_checkpoint_save"

#: G-DFIX-2 (WP12-R F2): `run_declared_train_step` gained a REQUIRED `caps_provider` and the
#: two drives below pass it. The values sit far past anything this file's fixtures can build,
#: so neither drive splits and every R173 cadence assertion exercises exactly what it did.
def _NON_BINDING_CAPS() -> MicrobatchCapsSpec:
    return MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)


#: `{run_id}_{step:08d}_{sha8}.ckpt` (`checkpoints.py:197`), decoded STRICTLY: `run_id` may
#: itself carry underscores, so a naive `split("_")` would decode the wrong field on a
#: renamed run. A name that does not match is an ERROR here, never a skipped file.
_CKPT_NAME = re.compile(r"\A(?P<run_id>.+)_(?P<step>\d{8})_(?P<sha8>[0-9a-f]{8})\.ckpt\Z")


# ── filename / census helpers ────────────────────────────────────────────────────────────
def _step_of(path: Path) -> int:
    m = _CKPT_NAME.fullmatch(path.name)
    assert m is not None, f"{path.name!r} is not a `{{run_id}}_{{step:08d}}_{{sha8}}.ckpt` name"
    return int(m.group("step"))


def _steps_of(paths: list[Path]) -> list[int]:
    return [_step_of(p) for p in paths]


def _source_of(src: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(src, node)
    assert seg is not None, f"ast.get_source_segment returned no text for {ast.dump(node)[:80]}"
    return seg


def _enclosing_fn(fns: list[Any], lineno: int) -> str:
    """The INNERMOST function whose span contains `lineno`. A read at module scope has no
    enclosing function and RAISES here rather than being counted under a placeholder — a
    census that silently buckets an unattributable read is a census that can be defeated."""
    best_line, best_name = -1, None
    for fn in fns:
        end = fn.end_lineno
        assert end is not None, f"ast reported no end_lineno for {fn.name!r}"
        if fn.lineno <= lineno <= end and fn.lineno > best_line:
            best_line, best_name = fn.lineno, fn.name
    assert best_name is not None, f"no enclosing function contains line {lineno}"
    return best_name


def _interval_read_census() -> collections.Counter:
    """`Counter` over `(module, receiver_source, enclosing_function)` for every
    `ast.Attribute` in `src/mantis/` whose `attr` is `checkpoint_interval`. A COUNTER, not a
    set: two identical triples (`coordinator/step.py:481,482`) are two reads, and a second
    read planted inside one function must move a COUNT rather than vanish into a dedupe."""
    counts: collections.Counter = collections.Counter()
    for path in sorted((_SRC / "mantis").rglob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        fns = [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "checkpoint_interval":
                counts[(path.relative_to(_REPO).as_posix(),
                        _source_of(src, node.value),
                        _enclosing_fn(fns, node.lineno))] += 1
    return counts


def _core_functions(name: str) -> list[Any]:
    tree = ast.parse(_CORE.read_text())
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]


def _the_core_function(name: str) -> Any:
    hits = _core_functions(name)
    assert len(hits) == 1, f"core.py defines {len(hits)} functions named {name!r}, expected 1"
    return hits[0]


def _calls_in(fn: Any, receiver: str, attr: str) -> int:
    """Count `<receiver>.<attr>(...)` Call nodes inside `fn`'s span. A grep cannot tell a
    reader from a `pop` (R93/DR-11) — hence AST."""
    n = 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == attr
                and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver):
            n += 1
    return n


# ── drive builders (re-created LOCALLY — R5 bars cross-test imports) ─────────────────────
def _graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real HexgBuffer fed through the real graph push path."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


def _dense_buffer(n_records: int = 8, capacity: int = 64) -> ReplayBuffer:
    rb = ReplayBuffer(capacity, GRID_ENCODING)
    s = int(_DSPEC.board_size)
    n_cells = s * s
    for i in range(n_records):
        state = np.zeros((int(_DSPEC.n_planes), s, s), dtype=np.float16)
        state[0, 0, i % s] = 1.0
        chain = np.zeros((6, s, s), dtype=np.float16)
        policy = np.zeros(int(_DSPEC.policy_stride), dtype=np.float32)
        policy[i % n_cells] = 1.0
        own = np.zeros(n_cells, dtype=np.uint8)
        wl = np.zeros(n_cells, dtype=np.uint8)
        rb.push(state, chain, policy, 1.0 if i % 2 == 0 else -1.0, own, wl)
    return rb


def _graph_arch() -> GnnArch:
    return GnnArch(in_dim=_GSPEC.node_feat_dim, edge_dim=_GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)


def _graph_trainer(tmp_path, config, hparams, sink) -> Trainer:
    torch.manual_seed(20260803)
    arch = _graph_arch()
    return Trainer(build_net(arch), config, arch=arch, checkpoint_dir=tmp_path,
                   device=torch.device("cpu"), train_hparams=hparams, sink=sink)


def _drive_graph(trainer: Trainer, spec: Any, n_steps: int) -> None:
    """`n_steps` REAL gradient updates through the REAL declared dispatcher
    (`run_declared_train_step` → `_graph_step` → `train_step_from_graph_batch`), which is the
    route `coordinator/step.py`'s burst loop takes. Reachability (R166): the tail calls the
    resolver UNCONDITIONALLY, so the resolver is an executed statement on every step; whether
    it WRITES is the predicate under test."""
    buffer = _graph_buffer()
    for _ in range(n_steps):
        run_declared_train_step(trainer, buffer, spec, batch_size=4, augment=False,
                                recency_weight=0.0, recent_buffer=None,
                                caps_provider=_NON_BINDING_CAPS)


def _drive_dense(trainer: Trainer, n_steps: int) -> None:
    buffer = _dense_buffer()
    for _ in range(n_steps):
        run_declared_train_step(trainer, buffer, _DSPEC, batch_size=4, augment=False,
                                recency_weight=0.0, recent_buffer=None,
                                caps_provider=_NON_BINDING_CAPS)


# ── OP-1 ⊕ — R173 arm 1: interval N → checkpoints at N and 2N (the GRAPH route) ──────────
def test_graph_periodic_checkpoints_land_at_n_and_2n(tmp_path, mk_config, full_train_hparams,
                                                     spy_sink) -> None:
    config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=2),
                             spy_sink)
    _drive_graph(trainer, _GSPEC, 4)
    residents = sorted(tmp_path.glob("*.ckpt"))
    events = spy_sink.named(_EVENT)

    assert trainer.step == 4, "premise: four real gradient updates ran"          # 1
    assert len(residents) == 2, f"N=2 over 4 steps writes exactly 2, got {residents}"  # 2
    assert set(_steps_of(residents)) == {2, 4}, (                                # 3
        "R173's arm is *at N and 2N* — a COUNT is blind to an off-by-one, which writes the "
        f"same NUMBER of files at {{1, 3}}; got {_steps_of(residents)}")
    assert len(events) == 2, f"one {_EVENT} per write (LAW-18), got {events}"    # 4
    assert [e["step"] for e in events] == [2, 4]                                 # 5
    assert [e["interval"] for e in events] == [2, 2]                             # 6
    assert [e["representation"] for e in events] == ["graph", "graph"]           # 7


# ── OP-2 ⊕ — R173 arm 2: interval 0 → none but the final ────────────────────────────────
def test_zero_interval_writes_nothing_and_the_final_save_is_not_periodic(
        tmp_path, mk_config, full_train_hparams, spy_sink) -> None:
    """⊕-EXEMPT from the pre-fix RED gate BY NAME (PREREG §6 G-0b): this row asserts the
    ABSENCE of periodic behaviour, so it passes at `5a519e6` — correctly. Its ORACLE-FIRST
    proof is its MUTATION: MP-3 (a guard that fires on every step when the cadence is
    disabled) is its sole killer, and executing it is what shows the row can fail."""
    config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=0),
                             spy_sink)
    _drive_graph(trainer, _GSPEC, 4)

    assert trainer.step == 4, "premise: four real gradient updates ran"          # 1
    assert len(sorted(tmp_path.glob("*.ckpt"))) == 0, (                          # 2
        "`0` disables the cadence — the value every committed config mints today")
    assert spy_sink.named(_EVENT) == []                                          # 3

    trainer.save_checkpoint({"loss": 0.0})   # stands for the final save (leg 2 / leg 3)
    residents = sorted(tmp_path.glob("*.ckpt"))
    assert len(residents) == 1, f"the explicit final save still writes, got {residents}"  # 4
    assert _steps_of(residents) == [4]                                           # 5
    assert spy_sink.named(_EVENT) == [], (                                       # 6
        "the final save must NOT emit a periodic event — an emit that migrated into "
        "`save_checkpoint` would make every leg's write look periodic in the stream")


# ── OP-3 ⊕ — the dense convergence: the same N/2N property through the grid route ────────
def test_dense_periodic_checkpoints_land_at_n_and_2n(tmp_path, mk_config, full_train_hparams,
                                                     tiny_arch, spy_sink) -> None:
    torch.manual_seed(20260803)
    trainer = Trainer(build_net(tiny_arch), mk_config(), arch=tiny_arch,
                      checkpoint_dir=tmp_path, device=torch.device("cpu"),
                      train_hparams=full_train_hparams(checkpoint_interval=2), sink=spy_sink)
    _drive_dense(trainer, 4)
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert trainer.step == 4, "premise: four real gradient updates ran"          # 1
    assert len(residents) == 2, f"N=2 over 4 dense steps writes exactly 2, got {residents}"  # 2
    assert set(_steps_of(residents)) == {2, 4}                                   # 3
    assert [e["representation"] for e in spy_sink.named(_EVENT)] == ["grid", "grid"]  # 4


# ── OP-4 ⊕ — the ONE-AUTHORITY census (`ast.parse` over src/mantis/, no import, no grep) ─
def test_exactly_one_checkpoint_interval_authority_in_src() -> None:
    """R173's most explicit prohibition — *no second interval authority* — is a property of
    the SOURCE, not of a run: two readers agree until they diverge, so no behavioural oracle
    can see one appear. The scope is `src/mantis/`, not `src/mantis/train/`, because the two
    homes a second authority would look NATIVE in are the composition root (`run.py` already
    reads something spelled `checkpoint_interval` — the BUFFER key) and `config/resolve/`
    (which §2.5(iii) rejects in prose). The expectation is line-free, so it cannot go stale.

    Honest limit, stated rather than implied: a read spelled differently
    (`getattr(self.hp, "checkpoint" + "_interval")`, an alias captured at `__init__`) defeats
    it. The census raises the cost of a second authority; it does not make one impossible."""
    # WP12-R R178(a) deleted `train.buffer_save_interval`, the replay-BUFFER cadence, and
    # with it `StepCoordinatorConfig.checkpoint_interval` and both no-op `_try_save_buffer`
    # arms. The two reads that vanish are exactly the BUFFER key's — `run.py`'s transport of
    # `knobs.checkpoint_interval` and `step.py`'s D4 gate (which read `cfg.checkpoint_interval`
    # twice on one line). The TRAINER's authority, the row this census exists to protect, is
    # UNCHANGED at one: `core.py`'s `self.hp` read inside the resolver. The census is now
    # STRONGER — a same-spelled second reader anywhere in `src/mantis/` reds it, where before
    # two legitimate buffer-key reads sat in the expectation as noise.
    _EXPECTED_READS = collections.Counter({
        ("src/mantis/train/trainer/core.py", "self.hp", _RESOLVER): 1,
    })
    resolvers = _core_functions(_RESOLVER)
    # 1
    assert len(resolvers) == 1, (
        f"premise: `Trainer.{_RESOLVER}` must exist exactly once in core.py, found "
        f"{len(resolvers)}")
    # 2
    assert _interval_read_census() == _EXPECTED_READS, (
        "a `train.checkpoint_interval` read moved, appeared, or was duplicated")
    dense, graph = _the_core_function(_DENSE_STEP), _the_core_function(_GRAPH_STEP)
    dense_saves = _calls_in(dense, "self", "save_checkpoint")
    graph_saves = _calls_in(graph, "self", "save_checkpoint")
    dense_calls = _calls_in(dense, "self", _RESOLVER)
    graph_calls = _calls_in(graph, "self", _RESOLVER)
    # 3
    assert (dense_saves, graph_saves) == (0, 0), (
        "neither step body may hold its own save trigger — both go through the resolver")
    # 4
    assert (dense_calls, graph_calls) == (1, 1), (
        "each step tail calls the ONE resolver exactly once")
    # 5
    assert _calls_in(resolvers[0], "self", "save_checkpoint") == 1, (
        "the resolver writes through `Trainer.save_checkpoint` — the SAME entry legs 2 and 3 "
        "call — exactly once (rule 3 / LAW-12: no second write surface)")


# ── OP-5 ⊕ — LAW-08 on run5's OWN config, key, identity, resolver and route ──────────────
def test_run5_config_produces_a_periodic_checkpoint_on_its_declared_route(
        tmp_path, spy_sink) -> None:
    """The card's reason to exist (R173(4)): a minted key with zero live consumers on the
    representation its own config declares. Everything here is run5's except the net's size —
    the config comes through the real loader, the KEY travels through the REAL
    `TrainHParams.from_config`, the spec through THE resolver, the step through the REAL
    dispatcher. The interval is overridden IN MEMORY ONLY; `configs/` is read-only to this
    card (R119) and R165 reserves the re-mint to the operator."""
    d = load_config(_REPO / "configs" / "run5.yaml").model_dump()
    d["train"]["checkpoint_interval"] = 2
    hp = TrainHParams.from_config(d)
    spec = resolve_step_spec(d)
    trainer = _graph_trainer(tmp_path, d, hp, spy_sink)
    _drive_graph(trainer, spec, 4)
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert (d["identity"]["representation"], spec.representation) == ("graph", "graph"), (  # 1
        "premise: `graph` is the route run5 DECLARES — without this the row is about nothing")
    assert hp.checkpoint_interval == 2, (                                        # 2
        "premise: the KEY reached the runtime hparams through the production resolver")
    assert len(residents) == 2, f"run5's route must honour its own cadence, got {residents}"  # 3
    assert set(_steps_of(residents)) == {2, 4}                                   # 4


# ── OP-6 ⊕ — LAW-14: a rigged writer failure PROPAGATES and is counted exactly once ──────
def test_periodic_save_failure_propagates_and_counts(tmp_path, mk_config, full_train_hparams,
                                                     spy_sink, monkeypatch) -> None:
    """`persist_errors_total` is a process-wide module GLOBAL and the `global … += 1` under
    test cannot be undone by an assertion; the monkeypatch pins it to 0 AND restores the
    pre-test value at teardown, so the increment cannot leak into a later suite (the
    heartbeat watchdog's persist-fatal rule is the literal `> 0`)."""
    monkeypatch.setattr(checkpoints, "persist_errors_total", 0)
    before = checkpoints.persist_errors_total

    def _boom(*_a: Any, **_k: Any) -> None:
        raise OSError("rigged")

    config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=1),
                             spy_sink)
    monkeypatch.setattr(torch, "save", _boom)

    assert trainer.step == 0 and before == 0, "premise: fresh trainer, counter snapshotted"  # 1
    with pytest.raises(OSError, match="rigged") as excinfo:                      # 2
        _drive_graph(trainer, _GSPEC, 1)
    # SR-6: every post-condition sits AFTER the block. Inside it they are unreachable exactly
    # when a mutation kills the raise, which is the case they exist to observe.
    assert checkpoints.persist_errors_total - before == 1, (                     # 3
        f"LAW-14: the write path counts the failure once and re-raises (saw {excinfo.value!r})")
    assert sorted(tmp_path.glob("*.ckpt")) == []                                 # 4
    assert spy_sink.named(_EVENT) == [], (                                       # 5
        "no event may claim a save that never happened — the emit follows the write")


# ── OP-7 ⊕ — rule 3 / LAW-12: the periodic artefact rides the ONE stamp path ─────────────
def test_periodic_artefact_loads_through_the_one_loader_with_its_stamp(
        tmp_path, mk_config, full_train_hparams, spy_sink) -> None:
    """ANTI-TAUTOLOGY (ORACLE_NOTES §attack): the run_id is DELIBERATELY not the factory's
    default `"run5"`. With the default, `metadata.run_id == config["run_id"]` is satisfied by
    a stamp that hardcoded the string, which is the assertion-that-cannot-fail class. A
    distinct id makes #5 a real read of the lineage the config declares. It moves no
    mutation mechanism — MP-10 reds this row at #1 and MP-5 at #3, neither of which touches
    the run_id."""
    config = mk_config(GRAPH_ENCODING, "graph", "op7_lineage")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=2),
                             spy_sink)
    _drive_graph(trainer, _GSPEC, 2)
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert len(residents) == 1, f"premise: one boundary crossed at N=2, got {residents}"  # 1
    ck = checkpoints.load_checkpoint(residents[0])   # raises on any provenance/stamp defect
    assert ck.kind == "full", "the periodic write is the FULL envelope, not a weights strip"  # 2
    assert ck.metadata.step == 2                                                 # 3
    assert ck.metadata.encoding_name == GRAPH_ENCODING                           # 4
    assert ck.metadata.run_id == config["run_id"]                                # 5
    assert ck.metadata.created_utc.endswith("Z") and len(ck.metadata.created_utc) > 1, (  # 6
        f"the stamp is built once and is ISO-Z, got {ck.metadata.created_utc!r}")


# ── OP-8 ⊕ — the leg-1 / leg-3 terminus coincidence, on a REAL StepCoordinator ───────────
def _coord_cfg(**over: Any) -> StepCoordinatorConfig:
    """The COORDINATOR's replay-BUFFER cadence field `checkpoint_interval` USED to be set
    here at 0, to keep the D4 buffer save out of this drive. WP12-R R178(a) deleted the field,
    its `train.buffer_save_interval` key and the D4 arm itself (the save was measured
    production-dead on every leg, F-CS-2), so there is nothing left to zero. The trainer's own
    `checkpoint_interval` is a DIFFERENT key and is unaffected — it is passed via
    `full_train_hparams`, not here."""
    base: dict[str, Any] = dict(
        # R242 (ADJ-D12): `gate_interval` joins `log_interval` — the ARMING cadence,
        # split off the narration one. 0 on both here for the same reason: this drive
        # is about the training step / the checkpoint terminus, not about either
        # boundary, and 0 keeps both quiet.
        eval_interval=0, log_interval=0, gate_interval=0, min_buf_size=1,
        capacity=64, buffer_schedule=(), training_steps_per_game=4.0, max_train_burst=4,
        batch_size=4, augment=False, recency_weight=0.0, mixing_initial_w=0.0,
        mixing_min_w=0.0, mixing_decay_steps=1.0, hard_gn_threshold=1e9,
        hard_gn_min_steps=10_000, stop_step=4, draw_rate_abort=None,
        final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
        eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0,
        terminal_eval_enabled=False, bot_batch_share=0.0,
        selfplay_stall_timeout_sec=1800.0,
    )
    base.update(over)
    return StepCoordinatorConfig(**base)


class _RunnerStats:
    """Minimal `RunnerStats` surface for `emit_iteration_complete_event`."""
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    """Minimal WorkerPoolLike stand-in for driving step() past O4/O5 (not the subject).

    WP12R Step 3 narration: gained the `PoolTelemetryLike` surface `iteration_complete`
    reads (`runner_stats`, `avg_game_length`, `x_winrate`, `o_winrate`, `draws`,
    `sims_per_sec`, `batch_fill_pct`, `gumbel_mcts`) because `iteration_complete` now emits
    per-burst (every O6 return) instead of only at `log_interval` boundaries, so this stub
    must satisfy `emit_iteration_complete_event` on every `step()` call.
    """

    def __init__(self, games_completed: int = 3) -> None:
        self.games_completed = games_completed
        self.n_workers = 1
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def runner_stats(self) -> Any:
        return _RunnerStats()


def test_terminus_holds_two_artefacts_and_leg_three_stays_exactly_once(
        tmp_path, mk_config, full_train_hparams, spy_sink) -> None:
    """`stop_step % interval == 0` is run5's real terminus at the architect's recommended N
    (1000000 % 5000 == 0). The burst writes a PERIODIC artefact at the ceiling step; the next
    `step()` takes the O2 arm and leg 3 writes the CLEAN-COMPLETION artefact at the SAME
    step. They are two different facts ("a resumption point" vs "the run FINISHED") and two
    distinct files — distinct ONLY because `metadata.created_utc` carries a sub-second field
    (`checkpoints.py:119-122`) which enters `content_sha8` and thence the filename. At second
    resolution the two payloads would hash identically and the second write would OVERWRITE
    the first, halving this row's subject intermittently on a fast host — which is why the
    resolution is asserted on the PRODUCER, before the count."""
    full_config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, full_config, full_train_hparams(checkpoint_interval=4),
                             spy_sink)
    coord = StepCoordinator(
        monitor_cfg=MonitorConfig(),
        trainer=trainer, buffer=_graph_buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=_Pool(), eval_pipeline=None, subsystems=None, anchor_state=None,
        shutdown=ShutdownState(), eval_model=None, bufs=None, config=_coord_cfg(),
        full_config=full_config, sink=spy_sink,
    )
    calls = 0
    while coord.shutdown.running and calls < 10:
        coord.step()
        calls += 1
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert coord.shutdown.running is False, (                                    # 1
        f"premise: the O2 clean-completion arm must fire within 10 step() calls (ran {calls})")
    assert trainer.step == 4, "premise: the burst reached the declared terminus"  # 2
    assert re.search(r"\.\d+Z$", checkpoints._now_iso()) is not None, (           # 3
        "the timestamp-RESOLUTION premise: `_now_iso()` must carry a sub-second field, or "
        "two same-step payloads hash identically and the second OVERWRITES the first")
    assert len(residents) == 2, f"leg 1 AND leg 3 both write at the terminus, got {residents}"  # 4
    assert _steps_of(residents) == [4, 4]                                        # 5
    assert len(spy_sink.named("clean_stop_save")) == 1, (                        # 6
        "Phase CS's exactly-once property is untouched: leg 1 neither reads nor sets the latch")
    periodic = spy_sink.named(_EVENT)
    assert len(periodic) == 1 and periodic[0]["step"] == 4                       # 7
    assert coord.clean_stop_saved is True                                        # 8
    assert coord.shutdown.abort_rule is None, "a clean stop is not an abort"     # 9


# ── OP-9 ⊕ — LAW-18: the event carries the WRITER's returned path ────────────────────────
def test_periodic_event_carries_the_writers_returned_path(tmp_path, mk_config,
                                                          full_train_hparams,
                                                          spy_sink) -> None:
    """The emit follows the write and publishes what the WRITER returned, never a path
    re-derived from the checkpoint dir before the fact. `loop.py:122` emits BEFORE its own
    save; copied here, a failed write would still put a claim of a save in the stream — on
    every boundary, ~200 times over run5, instead of once."""
    config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=1),
                             spy_sink)
    _drive_graph(trainer, _GSPEC, 1)
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert len(residents) == 1, f"premise: interval 1 writes on the first step, got {residents}"
    events = spy_sink.named(_EVENT)
    assert len(events) == 1, f"exactly one {_EVENT}, got {events}"
    assert events[0]["path"] == str(residents[0])
