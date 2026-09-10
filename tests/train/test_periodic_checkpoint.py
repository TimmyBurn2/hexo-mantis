# >300 justify (R8). The rows are ONE claim: `train.checkpoint_interval` is read in exactly ONE
# place, both training-step tails converge on it, and the write it triggers rides the ONE
# stamped writer. Splitting them would separate the mutation arms from the dense convergence
# they are converged WITH, and would fork the graph harness into copies that drift while both
# stay green; the bar on cross-test imports is why the harness is local rather than shared.
"""ORACLE — the periodic checkpoint seam.

`train.checkpoint_interval` had exactly one reader in `src/`, inside the dense step tail, while
the graph step held no interval read and no `save_checkpoint` call — so on run5's DECLARED graph
representation the minted knob had no consumer at any value. The fix is ONE resolver both step
tails call, with two arms: interval N writes at N and 2N (a step SET, never a count — an
off-by-one writes the same NUMBER of files), and interval 0 writes none but the final.

Each row is the only witness to one defect: a graph run that never checkpoints; a cadence that
fires when DISABLED (asserted as an ABSENCE, so its oracle-first proof is its mutation); the
dense leg regressing; a SECOND interval authority, which no behavioural oracle can see because
two readers agree until they diverge; the live-consumer claim on run5's own wiring; a swallowed
save failure; an artefact outside the ONE stamp path; the leg-1/leg-3 terminus coincidence; and
the event carrying the WRITER's returned path. Real everywhere except the ARCH and the SINK.
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
from mantis._engine import HexgBuffer
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

#: The two step-function tails the ONE resolver converges.
_GRAPH_STEP = "train_step_from_graph_batch"
_RESOLVER = "_maybe_periodic_checkpoint"
_EVENT = "periodic_checkpoint_save"

#: `{run_id}_{step:08d}_{sha8}.ckpt`, decoded STRICTLY: `run_id` may itself carry underscores,
#: so a naive split would decode the wrong field. A non-matching name is an ERROR, never a skip.
def _NON_BINDING_CAPS() -> MicrobatchCapsSpec:
    return MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)


#: `run_declared_train_step` requires a `caps_provider`; these values sit far past anything
#: this file's fixtures can build, so neither drive splits.
_CKPT_NAME = re.compile(r"\A(?P<run_id>.+)_(?P<step>\d{8})_(?P<sha8>[0-9a-f]{8})\.ckpt\Z")


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
    """The INNERMOST function whose span contains `lineno`. A module-scope read RAISES rather
    than bucketing under a placeholder — a census that silently buckets an unattributable read
    is a census that can be defeated."""
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
    `checkpoint_interval` attribute read in `src/mantis/`. A COUNTER, not a set: a second read
    planted inside one function must move a COUNT rather than vanish into a dedupe."""
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
    """Count `<receiver>.<attr>(...)` calls inside `fn`'s span. A grep cannot tell a reader
    from a `pop`, hence AST."""
    n = 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == attr
                and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver):
            n += 1
    return n


def _graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real HexgBuffer fed through the real graph push path."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


def _graph_arch() -> GnnArch:
    return GnnArch(in_dim=_GSPEC.node_feat_dim, edge_dim=_GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)


def _graph_trainer(tmp_path, config, hparams, sink) -> Trainer:
    torch.manual_seed(20260803)
    arch = _graph_arch()
    return Trainer(build_net(arch), config, arch=arch, checkpoint_dir=tmp_path,
                   device=torch.device("cpu"), train_hparams=hparams, sink=sink)


def _drive_graph(trainer: Trainer, spec: Any, n_steps: int) -> None:
    """`n_steps` REAL gradient updates through the REAL declared dispatcher, the route the
    burst loop takes. Reachability: the tail calls the resolver UNCONDITIONALLY, so it is an
    executed statement on every step; whether it WRITES is the predicate under test."""
    buffer = _graph_buffer()
    for _ in range(n_steps):
        run_declared_train_step(trainer, buffer, spec, batch_size=4, augment=False,
                                recency_weight=0.0, recent_buffer=None,
                                caps_provider=_NON_BINDING_CAPS, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)


# arm 1: interval N writes at N and 2N (the GRAPH route)
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


# arm 2: interval 0 writes none but the final
def test_zero_interval_writes_nothing_and_the_final_save_is_not_periodic(
        tmp_path, mk_config, full_train_hparams, spy_sink) -> None:
    """Exempt from the pre-fix RED gate by name: this row asserts the ABSENCE of periodic
    behaviour, so it passes before the fix — correctly. Its oracle-first proof is its mutation,
    a guard that fires on every step when the cadence is disabled."""
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


# the ONE-AUTHORITY census (`ast.parse` over src/mantis/, no import, no grep)
def test_exactly_one_checkpoint_interval_authority_in_src() -> None:
    """No second interval authority is a property of the SOURCE, not of a run: two readers agree
    until they diverge, so no behavioural oracle can see one appear. The scope is `src/mantis/`
    because the two homes a second authority would look NATIVE in are the composition root and
    `config/resolve/`, and the expectation is line-free so it cannot go stale. Honest limit: a
    differently-spelled read defeats it — this raises the cost, not the possibility."""
    # The replay-BUFFER key and its two reads are deleted, so the TRAINER's authority — the
    # row this census protects — stands alone at one, and a same-spelled second reader anywhere
    # in `src/mantis/` now reds it where before two legitimate buffer reads sat as noise.
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
    graph = _the_core_function(_GRAPH_STEP)
    graph_saves = _calls_in(graph, "self", "save_checkpoint")
    graph_calls = _calls_in(graph, "self", _RESOLVER)
    # 3
    assert graph_saves == 0, (
        "the step body may not hold its own save trigger — it goes through the resolver")
    # 4
    assert graph_calls == 1, "the step tail calls the ONE resolver exactly once"
    # 5
    assert _calls_in(resolvers[0], "self", "save_checkpoint") == 1, (
        "the resolver writes through `Trainer.save_checkpoint` — the SAME entry legs 2 and 3 "
        "call — exactly once (rule 3 / LAW-12: no second write surface)")


# the live-consumer claim on run5's OWN config, key, identity, resolver and route
def test_run5_config_produces_a_periodic_checkpoint_on_its_declared_route(
        tmp_path, spy_sink) -> None:
    """A minted key with zero live consumers on the representation its own config declares.
    Everything here is run5's except the net's size — the config comes through the real loader,
    the key through the real hparams resolver, the spec through THE resolver, the step through
    the real dispatcher. The interval is overridden IN MEMORY ONLY; `configs/` is read-only
    here."""
    d = load_config(_REPO / "configs" / "run6.yaml").model_dump()
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


# a rigged writer failure PROPAGATES and is counted exactly once
def test_periodic_save_failure_propagates_and_counts(tmp_path, mk_config, full_train_hparams,
                                                     spy_sink, monkeypatch) -> None:
    """`persist_errors_total` is a process-wide module GLOBAL and the increment under test
    cannot be undone by an assertion, so the monkeypatch pins it to 0 AND restores the pre-test
    value at teardown — the persist-fatal rule is a literal `> 0`."""
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
    # Every post-condition sits AFTER the block: inside it they are unreachable exactly when a
    # mutation kills the raise, which is the case they exist to observe.
    assert checkpoints.persist_errors_total - before == 1, (                     # 3
        f"LAW-14: the write path counts the failure once and re-raises (saw {excinfo.value!r})")
    assert sorted(tmp_path.glob("*.ckpt")) == []                                 # 4
    assert spy_sink.named(_EVENT) == [], (                                       # 5
        "no event may claim a save that never happened — the emit follows the write")


# the periodic artefact rides the ONE stamp path
def test_periodic_artefact_loads_through_the_one_loader_with_its_stamp(
        tmp_path, mk_config, full_train_hparams, spy_sink) -> None:
    """ANTI-TAUTOLOGY: the run_id is DELIBERATELY not the factory default. With the default,
    `metadata.run_id == config["run_id"]` is satisfied by a stamp that hardcoded the string —
    the assertion-that-cannot-fail class. A distinct id makes it a real read of the declared
    lineage, and it moves no mutation mechanism."""
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


# the leg-1 / leg-3 terminus coincidence, on a REAL StepCoordinator
def _coord_cfg(**over: Any) -> StepCoordinatorConfig:
    """The coordinator's replay-BUFFER cadence field USED to be zeroed here to keep the D4
    buffer save out of this drive; the field, its config key and the D4 arm are all deleted, so
    there is nothing left to zero. The trainer's own `checkpoint_interval` is a DIFFERENT key,
    passed through `full_train_hparams`."""
    base: dict[str, Any] = dict(
        # 0 on both cadence knobs: this drive is about the training step and the checkpoint
        # terminus, not about either boundary, and 0 keeps both quiet.
        eval_interval=0, log_interval=0, gate_interval=0, min_buf_size=1,
        capacity=64, buffer_schedule=(), training_steps_per_game=4.0, max_train_burst=4,
        batch_size=4, augment=False, recency_weight=0.0, hard_gn_threshold=1e9,
        hard_gn_min_steps=10_000, stop_step=4, draw_rate_abort=None,
        final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
        eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0,
        terminal_eval_enabled=False,
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
    """Minimal WorkerPoolLike stand-in for driving step() past O4/O5 (not the subject). Carries
    the full telemetry surface because `iteration_complete` emits per burst rather than only at
    `log_interval` boundaries."""

    def __init__(self, games_completed: int = 3) -> None:
        self.games_completed = games_completed
        self.n_workers = 1
        self.search_kind = "gumbel"
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
    """`stop_step % interval == 0` is run5's real terminus at the recommended N: the burst writes
    a PERIODIC artefact at the ceiling step and the next `step()` takes the O2 arm, so leg 3
    writes the CLEAN-COMPLETION artefact at the SAME step. Two facts, two files, distinct ONLY
    because `created_utc` carries a sub-second field that enters the content hash — at second
    resolution the second write would OVERWRITE the first, which is why the resolution is
    asserted on the PRODUCER first."""
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


# the event carries the WRITER's returned path
def test_periodic_event_carries_the_writers_returned_path(tmp_path, mk_config,
                                                          full_train_hparams,
                                                          spy_sink) -> None:
    """The emit follows the write and publishes what the WRITER returned, never a path
    re-derived from the checkpoint dir before the fact. A pre-emit would put a claim of a save
    in the stream on every failed write, once per boundary instead of once."""
    config = mk_config(GRAPH_ENCODING, "graph")
    trainer = _graph_trainer(tmp_path, config, full_train_hparams(checkpoint_interval=1),
                             spy_sink)
    _drive_graph(trainer, _GSPEC, 1)
    residents = sorted(tmp_path.glob("*.ckpt"))

    assert len(residents) == 1, f"premise: interval 1 writes on the first step, got {residents}"
    events = spy_sink.named(_EVENT)
    assert len(events) == 1, f"exactly one {_EVENT}, got {events}"
    assert events[0]["path"] == str(residents[0])
