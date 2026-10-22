"""`selfplay.decided_tails.value_weight`: a decided tail's kept value row pulls w of its z CE, every other row 1, over the row count."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _microbatch_harness as H
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from mantis._engine import HexgBuffer
from mantis.config.resolve.decided_tails import DecidedTailsShapeError
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.config.schema import RunConfig
from mantis.selfplay.graph_wire_split import plan_microbatches
from mantis.train import losses
from mantis.train.coordinator.dispatch import run_declared_eval_step, run_declared_train_step
from mantis.train.trainer.core import TrainHParams

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_VISITS = [(2, 0, 0.6), (1, 1, 0.4)]
_ONE_PART = MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)
_ROWS = 32
_TACTICS = {"kind": "strict_turn", "leaf_turns": 3, "leaf_nodes": 256, "root_turns": 8, "root_nodes": 20000,
            "audit": None}


def _ring() -> HexgBuffer:
    """Every fourth row a decided tail's value-only row the side to move loses; every other row searched and won."""
    buf = HexgBuffer(2 * _ROWS, H.GRAPH_ENCODING, 128)
    for i in range(_ROWS):
        if i % 4 == 0:
            buf.push_graph_position(_STONES, [], 1, 1, 4 + i, False, -1.0, True, 40, game_id=i, proven_tail=True)
        else:
            buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, 1.0, True, 10 + i, game_id=i)
    buf.seed_sampler(H.SEED)
    return buf


class _TargetsView:
    """The sampler's targets with named members replaced; every other member read through."""

    def __init__(self, targets: Any, **over: Any) -> None:
        self._targets, self._over = targets, over

    def __getattr__(self, name: str) -> Any:
        over = self.__dict__["_over"]
        return over[name] if name in over else getattr(self.__dict__["_targets"], name)


def _replay(**over: Any) -> H.ReplayWireBuffer:
    replay = H.ReplayWireBuffer(_ring(), _ROWS)
    if over:
        replay._pair = (replay._pair[0], _TargetsView(replay.targets, **over))
    return replay


def _tails(replay: H.ReplayWireBuffer) -> np.ndarray:
    return np.asarray(replay.targets.proven_tail) != 0


def _parts_caps(replay: H.ReplayWireBuffer) -> MicrobatchCapsSpec:
    """Caps that split the replayed wire into several micro-batches (asserted), so each part slices its own weights."""
    ec, nc = H.per_graph_counts(replay.wire)
    caps = MicrobatchCapsSpec(max_edges=int(ec.max()) * 4, max_nodes=int(nc.max()) * 4)
    assert len(plan_microbatches(replay.wire.edge_offsets, replay.wire.node_offsets, caps.max_edges,
                                 caps.max_nodes)) >= 4
    return caps


def _step(tmp_path: Path, replay: Any, weight: float, *, caps: MicrobatchCapsSpec = _ONE_PART, train: bool = True,
          sink: Any = None) -> tuple[dict[str, float], Any]:
    """One production step from the seeded tiny trainer at tail weight `weight`: its losses and its state after."""
    trainer = H.tiny_graph_trainer(tmp_path, tail_value_weight=weight, checkpoint_interval=0, sink=sink)
    if not train:
        return run_declared_eval_step(trainer, replay, H.GSPEC, batch_size=_ROWS, caps_provider=lambda: caps,
                                      sample_threads_provider=lambda: 1), trainer.model.state_dict()
    out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=_ROWS, augment=False,
                                  caps_provider=lambda: caps, sample_threads_provider=lambda: 1)
    return out, trainer.model.state_dict()


def test_the_sampler_carries_each_rows_tail_flag() -> None:
    """The flag rides its own row through the shuffled, augmented draw: tail rows are exactly the lost value-only ones."""
    buf = _ring()
    seen = np.zeros(2, dtype=np.int64)
    for _ in range(8):
        _wire, targets = buf.sample_graph_batch(_ROWS, augment=True, n_threads=2)
        tail = np.asarray(targets.proven_tail) != 0
        assert (tail == (np.asarray(targets.outcomes) == -1.0)).all()
        assert (tail == (np.asarray(targets.is_full_search) == 0)).all()
        seen += [int(tail.sum()), int((~tail).sum())]
    assert seen.min() > 0, f"a draw of 256 rows saw {seen.tolist()} tail/other rows"


def test_weight_one_leaves_the_flag_inert(tmp_path: Path) -> None:
    """At 1 a flagged row trains as any other: the step equals the step over the same rows with no flag set."""
    one, one_state = _step(tmp_path, _replay(), 1.0)
    plain, plain_state = _step(tmp_path, _replay(proven_tail=np.zeros(_ROWS, dtype=np.uint8)), 1.0)
    assert one == plain and all(torch.equal(one_state[k], plain_state[k]) for k in one_state)


def _assert_weighted_sum_over_the_count(tmp_path: Path, w: float, caps: Any = None) -> None:
    """The z CE at `w` is (A + w·B) / (n + t): A, n over the other rows, B, t over the tails, each read at weight 1."""
    replay = _replay()
    caps = _ONE_PART if caps is None else caps(replay)
    tail = _tails(replay)
    n, t = int((~tail).sum()), int(tail.sum())
    masked_vv = np.where(tail, 0, np.asarray(replay.targets.value_valid)).astype(np.uint8)
    others = _step(tmp_path, _replay(value_valid=masked_vv), 1.0, caps=caps)[0]["value_loss"]
    every = _step(tmp_path, replay, 1.0, caps=caps)[0]["value_loss"]
    a, b = n * others, (n + t) * every - n * others
    assert abs(others - every) > 1e-3, "the fixture's tail and other rows read one CE: the check would see nothing"
    got = _step(tmp_path, replay, w, caps=caps)[0]["value_loss"]
    want = (a + w * b) / (n + t)
    assert got == pytest.approx(want, rel=1e-5), f"the z CE at w={w} is {got}, not the weighted sum over the count {want}"


@pytest.mark.parametrize("caps", [None, _parts_caps], ids=["one_part", "several_parts"])
@pytest.mark.parametrize("w", [0.0, 0.1])
def test_a_tails_pull_is_its_weight_over_the_row_count(tmp_path: Path, w: float, caps: Any) -> None:
    """At 0 the tails pull nothing yet stay counted; at 0.1 a tenth; every other row's pull is the weight-1 one."""
    _assert_weighted_sum_over_the_count(tmp_path, w, caps)


def test_a_dispatcher_ignoring_the_flag_reds_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: weights that ignore the flag train the tails at 1 and red the weighted-sum check."""
    monkeypatch.setattr(losses, "tail_value_weights", lambda tail, weight: np.ones(np.asarray(tail).shape,
                                                                                   dtype=np.float32))
    with pytest.raises(AssertionError, match="weighted sum over the count"):
        _assert_weighted_sum_over_the_count(tmp_path, 0.1)


def test_the_eval_step_never_weighs(tmp_path: Path) -> None:
    """A reading must not move with the lever: the forward-only loss at 0.1 is the one at 1, bit for bit."""
    replay = _replay()
    assert _step(tmp_path, replay, 0.1, train=False)[0] == _step(tmp_path, replay, 1.0, train=False)[0]


def test_the_step_event_counts_the_tail_rows_and_echoes_the_weight(tmp_path: Path) -> None:
    """The lever's fire rate: the kept value rows it weighs, beside the weight; the kept count is the weight's own at any w."""
    replay = _replay()
    tail = _tails(replay)
    for w in (0.0, 1.0):
        sink = H.SpySink()
        _step(tmp_path, replay, w, sink=sink)
        event = sink.named("trainer_step")[0]
        assert event["value_rows_tail"] == int(tail.sum()) > 0 and event["tail_value_weight"] == w
        assert event["value_rows_kept"] == event["value_rows_valid"] == _ROWS


def test_the_hparams_read_the_block_and_refuse_any_other_shape() -> None:
    """Off reads 1 (a preloaded tail row then trains as any other); a block reads its weight; a v60 bool is refused by name."""
    config = H.graph_config()
    assert config["selfplay"]["decided_tails"] is None
    assert TrainHParams.from_config(config).tail_value_weight == 1.0
    config["selfplay"]["decided_tails"] = {"value_weight": 0.1}
    assert TrainHParams.from_config(config).tail_value_weight == 0.1
    for bad in (False, True, {"w": 0.1}):
        config["selfplay"]["decided_tails"] = bad
        with pytest.raises(DecidedTailsShapeError, match="decided_tails"):
            TrainHParams.from_config(config)


@pytest.mark.parametrize("w", [-0.1, 1.5, float("nan")], ids=["negative", "above_one", "nan"])
def test_a_hand_built_hparams_refuses_a_weight_off_zero_to_one(w: float) -> None:
    with pytest.raises(ValueError, match="outside"):
        H.graph_hparams(tail_value_weight=w)


def test_a_tail_weight_below_one_refuses_the_search_value_term() -> None:
    """The search term trains every row carrying a search value, a tail's ±1 included: a weight below 1 beside it is refused."""
    with pytest.raises(ValueError, match="search term"):
        H.graph_hparams(tail_value_weight=0.1, value_target_search_weight=0.5, value_target_search_encoding="end_bin")
    H.graph_hparams(tail_value_weight=1.0, value_target_search_weight=0.5, value_target_search_encoding="end_bin")
    config = H.graph_config()
    config["selfplay"]["search"]["tactics"] = _TACTICS
    config["selfplay"]["decided_tails"] = {"value_weight": 0.1}
    RunConfig.model_validate(config)
    config["train"].update(value_target_search_weight=0.5, value_target_search_encoding="end_bin")
    with pytest.raises(ValidationError, match="search term"):
        RunConfig.model_validate(config)
