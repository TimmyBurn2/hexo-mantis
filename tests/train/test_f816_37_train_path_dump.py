"""LAW-07 mutation self-test for the TRAINING path's `F-816-37` dump-on-fire (R340 leg 3).

WHY THIS SUITE EXISTS. R339(c) armed dump-on-fire on the EVAL path, because that is where the
class had been seen. At R340 leg 3 `F-816-37` fired in the TRAINER — `dispatch.py::_materialise`
— and the run halted with **no artifact**, which is precisely what R340(c)'s *"a firing halts
with the artifact"* forbids. `tests/eval/test_f816_37_instrument.py` is the eval-path twin; this
is the same contract on the path that actually took the run down.

THE PLANT GOES IN THE WIRE, NOT IN THE CHECK. Corrupting the check would prove nothing about the
half that has to notice. The plant sits on `slice_graph_wire`'s output — the exact object
`collate_graph_batch` reads — so the production `verify_edge_geometry` is what fires, and the
production `except GraphContractError` is what writes.

THE NEGATIVE CONTROL IS HALF THE FILE. A dump test that only ever runs a corrupted step would be
green against an instrument that dumps on every step, so the clean step is asserted to leave the
directory empty. Either assertion alone is satisfiable by a broken instrument.

`[0.5, 0.0, 1.1754944e-38]` is not an invented value: it is byte-for-byte what leg 3 and leg 1
both observed — bit 23 of word 0 cleared and bit 23 of word 2 set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import _microbatch_harness as H
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.selfplay.graph_collate import GraphContractError
from mantis.train.coordinator.dispatch import run_declared_train_step

#: F-816-37's observed corruption, both firings identical.
_OBSERVED = [0.5, 0.0, 1.1754944e-38]


def _drive(trainer: Any, replay: Any) -> None:
    caps = H.non_binding_caps(replay.wire)
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
        recent_buffer=None, sample_threads_provider=lambda: 1,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))


def _plant(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Corrupt ONE edge's axis one-hot in the first wire slice the trainer collates."""
    from mantis.selfplay import graph_wire_split

    state: dict[str, Any] = {"planted": 0}
    real = graph_wire_split.slice_graph_wire

    def _spy(payload: Any, g0: int, g1: int) -> Any:
        sub = real(payload, g0, g1)
        if state["planted"] == 0:
            attr = np.asarray(sub.edge_attr)
            if attr.size >= 5:
                attr[0:3] = _OBSERVED
                state["planted"] = 1
        return sub

    monkeypatch.setattr(graph_wire_split, "slice_graph_wire", _spy)
    return state


def _dumps(trainer: Any) -> list[Path]:
    return sorted((Path(trainer.checkpoint_dir).parent / "collate_dumps").glob("collate_dump_*.json"))


def test_a_clean_train_step_dumps_nothing(tmp_path: Path) -> None:
    """The negative control: without a plant the instrument is silent."""
    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(), 4)
    _drive(trainer, replay)
    assert _dumps(trainer) == [], "a clean training step wrote a dump"


def test_a_planted_corruption_DUMPS_and_REDS(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves: the trainer's check notices, AND the offending batch lands on disk.

    A step that reds without dumping is exactly HEAD's behaviour before R340 leg 3 — the
    failure that cost that leg its artifact — so the dump is the assertion that matters.
    """
    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(), 4)
    state = _plant(monkeypatch)

    with pytest.raises(GraphContractError, match="one-hot"):
        _drive(trainer, replay)
    assert state["planted"] == 1, "the corruption never reached a wire slice"

    dumps = _dumps(trainer)
    assert len(dumps) == 1, f"expected exactly one dump, got {[p.name for p in dumps]}"

    import json

    sidecar = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert sidecar["path"] == "train", "the dump does not identify the training path"
    assert sidecar["phase"] == "train_step"
    assert "one-hot" in sidecar["error"]
    assert sidecar["error_type"] == "EdgeAttrGeometryMismatch"

    batch = dumps[0].with_suffix(".npz")
    assert batch.exists(), "the sidecar names a batch that was not written"
    # The corruption must be IN the saved arrays — a dump that does not carry the offending
    # edge is a file, not evidence.
    attr = np.load(batch)["edge_attr"].reshape(-1, 5)
    assert np.allclose(attr[0][:3], _OBSERVED), (
        f"the saved batch does not carry the planted edge: {attr[0][:3]}")


def test_the_dump_never_replaces_the_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic that can convert a named contract failure into its own error destroys the
    evidence it exists to keep. With the writer itself broken, the ORIGINAL error must survive.
    """
    from mantis.train.coordinator import dispatch as dispatch_mod

    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(), 4)
    _plant(monkeypatch)

    def _explode(*_a: Any, **_k: Any) -> None:
        raise OSError("simulated dump-writer failure")

    monkeypatch.setattr(dispatch_mod, "_dump_train_collate", _explode)
    with pytest.raises(GraphContractError, match="one-hot"):
        _drive(trainer, replay)
