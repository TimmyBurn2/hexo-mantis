"""O4b — replay byte-parity vs the committed old-side replay npz (per-encoding tuples).

The frozen game ``g3`` (two-region K>1, off-window skip) is replayed through the
registry dispatcher for each of the three registered corpus encodings; each output must
be byte-identical to the committed ``value_probes/replay/{v6,v6w25,v6_live2_ls}.npz``
(COPIED from the frozen old-side capture). The tuple gated is the one the code actually
emits PER ENCODING (a real old asymmetry):
  - v6 / v6w25 → (states, chain_planes, policies, outcomes); states 18 / 8 planes.
  - v6_live2_ls → (states, policies, outcomes, ply_index); 4 planes, NO chain_planes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _frozen_games import FROZEN_GAMES

from mantis.data.replay import ReplayTriples, replay_game_to_triples
from mantis.encoding import lookup

_REPLAY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "replay"
_COMMITTED_GAME = "g3"  # the committed replay/*.npz are this game's frozen arrays


def _replay(encoding: str) -> ReplayTriples:
    moves, winner = FROZEN_GAMES[_COMMITTED_GAME]
    return replay_game_to_triples(moves, winner, lookup(encoding))


@pytest.mark.parametrize("encoding", ["v6", "v6w25"])
def test_replay_byte_parity_v6_family(encoding: str) -> None:
    ref = np.load(_REPLAY_DIR / f"{encoding}.npz")
    out = _replay(encoding)
    assert out.chain_planes is not None
    assert out.ply_index is None
    for name, arr in (
        ("states", out.states),
        ("chain_planes", out.chain_planes),
        ("policies", out.policies),
        ("outcomes", out.outcomes),
    ):
        assert np.array_equal(arr, ref[name]), f"{encoding} {name} byte mismatch"
        assert arr.dtype == ref[name].dtype


def test_replay_byte_parity_v6_live2_ls() -> None:
    ref = np.load(_REPLAY_DIR / "v6_live2_ls.npz")
    out = _replay("v6_live2_ls")
    # LS path emits ply_index and NO chain_planes.
    assert out.chain_planes is None
    assert out.ply_index is not None
    for name, arr in (
        ("states", out.states),
        ("policies", out.policies),
        ("outcomes", out.outcomes),
        ("ply_index", out.ply_index),
    ):
        assert np.array_equal(arr, ref[name]), f"v6_live2_ls {name} byte mismatch"
        assert arr.dtype == ref[name].dtype


def test_per_encoding_plane_counts_and_dtypes() -> None:
    # v6 states are the FULL 18-plane tensor (unsliced); v6w25=8; v6_live2_ls=4.
    assert _replay("v6").states.shape[1] == 18
    assert _replay("v6w25").states.shape[1] == 8
    assert _replay("v6_live2_ls").states.shape[1] == 4
    v6 = _replay("v6")
    ls = _replay("v6_live2_ls")
    assert v6.chain_planes is not None
    assert ls.ply_index is not None
    assert v6.states.dtype == np.float16
    assert v6.chain_planes.dtype == np.float16
    assert v6.policies.dtype == np.float32
    assert v6.outcomes.dtype == np.float32
    assert ls.ply_index.dtype == np.int32


def test_unknown_encoding_raises() -> None:
    class _FakeSpec:
        name = "v8"

    with pytest.raises(ValueError, match="no replayer for encoding"):
        replay_game_to_triples([(0, 0)], 1, _FakeSpec())  # type: ignore[arg-type]
