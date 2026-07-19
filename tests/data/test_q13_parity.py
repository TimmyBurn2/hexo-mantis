"""O5 — Q13 pretrain-parity (sha256 hash-manifest).

Regenerates the 12-fold hex-dihedral augmented training arrays IN-TEST — composing
WP9 ``replay_game_to_triples`` + the engine symmetry kernel + WP9 ``get_policy_scatters``
+ the ported ``_compute_chain_planes`` (post-augment recompute) — hashes each
(game, encoding, sym) tuple, and asserts equality with the committed
``value_probes/q13_manifest.tsv``. Guards the §97/F-10 silent-unport class: a wrong
slice, wrong opp_slot, or chain planes scattered (instead of recomputed) → hash mismatch.

Exact augment mechanics reproduced (the capture reality — matches the production
``make_augmented_collate`` per encoding):
  (1) slice v6 replay states to the 8 kept planes [0,1,2,3,8,9,10,11] before augment;
  (2) STATE scatter — v6 via the engine ``apply_symmetries_batch`` kernel (19×19);
      v6w25 / v6_live2_ls via the numpy policy-scatter LUT sliced to spatial-only
      (the kernel hardcodes 19×19, so the 25×25 / LS paths use numpy — as production does);
  (3) RECOMPUTE chain planes POST-augment via ``_compute_chain_planes`` at
      opp_slot = 4 (v6/v6w25) / 1 (v6_live2_ls) — NOT by scattering the replay chain planes;
  (4) policy scatter via the WP9 ``get_policy_scatters`` LUT.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from _frozen_games import ENCODINGS, FROZEN_GAMES

from mantis._engine import apply_symmetries_batch
from mantis.data.augment import get_policy_scatters
from mantis.data.replay import replay_game_to_triples
from mantis.encoding import lookup
from mantis.env.game_state import _compute_chain_planes

_MANIFEST = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "q13_manifest.tsv"
_KEPT_V6 = [0, 1, 2, 3, 8, 9, 10, 11]  # v6 18-plane → 8 kept planes before augment
_N_SYM = 12

# Per-encoding augment config: (board_size, opp_slot, use_engine_kernel).
_AUG_CFG = {
    "v6": (19, 4, True),
    "v6w25": (25, 4, False),
    "v6_live2_ls": (19, 1, False),
}


def _load_manifest() -> dict[tuple[str, str, int], str]:
    rows = _MANIFEST.read_text().splitlines()
    assert rows[0].split("\t") == ["game", "encoding", "sym", "sha256"]
    out: dict[tuple[str, str, int], str] = {}
    for line in rows[1:]:
        game, enc, sym, sha = line.split("\t")
        out[(game, enc, int(sym))] = sha
    return out


def _hash_sym(states_sym: np.ndarray, chain_sym: np.ndarray,
              pol_sym: np.ndarray, outcomes: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(states_sym).tobytes())
    h.update(np.ascontiguousarray(chain_sym).tobytes())
    h.update(np.ascontiguousarray(pol_sym).tobytes())
    h.update(np.ascontiguousarray(outcomes).tobytes())
    return h.hexdigest()


def _augment_game(game: str, encoding: str) -> dict[int, str]:
    """Compose the 12 augmented syms for one (game, encoding); return {sym: sha256}."""
    moves, winner = FROZEN_GAMES[game]
    board_size, opp_slot, use_kernel = _AUG_CFG[encoding]
    triples = replay_game_to_triples(moves, winner, lookup(encoding))

    states = triples.states
    if encoding == "v6":
        states = states[:, _KEPT_V6]  # slice 18 → 8 kept planes BEFORE augment
    policies = triples.policies
    outcomes = triples.outcomes

    n, channels = states.shape[0], states.shape[1]
    spatial = board_size * board_size
    scatters = get_policy_scatters(board_size, has_pass=True)

    out: dict[int, str] = {}
    for sym in range(_N_SYM):
        sc = scatters[sym]
        if use_kernel:
            st = apply_symmetries_batch(
                np.ascontiguousarray(states, dtype=np.float32), [sym] * n
            ).astype(np.float16, copy=False)
        else:
            state_scatter = sc[:spatial]  # has_pass → drop the pass row for state scatter
            flat = states.reshape(n, channels, spatial)
            st = flat[:, :, state_scatter].reshape(n, channels, board_size, board_size)

        chain = np.zeros((n, 6, board_size, board_size), dtype=np.float16)
        for i in range(n):
            chain[i] = _compute_chain_planes(
                st[i, 0].astype(np.float32),
                st[i, opp_slot].astype(np.float32),
            ).astype(np.float16) / 6.0

        pol = policies[:, sc]
        out[sym] = _hash_sym(st, chain, pol, outcomes)
    return out


@pytest.mark.parametrize("game", list(FROZEN_GAMES))
@pytest.mark.parametrize("encoding", list(ENCODINGS))
def test_q13_augment_hash_parity(game: str, encoding: str) -> None:
    manifest = _load_manifest()
    regenerated = _augment_game(game, encoding)
    assert len(regenerated) == _N_SYM
    for sym in range(_N_SYM):
        expected = manifest[(game, encoding, sym)]
        assert regenerated[sym] == expected, (
            f"Q13 hash mismatch for {game}/{encoding}/sym{sym}"
        )


def test_manifest_row_count() -> None:
    manifest = _load_manifest()
    assert len(manifest) == len(FROZEN_GAMES) * len(ENCODINGS) * _N_SYM  # 4 × 3 × 12 = 144
