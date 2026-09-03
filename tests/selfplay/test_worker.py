"""Suite I (worker half) — `mantis.selfplay.worker.SelfPlayWorker` + the temperature table.

IMPL-written (non-⊕). I-05 pins the MCTS/sampling glue (legal policy, legal-restricted
sampling with the uniform fallback, and the `get_temperature` modes); I-06 pins
`quarter_cosine_temperature` as a numeric table.

It also pins the CONFIG-KEY trace for this module. Two field↔key divergences are already
on record in this WP; the worker carries a third instance of the SAME class:

    field  `dirichlet_eps`   ←  config key  `mcts.epsilon`     (NOT `dirichlet_eps`)

Reading the field's own spelling returns `None` on every config and substitutes the 0.25
default for the operator's exploration noise — silently. `test_dirichlet_epsilon_reads_
the_real_config_key` is the bite.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mantis._engine import Board
from mantis.encoding import lookup
from mantis.model import CnnArch, build_net
from mantis.selfplay.utils import get_temperature, quarter_cosine_temperature
from mantis.selfplay.worker import SelfPlayWorker

_SPEC = lookup("v6")
_CPU = torch.device("cpu")
_N_ACTIONS = _SPEC.policy_logit_count


def _tiny_net() -> torch.nn.Module:
    torch.manual_seed(20260723)
    net = build_net(
        CnnArch(
            board_size=_SPEC.board_size,
            in_channels=_SPEC.n_planes,
            filters=8,
            res_blocks=1,
        )
    )
    net.eval()
    return net


#: The `train` section this worker's config must carry. AUDIT-1 F-31 made the engine's
#: autocast dtype a REQUIRED read (`config["train"]["amp_dtype"]`) rather than a `dtype=`-less
#: autocast running at torch's device default, so a config with no `train` section is now a
#: loud `KeyError` — which is the R1 posture, and the reason it is written here rather than
#: softened back to a `.get`.
_TRAIN: dict = {"train": {"amp_dtype": "bf16"}}


def _worker(**mcts: object) -> SelfPlayWorker:
    cfg: dict = {"encoding": {"version": "v6"}, "mcts": dict(mcts), **_TRAIN}
    return SelfPlayWorker(_tiny_net(), cfg, _CPU, encoding_spec=_SPEC)


# ══ I-05a — config wiring, incl. the field↔key trace ═════════════════════════════
def test_mcts_knobs_are_read_from_the_mcts_namespace() -> None:
    w = _worker(n_simulations=7, c_puct=2.5, dirichlet_alpha=0.11)
    assert w.n_sims == 7
    assert w.c_puct == pytest.approx(2.5)
    assert w.dirichlet_alpha == pytest.approx(0.11)


def test_dirichlet_epsilon_reads_the_real_config_key() -> None:
    """⚠ field name != config key. `mcts.epsilon` is the operator's knob and it must WIN;
    the field spelling must NOT be what is read, or the default silently substitutes."""
    w = _worker(epsilon=0.42)
    assert w.dirichlet_eps == pytest.approx(0.42), (
        "mcts.epsilon did not reach dirichlet_eps — the port read the field's own "
        "spelling and the 0.25 default fired instead of the operator's value"
    )

    # The field-name spelling is NOT a config key: setting it must change nothing.
    w_wrong = _worker(dirichlet_eps=0.42, dirichlet_epsilon=0.42)
    assert w_wrong.dirichlet_eps == pytest.approx(0.25)


def test_defaults_when_the_mcts_namespace_is_empty() -> None:
    w = _worker()
    assert w.n_sims == 50
    assert w.c_puct == pytest.approx(1.5)
    assert w.dirichlet_alpha == pytest.approx(0.3)
    assert w.dirichlet_eps == pytest.approx(0.25)


def test_n_simulations_falls_back_to_the_top_level_key() -> None:
    """`n_simulations` is read from the `mcts` namespace with a TOP-LEVEL fallback —
    a two-step chain, not a single lookup."""
    cfg = {"encoding": {"version": "v6"}, "mcts": {}, "n_simulations": 13, **_TRAIN}
    w = SelfPlayWorker(_tiny_net(), cfg, _CPU, encoding_spec=_SPEC)
    assert w.n_sims == 13


def test_geometry_comes_from_the_spec() -> None:
    w = _worker()
    assert w._board_size == _SPEC.board_size
    assert w._n_actions == _SPEC.policy_logit_count


def test_spec_like_object_is_adapted_by_name() -> None:
    class _SpecLike:
        name = "v6w25"

    w = SelfPlayWorker(
        _tiny_net(),
        {"encoding": {"version": "v6"}, "mcts": {}, **_TRAIN},
        _CPU,
        encoding_spec=_SpecLike(),
    )
    assert w.encoding_spec.name == "v6w25"
    assert w._n_actions == lookup("v6w25").policy_logit_count


def test_unadaptable_spec_raises() -> None:
    with pytest.raises(TypeError, match="cannot adapt"):
        SelfPlayWorker(
            _tiny_net(), {"encoding": {"version": "v6"}, "mcts": {}, **_TRAIN}, _CPU,
            encoding_spec=42,
        )


# ══ I-05b — `_run_mcts` produces a legal policy ══════════════════════════════════
def test_run_mcts_returns_a_legal_policy() -> None:
    w = _worker(n_simulations=8)
    board = Board()
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    policy = w._run_mcts(board, use_dirichlet=False)

    assert len(policy) == _N_ACTIONS
    arr = np.asarray(policy, dtype=np.float64)
    assert np.all(np.isfinite(arr))
    assert arr.min() >= 0.0
    assert arr.sum() == pytest.approx(1.0, abs=1e-5)

    legal_flat = {board.to_flat(q, r) for q, r in board.legal_moves()}
    illegal_mass = float(
        sum(arr[i] for i in range(_N_ACTIONS) if i not in legal_flat)
    )
    assert illegal_mass == pytest.approx(0.0, abs=1e-9), (
        "MCTS assigned probability mass to an illegal cell"
    )


def test_run_mcts_with_dirichlet_still_legal() -> None:
    """Root noise is exploration, not a legality escape: the noised policy must remain
    supported on legal cells only."""
    # Enough sims that the visit distribution is non-degenerate: below ~16 sims the
    # root has too few visits for a tau>0 policy to normalise (engine behaviour).
    w = _worker(n_simulations=32, epsilon=0.5, dirichlet_alpha=0.5)
    board = Board()
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    policy = np.asarray(w._run_mcts(board, use_dirichlet=True), dtype=np.float64)
    legal_flat = {board.to_flat(q, r) for q, r in board.legal_moves()}
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert all(policy[i] == pytest.approx(0.0, abs=1e-9)
               for i in range(_N_ACTIONS) if i not in legal_flat)


# ══ I-05c — `_sample_action` legal restriction + uniform fallback ════════════════
def test_sample_action_is_restricted_to_legal_moves() -> None:
    w = _worker()
    board = Board()
    board.apply_move(0, 0)
    legal = board.legal_moves()
    target = legal[3]

    policy = [0.0] * _N_ACTIONS
    policy[board.to_flat(*target)] = 1.0
    for _ in range(20):
        assert w._sample_action(policy, legal, board) == target


def test_sample_action_uniform_fallback_when_all_legal_mass_is_zero() -> None:
    """Degenerate case: MCTS gave every legal move zero probability. Sampling must fall
    back to uniform over the legal moves, never divide by zero or pick an illegal cell."""
    w = _worker()
    board = Board()
    board.apply_move(0, 0)
    legal = board.legal_moves()
    policy = [0.0] * _N_ACTIONS

    picks = {w._sample_action(policy, legal, board) for _ in range(50)}
    assert picks, "no move was sampled"
    assert picks <= set(legal)
    assert len(picks) > 1, "uniform fallback collapsed onto a single move"


def test_sample_action_ignores_out_of_range_flat_indices() -> None:
    """Legal moves whose flat index falls outside the action space contribute zero mass
    (the `i < n_actions` guard) instead of indexing past the policy vector."""
    w = _worker()
    board = Board()
    for q, r in [(0, 0), (1, 0), (0, 1), (8, 0), (9, 0), (8, 1)]:
        board.apply_move(q, r)
    legal = board.legal_moves()
    assert any(board.to_flat(q, r) >= _N_ACTIONS for q, r in legal), (
        "fixture must include a move whose flat index is outside the action space"
    )
    in_range = [(q, r) for q, r in legal if board.to_flat(q, r) < _N_ACTIONS]
    policy = [0.0] * _N_ACTIONS
    policy[board.to_flat(*in_range[0])] = 1.0
    assert w._sample_action(policy, legal, board) == in_range[0]


# ══ I-05d — `get_temperature` modes ══════════════════════════════════════════════
def test_get_temperature_evaluation_is_deterministic() -> None:
    assert get_temperature(ply=0, mode="evaluation", config={}) == 0.0
    assert get_temperature(ply=17, mode="evaluation", config={"mcts": {}}) == 0.0


def test_get_temperature_bootstrap_is_moderate() -> None:
    assert get_temperature(ply=5, mode="bootstrap", config={}) == 0.5


def test_get_temperature_training_uses_the_compound_turn_cosine() -> None:
    cfg = {"mcts": {"temperature_threshold_compound_moves": 8, "temp_min": 0.25}}
    # ply 0 is compound move 0 -> cos(0) = 1.0
    assert get_temperature(ply=0, mode="training", config=cfg) == pytest.approx(1.0)
    # ply 7 -> compound move 4 -> cos(pi/2 * 4/8)
    assert get_temperature(ply=7, mode="training", config=cfg) == pytest.approx(
        quarter_cosine_temperature(4, 8, 0.25)
    )
    # beyond the threshold -> the floor
    assert get_temperature(ply=99, mode="training", config=cfg) == pytest.approx(0.25)


def test_get_temperature_training_missing_keys_means_schedule_off() -> None:
    """Absent keys => threshold 0 => constant `temp_min` (default 0.5). The schedule is
    OFF, not a division by zero."""
    for ply in (0, 1, 10, 250):
        assert get_temperature(ply=ply, mode="training", config={}) == pytest.approx(0.5)


def test_get_temperature_legacy_ply_alias_converts_to_compound_turns() -> None:
    """The legacy eval/bot alias is a PLY clock; it converts to compound turns as
    `(ply + 1) // 2`. A port that used it as a compound-turn threshold would decay twice
    as fast."""
    cfg = {"mcts": {"temperature_threshold_ply": 16, "temp_min": 0.1}}
    expected_threshold = (16 + 1) // 2
    assert get_temperature(ply=5, mode="training", config=cfg) == pytest.approx(
        quarter_cosine_temperature(3, expected_threshold, 0.1)
    )


def test_run_mcts_temperature_override_wins_over_the_resolver() -> None:
    w = _worker(n_simulations=32)
    board = Board()
    board.apply_move(0, 0)
    hot = np.asarray(w._run_mcts(board, use_dirichlet=False, temperature=1.0))
    cold = np.asarray(w._run_mcts(board, use_dirichlet=False, temperature=0.0))
    assert hot.sum() == pytest.approx(1.0, abs=1e-5)
    assert cold.sum() == pytest.approx(1.0, abs=1e-5)
    # tau=0 is argmax: all mass on one cell.
    assert float(cold.max()) == pytest.approx(1.0, abs=1e-6)


# ══ I-06 — `quarter_cosine_temperature` numeric table ════════════════════════════
@pytest.mark.parametrize(
    ("compound_move", "threshold", "temp_min", "expected"),
    [
        # threshold 0 => schedule OFF => constant temp_min at every compound move.
        (0, 0, 0.5, 0.5),
        (7, 0, 0.5, 0.5),
        (0, 0, 0.0, 0.0),
        # cos(0) = 1 at the start of the schedule.
        (0, 10, 0.25, 1.0),
        # cos(pi/2 * 5/10) = cos(pi/4) = 0.7071…
        (5, 10, 0.25, 0.7071067811865476),
        # cos(pi/2 * 9/10) = 0.15643…, still above the 0.1 floor.
        (9, 10, 0.1, 0.15643446504023092),
        # …but clamped by a higher floor.
        (9, 10, 0.5, 0.5),
        # at/after the threshold the floor holds.
        (10, 10, 0.25, 0.25),
        (11, 10, 0.25, 0.25),
    ],
)
def test_quarter_cosine_temperature_table(
    compound_move, threshold, temp_min, expected
) -> None:
    assert quarter_cosine_temperature(compound_move, threshold, temp_min) == pytest.approx(
        expected
    )


def test_quarter_cosine_is_monotone_non_increasing_within_the_schedule() -> None:
    values = [quarter_cosine_temperature(cm, 16, 0.1) for cm in range(0, 20)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
