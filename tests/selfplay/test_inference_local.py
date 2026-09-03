"""Suite I (engine half) — `mantis.selfplay.inference_local.LocalInferenceEngine`.

>300 justify: the four dense-decode invariants, the graph branch, the no-graph-server
dense arm, the arch-sniff ban and the two model/spec-disagreement arms all bind the SAME
class and share one board fixture plus the scripted-net stubs that make the invariants
numerically observable. Splitting by arm would duplicate the fixture that makes the
max-over-windows rule visible at all.

IMPL-written (non-⊕). NO old-side golden exists anywhere for the dense decode, so the
pins ARE the four frozen decode invariants, asserted numerically against a stub model
with hand-computable outputs (PREREG §3 I-01):

  1. board value  = **min** over the K cluster windows (not mean, not first, not max);
  2. per-legal-cell policy = **max** over the windows that cover the cell (not sum, not
     first-window);
  3. a legal move whose flat index is `>= n_actions - 1` is SKIPPED (the unbounded board
     produces such moves — without the skip the write would be out of range);
  4. renormalize iff `total > 1e-9`, else fill uniform `1 / n_actions`.

Each stub is built so the WRONG rule produces a DIFFERENT number, not merely a different
shape — an invariant test that passes under mean/sum/first would pin nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board
from mantis.encoding import lookup
from mantis.model import CnnArch, GnnArch, build_net
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.selfplay.inference_local import LocalInferenceEngine

_GRID_SPEC = lookup("v6")
_GRAPH_SPEC = lookup("gnn_axis_v1")

_BOARD_SIZE = _GRID_SPEC.board_size
_N_ACTIONS = _GRID_SPEC.policy_logit_count
_HALF = (_BOARD_SIZE - 1) // 2
_CPU = torch.device("cpu")
#: F-816-10 D-1: `fused_graph_caps` is REQUIRED and keyword-only on this class — it
#: hand-builds its `InferenceServer` config with no `RunConfig`, so the bound is
#: THREADED from a parent resolver and never hardcoded at the site. The graph engines
#: below get a NON-BINDING pair (nothing here splits); the grid ones pass `None`
#: explicitly, which is what "this route has no fused graph forward" looks like
#: written down rather than omitted.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)


# ── boards ───────────────────────────────────────────────────────────────────────
def _two_cluster_board() -> Board:
    """Two stone clusters 8 cells apart: K=2 with OVERLAPPING windows, so many legal
    cells are covered by both — which is what makes the max-over-windows rule
    observable — and some legal moves fall outside the flat action space, which is what
    makes the skip rule observable."""
    b = Board()
    for q, r in [(0, 0), (1, 0), (0, 1), (8, 0), (9, 0), (8, 1)]:
        b.apply_move(q, r)
    return b


def _centers(board: Board) -> list[tuple[int, int]]:
    _views, centers = board.get_cluster_views()
    return list(centers)


def _local_idx(q: int, r: int, center: tuple[int, int]) -> int | None:
    cq, cr = center
    wq, wr = q - cq + _HALF, r - cr + _HALF
    if 0 <= wq < _BOARD_SIZE and 0 <= wr < _BOARD_SIZE:
        return wq * _BOARD_SIZE + wr
    return None


def _cells_in_both_windows(board: Board) -> list[tuple[int, int, int]]:
    """(q, r, flat) for legal cells inside BOTH windows and inside the action space."""
    centers = _centers(board)
    out: list[tuple[int, int, int]] = []
    for q, r in board.legal_moves():
        flat = board.to_flat(q, r)
        if flat >= _N_ACTIONS - 1:
            continue
        if all(_local_idx(q, r, c) is not None for c in centers):
            out.append((q, r, flat))
    return out


# ── stub models ──────────────────────────────────────────────────────────────────
class _ScriptedNet(torch.nn.Module):
    """Returns caller-scripted per-cluster (policy, value) rows, in cluster order."""

    def __init__(self, probs: np.ndarray, values: np.ndarray) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self._probs = torch.from_numpy(probs.astype(np.float32))
        self._values = torch.from_numpy(values.astype(np.float32)).reshape(-1, 1)
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor):
        self.seen_shapes.append(tuple(x.shape))
        n = x.shape[0]
        assert n == self._probs.shape[0], (
            f"stub scripted for {self._probs.shape[0]} cluster rows, got {n}"
        )
        # The engine exponentiates the first output, so hand back log-probabilities.
        return torch.log(self._probs), self._values, self._values


class _ArchTrappingNet(torch.nn.Module):
    """Records every attribute lookup that falls through to `__getattr__`.

    `nn.Module` resolves its own machinery (`training`, `_call_impl`, parameters, …)
    before `__getattr__`, so anything landing here is a genuine sniff for metadata that
    should be travelling on the declared arch/spec instead.
    """

    _ARCH_NAMES = frozenset({
        "in_channels", "board_size", "representation", "n_actions", "filters",
        "res_blocks", "value_head_type", "input_channels", "n_value_bins", "trunk",
        "policy_logit_count", "n_planes", "arch",
    })

    def __init__(self, probs: np.ndarray, values: np.ndarray) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        object.__setattr__(self, "sniffed", [])
        self._probs_np = probs
        self._values_np = values

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self._ARCH_NAMES:
                object.__getattribute__(self, "sniffed").append(name)
            raise

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        probs = torch.from_numpy(self._probs_np.astype(np.float32))
        values = torch.from_numpy(self._values_np.astype(np.float32)).reshape(-1, 1)
        assert n == probs.shape[0]
        return torch.log(probs), values, values


def _engine_with(model: torch.nn.Module) -> LocalInferenceEngine:
    return LocalInferenceEngine(model, _CPU, encoding_spec=_GRID_SPEC,
                                fused_graph_caps=None,
                                inference_batching=None, max_in_flight=0, amp_dtype="bf16")


def _graph_engine() -> LocalInferenceEngine:
    torch.manual_seed(20260723)
    net = build_net(
        GnnArch(
            in_dim=_GRAPH_SPEC.node_feat_dim,
            edge_dim=_GRAPH_SPEC.edge_feat_dim,
            hidden=16,
            num_layers=1,
            policy_hidden=16,
            value_hidden=16,
        )
    ).to(_CPU)
    net.eval()
    return LocalInferenceEngine(net, _CPU, encoding_spec=_GRAPH_SPEC,
                                fused_graph_caps=_CAPS,
                                inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, amp_dtype="bf16")


# ══ I-01 — the four dense-decode invariants, numerically ═════════════════════════
def test_dense_decode_value_is_the_min_over_windows() -> None:
    """min-pool: the WORST window is the board value. mean (0.2), first (0.7) and max
    (0.7) are all different numbers, so this cannot pass under the wrong rule."""
    board = _two_cluster_board()
    centers = _centers(board)
    assert len(centers) == 2, "the fixture must produce two clusters"

    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[:, 0] = 1.0  # some mass so the vector is a valid distribution
    values = np.array([0.7, -0.3])

    engine = _engine_with(_ScriptedNet(probs, values))
    policies, board_values = engine.infer_batch([board])

    assert len(policies) == 1
    assert len(board_values) == 1
    assert board_values[0] == pytest.approx(-0.3)


def test_dense_decode_policy_is_the_max_over_windows() -> None:
    """max-over-windows, pinned against sum and against first-window-wins.

    Cell C reads 0.1 from window 0 and 0.6 from window 1; cell D reads 0.3 from window 0
    and nothing from window 1. Expected (max): 0.6/0.9 and 0.3/0.9. Under a sum rule the
    numbers would be 0.7/1.0 and 0.3/1.0; under first-window-wins, 0.25 and 0.75.
    """
    board = _two_cluster_board()
    centers = _centers(board)
    both = _cells_in_both_windows(board)
    assert len(both) >= 2, "fixture must expose at least two doubly-covered legal cells"
    (cq_, cr_, flat_c) = both[0]
    (dq_, dr_, flat_d) = both[1]

    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[0, _local_idx(cq_, cr_, centers[0])] = 0.1
    probs[1, _local_idx(cq_, cr_, centers[1])] = 0.6
    probs[0, _local_idx(dq_, dr_, centers[0])] = 0.3
    values = np.array([0.0, 0.0])

    engine = _engine_with(_ScriptedNet(probs, values))
    policies, _values = engine.infer_batch([board])
    policy = policies[0]

    assert len(policy) == _N_ACTIONS
    assert policy[flat_c] == pytest.approx(0.6 / 0.9)
    assert policy[flat_d] == pytest.approx(0.3 / 0.9)
    assert sum(policy) == pytest.approx(1.0)
    # Explicitly NOT the sum rule and NOT first-window-wins.
    assert policy[flat_c] != pytest.approx(0.7)
    assert policy[flat_c] != pytest.approx(0.25)


def test_dense_decode_skips_moves_outside_the_action_space() -> None:
    """The unbounded board yields legal moves whose flat index is outside the action
    space; they must be SKIPPED. Without the skip the decode writes out of range."""
    board = _two_cluster_board()
    out_of_range = [
        (q, r) for q, r in board.legal_moves() if board.to_flat(q, r) >= _N_ACTIONS - 1
    ]
    assert out_of_range, "fixture must contain out-of-action-space legal moves"

    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[:, 0] = 1.0
    engine = _engine_with(_ScriptedNet(probs, np.array([0.0, 0.0])))
    policies, _values = engine.infer_batch([board])
    assert len(policies[0]) == _N_ACTIONS
    assert all(np.isfinite(policies[0]))


def test_dense_decode_uniform_fallback_when_all_mass_is_zero() -> None:
    """`total > 1e-9` else uniform: a model that assigns nothing to any legal cell yields
    a flat `1 / n_actions`, never a zero vector and never a NaN division."""
    board = _two_cluster_board()
    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    # All mass on the final slot, which no legal cell ever reads back (every in-window
    # local index is < n_actions - 1), so the decoded total is exactly 0.
    probs[:, _N_ACTIONS - 1] = 1.0
    engine = _engine_with(_ScriptedNet(probs, np.array([0.0, 0.0])))
    policies, _values = engine.infer_batch([board])
    policy = np.asarray(policies[0])
    assert policy == pytest.approx(np.full(_N_ACTIONS, 1.0 / _N_ACTIONS))
    assert sum(policy) == pytest.approx(1.0)


def test_dense_decode_renormalizes_above_the_threshold() -> None:
    """LAW-07 clean twin for the fallback: nonzero mass renormalizes rather than being
    replaced by the uniform vector."""
    board = _two_cluster_board()
    centers = _centers(board)
    both = _cells_in_both_windows(board)
    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[0, _local_idx(both[0][0], both[0][1], centers[0])] = 0.25
    engine = _engine_with(_ScriptedNet(probs, np.array([0.0, 0.0])))
    policies, _values = engine.infer_batch([board])
    policy = np.asarray(policies[0])
    assert policy[both[0][2]] == pytest.approx(1.0)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[both[1][2]] == pytest.approx(0.0)


def test_infer_single_board_wrapper_matches_batch() -> None:
    board = _two_cluster_board()
    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[:, 0] = 1.0
    engine = _engine_with(_ScriptedNet(probs, np.array([0.4, -0.2])))
    p_batch, v_batch = engine.infer_batch([board])
    engine2 = _engine_with(_ScriptedNet(probs, np.array([0.4, -0.2])))
    p_single, v_single = engine2.infer(board)
    assert p_single == p_batch[0]
    assert v_single == v_batch[0]
    assert v_single == pytest.approx(-0.2)


def test_empty_board_list_is_a_no_op() -> None:
    probs = np.zeros((1, _N_ACTIONS), dtype=np.float64)
    engine = _engine_with(_ScriptedNet(probs, np.array([0.0])))
    assert engine.infer_batch([]) == ([], [])
    assert engine.infer_batch_per_cluster([]) == ([], [], [])


# ══ I-02 — graph branch ══════════════════════════════════════════════════════════
def test_graph_infer_batch_no_attributeerror_and_correct_shape() -> None:
    engine = _graph_engine()
    try:
        policies, values = engine.infer_batch([Board()])
        assert len(policies) == 1
        assert len(values) == 1
        assert len(policies[0]) == _GRAPH_SPEC.policy_logit_count
        assert all(np.isfinite(policies[0]))
        assert np.isfinite(values[0])
        # Whole-board encoding at ply 0: no off-window drop, so the dense half alone is
        # already a distribution.
        assert abs(sum(policies[0]) - 1.0) < 1e-3
    finally:
        engine.close()


def test_graph_infer_batch_produces_a_legal_argmax_move() -> None:
    engine = _graph_engine()
    try:
        board = Board()
        policies, _values = engine.infer_batch([board])
        best_idx = int(np.argmax(policies[0]))
        legal_flat = {board.to_flat(q, r) for q, r in board.legal_moves()}
        assert best_idx in legal_flat
    finally:
        engine.close()


def test_graph_infer_batch_empty_boards_no_op() -> None:
    engine = _graph_engine()
    try:
        assert engine.infer_batch([]) == ([], [])
    finally:
        engine.close()


def test_graph_engine_close_stops_server_thread_and_is_idempotent() -> None:
    engine = _graph_engine()
    assert engine._graph_server is not None
    server_thread = engine._graph_server
    engine.infer_batch([Board()])  # warm the thread up first
    engine.close()
    assert engine._graph_server is None
    assert engine._graph_batcher is None
    assert not server_thread.is_alive()
    engine.close()  # idempotent — a second close must not raise


def test_graph_infer_batch_per_cluster_raises_named_error() -> None:
    """No no-drop legal-set decode exists for a whole-board graph net: die loud with a
    named, actionable error rather than an `AttributeError` two lines later."""
    engine = _graph_engine()
    try:
        with pytest.raises(NotImplementedError, match="infer_batch_per_cluster"):
            engine.infer_batch_per_cluster([Board()])
    finally:
        engine.close()


# ══ I-03 — the dense engine spins up no graph machinery ══════════════════════════
def test_dense_engine_constructs_no_graph_server() -> None:
    net = build_net(
        CnnArch(board_size=_BOARD_SIZE, in_channels=_GRID_SPEC.n_planes, filters=8,
                res_blocks=1)
    )
    engine = LocalInferenceEngine(net, _CPU, encoding_spec=_GRID_SPEC,
                                  fused_graph_caps=None,
                                  inference_batching=None, max_in_flight=0, amp_dtype="bf16")
    try:
        assert engine._is_graph is False
        assert engine._graph_server is None
        assert engine._graph_batcher is None
        policies, values = engine.infer_batch([Board()])
        assert len(policies) == 1
        assert len(policies[0]) == _N_ACTIONS
        assert np.isfinite(values[0])
    finally:
        engine.close()  # no-op for a dense engine — must not raise


def test_dense_engine_matches_a_direct_forward() -> None:
    """The decode reads exactly the model rows a direct forward produces — no hidden
    re-scaling between the forward and the scatter-max."""
    board = _two_cluster_board()
    centers = _centers(board)
    both = _cells_in_both_windows(board)
    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[0, _local_idx(both[0][0], both[0][1], centers[0])] = 0.5
    probs[1, _local_idx(both[1][0], both[1][1], centers[1])] = 0.25
    net = _ScriptedNet(probs, np.array([0.9, 0.1]))
    engine = _engine_with(net)
    policies, values = engine.infer_batch([board])
    policy = np.asarray(policies[0])

    # Direct forward: exp(log p) is the identity on the scripted rows.
    direct = probs
    row0 = direct[0, _local_idx(both[0][0], both[0][1], centers[0])]
    row1 = direct[1, _local_idx(both[1][0], both[1][1], centers[1])]
    assert policy[both[0][2]] == pytest.approx(row0 / 0.75)
    assert policy[both[1][2]] == pytest.approx(row1 / 0.75)
    assert values[0] == pytest.approx(0.1)


# ══ I-04 — no arch sniffing off the live module ══════════════════════════════════
def test_engine_reads_no_arch_attributes_off_the_model() -> None:
    """Arch metadata travels on the declared spec/arch, never on the `nn.Module`. The
    trapping stub records any arch-shaped attribute lookup that reaches `__getattr__`;
    the list must stay empty across construction AND a full decode."""
    board = _two_cluster_board()
    probs = np.zeros((2, _N_ACTIONS), dtype=np.float64)
    probs[:, 0] = 1.0
    net = _ArchTrappingNet(probs, np.array([0.0, 0.0]))
    engine = LocalInferenceEngine(net, _CPU, encoding_spec=_GRID_SPEC,
                                  fused_graph_caps=None,
                                  inference_batching=None, max_in_flight=0, amp_dtype="bf16")
    engine.infer_batch([board])
    engine.infer_batch_per_cluster([board])
    assert net.sniffed == [], f"arch attributes were read off the model: {net.sniffed}"


# ══ I-07 — model/spec disagreement fails LOUD ════════════════════════════════════
def test_graph_model_with_dense_spec_fails_loud() -> None:
    """The arm the deleted model-sniff would have silently routed to the graph loop: a
    graph-built net bound to a DENSE spec must raise at construction or on the first
    `infer_batch`. It must never return a decoded policy."""
    net = build_net(
        GnnArch(in_dim=_GRAPH_SPEC.node_feat_dim, edge_dim=_GRAPH_SPEC.edge_feat_dim,
                hidden=16, num_layers=1, policy_hidden=16, value_hidden=16)
    )
    engine = None
    try:
        engine = LocalInferenceEngine(net, _CPU, encoding_spec=_GRID_SPEC,
                                  fused_graph_caps=None,
                                  inference_batching=None, max_in_flight=0, amp_dtype="bf16")
        with pytest.raises(Exception) as err:
            engine.infer_batch([Board()])
        assert not isinstance(err.value, AssertionError)
    finally:
        if engine is not None:
            engine.close()


def test_dense_model_with_graph_spec_fails_loud() -> None:
    """The inverse arm: a dense net bound to a GRAPH spec must not silently decode."""
    net = build_net(
        CnnArch(board_size=_BOARD_SIZE, in_channels=_GRID_SPEC.n_planes, filters=8,
                res_blocks=1)
    )
    engine = LocalInferenceEngine(net, _CPU, encoding_spec=_GRAPH_SPEC,
                                  fused_graph_caps=_CAPS,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, amp_dtype="bf16")
    try:
        board = Board()
        board.apply_move(0, 0)
        board.apply_move(1, 0)
        with pytest.raises(Exception) as err:
            engine.infer_batch([board])
        assert "forward_batch" in str(err.value) or "Graph inference failed" in str(err.value)
    finally:
        engine.close()
