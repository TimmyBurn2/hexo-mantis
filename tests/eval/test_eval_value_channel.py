"""WP12-R Phase EVALDECODE — RED-TEAM close of F-RT-1: the eval head's VALUE channel.

RED-TEAM built a real defect in the graph decode adapter — `values = [0.0 for _v in values]`
inside `mantis.eval.worker._graph_expand_fn`, one channel over from the one R138 ruled on —
that reds **0 of 25** oracles in the frozen parity bank and **0 of 560** across
tests/{eval,selfplay,arena} + the bridge round-trip, while changing the head's played move at
**4 of 4** dispersed positions at production simulation counts.

The root cause is NOT the mutation's edit site. It is the frozen bank's PARAMETERIZATION. Both
oracles that reach the adapter through the production entrance (P-1b
`test_deploy_head_entrance_reaches_the_same_children`, P-3b
`test_head_plays_an_off_window_move_against_random_bot`) run `n_sims=1`. After a single
expansion every root child has `visits=0, q=0`, so `select_argmax_child`'s
`sigma = (c_visit + max_n_all) * c_scale * clamp(q, -1, 1)` is identically 0 and the move
collapses to `argmax log(max(prior, 1e-8))`. **At n_sims=1 the value channel is provably inert**
— measured here as `test_the_value_channel_is_inert_at_one_simulation`, which is the frozen
bank's blindness pinned rather than described.

The frozen bank is also blind for a second, independent reason: its stub net's value head
returns a CONSTANT `torch.zeros((n_graphs, 1))`, so even at high sims zeroing the values would
change nothing there. The net below is deliberately different — its value head is
POSITION-DEPENDENT — which is what makes the channel observable at all.

LAW-07: `values` is an input the eval search consumes, and until this file it had no producer
test on the graph seam. These two flips are that producer test. Nothing here re-points or
weakens an existing oracle; all 12 frozen files are untouched.

Killing mutation for both flips: RED-TEAM's `M-RT1`. Measured effect and the n_sims grounds are
in `IMPL_NOTES_EVALDECODE.md` "RED-TEAM close".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: F-816-10 D-1: `LocalInferenceEngine` takes the fused-forward memory bound as a REQUIRED
#: keyword — it hand-builds its `InferenceServer` config with no `RunConfig`, so the spec is
#: THREADED from a parent resolver and never hardcoded at the site. Non-binding by
#: construction here: nothing in this file exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "eval_selfplay_parity" / "dispersed_r6_v1.json"
)

#: Production-representative simulation count. run5 plays the gate block at
#: `eval.gate.deploy_sims: 150` and the random floor at `eval.random_model_sims: 96`. 32 is
#: chosen, not 150, and the grounds are measured rather than asserted:
#:   * it is far outside the INERT regime (n_sims=1 yields exactly ONE distinct root q at
#:     4/4 positions; n_sims=32 yields 2-4), so these flips have margin rather than sitting
#:     on a knife-edge;
#:   * it is in the same MECHANISM regime as production — the tree branches, values back up
#:     through more than one level, and the completed-Q sigma term actually competes with
#:     `log(prior)` in `select_argmax_child`. That mechanism is what F-RT-1 showed is absent
#:     at n_sims=1, and it is fully engaged by 32;
#:   * M-RT1 reds both flips at EVERY n_sims >= 2 (measured), so 150 buys no additional
#:     discriminating power against the defect this file exists to catch;
#:   * cost, measured on this box: ~3.0 s here vs ~13.5 s at n_sims=150 for the same work.
#:     A 4.5x default-tier cost for zero added detection is not a trade this tier should take.
#: If a future card wants the full 150-sim instrument, it belongs in the `integration` tier,
#: not here.
_SIMS = 32

#: `select_argmax_child` reduces to `argmax log(prior)` when every child has `visits=0, q=0`.
_INERT_SIMS = 1


def _positions() -> list[dict]:
    """The committed dispersed fixture, re-nested from its flat form. A missing key is a
    KeyError, never a default — this file reads the fixture and never writes it."""
    fx = json.loads(_FIXTURE.read_text())
    out = []
    for i in range(fx["n_positions"]):
        prefix = f"p{i}_"
        out.append({k[len(prefix):]: v for k, v in fx.items() if k.startswith(prefix)})
    return out


def _board(pos: dict) -> Board:
    board = Board.with_encoding_name(_ENC)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


class _ValueVisibleNet(torch.nn.Module):
    """`GnnNet.forward_batch`'s contract with a deterministic policy head AND a
    POSITION-DEPENDENT value head.

    The policy rule is the parity bank's `logit_rule` verbatim, so the priors — and therefore
    the tree's shape before any value backs up — are identical to the bank's. The value head
    is the deliberate difference: it is a bounded, deterministic function of the leaf graph's
    own size, so distinct leaves carry distinct values. `sign` flips the whole value head
    without touching a single prior, which is what lets the second flip below attribute a
    changed move to the VALUE channel and to nothing else.

    Everything between this net and the children is production: the `InferenceServer` graph
    loop, `collate_graph_batch`, `segment_softmax`, `assemble_ls_from_gnn_probs`,
    `submit_graphs_and_wait_ls`, `infer_batch_ls` and `expand_and_backup_ls_graph`.
    """

    def __init__(self, sign: float) -> None:
        super().__init__()
        self.sign = float(sign)

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        values: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            # `legal_index` is the wire's `legal_node_gather` (R284 P-MASK): the ROWS of the
            # legal nodes, not a dense mask. Counting index entries that fall in this graph's
            # `[lo, hi)` row range is the same count as summing the mask's bits over it, for
            # every payload the contract admits — the gather is strictly ascending, hence
            # unique (wire check 13). The stub's OUTPUT is unchanged; nothing it asserts moves.
            n_legal = int(((legal_index >= lo) & (legal_index < hi)).sum().item())
            logits.extend(((i * 37) % 101) / 20.0 for i in range(n_legal))
            key = (hi - lo) * 31 + n_legal
            values.append(self.sign * (((key * 7919) % 2001) - 1000) / 1000.0)
        return (
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32).reshape(n_graphs, 1),
            torch.zeros((n_graphs, 65), dtype=torch.float32),
        )


def _engine(sign: float) -> LocalInferenceEngine:
    net = _ValueVisibleNet(sign)
    net.eval()
    return LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=lookup(_ENC),
                                fused_graph_caps=_CAPS,
                                inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, amp_dtype="bf16")


@pytest.fixture
def value_visible_engines():
    """Two REAL engines differing ONLY in the sign of the value head."""
    spec = lookup(_ENC)
    positive, negative = _engine(1.0), _engine(-1.0)
    try:
        yield positive, negative, spec
    finally:
        positive.close()
        negative.close()


def _search(engine, spec, board, n_sims):
    """Drive the PRODUCTION entrance and return (played move, root child q values)."""
    player = worker.build_candidate_player(engine, n_sims, spec=spec, leaf_batch_size=1)
    player.new_game()
    move = player.select_move(board)
    tree = player._tree
    assert tree is not None, "the deploy head must retain its tree after select_move"
    return move, [float(q) for _coord, _idx, _prior, _visits, q in tree.get_root_children_info()]


# ── the value channel reaches the tree ────────────────────────────────────────────────
def test_value_channel_reaches_the_tree_at_production_sims(value_visible_engines) -> None:
    """The producer's VALUE half must arrive in the search, not just its policy half.

    A non-degenerate root-child q vector is the smallest thing that cannot be true unless
    per-leaf values crossed the FFI and backed up. Under M-RT1 every leaf value is 0.0, so
    every root child reads q == 0.0 and the vector collapses to a single distinct value —
    which is exactly the shape this asserts against.
    """
    positive, _negative, spec = value_visible_engines
    for pos in _positions():
        _move, qs = _search(positive, spec, _board(pos), _SIMS)
        distinct = {round(q, 9) for q in qs}
        assert len(distinct) >= 2, (
            f"{pos['id']}: every root child reads q={distinct} after {_SIMS} sims — the "
            f"value channel never reached the tree"
        )


# ── flipping the value head moves the head's choice ───────────────────────────────────
def test_flipping_the_value_head_moves_the_heads_choice(value_visible_engines) -> None:
    """The value channel must be LOAD-BEARING on the move actually played, not merely
    present in the tree.

    The two engines share a byte-identical policy head, so every prior — and hence the
    `log(max(prior, 1e-8))` term of `select_argmax_child` — is identical between them. Only
    the sign of the value head differs. A changed move is therefore attributable to the
    value channel and to nothing else. Under M-RT1 both engines see all-zero values, both
    collapse to `argmax log(prior)`, and the moves agree at every position.
    """
    positive, negative, spec = value_visible_engines
    agreed = []
    for pos in _positions():
        board = _board(pos)
        move_pos, _q = _search(positive, spec, board, _SIMS)
        move_neg, _q = _search(negative, spec, _board(pos), _SIMS)
        if move_pos == move_neg:
            agreed.append((pos["id"], move_pos))
    assert not agreed, (
        f"sign-flipping the value head left the played move unchanged at {len(agreed)} of 4 "
        f"positions {agreed} — the value channel is not load-bearing on the head's choice"
    )


# ── ⊕ᶜ CONTROL — the frozen bank's blindness, pinned rather than described ─────────────
def test_the_value_channel_is_inert_at_one_simulation(value_visible_engines) -> None:
    """CONTROL, not an R72 flip: M-RT1 leaves this GREEN by design.

    This is F-RT-1's root cause as an executable statement. At `n_sims=1` every root child
    has `visits=0, q=0`, so `sigma` is identically 0 and the played move cannot depend on any
    value the net produced. The two frozen production-entrance oracles (P-1b, P-3b) both run
    at exactly this parameterization, which is why a value-channel defect survives them.

    Pinned so that a future author cannot add a value oracle at `n_sims=1` and believe the
    channel is covered: if `select_argmax_child` ever grows a value-sensitive term that is
    live at one simulation, this test fails and the analysis above must be revisited.
    """
    positive, negative, spec = value_visible_engines
    for pos in _positions():
        board = _board(pos)
        move_pos, qs = _search(positive, spec, board, _INERT_SIMS)
        move_neg, _q = _search(negative, spec, _board(pos), _INERT_SIMS)
        assert {round(q, 9) for q in qs} == {0.0}, (
            f"{pos['id']}: root q is not uniformly 0 after one simulation"
        )
        assert move_pos == move_neg, (
            f"{pos['id']}: the played move changed with the value sign at one simulation"
        )
