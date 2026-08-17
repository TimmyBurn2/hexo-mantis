"""⊕ WP12-R Phase EVALDECODE (operator ruling R138) — the eval leg of the parity class.

Oracle-first (PREREG_EVALDECODE §1), byte-frozen through IMPL. R138 adopts Option A —
**eval consumes what the shared producer already returns**, and **self-play semantics is
THE authority**. The defect at HEAD is `inference_local.py:260`,
`policies = [dense for dense, _overflow, _value in results]`: the eval graph leg keeps the
dense half of `assemble_ls_from_gnn_probs`'s `LegalSetPolicy` and throws the `overflow` half
away, then expands through `expand_and_backup` (the dense rule) instead of the
`expand_and_backup_ls_at` self-play expands through. Measured at HEAD over the four fixture
positions: **0 off-window root children out of 768**, while 27%-97% of the self-play child
budget goes to moves eval cannot see (PREREG §4).

This file binds to the SAME two committed fixtures the Rust leg binds to
(`crates/mantis-selfplay/tests/graph_child_parity.rs`) — one file, both sides of the FFI.
`expected_children` is a **self-play-authored golden**; it is not independent of the
production code and no mutation can make it so. What the mutations show is sensitivity
(M6', DESIGN §b.3).

Pre-registered HEAD verdicts (PREREG §1). The RED rows here fail because the fix's
production surfaces (`LocalInferenceEngine.infer_batch_ls` / `infer_ls`,
`MCTSTree.expand_and_backup_ls_graph`, `InferenceBatcher.submit_graphs_and_wait_ls`,
`DeployHeadPlayer(expand_fn=...)`, `_build_candidate_player(..., spec=...)`) do not exist
yet — a frozen oracle must bind to the POST-FIX surface, so "the measured 0-off-window
child set" is the *reason* these are red, not their HEAD traceback.

    RED   P-1   test_eval_child_set_equals_the_fixture
    RED   P-1b  test_deploy_head_entrance_reaches_the_same_children
    RED   P-1c  test_both_legs_agree_on_priors_to_1e_5
    RED   P-1d  test_overflow_order_does_not_change_the_child_set
    GREEN P-2a  test_fixture_positions_are_in_the_over_361_regime
    RED   P-2b  test_eval_root_children_include_off_window_moves
    RED   P-2c  test_eval_consumes_both_halves
    RED   P-2d  test_eval_child_set_equals_the_rust_leg_on_dispersed_positions
    GREEN P-2e  test_every_off_window_legal_coord_is_in_overflow
    RED   P-3a  test_head_children_are_drawn_from_the_full_legal_set
    RED   P-3b  test_head_plays_an_off_window_move_against_random_bot
    GREEN P-3c  test_there_is_exactly_one_child_cap_authority
    RED   C-1a-d, C-2, C-3, C-4, C-5a/b, C-6, C-7, C-10
    GREEN C-8   test_no_drop_pooling_encoding_is_still_refused   (CONTROL, not a flip)

R8 >300 justify: ONE class boundary — eval-vs-self-play
consumption of one producer — and R72 requires every conjunct of every predicate the card
ships to be flipped in the same flip-set. Splitting the P-rows from the C-rows would put
the flip-set in a different file from the behaviour it bounds, and would duplicate the
fixture reader, the deterministic stub net and the production expand adapter that every row
below shares. Every helper here is used by both halves.

P-2a/b/d/e and P-1..P-1d loop over their fixture's positions INTERNALLY and are NOT
`@pytest.mark.parametrize`d over them: PREREG §5 counts ONE collected test per id, with
`C-1(x4)` and `C-5(x2)` the only multipliers. Parametrising over positions would break the
pre-registered delta by 12 or more with no defect having occurred.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
import torch

from mantis._engine import Board, MCTSTree
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.bots.random_bot import RandomBot
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: F-816-10 D-1: `LocalInferenceEngine` takes the fused-forward memory bound as a REQUIRED
#: keyword — it hand-builds its `InferenceServer` config with no `RunConfig`, so the spec is
#: THREADED from a parent resolver and never hardcoded at the site. Non-binding by
#: construction here: nothing in this file exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "eval_selfplay_parity"
_P1_FIXTURE = _FIXTURES / "child_parity_v1.json"
_P2_FIXTURE = _FIXTURES / "dispersed_r6_v1.json"

#: `policy_logit_count` is 362 and `Board.to_flat` returns a sentinel above the window, so
#: `to_flat(q, r) >= 361` is exactly "off-window" — the same test the Rust leg applies.
_OFF_WINDOW_FLAT = 361
#: `MAX_CHILDREN_PER_NODE` (`crates/mantis-search/src/mcts/mod.rs:47`). Read here as a
#: literal ON PURPOSE: P-3c asserts that Python owns no second authority for it.
_EXPECTED_CHILDREN = 192
#: DESIGN §b.3 P-1c. Measured at mint over up to 1294 terms: the largest cross-language
#: disagreement between the torch-f32 softmax and the Rust-f32 softmax is 7.3e-10.
_PRIOR_TOL = 1e-5
#: DESIGN §b.3 P-2 / PREREG §2 gate 6: the two fixtures together, enforced not asserted.
_FIXTURE_BYTE_BUDGET = 65536


# ── fixture access ───────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _positions(fx: dict) -> list[dict]:
    """The fixture is minted FLAT (`p0_*`, `p1_*`, ...) so the Rust leg can read it without
    a JSON dependency; this re-nests it. A missing key is a KeyError, never a default."""
    out = []
    for i in range(fx["n_positions"]):
        prefix = f"p{i}_"
        out.append({k[len(prefix):]: v for k, v in fx.items() if k.startswith(prefix)})
    return out


def _board(pos: dict) -> Board:
    """Replay the recorded move sequence — the identical construction the Rust leg performs
    (`Board::with_geometry` at the spec's geometry, then `apply_move`)."""
    board = Board.with_encoding_name(_ENC)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


def _coords(flat: list[int]) -> list[tuple[int, int]]:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _packed(coord: tuple[int, int]) -> int:
    """`backup.rs:148`'s tie-break key — also the fixture's canonical ordering."""
    q, r = coord
    return ((q + 32768) << 16) | ((r + 32768) & 0xFFFF)


# ── the deterministic stub net (the ONE stand-in; everything else is production) ──────
def _rule_logit(i: int) -> float:
    """The fixture's `logit_rule`, over the BUILDER's per-graph legal-node index."""
    return ((i * 37) % 101) / 20.0


class _RuleNet(torch.nn.Module):
    """`GnnNet.forward_batch`'s contract with a deterministic policy head.

    Cross-language byte-parity needs determinism, so the net is the only stand-in in the
    chain: `InferenceServer`'s graph loop, `collate_graph_batch`, `segment_softmax`,
    `assemble_ls_from_gnn_probs` and the expand are all production. The legal rows of each
    graph are contiguous and in builder order (`[stones | legal | dummy]`), which is the
    order `legal_offsets` segments and `assemble` zips against.
    """

    def forward_batch(self, x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            n_legal = int(legal_mask[lo:hi].sum().item())
            logits.extend(_rule_logit(i) for i in range(n_legal))
        return (
            torch.tensor(logits, dtype=torch.float32),
            torch.zeros((n_graphs, 1), dtype=torch.float32),
            torch.zeros((n_graphs, 65), dtype=torch.float32),
        )


@pytest.fixture
def graph_engine():
    """A REAL `LocalInferenceEngine` on the graph spec, driving the production graph seam."""
    spec = lookup(_ENC)
    net = _RuleNet()
    net.eval()
    engine = LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=spec,
                                  fused_graph_caps=_CAPS)
    try:
        yield engine, spec
    finally:
        engine.close()


# ── the production eval decode, in ONE place ─────────────────────────────────────────
def _expand(engine, spec, tree, leaves, *, overflows=None) -> None:
    """The post-fix eval expand: the producer's BOTH halves into the self-play expand."""
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    tree.expand_and_backup_ls_graph(
        dense,
        overflow if overflows is None else overflows,
        values,
        centers,
        spec.policy_logit_count,
        spec.trunk_size,
    )


def _eval_children(engine, spec, board) -> list[tuple[tuple[int, int], float]]:
    """Root children the eval decode produces for `board`, canonically ordered."""
    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    assert len(leaves) == 1, "a fresh root must yield exactly one pending leaf"
    _expand(engine, spec, tree, leaves)
    info = tree.get_root_children_info()
    return sorted(
        ((tuple(coord), float(prior)) for coord, _idx, prior, _visits, _q in info),
        key=lambda row: _packed(row[0]),
    )


# ── ⊕ P-1 ────────────────────────────────────────────────────────────────────────────
def test_eval_child_set_equals_the_fixture(graph_engine) -> None:
    """The eval decode's root children equal the self-play-authored golden, at every
    position of the P-1 fixture. Killing mutations: M1 (the set differs at 4/4, executed),
    M2, M6'."""
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    for pos in _positions(fx):
        board = _board(pos)
        got = [coord for coord, _prior in _eval_children(engine, spec, board)]
        want = _coords(pos["expected_children"])
        assert got == want, f"{pos['id']}: eval child set != the frozen self-play golden"


# ── ⊕ P-1b ───────────────────────────────────────────────────────────────────────────
def test_deploy_head_entrance_reaches_the_same_children(graph_engine) -> None:
    """The PRODUCTION entrance reaches the same children: `_build_candidate_player`'s closed
    match on `spec.representation` must take the graph arm and hand the deploy head an
    `expand_fn`. This is the graph arm of C-7's closed match. Killing mutation: M2."""
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    pos = _positions(fx)[0]
    board = _board(pos)

    player = worker._build_candidate_player(engine, 1, spec=spec)
    assert isinstance(player, DeployHeadPlayer)
    player.new_game()
    player.select_move(board)

    got = sorted(
        (tuple(coord) for coord, _i, _p, _v, _q in player._tree.get_root_children_info()),
        key=_packed,
    )
    assert got == _coords(pos["expected_children"]), (
        f"{pos['id']}: the deploy-head entrance produced a different child set"
    )


# ── ⊕ P-1c ───────────────────────────────────────────────────────────────────────────
def test_both_legs_agree_on_priors_to_1e_5(graph_engine) -> None:
    """Cross-language exactness: the priors the eval leg computes (torch f32 segment
    softmax) equal the Rust leg's frozen priors to 1e-5 over up to 1294 terms.

    LAW-06 is not weakened: autocast is CUDA-gated (`inference_server.py:447-451`) and the
    segment softmax is forced to f32 (`:462-464`), so a CPU run is float32 end to end and
    the bf16 graph pin is not engaged here. Killing mutations: M1, M2.
    """
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    for pos in _positions(fx):
        got = _eval_children(engine, spec, _board(pos))
        want_coords = _coords(pos["expected_children"])
        want_priors = pos["expected_child_priors"]
        assert [c for c, _p in got] == want_coords, f"{pos['id']}: child set precondition"
        for idx, ((coord, prior), want) in enumerate(zip(got, want_priors)):
            assert abs(prior - want) <= _PRIOR_TOL, (
                f"{pos['id']}: child {idx} {coord} prior {prior!r} != frozen {want!r}"
            )


# ── ⊕ P-1d ───────────────────────────────────────────────────────────────────────────
def test_overflow_order_does_not_change_the_child_set(graph_engine) -> None:
    """D-22: the overflow half crosses the FFI as a Vec materialised from map iteration, so
    ORDER enters Python. The bridge must rebuild a map, never scan the vector in order.
    Killing mutation: replace the bridge's map rebuild with an order-honouring scan."""
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    pos = _positions(fx)[0]
    board = _board(pos)

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    _dense, overflow, _values, _centers = engine.infer_batch_ls(leaves)
    shuffled = []
    rng = random.Random(20260731)
    for half in overflow:
        entries = list(half)
        rng.shuffle(entries)
        assert entries != list(half), "the permutation must actually permute"
        shuffled.append(entries)

    tree2 = MCTSTree()
    tree2.new_game(board)
    leaves2 = tree2.select_leaves(1)
    _expand(engine, spec, tree2, leaves2, overflows=shuffled)
    got = sorted(
        (tuple(coord) for coord, _i, _p, _v, _q in tree2.get_root_children_info()),
        key=_packed,
    )
    assert got == _coords(pos["expected_children"]), (
        f"{pos['id']}: the child set depends on overflow wire order"
    )


# ── ⊕ P-2a ───────────────────────────────────────────────────────────────────────────
def test_fixture_positions_are_in_the_over_361_regime() -> None:
    """The dispersed fixture's PRECONDITION, re-derived from the replayed board rather than
    trusted: >361 legal moves and at least one off-window legal move at every position, so
    the regression oracles cannot silently drift into a regime where they cannot fail. The
    R7 byte budget is enforced here too. Killing mutation: M7 (an in-window-only position).
    """
    total = 0
    for path in (_P1_FIXTURE, _P2_FIXTURE):
        size = path.stat().st_size
        total += size
        assert size <= _FIXTURE_BYTE_BUDGET, f"{path.name} is {size} B"
    assert total <= _FIXTURE_BYTE_BUDGET, f"fixtures total {total} B > {_FIXTURE_BYTE_BUDGET}"

    fx = _load(_P2_FIXTURE)
    assert fx["n_positions"] == 4, "PREREG pre-registers four dispersed positions"
    for pos in _positions(fx):
        board = _board(pos)
        legal = board.legal_moves()
        n_off = sum(1 for q, r in legal if board.to_flat(q, r) >= _OFF_WINDOW_FLAT)
        assert len(legal) == pos["n_legal"], f"{pos['id']}: recorded n_legal is a lie"
        assert n_off == pos["n_off_window"], f"{pos['id']}: recorded n_off_window is a lie"
        assert len(legal) > 361, f"{pos['id']}: only {len(legal)} legal moves"
        assert n_off > 0, f"{pos['id']}: no off-window legal move"


# ── ⊕ P-2b ───────────────────────────────────────────────────────────────────────────
def test_eval_root_children_include_off_window_moves(graph_engine) -> None:
    """Measured at HEAD: 0 off-window children of 192 at 4/4 positions. Post-fix at least
    one, and EXACTLY the count self-play produces. M2 reds both conjuncts; M1 reds the count
    conjunct only — not "count -> 0", which is the opposite of what M1 does."""
    engine, spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        board = _board(pos)
        children = [coord for coord, _prior in _eval_children(engine, spec, board)]
        n_off = sum(1 for q, r in children if board.to_flat(q, r) >= _OFF_WINDOW_FLAT)
        assert n_off >= 1, f"{pos['id']}: eval kept {len(children)} children, none off-window"
        assert n_off == pos["expected_off_window_children"], (
            f"{pos['id']}: {n_off} off-window children != self-play's "
            f"{pos['expected_off_window_children']}"
        )


# ── ⊕ P-2c ───────────────────────────────────────────────────────────────────────────
def test_eval_consumes_both_halves(graph_engine) -> None:
    """A "did eval keep BOTH halves" sentinel — not coverage of a defect survivor-counts are
    blind to (REV 1's framing rested on the withdrawn D-2; every in-window child's prior is
    byte-identical between the two consumers at HEAD). Sigma dense alone measures
    0.7155 / 0.4356 / 0.3086 / 0.2363; both halves together sum to 1 because the producer
    validates it always-on (`records.rs:455-463`). Killing mutation: M1."""
    engine, _spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        dense, overflow, _values, _centers = engine.infer_batch_ls([_board(pos)])
        total = sum(dense[0]) + sum(prob for _coord, prob in overflow[0])
        assert abs(total - 1.0) <= 1e-3, f"{pos['id']}: eval consumes mass {total!r}, not 1"


# ── ⊕ P-2d ───────────────────────────────────────────────────────────────────────────
def test_eval_child_set_equals_the_rust_leg_on_dispersed_positions(graph_engine) -> None:
    """The cross-FFI parity claim on the dispersed positions: the eval child set equals the
    set `crates/mantis-selfplay/tests/graph_child_parity.rs` produces from the same file.
    Killing mutations: M1, M2."""
    engine, spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        got = [coord for coord, _prior in _eval_children(engine, spec, _board(pos))]
        assert got == _coords(pos["expected_children"]), (
            f"{pos['id']}: eval child set != the Rust leg's set"
        )


# ── ⊕ P-2e ───────────────────────────────────────────────────────────────────────────
def test_every_off_window_legal_coord_is_in_overflow(graph_engine) -> None:
    """The assumption that keeps the legal-set floor unreachable, pinned.

    `pick_topk_children_ls` reads a coord absent from BOTH halves at
    `1/min(n_legal, 192) = 0.00521` (`legal_set.rs:35`) — about 7x the mean in-window prior
    — so any coverage gap between the builder's legal-node emission and `board.legal_moves()`
    would silently promote the uncovered cells to the TOP of the child list. Measured at
    HEAD: 0 absent coords at 4/4. Driven through `submit_graphs_and_wait`, the producer
    surface that exists at HEAD and is unchanged by the fix, because this row is
    pre-registered GREEN on both sides. Killing mutation: M11.
    """
    engine, _spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        board = _board(pos)
        dense, overflow, _value = engine._graph_batcher.submit_graphs_and_wait(
            [(list(board.get_stones()), int(board.current_player), int(board.moves_remaining))]
        )[0]
        covered = {tuple(coord) for coord, _prob in overflow}
        absent = [
            (q, r)
            for q, r in board.legal_moves()
            if board.to_flat(q, r) >= len(dense) and (q, r) not in covered
        ]
        assert not absent, f"{pos['id']}: {len(absent)} off-window coords read the floor"


# ── ⊕ P-3a ───────────────────────────────────────────────────────────────────────────
def test_head_children_are_drawn_from_the_full_legal_set(graph_engine) -> None:
    """Oracle (iii), tree form: the head's candidates are the top-192 of the FULL legal set
    by true prior, not the top-192 of a 361-cell window. The 192 cap survives and is SHARED
    with self-play (P-3c), so the residual asymmetry is symmetric. Killing mutation: M2 —
    M1 leaves this GREEN, measured, and that is the prediction."""
    engine, spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        board = _board(pos)
        children = [coord for coord, _prior in _eval_children(engine, spec, board)]
        assert len(children) == _EXPECTED_CHILDREN, f"{pos['id']}: {len(children)} children"
        assert any(board.to_flat(q, r) >= _OFF_WINDOW_FLAT for q, r in children), (
            f"{pos['id']}: every root child is inside the 361-cell window"
        )


# ── ⊕ P-3b ───────────────────────────────────────────────────────────────────────────
def test_head_plays_an_off_window_move_against_random_bot(graph_engine) -> None:
    """Oracle (iii), in play: the ladder asymmetry is dead at the head's own seat.

    `RandomBot` samples the FULL legal set (`random_bot.py:24-26`) while a window-confined
    head cannot answer off-window at all. Under R147 the RandomBot floor is ARMED for run5,
    so this oracle's subject is a production rung (`_play_random_floor` reaches the same
    `DeployHeadPlayer`); the armed VALUE is mint-prereg and is set nowhere here. At HEAD
    this is structurally impossible: `select_argmax_child` iterates
    `get_root_children_info()` only, and none of those is off-window. Killing mutation: M2.
    """
    engine, spec = graph_engine
    pos = _positions(_load(_P2_FIXTURE))[3]
    board = _board(pos)
    head_seat = int(board.current_player)
    player = worker._build_candidate_player(engine, 1, spec=spec)
    player.new_game()
    bot = RandomBot(seed=20260731)

    off_window_head_moves = []
    for _ply in range(8):
        if board.winner() is not None or not board.legal_moves():
            break
        if int(board.current_player) == head_seat:
            move = player.select_move(board)
            if board.to_flat(*move) >= _OFF_WINDOW_FLAT:
                off_window_head_moves.append(move)
        else:
            move = bot.select_move(board)
        board.apply_move(*move)

    assert off_window_head_moves, (
        f"{pos['id']}: the head played no off-window move in 8 plies from its own seat"
    )


# ── ⊕ P-3c ───────────────────────────────────────────────────────────────────────────
def test_there_is_exactly_one_child_cap_authority() -> None:
    """The 192 cap has ONE definition and Python owns no second one.

    The `^pub const` anchor is load-bearing: it stops the `pub` re-export at
    `mantis-search/src/lib.rs:21` and the uses in `tests/pool_overflow.rs` / `mcts/tests.rs`
    being miscounted as further authorities. Killing mutation: M9 — add
    `MAX_CHILDREN = 192` under `src/mantis/eval/`.
    """
    root = Path(__file__).resolve().parents[2]
    named = re.compile(r"\b(MAX_CHILDREN|max_children|CHILD_CAP|child_cap|n_children_cap)\b")
    bare = re.compile(r"(?<![\w.])192(?![\w.])")
    hits = []
    for package in ("eval", "arena", "bots"):
        for path in sorted((root / "src" / "mantis" / package).rglob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if named.search(line) or bare.search(line):
                    hits.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not hits, "a second child-cap authority appeared in Python:\n" + "\n".join(hits)

    definitions = [
        f"{path.relative_to(root)}:{lineno}"
        for path in sorted((root / "crates").rglob("*.rs"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if line.startswith("pub const MAX_CHILDREN_PER_NODE")
    ]
    assert definitions == ["crates/mantis-search/src/mcts/mod.rs:47"], definitions


# ── ⊕ C-1a-d ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("short_arg", ["policies", "overflows", "values", "centers"])
def test_expand_ls_graph_arity_conjuncts_are_enforced(graph_engine, short_arg) -> None:
    """Each of the four arity conjuncts on the new tree surface, flipped one at a time. The
    inner `expand_and_backup_ls_at` takes the MIN of every length and silently expands
    fewer leaves, so the bridge guard must be always-on."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _EXPECTED_CHILDREN, "clean-call control"

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    args = {"policies": dense, "overflows": overflow, "values": values, "centers": centers}
    args[short_arg] = []
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            args["policies"], args["overflows"], args["values"], args["centers"],
            spec.policy_logit_count, spec.trunk_size,
        )


# ── ⊕ C-2 ────────────────────────────────────────────────────────────────────────────
def test_expand_ls_graph_refuses_a_centre_the_board_disagrees_with(graph_engine) -> None:
    """D-7: self-play frames its expand on the BUILDER's `g.window_center`; eval re-derives
    from `board.window_center()`. The producer now returns its own centre and the bridge
    cross-checks it — a pairing/drift tripwire, expected always-equal. Mutation M3."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _EXPECTED_CHILDREN, "clean-call control"

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    drifted = [(int(cq) + 1, int(cr)) for cq, cr in centers]
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            dense, overflow, values, drifted, spec.policy_logit_count, spec.trunk_size,
        )


# ── ⊕ C-3 ────────────────────────────────────────────────────────────────────────────
def test_expand_ls_graph_refuses_a_trunk_the_board_disagrees_with(graph_engine) -> None:
    """D-8: self-play asserts `agg_trunk_sz == spec.trunk_size` always-on
    (`search_drive.rs:415-419`); eval read `board.cluster_window_size()`. Both are 19 at
    run5 (measured), so this guard is a drift tripwire. Mutation M4."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _EXPECTED_CHILDREN, "clean-call control"

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            dense, overflow, values, centers, spec.policy_logit_count, 25,
        )


# ── ⊕ C-4 ────────────────────────────────────────────────────────────────────────────
def test_expand_ls_graph_refuses_a_dense_half_of_the_wrong_stride(graph_engine) -> None:
    """A 361-long dense half against a 362-wide policy stride is the v6w25 class of silent
    wrong-width decode Phase B killed on the grid seam; it must be loud here too."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _EXPECTED_CHILDREN, "clean-call control"

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    narrowed = [list(half[:-1]) for half in dense]
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            narrowed, overflow, values, centers, spec.policy_logit_count, spec.trunk_size,
        )


# ── ⊕ C-5a/b ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["neither", "both"])
def test_deploy_head_takes_exactly_one_collaborator(case) -> None:
    """`DeployHeadPlayer` takes EXACTLY one of `infer_fn=` (dense, unchanged) or
    `expand_fn=` (graph). Neither and both are named `ValueError`s — no default arm, no
    silent pick, which is the whole reason a second player class was rejected."""
    def _infer(_leaf):
        raise AssertionError("the guard must fire before any inference")

    def _expand_fn(_tree, _leaves):
        raise AssertionError("the guard must fire before any expand")

    kwargs = {} if case == "neither" else {"infer_fn": _infer, "expand_fn": _expand_fn}
    with pytest.raises(ValueError):
        DeployHeadPlayer(n_sims=1, **kwargs)


# ── ⊕ C-6 ────────────────────────────────────────────────────────────────────────────
def test_infer_batch_ls_refuses_a_dense_spec() -> None:
    """The no-drop graph decode has no grid analogue; a dense spec must die by name here
    rather than an `AttributeError` two lines down. The refusal reads the BOUND SPEC, never
    the live model object — which is why the model handed in below is an `Identity`."""
    engine = LocalInferenceEngine(
        torch.nn.Identity(), torch.device("cpu"), encoding_spec=lookup("v6"),
        fused_graph_caps=None,
    )
    try:
        with pytest.raises(NotImplementedError):
            engine.infer_batch_ls([Board.with_encoding_name("v6")])
    finally:
        engine.close()


# ── ⊕ C-7 ────────────────────────────────────────────────────────────────────────────
def test_build_candidate_player_closed_match_refuses_an_unknown_representation() -> None:
    """`_build_candidate_player` matches CLOSED on `spec.representation`: an unregistered
    representation raises by name and NEVER falls back to the dense arm. A silent dense
    fallback here is exactly the class R138 ruled on."""
    class _SpecWithRepresentation:
        def __init__(self, base, representation):
            self._base = base
            self.representation = representation

        def __getattr__(self, item):
            return getattr(self._base, item)

    spec = _SpecWithRepresentation(lookup("v6"), "quantum")
    engine = LocalInferenceEngine(
        torch.nn.Identity(), torch.device("cpu"), encoding_spec=lookup("v6"),
        fused_graph_caps=None,
    )
    try:
        with pytest.raises(EvalDecodeUnsupportedError):
            worker._build_candidate_player(engine, 2, spec=spec)
    finally:
        engine.close()


# ── ⊕ C-10 ───────────────────────────────────────────────────────────────────────────
def test_infer_ls_is_the_same_refusal_predicate_as_infer_batch_ls(graph_engine) -> None:
    """ONE refusal predicate with TWO entry points, asserted as a delegation rather than
    duplicated: `infer_ls` is a one-line delegation to `infer_batch_ls`, so a future edit
    cannot give the single-board door a different (or absent) guard."""
    engine, _spec = graph_engine
    calls = []

    def _recording(boards):
        calls.append(list(boards))
        return ([[0.0]], [[]], [0.0], [(0, 0)])

    engine.infer_batch_ls = _recording
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    result = engine.infer_ls(board)
    assert calls == [[board]], "infer_ls did not delegate to infer_batch_ls"
    assert result == ([0.0], [], 0.0, (0, 0)), "infer_ls did not project the batch result"

    dense_engine = LocalInferenceEngine(
        torch.nn.Identity(), torch.device("cpu"), encoding_spec=lookup("v6"),
        fused_graph_caps=None,
    )
    try:
        with pytest.raises(NotImplementedError) as single:
            dense_engine.infer_ls(Board.with_encoding_name("v6"))
        with pytest.raises(NotImplementedError) as batch:
            dense_engine.infer_batch_ls([Board.with_encoding_name("v6")])
        assert str(single.value) == str(batch.value), "two entry points, two messages"
    finally:
        dense_engine.close()


# ── ⊕ᶜ C-8 (CONTROL — an unchanged predicate still fires; NOT an R72 flip) ────────────
def test_no_drop_pooling_encoding_is_still_refused() -> None:
    """R20 boundary: `v6_live2_ls` declares the no-drop grid pool and stays REFUSED. This
    card wires the no-drop GRAPH decode and does not touch ADJ-WP12R-4's grid seam; the
    refusal set `{"none", "scatter_max"}` is unchanged and both asserted substrings survive
    the D-20 message re-point. The full-round form of this control is O-8
    (`tests/eval/test_graph_round_encoding.py`, committed and NOT edited by this card); the
    guard entrance is asserted here so the control costs no round."""
    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        worker._assert_decode_implements_declared_pooling(lookup("v6_live2_ls"))
    message = str(excinfo.value)
    assert "v6_live2_ls" in message, message
    assert "legal_set_scatter_max" in message, message
    assert worker._DECODE_IMPLEMENTED_POLICY_POOLS == frozenset({"none", "scatter_max"})
