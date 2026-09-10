"""The eval leg of the eval-vs-self-play parity class: eval consumes what the producer returns.

Self-play semantics is THE authority. The defect this file bounds is an eval graph leg that kept
the dense half of the producer's `LegalSetPolicy`, threw the `overflow` half away, and expanded
through the dense rule. Measured over the four fixture positions: 0 off-window root children out
of 768, while 27%-97% of the self-play child budget went to moves eval could not see.

This file binds to the SAME two committed fixtures the Rust leg binds to — one file, both sides of
the FFI. `expected_children` is a self-play-authored golden, not independent of the production
code; what the mutations show is sensitivity.

>300 justify (R8): ONE class boundary, and every conjunct of every predicate the card ships must
be flipped in the same flip-set. Splitting the parity rows from the conjunct rows would put the
flip-set in a different file from the behaviour it bounds and duplicate the fixture reader, the
deterministic stub net and the production expand adapter every row shares. The parity rows loop
over positions INTERNALLY rather than parametrising, because the prereg counts one test per id.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board, MCTSTree
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.bots.random_bot import RandomBot
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: `LocalInferenceEngine` takes the fused-forward memory bound as a REQUIRED keyword — it
#: hand-builds its `InferenceServer` config with no `RunConfig`, so the spec is THREADED from a
#: parent resolver and never hardcoded at the site. Nothing here exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "eval_selfplay_parity"
_P1_FIXTURE = _FIXTURES / "child_parity_v1.json"
_P2_FIXTURE = _FIXTURES / "dispersed_r6_v1.json"

#: `policy_logit_count` is 362 and `Board.to_flat` returns a sentinel above the window, so
#: `to_flat(q, r) >= 361` is exactly "off-window" — the same test the Rust leg applies.
_OFF_WINDOW_FLAT = 361
#: `MAX_CHILDREN_PER_NODE`, read here as a literal ON PURPOSE: the cap-authority row parses the
#: Rust source for the constant and asserts Python owns no second authority for it.
_CHILD_CAP = 1024


def _expected_children(board: Board) -> int:
    """How many root children a position must produce: `min(n_legal, K)`. A literal was correct
    while every fixture position sat above the cap; raising the cap past two of them would make a
    bare `_CHILD_CAP` assert a count no position can reach."""
    return min(len(board.legal_moves()), _CHILD_CAP)
#: Measured at mint over up to 1294 terms: the largest cross-language disagreement between the
#: torch-f32 softmax and the Rust-f32 softmax is 7.3e-10.
_PRIOR_TOL = 1e-5
#: The two fixtures together, enforced not asserted. RE-DERIVED when the per-node cap was raised,
#: not loosened: a position's frozen child set is `min(n_legal, K)` coords, so the fixtures grew by
#: the same factor. The budget's job is R7 hygiene — keep a committed fixture small enough to read.
_FIXTURE_BYTE_BUDGET = 131072


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _positions(fx: dict) -> list[dict]:
    """Re-nest the fixture, which is minted FLAT (`p0_*`, `p1_*`, ...) so the Rust leg can read it
    without a JSON dependency. A missing key is a KeyError, never a default."""
    out = []
    for i in range(fx["n_positions"]):
        prefix = f"p{i}_"
        out.append({k[len(prefix):]: v for k, v in fx.items() if k.startswith(prefix)})
    return out


def _board(pos: dict) -> Board:
    """Replay the recorded move sequence — the identical construction the Rust leg performs."""
    board = Board.with_encoding_name(_ENC)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


def _coords(flat: list[int]) -> list[tuple[int, int]]:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _packed(coord: tuple[int, int]) -> int:
    """The backup tie-break key, which is also the fixture's canonical ordering."""
    q, r = coord
    return ((q + 32768) << 16) | ((r + 32768) & 0xFFFF)


def _rule_logit(i: int) -> float:
    """The fixture's `logit_rule`, over the BUILDER's per-graph legal-node index."""
    return ((i * 37) % 101) / 20.0


class _RuleNet(torch.nn.Module):
    """`GnnNet.forward_batch`'s contract with a deterministic policy head — the ONE stand-in.

    Cross-language byte-parity needs determinism; the graph loop, `collate_graph_batch`,
    `segment_softmax`, `assemble_ls_from_gnn_probs` and the expand are all production. The legal
    rows of each graph are contiguous and in builder order, which is the order `legal_offsets`
    segments and `assemble` zips against.
    """

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            # `legal_index` is the wire's `legal_node_gather`: the ROWS of the legal nodes, not a
            # dense mask. The gather is strictly ascending, hence unique, so counting entries in
            # this graph's `[lo, hi)` row range equals summing a mask's bits over it.
            n_legal = int(((legal_index >= lo) & (legal_index < hi)).sum().item())
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
                                  fused_graph_caps=_CAPS,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8,
                                  )
    try:
        yield engine, spec
    finally:
        engine.close()


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


def test_eval_child_set_equals_the_fixture(graph_engine) -> None:
    """The eval decode's root children equal the self-play-authored golden at every position of
    the parity fixture."""
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    for pos in _positions(fx):
        board = _board(pos)
        got = [coord for coord, _prior in _eval_children(engine, spec, board)]
        want = _coords(pos["expected_children"])
        assert got == want, f"{pos['id']}: eval child set != the frozen self-play golden"


def test_deploy_head_entrance_reaches_the_same_children(graph_engine) -> None:
    """The PRODUCTION entrance reaches the same children: `build_candidate_player`'s closed match
    on `spec.representation` must take the graph arm and hand the deploy head an `expand_fn`."""
    engine, spec = graph_engine
    fx = _load(_P1_FIXTURE)
    pos = _positions(fx)[0]
    board = _board(pos)

    player = worker.build_candidate_player(engine, 1, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)
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


def test_both_legs_agree_on_priors_to_1e_5(graph_engine) -> None:
    """The priors the eval leg computes equal the Rust leg's frozen priors to 1e-5 over up to 1294
    terms. LAW-06 is not weakened: autocast is CUDA-gated and the segment softmax is forced to f32,
    so a CPU run is float32 end to end."""
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


def test_overflow_order_does_not_change_the_child_set(graph_engine) -> None:
    """The overflow half crosses the FFI as a Vec materialised from map iteration, so ORDER enters
    Python. The bridge must rebuild a map, never scan the vector in order."""
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


def test_fixture_positions_are_in_the_over_361_regime() -> None:
    """The dispersed fixture's PRECONDITION, re-derived from the replayed board rather than trusted:
    >361 legal moves and at least one off-window legal move at every position, so the oracles cannot
    drift into a regime where they cannot fail. The R7 byte budget is enforced here too."""
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


def test_eval_root_children_include_off_window_moves(graph_engine) -> None:
    """At least one off-window root child, and EXACTLY the count self-play produces. Measured
    before the fix: 0 off-window children of 192 at 4/4 positions."""
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


def test_eval_consumes_both_halves(graph_engine) -> None:
    """A "did eval keep BOTH halves" sentinel: the two halves sum to 1, because the producer
    validates that always-on. Dense alone measures 0.7155 / 0.4356 / 0.3086 / 0.2363."""
    engine, _spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        dense, overflow, _values, _centers = engine.infer_batch_ls([_board(pos)])
        total = sum(dense[0]) + sum(prob for _coord, prob in overflow[0])
        assert abs(total - 1.0) <= 1e-3, f"{pos['id']}: eval consumes mass {total!r}, not 1"


def test_eval_child_set_equals_the_rust_leg_on_dispersed_positions(graph_engine) -> None:
    """The cross-FFI parity claim on the dispersed positions: the eval child set equals the set
    `crates/mantis-selfplay/tests/graph_child_parity.rs` produces from the same file."""
    engine, spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        got = [coord for coord, _prior in _eval_children(engine, spec, _board(pos))]
        assert got == _coords(pos["expected_children"]), (
            f"{pos['id']}: eval child set != the Rust leg's set"
        )


def test_every_off_window_legal_coord_is_in_overflow(graph_engine) -> None:
    """Every off-window legal coord appears in the overflow half — the assumption that keeps the
    legal-set floor unreachable.

    A coord absent from BOTH halves reads the floor at `1/min(n_legal, 192)`, about 7x the mean
    in-window prior, so any coverage gap between the builder's legal-node emission and
    `board.legal_moves()` would silently promote the uncovered cells to the TOP of the child list.
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


def test_head_children_are_drawn_from_the_full_legal_set(graph_engine) -> None:
    """The head's candidates are the top-K of the FULL legal set by true prior, not the top-K of a
    361-cell window. The cap survives and is SHARED with self-play, so the residual asymmetry is
    symmetric."""
    engine, spec = graph_engine
    for pos in _positions(_load(_P2_FIXTURE)):
        board = _board(pos)
        children = [coord for coord, _prior in _eval_children(engine, spec, board)]
        assert len(children) == _expected_children(board), (
            f"{pos['id']}: {len(children)} children")
        assert any(board.to_flat(q, r) >= _OFF_WINDOW_FLAT for q, r in children), (
            f"{pos['id']}: every root child is inside the 361-cell window"
        )


def test_head_plays_an_off_window_move_against_random_bot(graph_engine) -> None:
    """The ladder asymmetry is dead at the head's own seat, in play.

    `RandomBot` samples the FULL legal set while a window-confined head cannot answer off-window at
    all, and the RandomBot floor is an armed production rung reaching this same player. Before the
    fix this was structurally impossible: the answer came from `get_top_visits(1)`, which ranks the
    root's own children, and none of those was off-window.
    """
    engine, spec = graph_engine
    pos = _positions(_load(_P2_FIXTURE))[3]
    board = _board(pos)
    head_seat = int(board.current_player)
    player = worker.build_candidate_player(engine, 1, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)
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


def test_there_is_exactly_one_child_cap_authority() -> None:
    """The per-node child cap has ONE definition and Python owns no second one. The `^pub const`
    anchor is load-bearing: it stops the crate's `pub` re-export and its own test uses being
    miscounted as further authorities."""
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

    # The subject is "exactly ONE definition, and it is in the search crate" — the FILE and the
    # COUNT, never the LINE. This once pinned a line number, and a doc comment added above the
    # constant moved it and reddened a test with no opinion about doc comments (derive-or-delete).
    definitions = [
        str(path.relative_to(root))
        for path in sorted((root / "crates").rglob("*.rs"))
        for line in path.read_text().splitlines()
        if line.startswith("pub const MAX_CHILDREN_PER_NODE")
    ]
    assert definitions == ["crates/mantis-search/src/mcts/mod.rs"], definitions


@pytest.mark.parametrize("short_arg", ["policies", "overflows", "values", "centers"])
def test_expand_ls_graph_arity_conjuncts_are_enforced(graph_engine, short_arg) -> None:
    """Each of the four arity conjuncts on the tree surface, flipped one at a time. The inner
    `expand_and_backup_ls_at` takes the MIN of every length and silently expands fewer leaves, so
    the bridge guard must be always-on."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _expected_children(board), (
        "clean-call control")

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


def test_expand_ls_graph_refuses_a_centre_the_board_disagrees_with(graph_engine) -> None:
    """Self-play frames its expand on the BUILDER's `g.window_center` while eval re-derives from
    `board.window_center()`, so the producer returns its own centre and the bridge cross-checks it.
    A pairing/drift tripwire, expected always-equal."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _expected_children(board), (
        "clean-call control")

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    drifted = [(int(cq) + 1, int(cr)) for cq, cr in centers]
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            dense, overflow, values, drifted, spec.policy_logit_count, spec.trunk_size,
        )


def test_expand_ls_graph_refuses_a_trunk_the_board_disagrees_with(graph_engine) -> None:
    """Self-play asserts `agg_trunk_sz == spec.trunk_size` always-on while eval read
    `board.cluster_window_size()`. Both measure 19 here, so this guard is a drift tripwire."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _expected_children(board), (
        "clean-call control")

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            dense, overflow, values, centers, spec.policy_logit_count, 25,
        )


def test_expand_ls_graph_refuses_a_dense_half_of_the_wrong_stride(graph_engine) -> None:
    """A 361-long dense half against a 362-wide policy stride is the silent wrong-width decode
    class; it must be loud here too."""
    engine, spec = graph_engine
    board = _board(_positions(_load(_P1_FIXTURE))[0])
    assert len(_eval_children(engine, spec, board)) == _expected_children(board), (
        "clean-call control")

    tree = MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    dense, overflow, values, centers = engine.infer_batch_ls(leaves)
    narrowed = [list(half[:-1]) for half in dense]
    with pytest.raises(ValueError):
        tree.expand_and_backup_ls_graph(
            narrowed, overflow, values, centers, spec.policy_logit_count, spec.trunk_size,
        )


@pytest.mark.parametrize("case", ["neither", "both"])
def test_deploy_head_takes_exactly_one_collaborator(case) -> None:
    """`DeployHeadPlayer` takes EXACTLY one of `infer_fn=` (dense) or `expand_fn=` (graph).
    Neither and both are named `ValueError`s — no default arm and no silent pick, which is the
    whole reason a second player class was rejected."""
    def _infer(_leaf):
        raise AssertionError("the guard must fire before any inference")

    def _expand_fn(_tree, _leaves):
        raise AssertionError("the guard must fire before any expand")

    kwargs = {} if case == "neither" else {"infer_fn": _infer, "expand_fn": _expand_fn}
    with pytest.raises(ValueError):
        DeployHeadPlayer(n_sims=1, **kwargs, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)


def test_build_candidate_player_closed_match_refuses_an_unknown_representation() -> None:
    """`build_candidate_player` matches CLOSED on `spec.representation`: an unregistered
    representation raises by name and NEVER falls back to the dense arm."""
    class _SpecWithRepresentation:
        def __init__(self, base, representation):
            self._base = base
            self.representation = representation

        def __getattr__(self, item):
            return getattr(self._base, item)

    spec = _SpecWithRepresentation(lookup("gnn_axis_v1"), "quantum")
    engine = LocalInferenceEngine(
        torch.nn.Identity(), torch.device("cpu"), encoding_spec=lookup("gnn_axis_v1"),
        fused_graph_caps=_CAPS,
        inference_batching=InferenceBatchingSpec(inference_batch_size=64,
                                                 inference_max_wait_ms=10),
        max_in_flight=8, )
    try:
        with pytest.raises(EvalDecodeUnsupportedError):
            worker.build_candidate_player(engine, 2, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)
    finally:
        engine.close()


def test_infer_ls_is_the_same_refusal_predicate_as_infer_batch_ls(graph_engine) -> None:
    """ONE predicate with TWO entry points, asserted as a delegation rather than duplicated:
    `infer_ls` is a one-line delegation to `infer_batch_ls`, so a future edit cannot give the
    single-board door a different (or absent) guard."""
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


def test_no_drop_pooling_encoding_is_still_refused() -> None:
    """A spec declaring the no-drop pool stays REFUSED, and the refusal set is unchanged. No
    registered encoding declares an unimplemented pool any more, so the case is SYNTHESISED from a
    registered spec; the guard entrance is asserted here so the control costs no round."""
    import dataclasses

    base = lookup("gnn_axis_v1")
    if dataclasses.is_dataclass(base):
        spec = dataclasses.replace(base, policy_pool="legal_set_scatter_max")
    else:
        class _Shim:
            def __init__(self, inner):
                self._inner = inner
                self.policy_pool = "legal_set_scatter_max"

            def __getattr__(self, item):
                return getattr(self._inner, item)

        spec = _Shim(base)

    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        worker._assert_decode_implements_declared_pooling(spec)
    message = str(excinfo.value)
    assert "gnn_axis_v1" in message, message
    assert "legal_set_scatter_max" in message, message
    assert worker._DECODE_IMPLEMENTED_POLICY_POOLS == frozenset({"none", "scatter_max"})
