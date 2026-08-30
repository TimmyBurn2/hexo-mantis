# >300 justify: the replay, the label stamp, the manifest handshake and the provenance are
# ONE producer's contract, and the rows that would let a defect through are the ones that
# span two of them (a desynchronised replay produces valid-looking labels). Split, each half
# would pass over the seam the other owns.
"""NIGHTRUN-1 Leg 3 — the bootstrap corpus encoder is CAPABILITY, and it refuses.

WHAT THIS FILE IS THE ONLY WITNESS TO. SITTING4-PREP-1 §3.5 measured three gaps between the
R279-CERTIFIED corpus and any bootstrap use of it, and the first was that **the encoder does
not exist**. This is that encoder's oracle. Its rows fall into three groups and each group
exists for a defect the others cannot see:

  * **REPLAY FIDELITY** — a move list replayed through the production `Board` must produce
    positions in step with their own labels. A skipped or coerced move desynchronises every
    LATER position from its outcome, which is invisible downstream: the arrays are the right
    shape, the values are in range, and the net learns a position/label pairing that never
    occurred.
  * **THE LABEL'S AUTHORITY** — `outcome` comes from `graph_row_outcome`, which IS
    `finalize_graph_outcome`. The row that matters asserts the SIGN ALTERNATES with the side
    to move, because a corpus encoder that stamped one sign for the whole game would train
    the value head to predict the winner rather than the position's value.
  * **UNARMEDNESS** — asserted STRUCTURALLY over the import graph (R296(f)), not by grep. The
    module must be reachable from nothing in `src/` but its own package: capability, not
    posture. Landing is not arming.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from mantis.data.bootstrap_encode import (
    CorpusEncodeError,
    encode_corpus,
    encode_game,
    sha256_of,
)

_REPO = Path(__file__).resolve().parents[2]
_ENC = "gnn_axis_v1"


def _board_factory():
    from mantis._engine import Board

    return lambda: Board.with_encoding_name(_ENC)


def _legal_walk(n: int, *, seed: int = 20260831) -> list[tuple[int, int]]:
    """`n` stones of REAL legal play — the encoder replays through the production rules, so
    a synthetic coordinate list would be testing the refusal path, not the encode path."""
    import random

    from mantis._engine import Board

    rng = random.Random(seed)
    board = Board.with_encoding_name(_ENC)
    out: list[tuple[int, int]] = []
    for _ in range(n):
        legal = board.legal_moves()
        if not legal or board.check_win():
            break
        mv = rng.choice(legal)
        out.append((int(mv[0]), int(mv[1])))
        board.apply_move(*mv)
    return out


# ── replay fidelity ──────────────────────────────────────────────────────────────────────
def test_one_row_per_ply_in_placement_order() -> None:
    moves = _legal_walk(24)
    rows = list(encode_game(moves, 1, board_factory=_board_factory()))
    assert len(rows) == len(moves)
    for ply, (row, mv) in enumerate(zip(rows, moves, strict=True)):
        stones, visits, _cp, _mr, ply_index, _fs, _out, _vv, game_length = row
        assert ply_index == ply
        assert game_length == len(moves)
        assert len(stones) == ply, (
            "each row must carry the position BEFORE its own move; a row carrying the move "
            "it is the label for is a leak the net would learn instead of the game"
        )
        assert visits == [(mv[0], mv[1], 1.0)], (
            "the policy target must be a one-hot on the stone actually played"
        )


def test_the_policy_target_is_a_DISTRIBUTION_the_engine_accepts() -> None:
    """The ring refuses a `visits` row that is not a distribution
    (`refuse_non_distribution_row`). This drives the real push rather than asserting about
    the tuple, because "the engine accepts it" is the only claim that matters."""
    from mantis._engine import HexgBuffer

    moves = _legal_walk(12)
    buf = HexgBuffer(64, _ENC, 8)
    for row in encode_game(moves, -1, board_factory=_board_factory()):
        buf.push_graph_position(*row, game_id=-1)
    size, _capacity, _hist = buf.get_buffer_stats()
    assert size == len(moves)


# ── the label's authority ────────────────────────────────────────────────────────────────
def test_the_outcome_sign_ALTERNATES_with_the_side_to_move() -> None:
    """THE ROW THAT MATTERS. A corpus encoder that stamped one sign for the whole game would
    train the value head to predict the winner rather than the position's value, and every
    array would still be in range."""
    moves = _legal_walk(20)
    rows = list(encode_game(moves, 1, board_factory=_board_factory()))
    by_player = {1: set(), -1: set()}
    for _s, _v, cp, _mr, _pi, _fs, outcome, value_valid, _gl in rows:
        by_player[cp].add(outcome)
        assert value_valid is True, "a decided corpus game supervises every row"
    assert by_player[1] == {1.0}, by_player
    assert by_player[-1] == {-1.0}, by_player


def test_the_outcome_comes_from_the_RUST_authority_not_a_transcription() -> None:
    """One authority for the sign convention. If the module ever computes the value itself,
    the §178 split is transcribed and will drift the first time it moves."""
    source = (_REPO / "src" / "mantis" / "data" / "bootstrap_encode.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "graph_row_outcome"
    ]
    assert calls, "the encoder no longer calls graph_row_outcome"
    # and the value it yields must BE that call's result, never a literal beside it
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    assert any(
        isinstance(a.value, ast.Call)
        and getattr(a.value.func, "id", None) == "graph_row_outcome"
        for a in assigns
    ), "the outcome is not bound from the authority's own return"


def test_a_loss_flips_every_sign() -> None:
    moves = _legal_walk(10)
    won = list(encode_game(moves, 1, board_factory=_board_factory()))
    lost = list(encode_game(moves, -1, board_factory=_board_factory()))
    for w, l in zip(won, lost, strict=True):
        assert w[6] == -l[6], "the same position under opposite winners must flip its value"


# ── refusals: each one is a label that would otherwise be silently wrong ──────────────────
def test_an_illegal_move_is_REFUSED_and_names_its_ply() -> None:
    moves = _legal_walk(8)
    moves[4] = (10_000, 10_000)
    with pytest.raises(CorpusEncodeError, match="not legal on the replayed board"):
        list(encode_game(moves, 1, board_factory=_board_factory()))


def test_a_repeated_cell_is_REFUSED_rather_than_skipped() -> None:
    """The sharpest replay desync: a move on an occupied cell. Skipping it would keep every
    later position one ply ahead of its own index and one behind its own stones."""
    moves = _legal_walk(8)
    moves[5] = moves[0]
    with pytest.raises(CorpusEncodeError, match="not legal on the replayed board"):
        list(encode_game(moves, 1, board_factory=_board_factory()))


@pytest.mark.parametrize("winner", [0, 2, -2, None, "1", True])
def test_a_winner_outside_the_contract_is_REFUSED(winner) -> None:
    from mantis.data.bootstrap_encode import _require_record

    with pytest.raises(CorpusEncodeError, match="winner"):
        _require_record({"game_hash": "g", "winner": winner, "moves": [[0, 0]]}, 0)


@pytest.mark.parametrize("bad", [
    {"winner": 1, "moves": [[0, 0]]},                       # no game_hash
    {"game_hash": "", "winner": 1, "moves": [[0, 0]]},      # empty game_hash
    {"game_hash": "g", "winner": 1},                        # no moves
    {"game_hash": "g", "winner": 1, "moves": []},           # empty moves
    {"game_hash": "g", "winner": 1, "moves": [[0]]},        # short move
    {"game_hash": "g", "winner": 1, "moves": [["0", 0]]},   # non-int coord
])
def test_a_record_missing_its_contract_is_REFUSED(bad) -> None:
    from mantis.data.bootstrap_encode import _require_record

    with pytest.raises(CorpusEncodeError):
        _require_record(bad, 0)


# ── the manifest handshake + provenance ──────────────────────────────────────────────────
def _dataset(tmp_path: Path, records: list[dict], *, shape: str = "B",
             corrupt_sha: bool = False) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    rec_path = d / "games.jsonl"
    rec_path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    sha = sha256_of(rec_path)
    if corrupt_sha:
        sha = "0" * 64
    manifest = ({"file": "games.jsonl", "sha256": sha}
                if shape == "B" else {"files": [{"path": "games.jsonl", "sha256": sha}]})
    (d / "dataset_metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _records(n: int) -> list[dict]:
    return [
        {"game_hash": f"g{i}", "winner": 1 if i % 2 == 0 else -1,
         "moves": [[q, r] for q, r in _legal_walk(10, seed=100 + i)]}
        for i in range(n)
    ]


@pytest.mark.parametrize("shape", ["A", "B"])
def test_both_declared_manifest_shapes_are_accepted(tmp_path: Path, shape: str) -> None:
    """The audit declares two shapes and requires exactly one. Accepting either here is the
    audit's rule reused, not a widening invented at the encoder."""
    d = _dataset(tmp_path, _records(3), shape=shape)
    prov = encode_corpus(d, tmp_path / "out.hexg", encoding=_ENC, capacity=512,
                         visit_capacity=8)
    assert prov["games"] == 3


def test_a_manifest_matching_BOTH_shapes_is_REFUSED(tmp_path: Path) -> None:
    d = _dataset(tmp_path, _records(1))
    m = json.loads((d / "dataset_metadata.json").read_text(encoding="utf-8"))
    m["files"] = [{"path": "games.jsonl", "sha256": m["sha256"]}]
    (d / "dataset_metadata.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(CorpusEncodeError, match="exactly one of the declared shapes"):
        encode_corpus(d, tmp_path / "out.hexg", encoding=_ENC, capacity=64, visit_capacity=8)


def test_a_source_sha_MISMATCH_is_REFUSED(tmp_path: Path) -> None:
    """R279's certification handshake. A corpus that does not match its pin is a DIFFERENT
    corpus, and the certification is about the pinned bytes."""
    d = _dataset(tmp_path, _records(2), corrupt_sha=True)
    with pytest.raises(CorpusEncodeError, match="certification handshake"):
        encode_corpus(d, tmp_path / "out.hexg", encoding=_ENC, capacity=64, visit_capacity=8)


def test_the_provenance_sidecar_makes_the_artifact_checkable(tmp_path: Path) -> None:
    from mantis._engine import registry_sha_hex

    d = _dataset(tmp_path, _records(4))
    out = tmp_path / "corpus.hexg"
    prov = encode_corpus(d, out, encoding=_ENC, capacity=512, visit_capacity=8)
    sidecar = out.with_name(out.name + ".provenance.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == prov
    assert prov["artifact_sha256"] == sha256_of(out)
    assert prov["source_sha256"] == prov["source_sha256_declared"]
    assert prov["encoding"] == _ENC
    assert prov["registry_sha"] == registry_sha_hex(), (
        "an artifact encoded under a different registry is a different artifact; the sha is "
        "what lets a consumer notice"
    )
    assert prov["games"] == 4 and prov["plies"] > 0
    assert prov["winners"] == {"p1": 2, "p2": 2}
    assert prov["game_hash_set_sha256"] == hashlib.sha256(
        "\n".join(sorted(f"g{i}" for i in range(4))).encode("utf-8")
    ).hexdigest()


def test_a_TRUNCATED_artifact_says_so_in_its_own_provenance(tmp_path: Path) -> None:
    """A smoke-sized artifact must never read as a whole corpus. `truncated_at_max_games` is
    `None` on a full encode and the cap on a partial one, so the distinction is a KEY rather
    than an inference from a count nobody has the denominator for."""
    d = _dataset(tmp_path, _records(6))
    full = encode_corpus(d, tmp_path / "a.hexg", encoding=_ENC, capacity=512,
                         visit_capacity=8)
    part = encode_corpus(d, tmp_path / "b.hexg", encoding=_ENC, capacity=512,
                         visit_capacity=8, max_games=2)
    assert full["truncated_at_max_games"] is None and full["games"] == 6
    assert part["truncated_at_max_games"] == 2 and part["games"] == 2
    assert part["artifact_sha256"] != full["artifact_sha256"]


def test_the_written_ring_LOADS_BACK_through_the_production_loader(tmp_path: Path) -> None:
    """The artifact is only capability if the trainer's own loader can read it. No new format
    is invented here: this is the ring `sample_graph_batch` samples."""
    from mantis._engine import HexgBuffer

    d = _dataset(tmp_path, _records(5))
    out = tmp_path / "corpus.hexg"
    prov = encode_corpus(d, out, encoding=_ENC, capacity=512, visit_capacity=8)
    buf = HexgBuffer(512, _ENC, 8)
    loaded = buf.load_from_path(str(out))
    assert loaded == prov["plies"]
    size, _cap, _hist = buf.get_buffer_stats()
    assert size == prov["plies"]


# ── unarmedness, asserted structurally ───────────────────────────────────────────────────
def test_NOTHING_under_src_imports_the_encoder() -> None:
    """LANDING IS NOT ARMING. An `ast` import census, never a grep: a grep passes on a
    commented-out import and fails on a docstring that names the module."""
    hits: list[str] = []
    for path in (_REPO / "src").rglob("*.py"):
        if path.name == "bootstrap_encode.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("bootstrap_encode" in n for n in names):
                hits.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert not hits, (
        f"the bootstrap encoder is reachable from production code at {hits}. It is "
        "CAPABILITY: selected by nothing, armed by nothing, until the operator's bootstrap "
        "adjudication says otherwise."
    )


def test_no_config_key_selects_the_encoder() -> None:
    """The other half of unarmedness: no shipped config may name this artifact producer."""
    for cfg in sorted((_REPO / "configs").glob("*.yaml")):
        text = cfg.read_text(encoding="utf-8")
        assert "bootstrap_encode" not in text, f"{cfg.name} names the encoder"
