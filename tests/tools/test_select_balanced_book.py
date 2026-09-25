"""`tools/select_balanced_book.py` (BOOK_V2): an opening is kept only when a paired anchor-vs-itself result SPLITS by seat in at least one replay."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mantis.arena.books import round_openings
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
_SIMS = 256
_SEEDS = (11, 12, 13, 14)


@pytest.fixture(scope="module")
def selector():
    path = _REPO / "tools" / "select_balanced_book.py"
    return load_module_by_path("select_balanced_book_under_test", path)


def _pool(n: int) -> list[dict]:
    """`n` distinct 4-ply openings; legality is not the selector's concern."""
    return [{"id": i, "moves": [[i, 0], [i, 1], [i + 1, 0], [i - 1, 1]]} for i in range(n)]


def _game(opening: dict, *, seed: int, candidate: int, result: str, index: int,
          tail: list[list[int]] | None = None, sims: int = _SIMS) -> dict:
    moves = list(opening["moves"]) + (tail if tail is not None else [[50, 50], [51, 51]])
    return {
        "contract": "game-record-v1", "game_id": f"g{seed}_{index}", "run_id": "frontier1",
        "channel": "promotion", "rung": "anchor", "phase": "gate_screen", "step": 0,
        "step_kind": "round", "game_index": index,
        "colors": {"candidate": candidate, "opponent": -candidate},
        "seed": seed, "served_sims": sims, "plies": len(moves), "result": result,
        "termination": "six_in_a_row", "moves": moves,
        "trajectory_hash": hashlib.sha256(json.dumps(moves).encode()).hexdigest(),
    }


def _replay(opening: dict, seed: int, legs: tuple[str, str], *, index0: int = 0,
            tails: tuple[list[list[int]], list[list[int]]] | None = None) -> list[dict]:
    """One replay's colour-swapped pair: candidate first (`p1` seat), then candidate second."""
    tail_a, tail_b = tails if tails is not None else (None, None)
    return [_game(opening, seed=seed, candidate=1, result=legs[0], index=index0, tail=tail_a),
            _game(opening, seed=seed, candidate=-1, result=legs[1], index=index0 + 1, tail=tail_b)]


def _rows(opening: dict, per_seed: dict[int, tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for seed, legs in per_seed.items():
        rows.extend(_replay(opening, seed, legs))
    return rows


def _write_books_dir(tmp_path: Path, book_id: str, payload: dict) -> Path:
    books_dir = tmp_path / "books"
    books_dir.mkdir(parents=True)
    book_file = books_dir / f"{book_id}.json"
    book_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    sha = hashlib.sha256(book_file.read_bytes()).hexdigest()
    (books_dir / "manifest.toml").write_text(
        f'[books."{book_id}"]\nfile = "{book_file.name}"\nsha256 = "{sha}"\n', encoding="utf-8")
    return books_dir


@pytest.mark.parametrize("legs,verdict", [
    (("p1", "p1"), "seat_decided"), (("p2", "p2"), "seat_decided"),
    (("p1", "p2"), "split"), (("p2", "p1"), "split"),
    (("draw", "p1"), "undecided"), (("p1", "unknown"), "undecided"), (("draw", "draw"), "undecided"),
])
def test_a_pair_is_seat_decided_split_or_undecided_by_its_two_seat_results(selector, legs, verdict) -> None:
    assert selector.classify_pair(legs[0], legs[1]) == verdict


def test_two_games_with_the_same_colour_are_not_a_pair(selector) -> None:
    opening = _pool(1)[0]
    rows = [_game(opening, seed=1, candidate=1, result="p1", index=0),
            _game(opening, seed=1, candidate=1, result="p2", index=1)]
    with pytest.raises(selector.SelectionError, match="candidate seat"):
        selector.assess(_pool(1), rows, sims=_SIMS, min_replays=1)


def test_an_opening_covered_by_fewer_replays_than_required_is_a_named_failure(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("p1", "p2"), 12: ("p1", "p2"), 13: ("p1", "p2")})
    with pytest.raises(selector.ReplayCoverageError, match="3 replay"):
        selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)


def test_a_replay_missing_one_leg_does_not_count_as_coverage(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("p1", "p2"), 12: ("p1", "p2"), 13: ("p1", "p2")})
    rows.append(_game(opening, seed=14, candidate=1, result="p1", index=0))
    with pytest.raises(selector.ReplayCoverageError, match="3 replay"):
        selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)


def test_every_replay_first_mover_wins_both_is_seat_decided_and_dropped(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {s: ("p1", "p1") for s in _SEEDS})
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)
    assert (row.verdict, row.replays, row.pairs_seat_decided, row.pairs_split) == ("seat_decided", 4, 4, 0)


def test_the_seat_differing_between_replays_is_still_seat_decided(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("p1", "p1"), 12: ("p2", "p2"), 13: ("p1", "p1"), 14: ("p2", "p2")})
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)
    assert row.verdict == "seat_decided"


def test_one_splitting_replay_keeps_the_opening(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("p1", "p1"), 12: ("p1", "p1"), 13: ("p1", "p2"), 14: ("p1", "p1")})
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)
    assert (row.verdict, row.pairs_split, row.pairs_seat_decided) == ("split", 1, 3)


def test_draws_neither_split_nor_decide_so_an_opening_of_draws_is_undecided(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("draw", "p1"), 12: ("p1", "p1"), 13: ("p1", "unknown"), 14: ("p2", "p2")})
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)
    assert (row.verdict, row.pairs_undecided, row.pairs_seat_decided, row.pairs_split) == ("undecided", 2, 2, 0)


def test_min_split_raises_the_bar(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {11: ("p1", "p2"), 12: ("p1", "p1"), 13: ("p1", "p1"), 14: ("p1", "p1")})
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4, min_split=2)
    assert row.verdict == "seat_decided"


def test_the_colour_swap_is_read_from_the_candidate_seat_not_the_row_order(selector) -> None:
    opening = _pool(1)[0]
    rows: list[dict] = []
    for seed in _SEEDS:
        pair = _replay(opening, seed, ("p1", "p1"))
        rows.extend(reversed(pair))
    [row] = selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)
    assert row.verdict == "seat_decided"


def test_records_at_another_sims_are_refused_not_folded_in(selector) -> None:
    opening = _pool(1)[0]
    rows = _rows(opening, {s: ("p1", "p2") for s in _SEEDS})
    rows[0]["served_sims"] = 128
    with pytest.raises(selector.SelectionError, match="served_sims"):
        selector.assess(_pool(1), rows, sims=_SIMS, min_replays=4)


def test_games_not_in_the_pool_are_counted_and_ignored(selector) -> None:
    pool = _pool(1)
    rows = _rows(pool[0], {s: ("p1", "p2") for s in _SEEDS})
    stray = _game({"id": 99, "moves": [[9, 9], [9, 8], [8, 9], [7, 7]]}, seed=11, candidate=1,
                  result="p1", index=40)
    rows.append(stray)
    rows.append({**stray, "channel": "random_floor"})
    rows_out, skipped = selector.match_pool(pool, rows, sims=_SIMS)
    assert len(rows_out) == 8 and skipped == {"unmatched": 1, "other_channel": 1}


def test_identical_replays_are_reported_as_one_measurement(selector) -> None:
    pool = _pool(2)
    same = _rows(pool[0], {s: ("p1", "p2") for s in _SEEDS})
    varied: list[dict] = []
    for k, seed in enumerate(_SEEDS):
        tails = ([[60 + k, 0], [61, 1]], [[70 + k, 0], [71, 1]])
        varied.extend(_replay(pool[1], seed, ("p1", "p2"), tails=tails))
    rows = selector.assess(pool, same + varied, sims=_SIMS, min_replays=4)
    assert [r.distinct_trajectories for r in rows] == [(1, 1), (4, 4)]
    assert [r.identical_replays for r in rows] == [True, False]


def test_the_cut_takes_the_first_n_passing_in_pool_order(selector) -> None:
    pool = _pool(6)
    rows: list[dict] = []
    for i, opening in enumerate(pool):
        legs = ("p1", "p1") if i in (0, 3) else ("p1", "p2")
        rows.extend(_rows(opening, {s: legs for s in _SEEDS}))
    assessed = selector.assess(pool, rows, sims=_SIMS, min_replays=4)
    kept = selector.select_book(assessed, n=3)
    assert [r.pool_id for r in kept] == [1, 2, 4]
    assert [r.kept for r in assessed] == [False, True, True, False, True, False]


def test_fewer_than_n_passing_is_a_named_failure(selector) -> None:
    pool = _pool(3)
    rows: list[dict] = []
    for i, opening in enumerate(pool):
        legs = ("p1", "p2") if i == 0 else ("p2", "p2")
        rows.extend(_rows(opening, {s: legs for s in _SEEDS}))
    assessed = selector.assess(pool, rows, sims=_SIMS, min_replays=4)
    with pytest.raises(selector.BalancedCountError, match="1 of 3"):
        selector.select_book(assessed, n=2)


def _end_to_end(selector, tmp_path: Path, *, n_pool: int, n_book: int, capsys):
    pool = _pool(n_pool)
    pool_id = "pool_under_test"
    books_dir = _write_books_dir(tmp_path, pool_id, {"openings": pool})
    shard_dir = tmp_path / "games"
    shard_dir.mkdir()
    for k, seed in enumerate(_SEEDS):
        lines = [json.dumps({"record": "shard_opened", "run_id": "frontier1"})]
        index = 0
        for i, opening in enumerate(pool):
            legs = ("p1", "p1") if i % 3 == 0 else ("p1", "p2")
            for row in _replay(opening, seed, legs, index0=index):
                lines.append(json.dumps(row))
            index += 2
        (shard_dir / f"games_frontier1_seg000{k}_2026091500.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "book_v2_test_p4.json"
    report = tmp_path / "report.jsonl"
    argv = ["--pool", pool_id, "--books-dir", str(books_dir), "--sims", str(_SIMS),
            "--n", str(n_book), "--out", str(out), "--report", str(report),
            "--book-id", "book_v2_test_p4", "--games", *sorted(str(p) for p in shard_dir.iterdir())]
    rc = selector.main(argv)
    return rc, out, report, capsys.readouterr().out


def test_the_cli_writes_the_book_its_sha_the_manifest_lines_and_the_report(selector, tmp_path, capsys) -> None:
    rc, out, report, stdout = _end_to_end(selector, tmp_path, n_pool=12, n_book=8, capsys=capsys)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [o["id"] for o in payload["openings"]] == list(range(8))
    assert payload["provenance"]["pool_ids"] == [1, 2, 4, 5, 7, 8, 10, 11]
    assert payload["provenance"]["replay_seeds"] == list(_SEEDS)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert sha in stdout
    assert f'[books."book_v2_test_p4"]\nfile = "book_v2_test_p4.json"\nsha256 = "{sha}"' in stdout
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    per_opening = [r for r in rows if r.get("row") == "opening"]
    assert len(per_opening) == 12
    assert sum(r["kept"] for r in per_opening) == 8
    assert {r["verdict"] for r in per_opening} == {"split", "seat_decided"}
    [summary] = [r for r in rows if r.get("row") == "summary"]
    assert summary["openings"] == 12 and summary["kept"] == 8 and summary["identical_replays"] == 12


def test_the_cli_fails_loud_when_too_few_openings_pass(selector, tmp_path, capsys) -> None:
    with pytest.raises(selector.BalancedCountError):
        _end_to_end(selector, tmp_path, n_pool=6, n_book=5, capsys=capsys)


def test_the_selected_book_round_trips_through_books_resolution(selector, tmp_path, capsys) -> None:
    _rc, out, _report, _stdout = _end_to_end(selector, tmp_path, n_pool=12, n_book=8, capsys=capsys)
    payload = json.loads(out.read_text(encoding="utf-8"))
    books_dir = _write_books_dir(tmp_path / "v2", "book_v2_test_p4", payload)
    openings = round_openings("book_v2_test_p4", n_pairs=8, seed_base=1, round_index=0,
                              books_dir=books_dir)
    assert len(openings) == 8
    assert {o.opening_id for o in openings} == {str(i) for i in range(8)}
    by_id = {int(o.opening_id): o.moves for o in openings}
    assert by_id[0] == [(1, 0), (1, 1), (2, 0), (0, 1)]


def test_a_markdown_report_is_a_table(selector, tmp_path) -> None:
    pool = _pool(1)
    rows = _rows(pool[0], {s: ("p1", "p2") for s in _SEEDS})
    assessed = selector.assess(pool, rows, sims=_SIMS, min_replays=4)
    selector.select_book(assessed, n=1)
    path = tmp_path / "report.md"
    selector.write_report(path, assessed, selector.summarize(assessed))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("| pool_id |") and "| split |" in text


def test_the_box_procedure_is_served_by_a_flag(selector, capsys) -> None:
    assert selector.main(["--procedure"]) == 0
    text = capsys.readouterr().out
    assert "strength_frontier.py" in text and "4096 games" in text and '"seed_base"' in text
