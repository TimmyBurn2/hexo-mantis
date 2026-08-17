# >300 justify (R8): one oracle for one tool. The synthetic-game builder is shared by every
# case here and by both mutation self-tests, and each mutation is only evidence because the
# SAME builder produced the clean baseline it is perturbed from — split the file and the
# builder becomes a second authority on what "a conforming record" is, which is exactly the
# drift the tool's DECLARED INPUT CONTRACT exists to prevent.
"""Oracle for tools/audit_bootstrap_corpus.py (R247).

Every fixture is built inline in ``tmp_path`` — no committed data (R7). The two mutation
self-tests (LAW-07) are ``test_a_flipped_byte_...`` and ``test_a_renamed_field_...``: each
first proves the clean fixture passes, then perturbs exactly one thing and proves the leg
bites. A gate whose trigger is never fired is a gate nobody can trust.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "audit_bootstrap_corpus.py"


def _load_tool():
    """Load the audit tool by PATH.

    R5/LAW-17 ban `sys.path` mutation and `tools/` is not an importable package, so the tool
    is spec-loaded from its file exactly as the other `tests/tools/` oracles do it.
    """
    spec = importlib.util.spec_from_file_location("_audit_bootstrap_corpus", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()
EXIT_OK: int = TOOL.EXIT_OK
EXIT_SHA: int = TOOL.EXIT_SHA
EXIT_CONTRACT: int = TOOL.EXIT_CONTRACT
EXIT_USAGE: int = TOOL.EXIT_USAGE
main = TOOL.main
mover_of_ply = TOOL.mover_of_ply
has_six_run = TOOL.has_six_run
nearest_rank = TOOL.nearest_rank
turns_from_plies = TOOL.turns_from_plies

# --------------------------------------------------------------------------------------
# fixture builders — tiny, inline, never committed
# --------------------------------------------------------------------------------------

#: A synthetic game whose FINAL position gives Player::One a six-run along the E/W axis
#: (HEX_AXES[0]); P2's stones are parked far away so they form no run. Stone i is placed by
#: `mover_of_ply(i)`: 0->P1, 1,2->P2, 3,4->P1, 5,6->P2, 7,8->P1, 9,10->P2, 11,12->P1, ...
#: P1 owns indices {0, 3, 4, 7, 8, 11, 12, 15, 16, ...}.
#: contract v2: the baseline fixture is the OBSERVED record shape — game_hash as identity,
#: `source` alongside it, `elo` a per-player pair, and NO `game_id` (OPEN-5 / OPEN-7).
def _decisive_p1_game(gid: str, *, plies: int = 24) -> dict[str, Any]:
    p1_line = [(q, 0) for q in range(6)]  # the six-run P1 must end up holding
    moves: list[list[int]] = []
    p1_next = 0
    p2_next = 0
    for i in range(plies):
        if mover_of_ply(i) == 1:
            if p1_next < len(p1_line):
                q, r = p1_line[p1_next]
            else:
                q, r = (100 + p1_next, 50)
            p1_next += 1
        else:
            q, r = (-100 - p2_next, -50 - (p2_next % 3) * 7)
            p2_next += 1
        moves.append([q, r])
    return {
        "game_hash": hashlib.sha256(gid.encode("utf-8")).hexdigest(),
        "winner": 1,
        "elo": [1234, 1300],
        "moves": moves,
        "source": "human",
    }


def _write_dataset(root: Path, records: list[dict[str, Any]]) -> Path:
    """SHAPE A — the multi-file `files: [{path, sha256}]` manifest."""
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    data = ds / "games.jsonl"
    data.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
    )
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    (ds / "dataset_metadata.json").write_text(
        json.dumps({"files": [{"path": "games.jsonl", "sha256": sha}]}, indent=2),
        encoding="utf-8",
    )
    return ds


def _write_flat_dataset(
    root: Path, records: list[dict[str, Any]], *, declared_bytes: int | None = None
) -> Path:
    """SHAPE B — the flat single-file manifest the R247 dataset actually ships (OPEN-2).

    Field-for-field the observed shape: `file` / `bytes` / `sha256` plus provenance keys
    the tool must record and never parse for meaning.
    """
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    data = ds / "hexo_human_corpus.jsonl"
    data.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
    )
    raw = data.read_bytes()
    (ds / "dataset_metadata.json").write_text(
        json.dumps({
            "name": "hexo-human-corpus",
            "schema_version": 1,
            "format": "jsonl",
            "n_games": len(records),
            "n_duplicates_dropped": 0,
            "dedup_key": "game_hash (16-hex SHA-256 of move sequence)",
            "source_filter": "rated, >=20 moves, decisive by six-in-a-row",
            "file": "hexo_human_corpus.jsonl",
            "bytes": len(raw) if declared_bytes is None else declared_bytes,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }, indent=2),
        encoding="utf-8",
    )
    return ds


def _run(ds: Path, out: Path, *extra: str) -> int:
    return main([str(ds), "--out", str(out), *extra])


def _report(out: Path) -> dict[str, Any]:
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# in-repo-fact pins (the constants the script maps onto mantis-core)
# --------------------------------------------------------------------------------------

def test_turn_structure_matches_board_mod_rs_24_26():
    """ply 0 = one P1 stone, then two stones per turn, alternating."""
    assert [mover_of_ply(i) for i in range(9)] == [1, -1, -1, 1, 1, -1, -1, 1, 1]


def test_six_run_detector_needs_six_collinear_on_a_hex_axis():
    assert has_six_run({(q, 0) for q in range(6)})
    assert not has_six_run({(q, 0) for q in range(5)})
    assert has_six_run({(0, r) for r in range(6)})       # NE/SW axis
    assert has_six_run({(i, -i) for i in range(6)})      # SE/NW axis
    assert not has_six_run({(i, i) for i in range(6)})   # not a hex axis


# --------------------------------------------------------------------------------------
# the clean path
# --------------------------------------------------------------------------------------

def test_a_conforming_dataset_audits_clean(tmp_path: Path):
    ds = _write_dataset(tmp_path, [_decisive_p1_game(f"g{i}") for i in range(4)])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    rep = _report(out)
    assert rep["verdict"] == "CLEAN"
    assert rep["violations"] == []
    conv = rep["convention_audit"]
    assert conv["games"] == 4
    assert conv["winner_holds_six_run"] == 4
    assert conv["winner_lacks_six_run"] == 0
    assert "CONSISTENT" in conv["verdict"]
    assert rep["sha256_verification"]["clean"] is True


def test_both_selection_biases_are_recorded_with_measurements(tmp_path: Path):
    """R247 requires the biases in the OUTPUT, measured AND stated."""
    ds = _write_dataset(tmp_path, [_decisive_p1_game(f"g{i}") for i in range(3)])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    biases = {b["id"]: b for b in _report(out)["selection_biases"]}
    assert set(biases) == {"bias-a-decisive-only", "bias-b-min-move-floor"}

    a = biases["bias-a-decisive-only"]
    assert a["register_adjacency"] == ["F-07"]
    assert a["measured"]["draws_found"] == 0          # measured
    assert "F-07" in a["statement"]                    # stated

    b = biases["bias-b-min-move-floor"]
    assert b["register_adjacency"] == ["F-15", "F-38"]
    assert b["measured"]["min_plies_found"] == 24      # measured
    assert b["measured"]["declared_floor_plies"] == 20
    assert "F-15" in b["statement"] and "F-38" in b["statement"]


def test_distributions_report_count_min_median_mean_max_and_a_histogram(tmp_path: Path):
    recs = []
    for i in range(3):
        r = _decisive_p1_game(f"g{i}", plies=24 + 2 * i)
        r["elo"] = [1000 + 100 * i, 1000 + 100 * i]
        recs.append(r)
    ds = _write_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    dist = _report(out)["distributions"]
    elo = dist["elo_per_player"]["distribution"]
    assert elo["count"] == 6                       # per PLAYER, not per game
    assert elo["min"] == 1000.0 and elo["max"] == 1200.0
    assert elo["median"] == 1100.0
    assert len(elo["histogram"]) == 10
    assert dist["move_count_plies"]["min"] == 24.0
    assert dist["move_count_plies"]["max"] == 28.0


def test_the_dedupe_leg_says_no_in_repo_reference_available_by_default(tmp_path: Path):
    """There is no in-repo game_hash producer; the leg must SAY so, not compare nothing."""
    ds = _write_dataset(tmp_path, [_decisive_p1_game("g0"), _decisive_p1_game("g1")])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    leg = _report(out)["dedupe"]
    assert leg["overlap"] == "NO IN-REPO REFERENCE AVAILABLE"
    assert leg["in_repo_game_hash_producer"].startswith("ABSENT")
    assert leg["dataset_distinct_game_hash"] == 2


def test_the_dedupe_leg_labels_a_supplied_reference_as_derived(tmp_path: Path):
    """With --in-repo-corpus the comparison is DERIVED, never a game_hash comparison."""
    shared = _decisive_p1_game("shared")
    ds = _write_dataset(tmp_path, [shared, _decisive_p1_game("only-in-dataset")])
    repo_corpus = tmp_path / "human"
    repo_corpus.mkdir()
    # in-repo human corpus shape: moves[i].x / moves[i].y ARE the axial (q, r)
    # (src/mantis/data/sources/human.py:85).
    moves = shared["moves"]
    assert isinstance(moves, list)
    (repo_corpus / "uuid-1.json").write_text(
        json.dumps({"moves": [{"x": m[0], "y": m[1]} for m in moves]}), encoding="utf-8"
    )
    out = tmp_path / "report.json"
    assert _run(ds, out, "--in-repo-corpus", str(repo_corpus)) == EXIT_OK
    leg = _report(out)["dedupe"]
    assert leg["overlap"].startswith("DERIVED KEY COMPARISON")
    assert leg["in_repo_games_keyed"] == 1
    assert leg["overlapping_games"] == 1


# --------------------------------------------------------------------------------------
# MUTATION SELF-TEST 1 (LAW-07) — the sha leg bites
# --------------------------------------------------------------------------------------

def test_a_flipped_byte_in_a_data_file_is_caught_by_the_sha_leg(tmp_path: Path):
    recs = [_decisive_p1_game(f"g{i}") for i in range(3)]
    ds = _write_dataset(tmp_path, recs)
    out = tmp_path / "report.json"

    # precondition: the UNMUTATED fixture passes, so the failure below is the mutation's
    assert _run(ds, out) == EXIT_OK

    data = ds / "games.jsonl"
    raw = data.read_text(encoding="utf-8")
    mutated = raw.replace('"elo": [1234, 1300]', '"elo": [1235, 1300]', 1)
    assert mutated != raw, "the mutation itself must land, or this oracle proves nothing"
    data.write_text(mutated, encoding="utf-8")

    assert _run(ds, out) == EXIT_SHA
    rep = _report(out)
    assert rep["verdict"] == "SHA_VERIFICATION_FAILED"
    sha = rep["sha256_verification"]
    assert len(sha["mismatch"]) == 1
    assert sha["mismatch"][0]["path"] == "games.jsonl"
    assert sha["mismatch"][0]["expected"] != sha["mismatch"][0]["actual"]
    # no statistics over unpinned bytes
    assert "convention_audit" not in rep
    assert rep["record_audit"].startswith("NOT RUN")


def test_a_missing_listed_file_and_an_unlisted_record_file_both_bite(tmp_path: Path):
    ds = _write_dataset(tmp_path, [_decisive_p1_game("g0")])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK

    (ds / "extra_games.jsonl").write_text("{}\n", encoding="utf-8")
    assert _run(ds, out) == EXIT_SHA
    assert _report(out)["sha256_verification"]["unlisted_record_files"] == ["extra_games.jsonl"]

    (ds / "extra_games.jsonl").unlink()
    (ds / "games.jsonl").unlink()
    assert _run(ds, out) == EXIT_SHA  # the pinned file is gone
    assert _report(out)["sha256_verification"]["missing"] == ["games.jsonl"]


# --------------------------------------------------------------------------------------
# MUTATION SELF-TEST 2 (LAW-07) — the contract leg bites, naming the field
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "renamed"),
    [("elo", "rating"), ("winner", "result"), ("moves", "move_list"),
     ("game_hash", "hash")],
)
def test_a_renamed_field_is_refused_by_name_never_coerced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, renamed: str
):
    rec = _decisive_p1_game("g0")
    rec[renamed] = rec.pop(field)
    ds = _write_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"

    assert _run(ds, out) == EXIT_CONTRACT
    err = capsys.readouterr().err
    assert "CONTRACT VIOLATION" in err
    assert repr(field) in err          # the message names the MISSING field, precisely
    assert renamed in err              # and shows what was actually present
    assert not out.exists()            # a refused audit writes no report


def test_an_object_form_move_list_is_refused_and_names_the_in_repo_alternative(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The in-repo corpus uses {'x','y'} objects. The tool must NOT silently accept them."""
    rec = _decisive_p1_game("g0")
    moves = rec["moves"]
    assert isinstance(moves, list)
    rec["moves"] = [{"x": m[0], "y": m[1]} for m in moves]
    ds = _write_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"

    assert _run(ds, out) == EXIT_CONTRACT
    err = capsys.readouterr().err
    assert "list of objects with keys ['x', 'y']" in err
    assert "human.py:85" in err        # points the operator at the in-repo alternative
    assert not out.exists()


def test_a_draw_is_measured_and_then_refused_as_unmappable(tmp_path: Path):
    """winner == 0 has no Player member (core.rs:59-64): counted, reported, refused."""
    rec = _decisive_p1_game("g0")
    rec["winner"] = 0
    ds = _write_dataset(tmp_path, [rec, _decisive_p1_game("g1")])
    out = tmp_path / "report.json"

    assert _run(ds, out) == EXIT_CONTRACT
    rep = _report(out)
    assert rep["verdict"] == "CONTRACT_VIOLATION"
    assert rep["convention_audit"]["draws"] == 1
    # measured AND stated: the bias row still carries the measurement
    bias_a = next(b for b in rep["selection_biases"] if b["id"] == "bias-a-decisive-only")
    assert bias_a["measured"]["draws_found"] == 1
    assert any("winner == 0" in v for v in rep["violations"])


def test_a_winner_outside_the_mapped_domain_is_refused(tmp_path: Path):
    rec = _decisive_p1_game("g0")
    rec["winner"] = 2
    ds = _write_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    rep = _report(out)
    assert rep["convention_audit"]["winner_outside_mapped_domain"][0]["winner"] == 2


def test_an_inverted_winner_convention_is_diagnosed_not_silently_flipped(tmp_path: Path):
    """The whole point of CHECKING the mapping rather than asserting it."""
    recs = []
    for i in range(3):
        r = _decisive_p1_game(f"g{i}")
        r["winner"] = -1  # P2 declared the winner, but P1 holds the six-run
        recs.append(r)
    ds = _write_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    conv = _report(out)["convention_audit"]
    assert conv["winner_lacks_six_run"] == 3
    assert conv["loser_holds_six_run_when_winner_does_not"] == 3
    assert "LIKELY INVERTED" in conv["verdict"]


def test_a_manifest_without_a_files_array_is_refused_by_name(tmp_path: Path):
    ds = _write_dataset(tmp_path, [_decisive_p1_game("g0")])
    (ds / "dataset_metadata.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT


# --------------------------------------------------------------------------------------
# OPEN-2 (contract v2) — the manifest ships FLAT and SINGLE-FILE, not a files[] array
# --------------------------------------------------------------------------------------

def test_the_flat_single_file_manifest_is_accepted_as_shape_b(tmp_path: Path):
    """The shape the R247 dataset actually ships: `file` + `sha256` + `bytes`."""
    ds = _write_flat_dataset(tmp_path, [_decisive_p1_game(f"g{i}") for i in range(3)])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    rep = _report(out)
    assert rep["verdict"] == "CLEAN"
    assert rep["contract_version"] == 2
    sha = rep["sha256_verification"]
    assert sha["clean"] is True
    assert sha["listed"] == 1
    assert sha["match"] == ["hexo_human_corpus.jsonl"]
    assert rep["convention_audit"]["games"] == 3


def test_a_manifest_in_neither_declared_shape_is_refused_naming_the_observed_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """No silent widening: the refusal names BOTH declared shapes and what was found."""
    ds = _write_flat_dataset(tmp_path, [_decisive_p1_game("g0")])
    (ds / "dataset_metadata.json").write_text(
        json.dumps({"shards": [], "digest": "deadbeef"}), encoding="utf-8"
    )
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    err = capsys.readouterr().err
    assert "SHAPE A" in err and "SHAPE B" in err
    assert "OBSERVED top-level fields: ['digest', 'shards']" in err
    assert not out.exists()


def test_a_manifest_carrying_both_shapes_is_refused_as_ambiguous(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Accepting either shape must not mean accepting both at once."""
    ds = _write_flat_dataset(tmp_path, [_decisive_p1_game("g0")])
    meta_path = ds / "dataset_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"] = [{"path": "hexo_human_corpus.jsonl", "sha256": meta["sha256"]}]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    assert "ambiguous" in capsys.readouterr().err
    assert not out.exists()


def test_a_declared_byte_length_that_disagrees_with_disk_is_a_pin_failure(tmp_path: Path):
    """MUTATION (LAW-07): shape B's `bytes` is verified, not decoration."""
    recs = [_decisive_p1_game("g0")]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK          # the unmutated fixture passes

    meta_path = ds / "dataset_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["bytes"] = int(meta["bytes"]) + 1   # sha still matches; only the length lies
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert _run(ds, out) == EXIT_SHA
    rep = _report(out)
    assert rep["verdict"] == "SHA_VERIFICATION_FAILED"
    bad = rep["sha256_verification"]["declared_bytes_mismatch"]
    assert len(bad) == 1 and bad[0]["path"] == "hexo_human_corpus.jsonl"
    assert bad[0]["expected"] == bad[0]["actual"] + 1


# --------------------------------------------------------------------------------------
# OPEN-5 (contract v2) — `elo` is a per-PLAYER pair, not one number per game
# --------------------------------------------------------------------------------------

def test_the_elo_leg_reports_per_player_and_says_so(tmp_path: Path):
    recs = [_decisive_p1_game("g0"), _decisive_p1_game("g1")]
    recs[1]["elo"] = [900, 1100]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    leg = _report(out)["distributions"]["elo_per_player"]
    assert "per PLAYER" in leg["unit"]
    assert leg["games"] == 2
    assert leg["individual_ratings"] == 4
    assert leg["games_with_complete_pair"] == 2
    assert leg["null_ratings"] == 0
    assert leg["distribution"]["min"] == 900.0
    assert leg["distribution"]["max"] == 1300.0


def test_a_scalar_elo_is_refused_and_the_message_names_the_pair_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """MUTATION (LAW-07): v1's declared shape is now the failure, named as such."""
    rec = _decisive_p1_game("g0")
    rec["elo"] = 1234.0
    ds = _write_flat_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    err = capsys.readouterr().err
    assert "ONE RATING PER PLAYER" in err
    assert "OPEN-5" in err
    assert "will not average, pick a seat, or broadcast" in err
    assert not out.exists()


def test_a_null_rating_is_carried_not_counted_as_zero(tmp_path: Path):
    """SCHEMA.md: `[int or null, int or null]` — null means exported without metadata."""
    rec = _decisive_p1_game("g0")
    rec["elo"] = [None, 1100]
    ds = _write_flat_dataset(tmp_path, [rec, _decisive_p1_game("g1")])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    leg = _report(out)["distributions"]["elo_per_player"]
    assert leg["null_ratings"] == 1
    assert leg["games_with_complete_pair"] == 1
    assert leg["individual_ratings"] == 3
    assert leg["distribution"]["min"] == 1100.0   # the null did not become a 0


# --------------------------------------------------------------------------------------
# OPEN-7 (contract v2) — `game_id` is ABSENT; identity is `game_hash`
# --------------------------------------------------------------------------------------

def test_a_record_without_game_id_audits_clean_and_is_named_by_game_hash(tmp_path: Path):
    rec = _decisive_p1_game("g0")
    rec["winner"] = 2                      # force a diagnostic that must name the game
    assert "game_id" not in rec
    ds = _write_flat_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT   # refused on the winner, NOT on the missing id
    named = _report(out)["convention_audit"]["winner_outside_mapped_domain"][0]
    assert named["game"] == rec["game_hash"]


def test_a_present_but_empty_game_id_is_still_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Optional does not mean unvalidated: present-and-broken is a contract violation."""
    rec = _decisive_p1_game("g0")
    rec["game_id"] = ""
    ds = _write_flat_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    err = capsys.readouterr().err
    assert "OPTIONAL" in err and "OPEN-7" in err
    assert not out.exists()


def test_a_present_game_id_rides_the_identity_alongside_the_hash(tmp_path: Path):
    rec = _decisive_p1_game("g0")
    rec["game_id"] = "uuid-42"
    rec["winner"] = 2
    ds = _write_flat_dataset(tmp_path, [rec])
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    named = _report(out)["convention_audit"]["winner_outside_mapped_domain"][0]
    assert named["game"] == f"uuid-42 ({rec['game_hash']})"


# --------------------------------------------------------------------------------------
# the certified distribution table (R278(d)) — plies AND turns, the ladder, truncation
# --------------------------------------------------------------------------------------

def test_turns_derive_from_plies_by_the_compound_turn_rule(tmp_path: Path):
    """LAW-03: ply 0 places ONE stone, every later turn places two."""
    assert turns_from_plies(1) == 1.0
    assert turns_from_plies(20) == 10.5     # the card's "20 moves ~ 10.5 turns"
    assert turns_from_plies(128) == 64.5
    assert turns_from_plies(200) == 100.5


def test_percentiles_are_nearest_rank_and_never_interpolate():
    """A p99 of 128.5 plies is a length no game can have — so no interpolation."""
    sample = [float(v) for v in range(1, 101)]
    assert nearest_rank(sample, 50.0) == 50.0
    assert nearest_rank(sample, 99.0) == 99.0
    assert nearest_rank(sample, 99.9) == 100.0
    assert nearest_rank(sample, 0.0) == 1.0
    odd = [1.0, 2.0, 4.0]
    assert nearest_rank(odd, 50.0) == 2.0    # an OBSERVED value, not the 2.0/4.0 midpoint


def test_the_ply_ladder_carries_both_units_and_the_declared_percentiles(tmp_path: Path):
    recs = [_decisive_p1_game(f"g{i}", plies=20 + 2 * i) for i in range(10)]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    ply = _report(out)["distributions"]["ply_length"]
    assert ply["count"] == 10
    assert "PLIES" in ply["unit"]
    ladder = {row["statistic"]: row for row in ply["ladder"]}
    assert set(ladder) == {"p50", "p90", "p95", "p99", "p999", "min", "max", "mean"}
    assert ladder["min"]["plies"] == 20.0 and ladder["max"]["plies"] == 38.0
    for row in ply["ladder"]:
        assert row["turns"] == turns_from_plies(row["plies"])


def test_truncation_is_measured_at_every_candidate_cap(tmp_path: Path):
    """The number a cap decision is actually made from."""
    recs = [_decisive_p1_game("short", plies=24), _decisive_p1_game("long", plies=300)]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    rows = {r["cap_plies"]: r for r in _report(out)["distributions"]["ply_length"]
            ["truncation_by_cap"]}
    assert set(rows) == {128, 200}          # the declared pair: the live literal + R276(g)'s
    assert rows[128]["games_exceeding"] == 1
    assert rows[128]["fraction_truncated"] == 0.5
    assert rows[128]["fraction_fitting"] == 0.5
    assert rows[128]["cap_turns"] == 64.5


def test_a_ply_cap_override_replaces_the_declared_pair(tmp_path: Path):
    ds = _write_flat_dataset(tmp_path, [_decisive_p1_game("g0", plies=24)])
    out = tmp_path / "report.json"
    assert _run(ds, out, "--ply-cap", "50") == EXIT_OK
    rows = _report(out)["distributions"]["ply_length"]["truncation_by_cap"]
    assert [r["cap_plies"] for r in rows] == [50]


def test_the_modal_bucket_is_reported_next_to_the_exact_mode(tmp_path: Path):
    """A selection floor makes a spike; a spike inside a wide bucket reads as a broad mode."""
    recs = [_decisive_p1_game(f"spike{i}", plies=20) for i in range(6)]
    recs += [_decisive_p1_game(f"spread{i}", plies=22 + 2 * i) for i in range(4)]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    ply = _report(out)["distributions"]["ply_length"]
    assert ply["histogram_bucket_plies"] == 20
    modal = ply["modal_bucket"]
    assert (modal["lo_plies"], modal["hi_plies"]) == (20, 39)
    assert modal["count"] == 10 and modal["share"] == 1.0
    # the whole point: the bucket says 100%, the EXACT mode says the spike is 60% of it
    assert ply["mode_exact_plies"] == 20
    assert ply["mode_exact_count"] == 6
    assert ply["mode_exact_share"] == 0.6
    assert ply["mode_exact_share_of_modal_bucket"] == 0.6


def test_the_floor_spike_is_measured_at_the_minimum_not_inferred_from_a_bucket(tmp_path: Path):
    """A binding selection floor piles mass on the floor VALUE; the report must say so."""
    recs = [_decisive_p1_game(f"floor{i}", plies=20) for i in range(3)]
    recs += [_decisive_p1_game(f"other{i}", plies=24 + 2 * i) for i in range(7)]
    ds = _write_flat_dataset(tmp_path, recs)
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_OK
    ply = _report(out)["distributions"]["ply_length"]
    assert ply["count_at_min_plies"] == {"plies": 20, "count": 3, "share": 0.3}
    head = ply["most_common_exact_plies"]
    assert head[0] == {"plies": 20, "count": 3, "share": 0.3}
    assert len(head) == 5
    assert [e["plies"] for e in head[1:]] == [24, 26, 28, 30]   # ties broken low-first


def test_a_manifest_pinning_no_record_file_at_all_is_refused(tmp_path: Path):
    """A sha-clean dataset with nothing parseable must not audit as CLEAN over zero games."""
    ds = tmp_path / "dataset"
    ds.mkdir()
    readme = ds / "README.txt"
    readme.write_text("no games here\n", encoding="utf-8")
    sha = hashlib.sha256(readme.read_bytes()).hexdigest()
    (ds / "dataset_metadata.json").write_text(
        json.dumps({"files": [{"path": "README.txt", "sha256": sha}]}), encoding="utf-8"
    )
    out = tmp_path / "report.json"
    assert _run(ds, out) == EXIT_CONTRACT
    assert not out.exists()


def test_a_bad_in_repo_corpus_path_is_a_usage_error_not_a_dataset_verdict(tmp_path: Path):
    ds = _write_dataset(tmp_path, [_decisive_p1_game("g0")])
    out = tmp_path / "report.json"
    assert _run(ds, out, "--in-repo-corpus", str(tmp_path / "nope")) == EXIT_USAGE
    assert not out.exists()


def test_an_output_path_inside_the_repo_is_refused(tmp_path: Path):
    """R7: the report is a run artifact; it never lands in the tree."""
    ds = _write_dataset(tmp_path, [_decisive_p1_game("g0")])
    inside = Path(__file__).resolve().parents[2] / "_bootstrap_audit_probe.json"
    assert _run(ds, inside) == EXIT_USAGE
    assert not inside.exists()
