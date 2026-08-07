#!/usr/bin/env python3
# >300 justify (R8): the DECLARED input contract, the validator that enforces it, the
# mantis-core convention audit, the distribution pass and the two recorded selection biases
# are ONE offline instrument over ONE external artifact. Splitting them would put the
# contract in one file and the code that enforces it in another — which is precisely how a
# schema and its checker drift apart — and here the drift would be INVISIBLE, because the
# dataset is not in the tree and there is nothing to re-check the split halves against.
"""Offline audit of the external human bootstrap corpus (R247).

R247 adopts a Hugging Face human-game corpus as a sha-pinned EXTERNAL bootstrap artifact
(R7: outside the repo, manifest-indexed) *pending an audit*. This tool is that audit.

It NEVER downloads anything, NEVER writes inside the repository tree, and NEVER coerces,
defaults or guesses a field name. A schema mismatch is an error the operator resolves by
amending the DECLARED INPUT CONTRACT below — it is not something this tool papers over.

================================================================================
DECLARED INPUT CONTRACT  (contract_version 1)
================================================================================
THIS SECTION IS DECLARED, NOT VERIFIED. The dataset has never been seen by the author of
this file. Every field name and shape below is derived from R247's own wording
("human-only, rated, per-game Elo, sha256'd, encoding-free axial move lists") plus the
shape the in-repo human corpus already uses. If the real dataset differs, this tool exits
non-zero naming the exact field, and THIS BLOCK is the single place to amend.

Layout::

    <dataset_dir>/
        dataset_metadata.json      # required, the sha256 manifest
        <one or more *.jsonl or *.json record files>
        <any other files>          # sha-verified if listed; never parsed

``dataset_metadata.json`` -- a JSON object carrying::

    {"files": [{"path": "<relative path>", "sha256": "<64 lowercase hex>"}, ...]}

Record files -- ``*.jsonl`` (one JSON object per non-blank line) or ``*.json`` (a JSON
array of objects). Every record is an object carrying EXACTLY these declared fields
(additional fields are permitted, recorded, and ignored)::

    game_id    str, non-empty     opaque per-game identifier
    game_hash  str, non-empty     R247's named dedupe key
    winner     int, 1 or -1       1 => Player::One, -1 => Player::Two (see MAPPING)
    elo        int or float       the per-game Elo R247 names
    moves      list of [q, r]     axial move list; each entry a 2-element array of ints,
                                  in placement order, one entry per STONE (ply), not turn

================================================================================
MAPPING TO mantis-core  (verified in-repo; file:line cited at each constant below)
================================================================================
winner
    ``crates/mantis-core/src/board/state/core.rs:59-64`` -- ``#[repr(i8)] enum Player {
    One = 1, Two = -1 }``. There is NO draw member. ``core.rs:77-84`` -- ``#[repr(i8)]
    enum Cell { Empty = 0, P1 = 1, P2 = -1 }``.
    ``src/mantis/data/sources/base.py:20,28`` -- ``GameRecord.winner: int`` documented
    "+1 if player 1 won, -1 if player 2 won".
    => The mapped domain is EXACTLY {1, -1}. ``winner == 0`` is a DRAW, which is
    unrepresentable as a ``Player``; it is counted (bias measurement, below) and then
    reported as a contract violation rather than silently mapped.

coordinates
    ``crates/mantis-core/src/board/mod.rs:3-14`` -- sparse axial (q, r), directions
    ``E (+1,0) / W (-1,0) / NE (0,+1) / SW (0,-1) / NW (-1,+1) / SE (+1,-1)``, storage
    ``FxHashMap<(q,r), Cell>``, UNBOUNDED. The 19x19 tensor is a *sliding view window*
    centred on the stone bounding-box centroid -- it is NOT a coordinate bound and "it
    never clips stones".
    ``core.rs:113-115`` -- the map key is ``(i32, i32)``.
    => The only hard bound is i32. Coordinates outside the nominal window
    ``[-9, 9]`` (``core.rs:40-42``, ``zobrist.rs:77-81``) are LEGAL and are reported as a
    statistic, never rejected.

turn structure (LAW-03: this tool counts PLIES, and says so everywhere)
    ``board/mod.rs:24-26`` and ``core.rs:118-121`` -- ply 0: player 1 places exactly ONE
    stone; from then on each player places exactly TWO stones before the turn passes.
    => mover(0) = P1; for i >= 1, mover(i) = P2 when ((i-1)//2) is even, else P1.

win condition
    ``board/mod.rs:21-22`` -- six stones of one player in a row along one of the three hex
    axes. ``core.rs:51-55`` -- ``HEX_AXES = [(1,0), (0,1), (1,-1)]`` (positive directions;
    the scan uses +/-).

The winner/coordinate mapping is CHECKED, not asserted: every game is replayed and the
declared winner must actually hold a six-run in the final position under the identity
mapping. KNOWN LIMIT, stated rather than hidden: the axis SET {(1,0),(0,1),(1,-1)} is
invariant under a q<->r relabelling (it maps (1,0)<->(0,1) and (1,-1)<->-(1,-1)), so a
transposed axial convention CANNOT be detected by a win-line check. It would be detected
only by comparing against the source renderer, which is out of scope for an offline audit.

================================================================================
DEDUPE -- read this before trusting the dedupe leg
================================================================================
R247 says "dedupe overlap vs the in-repo corpus by game_hash". THERE IS NO IN-REPO
``game_hash`` PRODUCER FOR THE HUMAN CORPUS. Verified:
  * ``src/mantis/data/sources/human.py:91`` -- a human game's identity is
    ``game_id_str=path.stem``, i.e. the source UUID of the JSON filename. Not a hash.
  * ``src/mantis/data/generate.py:132-135`` -- ``_game_hash`` is the ONLY hash-of-moves in
    the repo. It hashes BOT self-play games and is used as a FILENAME stem; F-06 rules bot
    games non-canonical, so it is not the human corpus's key.
Therefore this tool never compares the dataset's ``game_hash`` against an in-repo
``game_hash``: there is nothing to compare it to. Without ``--in-repo-corpus`` the leg
reports ``NO IN-REPO REFERENCE AVAILABLE``. With it, the leg derives a comparable key on
BOTH sides using the canonical move-sequence digest defined in ``derived_move_key`` and
labels the whole result DERIVED -- it is this tool's construction, not an in-repo contract.

================================================================================
EXIT CODES
================================================================================
    0  clean
    2  sha256 verification failure (mismatch, missing, or unlisted record file)
    3  contract violation
    4  usage / IO error (unreadable dataset dir, refused output path)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1

#: The declared per-record fields. Amending this list means amending the contract block.
DECLARED_RECORD_FIELDS = ("game_id", "game_hash", "winner", "elo", "moves")

#: crates/mantis-core/src/board/state/core.rs:51-55 (positive directions; scan uses +/-).
HEX_AXES = ((1, 0), (0, 1), (1, -1))
#: crates/mantis-core/src/board/mod.rs:21-22.
WIN_RUN = 6
#: crates/mantis-core/src/board/state/core.rs:59-64 -- Player::One / Player::Two.
PLAYER_ONE = 1
PLAYER_TWO = -1
WINNER_DOMAIN = (PLAYER_ONE, PLAYER_TWO)
#: crates/mantis-core/src/board/state/core.rs:113-115 -- the cell map key is (i32, i32).
I32_MIN = -(2**31)
I32_MAX = 2**31 - 1
#: crates/mantis-core/src/board/state/core.rs:40-42 and board/zobrist.rs:77-81. The window
#: SLIDES (board/mod.rs:12-14) -- this is a reporting reference, never a rejection bound.
NOMINAL_WINDOW_HALF = 9
#: src/mantis/data/sources/human.py:108-112 -- the in-repo ingestion filter's move floor,
#: in PLIES. R247 names the same floor as selection bias (b).
DECLARED_MIN_PLIES = 20

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK = 1 << 20
_RECORD_SUFFIXES = (".jsonl", ".json")

EXIT_OK = 0
EXIT_SHA = 2
EXIT_CONTRACT = 3
EXIT_USAGE = 4


class ContractViolation(Exception):
    """The dataset does not match the DECLARED INPUT CONTRACT. Names the exact field."""


class UsageError(Exception):
    """Bad invocation or unreadable input. Never a statement about the dataset."""


# ---------------------------------------------------------------------------
# sha256 leg
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Streaming sha256 hex digest. Binary mode -- takes no ``encoding`` by design."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_metadata(dataset_dir: Path) -> list[tuple[str, str]]:
    """Parse ``dataset_metadata.json`` into (relative path, expected sha256) rows."""
    meta_path = dataset_dir / "dataset_metadata.json"
    if not meta_path.is_file():
        raise ContractViolation(
            f"missing required manifest {meta_path}. The DECLARED INPUT CONTRACT requires a "
            "'dataset_metadata.json' at the dataset root carrying a 'files' array of "
            "{'path','sha256'} objects. If the real dataset names it differently, amend the "
            "contract block in this file -- do not rename the file to fit the tool."
        )
    try:
        raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"could not parse {meta_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ContractViolation(
            f"{meta_path}: root must be a JSON object carrying 'files'; found "
            f"{type(raw).__name__}"
        )
    if "files" not in raw:
        raise ContractViolation(
            f"{meta_path}: required field 'files' is ABSENT. Present top-level fields: "
            f"{sorted(raw)}. Amend the contract block if the manifest names it otherwise."
        )
    files: object = raw["files"]
    if not isinstance(files, list):
        raise ContractViolation(
            f"{meta_path}: 'files' must be an array; found {type(files).__name__}"
        )

    rows: list[tuple[str, str]] = []
    for idx, entry in enumerate(files):  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, dict):
            raise ContractViolation(
                f"{meta_path}: files[{idx}] must be an object with 'path' and 'sha256'; "
                f"found {type(entry).__name__}"
            )
        for field in ("path", "sha256"):
            if field not in entry:
                raise ContractViolation(
                    f"{meta_path}: files[{idx}] is missing required field {field!r}. "
                    f"Present fields: {sorted(entry)}."
                )
        rel: object = entry["path"]
        sha: object = entry["sha256"]
        if not isinstance(rel, str) or not rel:
            raise ContractViolation(f"{meta_path}: files[{idx}].path must be a non-empty string")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            raise ContractViolation(
                f"{meta_path}: files[{idx}].sha256 must be 64 lowercase hex chars; "
                f"found {sha!r}"
            )
        rows.append((rel, sha))
    if not rows:
        raise ContractViolation(f"{meta_path}: 'files' is empty -- nothing is pinned")
    return rows


def verify_shas(dataset_dir: Path, rows: list[tuple[str, str]]) -> dict[str, Any]:
    """Verify each listed file, and flag record-bearing files the manifest does not pin."""
    match: list[str] = []
    mismatch: list[dict[str, str]] = []
    missing: list[str] = []
    for rel, expected in rows:
        path = dataset_dir / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(path)
        if actual == expected:
            match.append(rel)
        else:
            mismatch.append({"path": rel, "expected": expected, "actual": actual})

    listed = {rel for rel, _ in rows}
    unlisted: list[str] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dataset_dir).as_posix()
        if rel == "dataset_metadata.json" or rel in listed:
            continue
        if path.suffix in _RECORD_SUFFIXES:
            unlisted.append(rel)

    return {
        "listed": len(rows),
        "match": match,
        "mismatch": mismatch,
        "missing": missing,
        "unlisted_record_files": unlisted,
        "clean": not mismatch and not missing and not unlisted,
    }


# ---------------------------------------------------------------------------
# record intake -- strict, loud, never coercing
# ---------------------------------------------------------------------------

def _describe_moves(value: object) -> str:
    """Diagnostic for a 'moves' payload that is not the declared list-of-[q,r]."""
    if not isinstance(value, list) or not value:
        return f"type={type(value).__name__}"
    head: object = value[0]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(head, dict):
        return f"list of objects with keys {sorted(head)}"
    if isinstance(head, (list, tuple)):
        return f"list of {type(head).__name__} of length {len(head)}"  # pyright: ignore[reportUnknownArgumentType]
    return f"list of {type(head).__name__}"


def parse_record(obj: object, where: str) -> dict[str, Any]:
    """Validate one record against the DECLARED contract. Raises naming the exact field."""
    if not isinstance(obj, dict):
        raise ContractViolation(f"{where}: record must be a JSON object; found {type(obj).__name__}")

    for field in DECLARED_RECORD_FIELDS:
        if field not in obj:
            raise ContractViolation(
                f"{where}: declared field {field!r} is ABSENT. Present fields: {sorted(obj)}. "
                "The DECLARED INPUT CONTRACT in this file's docstring is the single place to "
                "amend -- this tool will not guess a replacement name."
            )

    game_id: object = obj["game_id"]
    if not isinstance(game_id, str) or not game_id:
        raise ContractViolation(f"{where}: 'game_id' must be a non-empty string; found {game_id!r}")

    game_hash: object = obj["game_hash"]
    if not isinstance(game_hash, str) or not game_hash:
        raise ContractViolation(
            f"{where}: 'game_hash' must be a non-empty string; found {game_hash!r}. R247 names "
            "game_hash as the dedupe key; see the DEDUPE block for what it can be compared to."
        )

    winner: object = obj["winner"]
    if isinstance(winner, bool) or not isinstance(winner, int):
        raise ContractViolation(
            f"{where}: 'winner' must be an int in {WINNER_DOMAIN}; found {winner!r} "
            f"({type(winner).__name__})"
        )

    elo: object = obj["elo"]
    if isinstance(elo, bool) or not isinstance(elo, (int, float)):
        raise ContractViolation(
            f"{where}: 'elo' must be a number (R247: per-game Elo); found {elo!r} "
            f"({type(elo).__name__})"
        )

    raw_moves: object = obj["moves"]
    if not isinstance(raw_moves, list) or not raw_moves:
        raise ContractViolation(
            f"{where}: 'moves' must be a non-empty list of [q, r] int pairs; found "
            f"{_describe_moves(raw_moves)}"
        )
    moves: list[tuple[int, int]] = []
    for i, mv in enumerate(raw_moves):  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(mv, list) or len(mv) != 2:  # pyright: ignore[reportUnknownArgumentType]
            raise ContractViolation(
                f"{where}: moves[{i}] must be a 2-element [q, r] array; found "
                f"{_describe_moves(raw_moves)}. The in-repo human corpus uses "
                "{'x': q, 'y': r} objects (src/mantis/data/sources/human.py:78); if the "
                "dataset does too, amend the contract block -- this tool will not coerce."
            )
        q: object = mv[0]  # pyright: ignore[reportUnknownVariableType]
        r: object = mv[1]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(q, bool) or isinstance(r, bool) or not isinstance(q, int) or not isinstance(r, int):
            raise ContractViolation(
                f"{where}: moves[{i}] coordinates must be ints; found ({q!r}, {r!r})"
            )
        if not (I32_MIN <= q <= I32_MAX) or not (I32_MIN <= r <= I32_MAX):
            raise ContractViolation(
                f"{where}: moves[{i}] = ({q}, {r}) does not fit i32; mantis-core keys its "
                "cell map with (i32, i32) (core.rs:113-115)"
            )
        moves.append((q, r))

    extra = sorted(set(obj) - set(DECLARED_RECORD_FIELDS))
    return {
        "game_id": game_id,
        "game_hash": game_hash,
        "winner": winner,
        "elo": float(elo),
        "moves": moves,
        "extra_fields": extra,
    }


def iter_records(dataset_dir: Path, rows: list[tuple[str, str]]) -> Iterator[dict[str, Any]]:
    """Yield validated records from every manifest-listed record file, in manifest order."""
    for rel, _ in rows:
        path = dataset_dir / rel
        if path.suffix not in _RECORD_SUFFIXES or rel == "dataset_metadata.json":
            continue
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    where = f"{rel}:{lineno}"
                    try:
                        obj: object = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ContractViolation(f"{where}: not valid JSON ({exc})") from exc
                    yield parse_record(obj, where)
        else:
            try:
                payload: object = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ContractViolation(f"{rel}: not valid JSON ({exc})") from exc
            if not isinstance(payload, list):
                raise ContractViolation(
                    f"{rel}: a .json record file must contain a JSON ARRAY of records; found "
                    f"{type(payload).__name__}"
                )
            for idx, obj in enumerate(payload):  # pyright: ignore[reportUnknownVariableType]
                yield parse_record(obj, f"{rel}[{idx}]")


# ---------------------------------------------------------------------------
# convention audit -- the mapping is CHECKED, not asserted
# ---------------------------------------------------------------------------

def mover_of_ply(i: int) -> int:
    """Which player places stone index ``i``.

    board/mod.rs:24-26 + core.rs:118-121: ply 0 is P1's single stone, then each player
    places two stones per turn.
    """
    if i == 0:
        return PLAYER_ONE
    return PLAYER_TWO if ((i - 1) // 2) % 2 == 0 else PLAYER_ONE


def has_six_run(cells: set[tuple[int, int]]) -> bool:
    """True if ``cells`` contains ``WIN_RUN`` collinear stones along a hex axis."""
    for dq, dr in HEX_AXES:
        for q, r in cells:
            if (q - dq, r - dr) in cells:
                continue  # not the start of a run
            n = 0
            cq, cr = q, r
            while (cq, cr) in cells:
                n += 1
                if n >= WIN_RUN:
                    return True
                cq, cr = cq + dq, cr + dr
    return False


def audit_conventions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay every game and check the declared winner against mantis-core's rules."""
    out: dict[str, Any] = {
        "games": len(records),
        "winner_values_seen": {},
        "draws": 0,
        "winner_outside_mapped_domain": [],
        "repeated_cell_games": [],
        "winner_holds_six_run": 0,
        "winner_lacks_six_run": 0,
        "loser_holds_six_run_when_winner_does_not": 0,
        "neither_holds_six_run": 0,
        "both_hold_six_run": 0,
        "q_min": None, "q_max": None, "r_min": None, "r_max": None,
        "games_with_cell_outside_nominal_window": 0,
        "nominal_window_half": NOMINAL_WINDOW_HALF,
    }
    seen: dict[str, int] = {}
    q_lo = r_lo = q_hi = r_hi = None

    for rec in records:
        winner: int = rec["winner"]
        seen[str(winner)] = seen.get(str(winner), 0) + 1
        if winner == 0:
            out["draws"] += 1
        elif winner not in WINNER_DOMAIN:
            if len(out["winner_outside_mapped_domain"]) < 20:
                out["winner_outside_mapped_domain"].append(
                    {"game_id": rec["game_id"], "winner": winner}
                )

        moves: list[tuple[int, int]] = rec["moves"]
        by_player: dict[int, set[tuple[int, int]]] = {PLAYER_ONE: set(), PLAYER_TWO: set()}
        occupied: set[tuple[int, int]] = set()
        repeated = False
        outside = False
        for i, (q, r) in enumerate(moves):
            if (q, r) in occupied:
                repeated = True
            occupied.add((q, r))
            by_player[mover_of_ply(i)].add((q, r))
            q_lo = q if q_lo is None else min(q_lo, q)
            q_hi = q if q_hi is None else max(q_hi, q)
            r_lo = r if r_lo is None else min(r_lo, r)
            r_hi = r if r_hi is None else max(r_hi, r)
            if abs(q) > NOMINAL_WINDOW_HALF or abs(r) > NOMINAL_WINDOW_HALF:
                outside = True
        if repeated and len(out["repeated_cell_games"]) < 20:
            out["repeated_cell_games"].append(rec["game_id"])
        if outside:
            out["games_with_cell_outside_nominal_window"] += 1

        if winner in WINNER_DOMAIN:
            win_run = has_six_run(by_player[winner])
            lose_run = has_six_run(by_player[-winner])
            if win_run and lose_run:
                out["both_hold_six_run"] += 1
            if win_run:
                out["winner_holds_six_run"] += 1
            else:
                out["winner_lacks_six_run"] += 1
                if lose_run:
                    out["loser_holds_six_run_when_winner_does_not"] += 1
                else:
                    out["neither_holds_six_run"] += 1

    out["winner_values_seen"] = seen
    out["q_min"], out["q_max"], out["r_min"], out["r_max"] = q_lo, q_hi, r_lo, r_hi

    lacks = out["winner_lacks_six_run"]
    flipped = out["loser_holds_six_run_when_winner_does_not"]
    if lacks and flipped == lacks:
        verdict = (
            "WINNER CONVENTION LIKELY INVERTED: in every game where the declared winner "
            "holds no six-run, the OTHER player does. Do NOT flip a sign in this tool -- "
            "resolve the convention at source and amend the contract block."
        )
    elif lacks:
        verdict = (
            "WINNER CONVENTION UNRESOLVED: some declared winners hold no six-run and the "
            "flip does not explain all of them (truncated records, a different win rule, or "
            "a transposed axial convention are all live explanations)."
        )
    else:
        verdict = (
            "winner convention CONSISTENT with mantis-core under the identity mapping "
            "(1 => Player::One, -1 => Player::Two) on every game replayed"
        )
    out["verdict"] = verdict
    out["known_limit"] = (
        "a q<->r transposition is UNDETECTABLE by this check: the axis set "
        "{(1,0),(0,1),(1,-1)} is invariant under it (core.rs:51-55)"
    )
    return out


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

def describe(values: list[float], *, bins: int = 10) -> dict[str, Any]:
    """count / min / median / mean / max plus an equal-width histogram.

    Bin edges are DERIVED from the observed range, never transcribed, so the histogram
    cannot go stale against a constant somebody edited elsewhere.
    """
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None,
                "histogram": []}
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins if hi > lo else 0.0
    counts = [0] * bins
    for v in values:
        idx = bins - 1 if width == 0.0 else min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    histogram = [
        {
            "lo": lo + i * width,
            "hi": (lo + (i + 1) * width) if width else hi,
            "count": counts[i],
        }
        for i in range(bins)
    ]
    return {
        "count": len(values),
        "min": lo,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": hi,
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------

def derived_move_key(moves: list[tuple[int, int]]) -> str:
    """A canonical digest of a move sequence, DERIVED BY THIS TOOL.

    NOT an in-repo contract and NOT the dataset's ``game_hash``. It exists solely so the
    two corpora can be compared at all -- see the DEDUPE block in the module docstring.
    Definition, pinned so it is reproducible: sha256 of the compact JSON encoding of the
    ordered ``[[q, r], ...]`` list, hex.
    """
    payload = json.dumps([[q, r] for q, r in moves], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _in_repo_move_keys(corpus_dir: Path) -> tuple[list[str], list[str]]:
    """Derive move keys from the in-repo human corpus JSON cache.

    Shape per src/mantis/data/sources/human.py:78 -- ``moves[i].x`` / ``moves[i].y`` ARE the
    axial (q, r); src/mantis/data/generate.py:125 writes the same pair back out.
    """
    keys: list[str] = []
    skipped: list[str] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            skipped.append(f"{path.name}: root is {type(data).__name__}, not an object")
            continue
        raw: object = data.get("moves")
        if not isinstance(raw, list) or not raw:
            skipped.append(f"{path.name}: no 'moves' list")
            continue
        moves: list[tuple[int, int]] = []
        bad = False
        for mv in raw:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(mv, dict) or "x" not in mv or "y" not in mv:
                bad = True
                break
            x: object = mv["x"]
            y: object = mv["y"]
            if not isinstance(x, int) or not isinstance(y, int):
                bad = True
                break
            moves.append((x, y))
        if bad:
            skipped.append(f"{path.name}: moves are not {{'x': int, 'y': int}} objects")
            continue
        keys.append(derived_move_key(moves))
    return keys, skipped


def dedupe_leg(records: list[dict[str, Any]], corpus_dir: Path | None) -> dict[str, Any]:
    """R247's dedupe leg. Loud about the absence of an in-repo ``game_hash`` producer."""
    hashes = [r["game_hash"] for r in records]
    derived = [derived_move_key(r["moves"]) for r in records]
    leg: dict[str, Any] = {
        "dataset_games": len(records),
        "dataset_distinct_game_hash": len(set(hashes)),
        "dataset_duplicate_game_hash": len(hashes) - len(set(hashes)),
        "dataset_distinct_derived_move_key": len(set(derived)),
        "dataset_duplicate_derived_move_key": len(derived) - len(set(derived)),
        "in_repo_game_hash_producer": (
            "ABSENT -- human corpus identity is the source UUID filename stem "
            "(src/mantis/data/sources/human.py:91); the only hash-of-moves in the repo is "
            "src/mantis/data/generate.py:132-135, which hashes BOT games for use as a "
            "filename and is non-canonical under F-06"
        ),
    }
    if corpus_dir is None:
        leg["overlap"] = "NO IN-REPO REFERENCE AVAILABLE"
        leg["overlap_detail"] = (
            "no --in-repo-corpus supplied, and there is no in-repo game_hash to compare "
            "against regardless; this leg made NO comparison"
        )
        return leg
    if not corpus_dir.is_dir():
        raise UsageError(f"--in-repo-corpus is not a directory: {corpus_dir}")
    repo_keys, skipped = _in_repo_move_keys(corpus_dir)
    overlap = set(derived) & set(repo_keys)
    leg["overlap"] = "DERIVED KEY COMPARISON -- not an in-repo game_hash comparison"
    leg["in_repo_corpus_dir"] = str(corpus_dir)
    leg["in_repo_games_keyed"] = len(repo_keys)
    leg["in_repo_files_skipped"] = len(skipped)
    leg["in_repo_skip_examples"] = skipped[:10]
    leg["overlapping_games"] = len(overlap)
    leg["overlap_fraction_of_dataset"] = (len(overlap) / len(derived)) if derived else None
    return leg


# ---------------------------------------------------------------------------
# the two recorded selection biases (R247 requires these in the OUTPUT)
# ---------------------------------------------------------------------------

def selection_biases(records: list[dict[str, Any]], conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Measured-plus-stated. The measurement is taken here; the statement is R247's."""
    ply_counts = [len(r["moves"]) for r in records]
    below = sum(1 for n in ply_counts if n < DECLARED_MIN_PLIES)
    decisive = sum(1 for r in records if r["winner"] in WINNER_DOMAIN)
    draws = conv["draws"]
    total = len(records) or 1
    return [
        {
            "id": "bias-a-decisive-only",
            "statement": (
                "The corpus is decisive-only: it carries zero draw mass. A value head "
                "trained on it sees no drawn outcome and so over-fits corpus-mode signal "
                "that self-play cannot reproduce -- the F-07 mechanism (more pretrain "
                "epochs regressed self-play; median plies 12 vs 17). Adjacency: F-07."
            ),
            "register_adjacency": ["F-07"],
            "measured": {
                "games": len(records),
                "draws_found": draws,
                "draw_fraction": draws / total,
                "decisive_games": decisive,
                "winner_values_seen": conv["winner_values_seen"],
            },
        },
        {
            "id": "bias-b-min-move-floor",
            "statement": (
                f"A >={DECLARED_MIN_PLIES}-ply selection floor drops SHORT tactical wins -- "
                "exactly the near-win class the register cares about. F-15: a forced-win "
                "short-circuit meant the network never evaluated near-win positions, so no "
                "fork learning. F-38: the deep value-blind losses are bounded-tactical "
                "properties provable at depth 6-8 turns, so short decisive games are "
                "signal, not noise. Adjacencies: F-15, F-38. LAW-03: these counts are "
                "PLIES (stones), not turns -- ply 0 places one stone, every later turn "
                "places two (board/mod.rs:24-26)."
            ),
            "register_adjacency": ["F-15", "F-38"],
            "measured": {
                "min_plies_found": min(ply_counts) if ply_counts else None,
                "games_below_declared_floor": below,
                "declared_floor_plies": DECLARED_MIN_PLIES,
                "floor_source": "src/mantis/data/sources/human.py:108-112",
            },
        },
    ]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """The repository this tool ships in (tools/<this file> -> parents[1])."""
    return Path(__file__).resolve().parents[1]


def checked_out_path(out: Path) -> Path:
    """Refuse an output path inside the repository tree (R7).

    The report is a run artifact over an artifact that deliberately lives outside the tree;
    writing it back in is the exact class R7 exists to stop. Both sides are ``resolve()``d
    so a symlink cannot walk around the check.
    """
    resolved = out.expanduser().resolve()
    root = _repo_root()
    if resolved == root or root in resolved.parents:
        raise UsageError(
            f"refusing to write the audit report inside the repository tree: {resolved} is "
            f"under {root}. R7 keeps run artifacts out of git; pass --out somewhere outside "
            "the checkout."
        )
    return resolved


def build_report(
    dataset_dir: Path,
    sha_leg: dict[str, Any],
    records: list[dict[str, Any]],
    corpus_dir: Path | None,
) -> tuple[dict[str, Any], int]:
    """Assemble the report and decide the exit code."""
    conv = audit_conventions(records)
    violations: list[str] = []

    if conv["draws"]:
        violations.append(
            f"{conv['draws']} record(s) carry winner == 0 (a draw). mantis-core's Player has "
            "no draw member (core.rs:59-64) and GameRecord.winner is +1/-1 only "
            "(data/sources/base.py:20) -- a draw is unmappable. R247 predicted zero draw "
            "mass; this is the measurement disagreeing with it."
        )
    if conv["winner_outside_mapped_domain"]:
        violations.append(
            f"winner values outside the mapped domain {WINNER_DOMAIN}: "
            f"{conv['winner_outside_mapped_domain']}"
        )
    if conv["repeated_cell_games"]:
        violations.append(
            f"{len(conv['repeated_cell_games'])} game(s) place two stones on the same cell "
            f"(examples: {conv['repeated_cell_games'][:5]}); mantis-core's board is a "
            "cell->Cell map (core.rs:113-115) and cannot represent that"
        )
    if conv["winner_lacks_six_run"]:
        violations.append(
            f"{conv['winner_lacks_six_run']} game(s): the declared winner holds no six-run "
            f"in the final position. {conv['verdict']}"
        )

    report: dict[str, Any] = {
        "tool": "tools/audit_bootstrap_corpus.py",
        "ruling": "R247",
        "contract_version": CONTRACT_VERSION,
        "dataset_dir": str(dataset_dir),
        "sha256_verification": sha_leg,
        "convention_audit": conv,
        "distributions": {
            "elo": describe([r["elo"] for r in records]),
            "move_count_plies": describe([float(len(r["moves"])) for r in records]),
        },
        "dedupe": dedupe_leg(records, corpus_dir),
        "selection_biases": selection_biases(records, conv),
        "unexpected_record_fields": sorted({f for r in records for f in r["extra_fields"]}),
        "violations": violations,
    }

    if violations:
        report["verdict"] = "CONTRACT_VIOLATION"
        return report, EXIT_CONTRACT
    report["verdict"] = "CLEAN"
    return report, EXIT_OK


def sha_only_report(dataset_dir: Path, sha_leg: dict[str, Any]) -> dict[str, Any]:
    """The report emitted when verification fails: no statistics over unpinned bytes.

    Deliberately short-circuits before any record is parsed. Distributions, the convention
    audit and the bias measurements over bytes that are NOT the pinned bytes would be
    numbers about an unknown artifact — worse than no numbers, because they read as
    evidence.
    """
    return {
        "tool": "tools/audit_bootstrap_corpus.py",
        "ruling": "R247",
        "contract_version": CONTRACT_VERSION,
        "dataset_dir": str(dataset_dir),
        "verdict": "SHA_VERIFICATION_FAILED",
        "sha256_verification": sha_leg,
        "record_audit": (
            "NOT RUN — the bytes on disk are not the bytes dataset_metadata.json pins, so "
            "every downstream measurement would describe an unknown artifact"
        ),
        "violations": [],
    }


def _print_summary(report: dict[str, Any]) -> None:
    sha = report["sha256_verification"]
    print(f"verdict: {report['verdict']}")
    print(
        f"sha256: {len(sha['match'])}/{sha['listed']} match, {len(sha['mismatch'])} mismatch, "
        f"{len(sha['missing'])} missing, {len(sha['unlisted_record_files'])} unlisted"
    )
    if "convention_audit" not in report:
        print(f"records: {report['record_audit']}")
        return
    conv = report["convention_audit"]
    print(f"games: {conv['games']}  |  {conv['verdict']}")
    for bias in report["selection_biases"]:
        print(f"\n[{bias['id']}] measured: {json.dumps(bias['measured'], sort_keys=True)}")
        print(f"  {bias['statement']}")
    for v in report["violations"]:
        print(f"\nVIOLATION: {v}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_bootstrap_corpus",
        description="Offline audit of the external human bootstrap corpus (R247). Downloads "
                    "nothing; writes nothing inside the repository.",
    )
    parser.add_argument("dataset_dir", type=Path, help="extracted dataset directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON report path; must be OUTSIDE the repository tree (R7)")
    parser.add_argument("--in-repo-corpus", type=Path, default=None,
                        help="optional in-repo human corpus JSON directory for the dedupe "
                             "leg; omitted => the leg reports NO IN-REPO REFERENCE AVAILABLE")
    args = parser.parse_args(argv)

    try:
        dataset_dir: Path = args.dataset_dir.expanduser()
        if not dataset_dir.is_dir():
            raise UsageError(f"dataset_dir is not a directory: {dataset_dir}")
        out_path = checked_out_path(args.out)
        rows = load_metadata(dataset_dir)
        sha_leg = verify_shas(dataset_dir, rows)
        # sha FIRST, and it short-circuits: an unpinned byte invalidates every measurement
        # that would follow it.
        if not sha_leg["clean"]:
            report, code = sha_only_report(dataset_dir, sha_leg), EXIT_SHA
        else:
            records = list(iter_records(dataset_dir, rows))
            if not records:
                raise ContractViolation(
                    f"{dataset_dir}: zero records parsed from the manifest-listed "
                    f"{'/'.join(_RECORD_SUFFIXES)} files. The DECLARED INPUT CONTRACT expects "
                    "the game records to live in manifest-listed .jsonl or .json files."
                )
            report, code = build_report(dataset_dir, sha_leg, records, args.in_repo_corpus)
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ContractViolation as exc:
        print(f"CONTRACT VIOLATION: {exc}", file=sys.stderr)
        return EXIT_CONTRACT

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"\nreport: {out_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
