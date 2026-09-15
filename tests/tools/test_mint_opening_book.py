"""`tools/mint_opening_book.py --exclude-book` mints the book_v2 CANDIDATE POOL: deterministic, duplicate-free, and disjoint from book_v1 by exact move list."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BOOKS_DIR = _REPO / "src" / "mantis" / "arena" / "books"
_MINTER = _REPO / "tools" / "mint_opening_book.py"
_BOOK_V1 = _BOOKS_DIR / "book_v1_s20260625_p4.json"
_POOL_ID = "book_v2_pool_s20260915_p4"
_POOL_ARGS = ["--seed", "20260915", "--plies", "4", "--n", "512", "--exclude-book", str(_BOOK_V1)]


@pytest.fixture(scope="module")
def minter():
    spec = importlib.util.spec_from_file_location("mint_opening_book_under_test", _MINTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _moves_of(payload: dict) -> list[list[list[int]]]:
    return [o["moves"] for o in payload["openings"]]


def test_the_pool_is_deterministic_in_its_seed(minter) -> None:
    exclude = _moves_of(json.loads(_BOOK_V1.read_text(encoding="utf-8")))
    a = minter.mint_pool(seed=7, plies=4, n=16, exclude=exclude)
    b = minter.mint_pool(seed=7, plies=4, n=16, exclude=exclude)
    c = minter.mint_pool(seed=8, plies=4, n=16, exclude=exclude)
    assert a == b and _moves_of(a) != _moves_of(c)
    assert [o["id"] for o in a["openings"]] == list(range(16))


def test_the_pool_excludes_book_v1_by_exact_move_list_and_carries_no_duplicates(minter) -> None:
    v1 = _moves_of(json.loads(_BOOK_V1.read_text(encoding="utf-8")))
    pool = minter.mint_pool(seed=20260915, plies=4, n=512, exclude=v1)
    keys = [json.dumps(m) for m in _moves_of(pool)]
    assert len(keys) == 512 and len(set(keys)) == 512
    assert not set(keys) & {json.dumps(m) for m in v1}


def test_an_excluded_opening_is_skipped_not_renumbered_around(minter) -> None:
    plain = minter.mint_pool(seed=3, plies=4, n=4, exclude=[])
    first = _moves_of(plain)[0]
    without_first = minter.mint_pool(seed=3, plies=4, n=3, exclude=[first])
    assert _moves_of(without_first) == _moves_of(plain)[1:4]


def test_the_pool_provenance_names_its_seed_and_exclusions(minter) -> None:
    pool = minter.mint_pool(seed=5, plies=4, n=2, exclude=[[[0, 0], [1, 0], [0, 1], [1, 1]]])
    assert pool["provenance"] == {"minter": "tools/mint_opening_book.py", "seed": 5, "plies": 4,
                                  "n": 2, "excluded_openings": 1}


def test_the_packaged_pool_reproduces_from_its_minter_args_and_matches_its_manifest_sha(tmp_path) -> None:
    out = tmp_path / "pool.json"
    result = subprocess.run([sys.executable, str(_MINTER), *_POOL_ARGS, "--out", str(out)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"minter failed: {result.stderr}"
    with (_BOOKS_DIR / "manifest.toml").open("rb") as handle:
        entry = tomllib.load(handle)["books"][_POOL_ID]
    packaged = _BOOKS_DIR / entry["file"]
    assert out.read_bytes() == packaged.read_bytes()
    assert hashlib.sha256(packaged.read_bytes()).hexdigest() == entry["sha256"]


def test_the_plain_minter_output_is_unchanged_by_the_pool_arm(minter) -> None:
    assert minter.mint_book(seed=20260625, plies=4, n=2) == {
        "openings": minter.mint_pool(seed=20260625, plies=4, n=2, exclude=[])["openings"]}
