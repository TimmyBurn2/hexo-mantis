# >300 justify (R8): five mechanical censuses (J-01..J-05) over the ONE package src/mantis/selfplay, each with its LAW-07 bite arm; J-01's mutation self-tests re-drive the same _loop_census/_find_function primitives they prove bite, and each remaining census is small — a per-census split would duplicate the AST walker or import test-from-test.
"""Suite J — census pins over `src/mantis/selfplay` (J-01 … J-05).

IMPL-written (non-⊕). These are mechanical, review-blocking censuses: each is a grep/AST
scan of the SHIPPED source with a named bug class, and the checker-shaped ones carry a
LAW-07 mutation self-test proving the census bites.

  J-01 — hot-path loop ban (Q6, M8-extended). AST census over the named hot-path functions
         counting For/While/AsyncFor + ListComp/SetComp/DictComp/GeneratorExp, plus a
         ZERO-count assertion on `map(`/`filter(` calls in those functions, compared against
         the table FROZEN in DESIGN §Q6 (review-measured old-side counts — the allowlist is
         anchored to old truth, not authored here). Two-armed mutation self-test: an injected
         `for` loop AND an injected comprehension must both be detected.
  J-02 — LAW-11 census: zero `"grid"`-default representation tokens (the WP9 _RE_DENSE_DEFAULT
         pattern set).
  J-03 — KILL census: the WP4/WP6/WP9 killed knobs have zero hits.
  J-04 — swallow census: `except …: pass` hits == exactly the DV-11 allowlisted `__del__`
         site in `inference_local.py`.
  J-05 — event-schema pin: the `game_complete` payload key set is frozen (WP13-A builds
         against it; a dropped/added/renamed key bites).

Nothing under `src/mantis/selfplay` is modified by this file; it only reads it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_SELFPLAY = Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"

# ── AST census primitives ────────────────────────────────────────────────────────────
_LOOP_FOR = (ast.For, ast.AsyncFor)
_COMP = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _find_function(tree: ast.Module, qual: str) -> ast.AST | None:
    """Resolve `name` (module-level function) or `Class.name` (method) inside `tree`."""
    parts = qual.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return node
        return None
    cls_name, fn_name = parts
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for member in node.body:
                if (isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and member.name == fn_name):
                    return member
    return None


def _loop_census(func: ast.AST) -> tuple[int, int, int]:
    """(for, while, comprehension) counts inside `func` (including nested closures — the
    convention the §Q6 table was measured under)."""
    fors = whiles = comps = 0
    for sub in ast.walk(func):
        if sub is func:
            continue
        if isinstance(sub, _LOOP_FOR):
            fors += 1
        elif isinstance(sub, ast.While):
            whiles += 1
        elif isinstance(sub, _COMP):
            comps += 1
    return fors, whiles, comps


def _mapfilter_census(func: ast.AST) -> int:
    """Count `map(...)` / `filter(...)` calls inside `func` — per-item Python loops named in
    the ban alongside `for`/`while`/comprehensions."""
    n = 0
    for sub in ast.walk(func):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id in ("map", "filter")):
            n += 1
    return n


# ── the FROZEN §Q6 table (review-measured old-side counts, NOT authored here) ─────────
# Each row: (label, [(module_filename, qualname), …], (for, while, comprehension)).
# Rows whose function set has >1 entry are the split/combined §Q6 rows: J-01 asserts the
# SUM over the named functions equals the frozen triple (counts are conserved under the
# §a.2 4-way pool relocation).
_Q6_TABLE: list[tuple[str, list[tuple[str, str]], tuple[int, int, int]]] = [
    ("collate_graph_batch",
     [("graph_collate.py", "collate_graph_batch")], (0, 0, 0)),
    ("_check_structural",
     [("graph_collate.py", "_check_structural")], (2, 0, 0)),
    ("_check_semantic",
     [("graph_collate.py", "_check_semantic")], (1, 0, 1)),
    ("segment_softmax / stone_mask_from_batch",
     [("graph_collate.py", "segment_softmax"),
      ("graph_collate.py", "stone_mask_from_batch")], (0, 0, 0)),
    ("InferenceServer.run (dense loop)",
     [("inference_server.py", "InferenceServer.run")], (0, 1, 0)),
    # F-816-10 (R276(f)) moves this row's `for` count 0 -> 1, and the movement is RULED, not
    # absorbed. The R43 frozen-table edit was DISCLOSED and queued the same event as
    # ADJUDICATION_QUEUE F-816-13, and GRANTED by R281(e) on the reasoning below — the
    # R276(c) shape, per-event, NEVER precedent. Cite the queue row, not this comment, for
    # the grant's scope. The ban this row enforces is on a PER-ITEM Python loop: the hot paths are
    # vectorized numpy/torch and a loop that touches one graph, one node or one edge at a time
    # re-introduces exactly the per-item overhead the port exists to keep out. The new `for`
    # is a loop over the PARTS OF ONE MEMORY-BOUNDED PLAN — `M` iterations where `M = 1`
    # whenever the caps do not bind, which is every config CI runs (their caps are non-binding
    # by construction) — and each iteration performs one WHOLE vectorized collate + forward +
    # segment-softmax over its part. Its count is a function of the minted cap, not of the
    # batch's item count, so the quantity the ban is about does not move.
    # It is also not optional: the bound this packet ships is a bound on the PEAK, so the
    # parts must run one at a time with the previous part's tensors freed. A vectorized
    # "all parts at once" is the un-split forward, i.e. the defect.
    ("InferenceServer._run_graph_loop",
     [("inference_server.py", "InferenceServer._run_graph_loop")], (1, 1, 0)),
    ("InferenceServer.submit_and_wait / load_state_dict_safe",
     [("inference_server.py", "InferenceServer.submit_and_wait"),
      ("inference_server.py", "InferenceServer.load_state_dict_safe")], (0, 0, 0)),
    ("run_stats_loop + pool_push arms (old _run_stats_loop, §a.2 split)",
     [("pool_drain.py", "run_stats_loop"),
      ("pool_push.py", "push_dense"),
      ("pool_push.py", "push_graph")], (3, 1, 2)),
    ("LocalInferenceEngine.infer_batch",
     [("inference_local.py", "LocalInferenceEngine.infer_batch")], (4, 0, 0)),
    ("LocalInferenceEngine._infer_batch_graph",
     [("inference_local.py", "LocalInferenceEngine._infer_batch_graph")], (0, 0, 3)),
]


def _trees() -> dict[str, ast.Module]:
    return {p.name: ast.parse(p.read_text()) for p in _SELFPLAY.glob("*.py")}


@pytest.mark.parametrize(
    "label,members,expected",
    [(label, members, expected) for label, members, expected in _Q6_TABLE],
    ids=[row[0].split(" ")[0] for row in _Q6_TABLE],
)
def test_j01_hot_path_loop_census(label, members, expected) -> None:
    """J-01 — PASS iff the summed (for, while, comprehension) census over the named hot-path
    function(s) equals the FROZEN §Q6 triple, AND those functions contain zero `map`/`filter`
    calls.

    FAIL = a new per-item Python loop, comprehension, generator expression, or map/filter on
    a hot path. This is a perf/behavior contract: the hot paths are vectorized numpy/torch,
    and a Python-level loop re-introduces the per-item overhead the port exists to keep out.
    Any deviation is a mechanical REVIEW-impl finding, not a judgment call."""
    trees = _trees()
    total = [0, 0, 0]
    for module, qual in members:
        tree = trees[module]
        func = _find_function(tree, qual)
        assert func is not None, f"{label}: {module}:{qual} not found (a rename broke the census)"
        f, w, c = _loop_census(func)
        total[0] += f
        total[1] += w
        total[2] += c
        assert _mapfilter_census(func) == 0, (
            f"{label}: {qual} contains a map()/filter() — banned per-item loop on a hot path"
        )
    assert tuple(total) == expected, (
        f"{label}: census {tuple(total)} != frozen §Q6 {expected}"
    )


def test_j01_census_covers_every_frozen_row() -> None:
    """J-01 (coverage arm) — every §Q6 row resolves to real functions in the shipped source,
    so the census can never silently cover less than the frozen table (a deleted/renamed
    hot-path function would make the parametrized rows disappear, not fail)."""
    trees = _trees()
    for label, members, _ in _Q6_TABLE:
        for module, qual in members:
            assert module in trees, f"{label}: module {module} missing from selfplay"
            assert _find_function(trees[module], qual) is not None, (
                f"{label}: {module}:{qual} not found"
            )
    assert len(_Q6_TABLE) == 10, "the frozen §Q6 table has exactly 10 rows"


# ── J-01 mutation self-test (LAW-07): the census must BITE, two-armed ─────────────────
# `segment_softmax` is the §Q6 (0,0,0) row; a doctored copy with an injected loop or
# comprehension must be detected as != (0,0,0) by the SAME census logic used above.
_SEGMENT_SOFTMAX_SRC = _SELFPLAY / "graph_collate.py"


def _doctored_func(inject: str) -> ast.AST:
    """Return the `segment_softmax` node from a copy of graph_collate.py whose body has had
    one line injected right after the signature."""
    src_lines = _SEGMENT_SOFTMAX_SRC.read_text().splitlines()
    tree = ast.parse("\n".join(src_lines))
    func = _find_function(tree, "segment_softmax")
    assert func is not None
    # Splice the injected statement in as the second body element (after the docstring),
    # then re-parse just that function so line/col offsets are consistent.
    doctored = ast.parse(inject).body[0]
    func.body.insert(1, doctored)
    ast.fix_missing_locations(func)
    return func


def test_j01_mutation_self_test_injected_for_loop_bites() -> None:
    """J-01 (LAW-07, arm 1) — an injected `for` loop into the (0,0,0) `segment_softmax`
    function makes the census report a nonzero `for` count, so the frozen-table comparison
    would FAIL. Proves the checker detects a planted loop rather than silently passing."""
    func = _doctored_func("for _ in range(1):\n    pass\n")
    census = _loop_census(func)
    assert census != (0, 0, 0), "census must bite an injected for-loop"
    assert census[0] >= 1, f"the injected for-loop must be counted (got {census})"


def test_j01_mutation_self_test_injected_comprehension_bites() -> None:
    """J-01 (LAW-07, arm 2) — an injected comprehension into the (0,0,0) `segment_softmax`
    function makes the census report a nonzero comprehension count. Proves the M8 extension
    (comprehensions/genexps are counted) actually closes the hole REVIEW-design found."""
    func = _doctored_func("_evade = [x for x in range(1)]\n")
    census = _loop_census(func)
    assert census != (0, 0, 0), "census must bite an injected comprehension"
    assert census[2] >= 1, f"the injected comprehension must be counted (got {census})"


def test_j01_mutation_self_test_injected_map_bites() -> None:
    """J-01 (LAW-07, arm 3) — an injected `map(...)` call is caught by the map/filter census
    (the M8 extension names map/filter as per-item loops alongside comprehensions)."""
    func = _doctored_func("_evade = list(map(str, range(1)))\n")
    assert _mapfilter_census(func) >= 1, "census must bite an injected map() call"


# ── J-02 — LAW-11: zero dense-default representation tokens ────────────────────────────
# The WP9 _RE_DENSE_DEFAULT pattern set (tests/model/test_arch_ban.py:38-42) — a "grid"
# default recovered off a spec/getattr is the silently-dense-by-default class LAW-11 kills.
_RE_DENSE_DEFAULT = re.compile(
    r"getattr\([^)]*,\s*[\"']grid[\"']\s*\)"
    r"|\.get\(\s*[\"']representation[\"']\s*,\s*[\"']grid[\"']"
    r"|,\s*[\"']grid[\"']\s*\)"
)


def _dense_default_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if _RE_DENSE_DEFAULT.search(line):
                hits.append(f"{path.name}:{i}: {line.strip()}")
    return hits


def test_j02_no_dense_default_representation_tokens() -> None:
    """J-02 — PASS iff no dense-by-default `"grid"` representation token appears anywhere in
    `src/mantis/selfplay`. FAIL = a spec/dict read that falls back to grid, which would route
    a graph run through the dense path silently (the LAW-11 class)."""
    assert _dense_default_hits(_SELFPLAY) == []


def test_j02_census_bites_planted_dense_default(tmp_path: Path) -> None:
    """J-02 (LAW-07) — a planted grid-default line makes the census fire."""
    (tmp_path / "planted.py").write_text(
        'def f(cfg):\n    return cfg.get("representation", "grid")\n'
    )
    assert _dense_default_hits(tmp_path), "census must bite a planted grid-default token"


# ── J-03 — KILL census: zero killed-knob tokens ───────────────────────────────────────
# The WP4/WP6/WP9 kills. S1 deliberately DESCRIBED rather than spelled these knob names in
# comments so this census stays at zero; a real re-introduction (read or set) bites.
_KILLED_TOKENS = (
    "legal_move_radius_jitter",
    "interior_selector",
    "gumbel_improved",
    "GumbelImproved",
    "model_representation",
    "KEPT_PLANE_INDICES",
    "turn_veto",
    "strength_aggregate",
    "cluster_pool.",
    "global_encoder.",
    "gpool_bias_branch.",
    "V8ArgmaxBot",
    "V8MCTSBot",
    "dataset_v8",
    "v8_canvas_realness",
)
# Bare `v8` is matched word-boundaried so it does not fire on unrelated substrings.
_RE_V8_WORD = re.compile(r"\bv8\b")


def _killed_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            for tok in _KILLED_TOKENS:
                if tok in line:
                    hits.append(f"{path.name}:{i}: killed token {tok!r}: {line.strip()}")
            if _RE_V8_WORD.search(line):
                hits.append(f"{path.name}:{i}: killed token 'v8': {line.strip()}")
    return hits


def test_j03_no_killed_tokens() -> None:
    """J-03 — PASS iff none of the WP4/WP6/WP9 KILLed knob names (jitter, interior_selector,
    the gumbel-improved placeholder, model_representation, KEPT_PLANE_INDICES, turn_veto, the
    cluster/global/gpool cluster-pool prefixes, strength_aggregate, the v8 family) appear in
    `src/mantis/selfplay`. FAIL = a killed knob resurfaced as a read or a set — the exact
    silently-disabled-opponent / dead-config class the register exists to keep dead."""
    assert _killed_hits(_SELFPLAY) == []


def test_j03_census_bites_planted_killed_token(tmp_path: Path) -> None:
    """J-03 (LAW-07) — a planted killed-knob reference makes the census fire."""
    (tmp_path / "planted.py").write_text('LEGAL = cfg["legal_move_radius_jitter"]\n')
    assert _killed_hits(tmp_path), "census must bite a planted killed token"


# ── J-04 — swallow census: exactly one allowlisted `__del__` site ─────────────────────
def _swallow_sites(root: Path) -> list[str]:
    """`except …: pass` handlers whose body is exactly `pass` (any number of pass stmts)."""
    sites: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.body and all(isinstance(s, ast.Pass) for s in node.body):
                    sites.append(f"{path.name}:{node.lineno}")
    return sites


def _enclosing_function_name(tree: ast.Module, lineno: int) -> str | None:
    """Name of the function that most tightly encloses `lineno`."""
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.name)
    return best[1] if best else None


def test_j04_swallow_census_is_the_single_del_site() -> None:
    """J-04 (DV-11) — PASS iff the ONLY `except …: pass` swallow in `src/mantis/selfplay` is
    the one sanctioned `__del__` best-effort cleanup in `inference_local.py`. FAIL = a second
    swallow anywhere (every other blind-except is a defect: the failure vanishes silently)."""
    sites = _swallow_sites(_SELFPLAY)
    assert len(sites) == 1, f"expected exactly one allowlisted swallow, found: {sites}"
    file_name, lineno = sites[0].split(":")
    assert file_name == "inference_local.py", (
        f"the swallow must live in inference_local.py, not {file_name}"
    )
    tree = ast.parse((_SELFPLAY / "inference_local.py").read_text())
    assert _enclosing_function_name(tree, int(lineno)) == "__del__", (
        "the sanctioned swallow must be inside __del__ (the GC-time best-effort site)"
    )


def test_j04_census_bites_planted_swallow(tmp_path: Path) -> None:
    """J-04 (LAW-07) — a planted `except: pass` makes the census fire."""
    (tmp_path / "planted.py").write_text(
        "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    )
    assert len(_swallow_sites(tmp_path)) == 1, "census must bite a planted swallow"


# ── J-05 — event-schema pin: the frozen `game_complete` payload key set ────────────────
# The FROZEN key set, extracted once from the shipped dict literal. WP13-A builds against
# this schema; a dropped/added/renamed key must bite here before it reaches a consumer.
_FROZEN_GAME_COMPLETE_KEYS = frozenset({
    "event",
    "game_id",
    "game_id_byte_hash",
    "winner",
    "moves",
    "moves_list",
    "worker_id",
    "moves_detail",
    "value_trace",
    "colony_extension_stone_count",
    "colony_extension_stone_total",
    "colony_extension_fraction",
    "longest_line_fraction",
    "n_components",
    "terminal_reason",
    "model_version_min",
    "model_version_max",
    "model_version_distinct",
    "stride5_run_p90",
    "row_max_density",
    "seeded",
    "solver_fires",
})


def _game_complete_dict_keys() -> set[str]:
    """The string keys of the `game_complete` payload dict literal in pool_drain.py, read
    off the SOURCE so a key change in the emitter bites without running the drain loop."""
    tree = ast.parse((_SELFPLAY / "pool_drain.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "event" in keys:
                # The game_complete payload is the dict whose "event" key maps to the
                # literal "game_complete".
                for k, v in zip(node.keys, node.values, strict=False):
                    if (isinstance(k, ast.Constant) and k.value == "event"
                            and isinstance(v, ast.Constant) and v.value == "game_complete"):
                        return set(keys)
    raise AssertionError("no game_complete payload dict literal found in pool_drain.py")


def test_j05_game_complete_source_key_set_frozen() -> None:
    """J-05 — PASS iff the `game_complete` payload dict literal in the emitter carries EXACTLY
    the frozen key set. FAIL = a dropped/added/renamed key. WP13-A's monitor builds against
    this schema, so an undeclared key change is a silent break of a downstream consumer."""
    assert _game_complete_dict_keys() == set(_FROZEN_GAME_COMPLETE_KEYS)


def test_j05_game_complete_golden_key_set_frozen() -> None:
    """J-05 (capture arm) — the captured old-side `game_complete` events (#C3b golden) carry
    the frozen key set MINUS the uuid `game_id` (excluded from the golden, per C-03). Binds
    the schema to the dispatcher capture as well as to the source, so the two cannot drift
    apart silently."""
    golden = json.loads((_FIXTURES / "drain" / "drain_goldens.json").read_text())
    expected = set(_FROZEN_GAME_COMPLETE_KEYS) - {"game_id"}
    seen = 0
    for variant in golden["variants"].values():
        for event in variant.get("events", []):
            if event.get("event") == "game_complete":
                assert set(event.keys()) == expected, (
                    f"captured game_complete key set drift: "
                    f"missing {expected - set(event.keys())}, extra {set(event.keys()) - expected}"
                )
                seen += 1
    assert seen > 0, "the drain golden must contain at least one game_complete event"
