# >300 justify (R8): five mechanical censuses (J-01..J-05) over the ONE package src/mantis/selfplay, each with its LAW-07 bite arm; J-01's mutation self-tests re-drive the same _loop_census/_find_function primitives they prove bite, and each remaining census is small — a per-census split would duplicate the AST walker or import test-from-test.
"""Suite J — census pins over `src/mantis/selfplay` (J-01 … J-05).

Mechanical, review-blocking censuses of the SHIPPED source, each with a named bug class and,
where checker-shaped, a LAW-07 mutation self-test proving it bites: J-01 the hot-path loop ban
against a frozen table, J-02 the LAW-11 zero-`"grid"`-default census, J-03 the killed-knob
census, J-04 the `except …: pass` swallow census against one allowlisted `__del__` site, and
J-05 the frozen `game_complete` key set. Nothing here modifies what it reads.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_SELFPLAY = Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"

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
    """(for, while, comprehension) counts inside `func`, including nested closures — the
    convention the frozen table was measured under."""
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
    """Count `map(...)` / `filter(...)` calls inside `func`, named in the ban alongside
    `for`/`while`/comprehensions."""
    n = 0
    for sub in ast.walk(func):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id in ("map", "filter")):
            n += 1
    return n


# The FROZEN table: (label, [(module_filename, qualname), …], (for, while, comprehension)),
# review-measured old-side counts and NOT authored here. Multi-entry rows assert the SUM over
# the named functions, because counts are conserved under the 4-way pool relocation.
_Q6_TABLE: list[tuple[str, list[tuple[str, str]], tuple[int, int, int]]] = [
    ("collate_graph_batch",
     [("graph_collate.py", "collate_graph_batch")], (0, 0, 0)),
    ("_check_structural",
     [("graph_collate.py", "_check_structural")], (2, 0, 0)),
    # DOWN-RATCHET `(1, 0, 1)` -> `(1, 0, 0)`: the comprehension over a graph's legal nodes is
    # gone, replaced by three linear numpy passes, and the surviving `for` is the O(B) cell-array
    # build rather than an O(Lg) per-legal-node walk. A NO-NEW-LOOPS contract, so lowering it
    # tightens the gate.
    ("_check_semantic",
     [("graph_collate.py", "_check_semantic")], (1, 0, 0)),
    ("segment_softmax / stone_mask_from_batch",
     [("graph_collate.py", "segment_softmax"),
      ("graph_collate.py", "stone_mask_from_batch")], (0, 0, 0)),
    # `run()` was the DENSE loop and is now a one-line delegation, so its `while` moved to
    # `_run_graph_loop`'s row below rather than vanishing.
    ("InferenceServer.run",
     [("inference_server.py", "InferenceServer.run")], (0, 0, 0)),
    # This row's `for` count moved 0 -> 1 under a RULED, per-event grant — cite the queue row
    # for its scope. The ban is on a PER-ITEM Python loop; this `for` iterates the PARTS OF ONE
    # MEMORY-BOUNDED PLAN (M = 1 whenever the caps do not bind), so its count follows the minted
    # cap. Not optional either: the bound is on the PEAK, so parts run one at a time.
    ("InferenceServer._run_graph_loop",
     [("inference_server.py", "InferenceServer._run_graph_loop")], (1, 1, 0)),
    ("InferenceServer.submit_and_wait / load_state_dict_safe",
     [("inference_server.py", "InferenceServer.submit_and_wait"),
      ("inference_server.py", "InferenceServer.load_state_dict_safe")], (0, 0, 0)),
    ("run_stats_loop + pool_push arms (old _run_stats_loop, §a.2 split)",
     [("pool_drain.py", "run_stats_loop"),
      ("pool_push.py", "push_dense"),
      ("pool_push.py", "push_graph")], (3, 1, 2)),
    # DOWN-RATCHET: the four `for`s were the K-cluster dense decode, and `infer_batch` now
    # delegates to `_infer_batch_graph`, whose own row below is unchanged.
    ("LocalInferenceEngine.infer_batch",
     [("inference_local.py", "LocalInferenceEngine.infer_batch")], (0, 0, 0)),
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
    functions equals the FROZEN triple and they contain zero `map`/`filter` calls. FAIL = a new
    per-item Python loop on a hot path, which re-introduces the overhead the port keeps out."""
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
    """J-01 (coverage arm) — every frozen row resolves to real functions in the shipped source,
    so a deleted or renamed hot-path function makes the parametrized rows disappear rather than
    fail."""
    trees = _trees()
    for label, members, _ in _Q6_TABLE:
        for module, qual in members:
            assert module in trees, f"{label}: module {module} missing from selfplay"
            assert _find_function(trees[module], qual) is not None, (
                f"{label}: {module}:{qual} not found"
            )
    assert len(_Q6_TABLE) >= 10, (
        "the frozen §Q6 table has lost rows. AUDIT-1 F-49: this read `== 10` against a\n"
        "literal in this same module, which could only notice someone editing that\n"
        "literal. What it is FOR is that a deleted hot-path function makes its\n"
        "parametrized row vanish rather than fail, so a FLOOR is the shape that catches\n"
        f"the case; rows may be added. Now: {len(_Q6_TABLE)}"
    )


# J-01 mutation self-test (LAW-07): `segment_softmax` is the (0,0,0) row, and a doctored copy
# with an injected loop or comprehension must be detected by the SAME census logic used above.
_SEGMENT_SOFTMAX_SRC = _SELFPLAY / "graph_collate.py"


def _doctored_func(inject: str) -> ast.AST:
    """Return the `segment_softmax` node from a copy of graph_collate.py whose body has had
    one line injected right after the signature."""
    src_lines = _SEGMENT_SOFTMAX_SRC.read_text().splitlines()
    tree = ast.parse("\n".join(src_lines))
    func = _find_function(tree, "segment_softmax")
    assert func is not None
    # Spliced in as the second body element (after the docstring), then re-parsed so line/col
    # offsets stay consistent.
    doctored = ast.parse(inject).body[0]
    func.body.insert(1, doctored)
    ast.fix_missing_locations(func)
    return func


def test_j01_mutation_self_test_injected_for_loop_bites() -> None:
    """J-01 (LAW-07, arm 1) — an injected `for` loop into the (0,0,0) function makes the census
    report a nonzero `for` count, so the frozen-table comparison would FAIL."""
    func = _doctored_func("for _ in range(1):\n    pass\n")
    census = _loop_census(func)
    assert census != (0, 0, 0), "census must bite an injected for-loop"
    assert census[0] >= 1, f"the injected for-loop must be counted (got {census})"


def test_j01_mutation_self_test_injected_comprehension_bites() -> None:
    """J-01 (LAW-07, arm 2) — an injected comprehension makes the census report a nonzero
    comprehension count, closing the hole a name-only loop census left."""
    func = _doctored_func("_evade = [x for x in range(1)]\n")
    census = _loop_census(func)
    assert census != (0, 0, 0), "census must bite an injected comprehension"
    assert census[2] >= 1, f"the injected comprehension must be counted (got {census})"


def test_j01_mutation_self_test_injected_map_bites() -> None:
    """J-01 (LAW-07, arm 3) — an injected `map(...)` call is caught by the map/filter census
    (the M8 extension names map/filter as per-item loops alongside comprehensions)."""
    func = _doctored_func("_evade = list(map(str, range(1)))\n")
    assert _mapfilter_census(func) >= 1, "census must bite an injected map() call"


# J-02 — LAW-11: a `"grid"` default recovered off a spec/getattr is the silently-dense-by-default
# class LAW-11 kills.
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
    `src/mantis/selfplay`. FAIL = a spec/dict read that falls back to grid, silently routing a
    graph run through the dense path."""
    assert _dense_default_hits(_SELFPLAY) == []


def test_j02_census_bites_planted_dense_default(tmp_path: Path) -> None:
    """J-02 (LAW-07) — a planted grid-default line makes the census fire."""
    (tmp_path / "planted.py").write_text(
        'def f(cfg):\n    return cfg.get("representation", "grid")\n'
    )
    assert _dense_default_hits(tmp_path), "census must bite a planted grid-default token"


# J-03 — KILL census. The killed knob names are deliberately DESCRIBED and not spelled in
# comments so this census stays at zero; a real re-introduction bites.
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
    """J-03 — PASS iff none of the KILLed knob names appear in `src/mantis/selfplay`. FAIL = a
    killed knob resurfaced as a read or a set, which is the silently-disabled-opponent /
    dead-config class the register exists to keep dead."""
    assert _killed_hits(_SELFPLAY) == []


def test_j03_census_bites_planted_killed_token(tmp_path: Path) -> None:
    """J-03 (LAW-07) — a planted killed-knob reference makes the census fire."""
    (tmp_path / "planted.py").write_text('LEGAL = cfg["legal_move_radius_jitter"]\n')
    assert _killed_hits(tmp_path), "census must bite a planted killed token"


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
    """J-04 — PASS iff the ONLY `except …: pass` swallow in `src/mantis/selfplay` is the
    sanctioned `__del__` best-effort cleanup in `inference_local.py`. FAIL = a second swallow
    anywhere, since every other blind-except makes a failure vanish silently."""
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


# J-05 — the FROZEN `game_complete` key set, extracted once from the shipped dict literal. A
# dropped, added or renamed key must bite here before it reaches a consumer.
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
})


def _game_complete_dict_keys() -> set[str]:
    """The string keys of the `game_complete` payload dict literal in pool_drain.py, read off the
    SOURCE so a key change in the emitter bites without running the drain loop."""
    tree = ast.parse((_SELFPLAY / "pool_drain.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "event" in keys:
                # The payload is the dict whose "event" key maps to "game_complete".
                for k, v in zip(node.keys, node.values, strict=False):
                    if (isinstance(k, ast.Constant) and k.value == "event"
                            and isinstance(v, ast.Constant) and v.value == "game_complete"):
                        return set(keys)
    raise AssertionError("no game_complete payload dict literal found in pool_drain.py")


def test_j05_game_complete_source_key_set_frozen() -> None:
    """J-05 — PASS iff the emitter's `game_complete` payload literal carries EXACTLY the frozen
    key set. The monitor builds against this schema, so an undeclared key change is a silent
    break of a downstream consumer."""
    assert _game_complete_dict_keys() == set(_FROZEN_GAME_COMPLETE_KEYS)


def test_j05_game_complete_golden_key_set_frozen() -> None:
    """J-05 (capture arm) — the captured old-side events carry the frozen key set MINUS the uuid
    `game_id`, which the golden excludes. Binds the schema to the capture as well as to the
    source, so the two cannot drift apart silently."""
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
