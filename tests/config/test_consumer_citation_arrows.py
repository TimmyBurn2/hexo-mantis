"""LAW-08 citation arrows are checked BY SYMBOL REFERENCE, never by prose (R291(c), R244).

WHAT WENT WRONG, AND WHY NOTHING CAUGHT IT. All four `monitor.supervisor_*` keys were cited as
`"resolve_monitor_config -> monitor/supervise.py <flag>"`. `resolve_monitor_config` runs in the
RUN process; `supervise.py::main` runs in the SUPERVISOR process and built its own bare
`MonitorConfig()`. The arrow crossed a process boundary no code crossed. Every existing check
passed: the registry is verified only for BIJECTION against schema leaves — a key-SET diff — so
the citation STRINGS are unverified prose and a false arrow stays green forever. That is
F-816-24's second evidence leg, and it is the SECOND instance of this class in the same file
(the `drain` block was the first, WPMINT Phase K-A / R93).

THE DISCRIMINATOR, AND WHY IT IS THIS ONE. The naive check — "the cited file must reference the
cited symbol" — was tried first and MEASURED: it flags 17 of the 21 citations that name a file,
and 17 of those are TRUTHFUL. `resolve_monitor_config -> monitor/rules.py` is correct precisely
because rules.py does NOT import the resolver: the run process builds a `MonitorConfig` and
PASSES it in. A multi-hop data-flow arrow is the normal shape here, so a check that treats every
hop as an import edge is a false-positive generator, and a check whose failures are usually
wrong teaches its own suppression.

What separates the defect from the 17 is a fact about PROCESSES, and it is derivable: an object
cannot be passed into a program that runs as its OWN process. So the rule is narrow and exact:

    if a citation names a file that is ITSELF a process entry point, that file must reference
    the symbols the citation says deliver the value to it.

Measured over the live registry: exactly ONE cited file is its own entry point — `supervise.py`,
the defect — and the other eight are in-process consumers the rule never touches. The check
therefore fires on the class and on nothing else, which is the difference between a gate and a
nuisance.

SYMBOLS ARE DERIVED, NEVER GUESSED (R244). A token in a citation counts as a symbol reference
only if it is a real top-level `def`/`class` name somewhere under `src/mantis` — the index is
built from the tree at run time, so English prose in a citation cannot be mistaken for a symbol
and a renamed symbol stops being one on the same commit that renames it.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mantis"

_FILE_RE = re.compile(r"((?:[a-z_][a-z0-9_]*/)+[a-z_][a-z0-9_]*\.py)")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _load_registry(name: str) -> dict[str, str]:
    """Both copies are loaded by path — they are deliberate duplicates of one another."""
    path = REPO_ROOT / "tests" / "config" / name
    spec = importlib.util.spec_from_file_location(f"_reg_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.CONSUMER_REGISTRY)


@lru_cache(maxsize=1)
def _symbol_index() -> frozenset[str]:
    """Every top-level `def`/`class` name under `src/mantis`, derived from the tree."""
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    return frozenset(names)


@lru_cache(maxsize=1)
def _files_by_suffix() -> dict[str, Path]:
    return {path.relative_to(SRC).as_posix(): path for path in SRC.rglob("*.py")}


def _resolve_cited_file(rel: str) -> Path | None:
    """Resolve a cited path by SUFFIX match against the tree — the citations write
    `trainer/core.py` for `src/mantis/train/trainer/core.py`, so no root may be assumed."""
    hits = [p for key, p in _files_by_suffix().items() if key == rel or key.endswith("/" + rel)]
    return hits[0] if len(hits) == 1 else None


def _is_process_entry_point(path: Path) -> bool:
    """A file with an `if __name__ == "__main__"` guard runs as its own process."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"):
            return True
    return False


def _cited_symbols(citation: str, cited_file: str) -> list[str]:
    remainder = citation.replace(cited_file, " ")
    return sorted({tok for tok in _TOKEN_RE.findall(remainder) if tok in _symbol_index()})


def _arrow_violations(registry: dict[str, str]) -> list[str]:
    """Every (key, file, missing symbols) where an entry-point file cannot see its own arrow."""
    bad: list[str] = []
    for key, citation in sorted(registry.items()):
        for rel in _FILE_RE.findall(citation):
            path = _resolve_cited_file(rel)
            if path is None or not _is_process_entry_point(path):
                continue
            source = path.read_text(encoding="utf-8")
            missing = [s for s in _cited_symbols(citation, rel) if s not in source]
            if missing:
                bad.append(f"{key}: {rel} is its own process and references none of {missing}")
    return bad


def test_no_citation_sends_a_value_into_a_separate_process_that_cannot_see_it():
    """The class fix. Both registry copies, one rule, derived from the tree."""
    for name in ("test_every_key_has_consumer.py", "test_every_key_has_consumer_p2.py"):
        violations = _arrow_violations(_load_registry(name))
        assert not violations, (
            f"{name}: a consumer citation names a symbol that its cited ENTRY-POINT file never "
            "references, so the value cannot reach it — the F-816-24 class:\n  "
            + "\n  ".join(violations)
        )


def test_the_arrow_check_bites_on_a_planted_false_arrow():
    """LAW-07 mutation self-test: a checker that cannot fail is a phantom gate.

    THE HISTORICAL ARROW CANNOT BE USED AS THE PLANTED CASE ANY MORE, AND THAT IS THE POINT.
    The obvious mutation is the defect verbatim — `resolve_monitor_config -> monitor/supervise.py`
    — but F-816-24's fix makes `supervise.py` reference `resolve_monitor_config`, so that arrow is
    now TRUE and the check correctly declines to flag it. Written down because it is a small proof
    the fix landed: the citation that was false is false no longer, measured by the same rule that
    would have caught it.

    So the planted arrow names a symbol this entry point genuinely does not touch. It stays false
    however the supervisor evolves, unless someone wires the drain resolver into it.
    """
    planted = {
        "monitor.drain.terminal_eval_hard_cap_sec":
            "resolve_drain_caps -> monitor/supervise.py terminal cap",
    }
    assert _arrow_violations(planted), (
        "a planted false arrow into an entry point was NOT detected — either supervise.py "
        "stopped being a process entry point, or the symbol index stopped resolving "
        "resolve_drain_caps, and in either case the check above has stopped meaning anything"
    )


def test_the_check_does_not_fire_on_in_process_multi_hop_arrows():
    """The negative control, and the reason the discriminator is process-shaped.

    `resolve_monitor_config -> monitor/rules.py` is TRUE and rules.py imports no resolver: the
    run process builds the object and passes it in. A check that flagged this would be wrong 17
    times out of 21 on the live registry — measured — and would be suppressed within a week.
    """
    in_process = {
        "monitor.alert_entropy_min": "resolve_monitor_config -> monitor/rules.py entropy WARN",
    }
    assert not _arrow_violations(in_process)


def test_exactly_the_expected_cited_files_are_process_entry_points():
    """Derived, not asserted: the rule's REACH is measured at HEAD rather than transcribed.

    This is the line that would go stale if it stated a count, so it states a SET and derives it
    (R192(e)). If a future citation names a new entry point, this test names it and the rule
    starts applying to it — which is the intended behaviour, not a regression.
    """
    cited: dict[str, bool] = {}
    for name in ("test_every_key_has_consumer.py", "test_every_key_has_consumer_p2.py"):
        for citation in _load_registry(name).values():
            for rel in _FILE_RE.findall(citation):
                path = _resolve_cited_file(rel)
                if path is not None:
                    cited[rel] = _is_process_entry_point(path)
    entry_points = {rel for rel, is_entry in cited.items() if is_entry}
    assert entry_points == {"monitor/supervise.py"}, (
        "the set of cited files that are their own process changed; the arrow rule now reaches "
        f"{sorted(entry_points)} and each newcomer needs its arrow checked by hand once"
    )
