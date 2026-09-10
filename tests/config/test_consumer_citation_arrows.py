""">300 justify (R8): ONE rule and the evidence that it is the RIGHT rule. The discriminator below
was chosen over a simpler one only because the simpler one was measured and found to be a
false-positive generator, and the tests that record that measurement are what stop the rule being
"simplified" back into the version that does not work.

Citation arrows are checked BY SYMBOL REFERENCE, never by prose. All four `monitor.supervisor_*`
keys were cited as `"resolve_monitor_config -> monitor/supervise.py <flag>"`, but
`resolve_monitor_config` runs in the RUN process while `supervise.py::main` runs in the SUPERVISOR
process and built its own bare `MonitorConfig()`. The arrow crossed a process boundary no code
crossed, and every existing check passed, because the registry is verified only for BIJECTION
against schema leaves — so the citation STRINGS were unverified prose.

THE DISCRIMINATOR. The naive check — the cited file must reference the cited symbol — was measured
first: it flags 17 of the 21 citations that name a file, of which 4 are the real defect and 13 are
TRUTHFUL arrows. A multi-hop data-flow arrow is the normal shape here, so a check that treats every
hop as an import edge is a false-positive generator, and a check whose failures are usually wrong
teaches its own suppression.

What separates the defect is a derivable fact about PROCESSES — an object cannot be passed into a
program that runs as its OWN process — so the rule is narrow and exact:

    if a citation names a file that is ITSELF a process entry point, that file must reference
    the symbols the citation says deliver the value to it.

Measured over the live registry, exactly ONE cited file is its own entry point, so the check fires
on the class and on nothing else. Symbols are DERIVED, never guessed: a token counts only if it is
a real top-level `def`/`class` name under `src/mantis`, indexed from the tree at run time.
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
    """Resolve a cited path by SUFFIX match against the tree — the citations write `trainer/core.py` for `src/mantis/train/trainer/core.py`, so no root may be assumed."""
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


def _unresolvable_citations(registry: dict[str, str]) -> list[str]:
    """Cited paths that match no file, or more than one."""
    bad: list[str] = []
    for key, citation in sorted(registry.items()):
        for rel in _FILE_RE.findall(citation):
            hits = [k for k in _files_by_suffix() if k == rel or k.endswith("/" + rel)]
            if len(hits) != 1:
                bad.append(f"{key}: cited path {rel!r} resolves to {len(hits)} files: {hits}")
    return bad


@lru_cache(maxsize=None)
def _referenced_names(path: Path) -> frozenset[str]:
    """Names this module actually REFERENCES in code — imports, calls, attributes, plain loads."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        # LOAD CONTEXT ONLY. A binding is not a reference: `resolve_drain_caps = None` is a STORE,
        # and counting it verified an arrow into a module that receives nothing while staying
        # ruff-clean. An unused import is likewise not a delivery, so only a name the module
        # actually READS survives this filter.
        #
        # THE LIMIT, STATED RATHER THAN LEFT TO BE FOUND: a Load inside code that never executes is
        # still a Load, so it counts. That residual is accepted — reachability is a different
        # instrument, and the two shapes that reach it are already caught by ruff F821/F401.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            names.add(node.attr)
    return frozenset(names)


def _arrow_violations(registry: dict[str, str]) -> list[str]:
    """Every (key, file, missing symbols) where an entry-point file cannot see its own arrow."""
    bad: list[str] = []
    for key, citation in sorted(registry.items()):
        for rel in _FILE_RE.findall(citation):
            path = _resolve_cited_file(rel)
            if path is None or not _is_process_entry_point(path):
                continue
            cited = _cited_symbols(citation, rel)
            if not cited:
                # A CITATION WITH NOTHING VERIFIABLE IN IT IS NOT A PASS: naming an entry point as
                # a destination while naming no symbol that carries the value there asserts a
                # delivery and offers no way to check it, so it used to pass VACUOUSLY.
                bad.append(
                    f"{key}: {rel} is its own process and the citation names no symbol at all, "
                    "so the arrow asserts a delivery nothing can verify"
                )
                continue
            referenced = _referenced_names(path)
            missing = [s for s in cited if s not in referenced]
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


def test_every_cited_path_resolves_to_exactly_one_file():
    """The checker's own precondition, asserted rather than assumed."""
    for name in ("test_every_key_has_consumer.py", "test_every_key_has_consumer_p2.py"):
        unresolvable = _unresolvable_citations(_load_registry(name))
        assert not unresolvable, (
            f"{name}: a consumer citation names a path the tree cannot resolve uniquely, so the "
            "arrow rule silently skipped it:\n  " + "\n  ".join(unresolvable)
        )


def test_the_resolution_check_bites_on_a_path_the_tree_cannot_resolve():
    """LAW-07 self-test for the precondition above."""
    planted = {"planted.key": "resolve_monitor_config -> monitor/no_such_module.py something"}
    assert _unresolvable_citations(planted), (
        "a cited path matching no file was treated as resolved; the precondition test above is "
        "inert and an unchecked citation would read as a checked one"
    )


def test_the_arrow_check_bites_on_a_planted_false_arrow():
    """LAW-07 mutation self-test: a checker that cannot fail is a phantom gate."""
    planted = {
        "monitor.drain.terminal_eval_hard_cap_sec":
            "resolve_drain_caps -> monitor/supervise.py terminal cap",
    }
    assert _arrow_violations(planted), (
        "a planted false arrow into an entry point was NOT detected — either supervise.py "
        "stopped being a process entry point, or the symbol index stopped resolving "
        "resolve_drain_caps, and in either case the check above has stopped meaning anything"
    )


def test_a_symbol_mentioned_only_in_PROSE_is_not_accepted_as_a_reference():
    """RED-TEAM #2's attack, kept as a regression."""
    planted = {
        "planted.prose_only": "force_teardown_all -> monitor/supervise.py stop ladder",
    }
    assert _arrow_violations(planted), (
        "a symbol appearing only in a docstring was accepted as evidence of delivery; the check "
        "has gone back to reading prose, which is the defect R291(c) ordered fixed"
    )
    # ...and the mention really is prose-only, so the test is testing what it says it is.
    source = (SRC / "monitor" / "supervise.py").read_text(encoding="utf-8")
    assert "force_teardown_all" in source, "the premise moved; pick another prose-only symbol"
    assert "force_teardown_all" not in _referenced_names(SRC / "monitor" / "supervise.py")


def test_a_bare_BINDING_is_not_accepted_as_a_reference(tmp_path):
    """RED-TEAM's bypass of the AST rule, kept as a regression."""
    mutated = tmp_path / "supervise.py"
    mutated.write_text(
        (SRC / "monitor" / "supervise.py").read_text(encoding="utf-8")
        + "\nresolve_drain_caps = None  # a binding, not a delivery\n",
        encoding="utf-8",
    )
    assert "resolve_drain_caps" not in _referenced_names(mutated), (
        "a name bound but never read counts as a reference again; the checker is back to "
        "accepting the appearance of a symbol as evidence that a value arrives"
    )


def test_a_citation_naming_NO_symbol_does_not_pass_vacuously():
    """RED-TEAM #2's second half: an assertion with nothing in it to check is not a passing one."""
    planted = {"planted.no_symbol": "monitor/supervise.py somehow delivers this via magic"}
    assert _arrow_violations(planted), (
        "a citation naming an entry point and no symbol was treated as verified; an arrow that "
        "cannot be checked must be reported, not counted as clean"
    )


def test_the_check_does_not_fire_on_in_process_multi_hop_arrows():
    """The negative control, and the reason the discriminator is process-shaped."""
    in_process = {
        "monitor.alert_entropy_min": "resolve_monitor_config -> monitor/rules.py entropy WARN",
    }
    assert not _arrow_violations(in_process)


def test_exactly_the_expected_cited_files_are_process_entry_points():
    """Derived, not asserted: the rule's REACH is measured at HEAD rather than transcribed."""
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
