"""⊕ the deploy head and the self-play workers read ONE selector, by construction.

WHAT LAW-15 ACTUALLY CLAIMS. "Deploy-matched eval is the DEFAULT promotion bar" is a claim
about the SEARCH: the head that decides a promotion plays the game the run's workers played.
Before `search.kind` the eval head's regime came from `DeployHeadPlayer`'s own body — a PUCT
tree with a Gumbel-scored root pick, an algorithm that appeared in no config at all — so the
claim was not false so much as unstatable. The key alone does not fix that: two call sites
each reading `config["search"]["kind"]` would agree today and drift the first time one of
them grew a fallback.

SO THE PROPERTY UNDER TEST IS "ONE READER", NOT "TWO EQUAL READS", and it is checked over the
composition root's own source. The behavioural half — that the kind reaches the head's TREE —
lives in `tests/arena/test_deploy_head.py`.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.config.resolve.search import (
    SEARCH_KINDS,
    MissingSearchKindError,
    resolve_search_kind,
)
from mantis.selfplay.hparams import SelfPlayHParams

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"


def _calls(path: Path) -> set[str]:
    """Every function NAME called anywhere in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_both_sides_call_the_same_resolver():
    """The self-play wire and the eval wire each call `resolve_search_kind`.

    MUTATION THAT REDS IT: either side reading `config["search"]["kind"]` directly again.
    That read is not wrong on its own — it is wrong as a SECOND authority, which is exactly
    what this cannot see the difference between and so forbids outright.
    """
    for module in ("selfplay/hparams.py", "run.py"):
        assert "resolve_search_kind" in _calls(_SRC / module), (
            f"{module} does not call `resolve_search_kind` — the search regime it hands "
            f"downstream is being read some other way, and a second reader is a second "
            f"authority (R1)"
        )


def test_the_deploy_head_reads_no_config_at_all():
    """The head TAKES the kind; it does not go and find one.

    A head that could read a config could read a different one from the workers', or fall
    back when the key was absent. It has no such reach: `search_kind` is a required
    constructor argument and `deploy_head.py` imports nothing from `mantis.config`.
    """
    tree = ast.parse((_SRC / "arena" / "deploy_head.py").read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    # The MODULE graph, not the prose: the docstring names the resolver on purpose, and a
    # text search would read that as a reach.
    assert not any(m.startswith("mantis.config") for m in imported), (
        f"the deploy head imports from `mantis.config` ({sorted(imported)}) — it must be "
        f"HANDED its search kind, not go looking for one"
    )
    sig = inspect.signature(DeployHeadPlayer.__init__)
    assert sig.parameters["search_kind"].default is inspect.Parameter.empty


@pytest.mark.parametrize("kind", SEARCH_KINDS)
def test_the_resolver_is_what_the_workers_are_configured_with(kind, smoke_run_config):
    """The value the resolver returns is the value `SelfPlayHParams` carries to the runner.

    Driven off a REAL minted config so the section shape is the one CI gate 7 validates.
    `dev_example.yaml` is the graph lineage, whose record format refuses `gumbel` — so the
    kind is injected into the DUMPED mapping rather than through `RunConfig`, which is the
    right seam for this test anyway: what is being checked is the resolver-to-hparams wire,
    not the mint rules that sit above it.
    """
    cfg = smoke_run_config("dev_example.yaml").model_dump()
    cfg["search"]["kind"] = kind
    cfg["encoding"] = cfg["identity"]["encoding"]
    assert resolve_search_kind(cfg) == kind
    assert SelfPlayHParams.from_config(cfg).search_kind == kind


def test_an_absent_or_unknown_kind_is_refused_by_the_selector():
    """No fallback, on either shape of input (R1/LAW-11)."""
    with pytest.raises(MissingSearchKindError):
        resolve_search_kind({})
    with pytest.raises(MissingSearchKindError):
        resolve_search_kind({"search": {}})
    with pytest.raises(MissingSearchKindError):
        resolve_search_kind({"search": {"kind": "mctx"}})
