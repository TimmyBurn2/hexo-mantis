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
lives in `tests/arena/test_deploy_head.py`. R351(c) SPLIT THE KEY (`selfplay.search.kind` is
what the workers run, `deploy.search.kind` what the bar plays), so the property is now ONE
READER PER KEY: each wire reads its own resolver and never the other's.
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
    resolve_deploy_search_kind,
    resolve_selfplay_search_kind,
)
from mantis.config.schema import RunConfig
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


def test_each_wire_calls_its_own_resolver_and_not_the_other():
    """The self-play wire resolves `selfplay.search.kind`; the eval wire `deploy.search.kind`.

    MUTATION THAT REDS IT: either side reading `config[...]["kind"]` directly again, or the
    self-play wire reaching for the DEPLOY key (the coupling R351(c) undid).
    """
    selfplay_calls = _calls(_SRC / "selfplay" / "hparams.py")
    assert "resolve_selfplay_search_kind" in selfplay_calls
    assert "resolve_deploy_search_kind" not in selfplay_calls, (
        "the self-play wire reads the DEPLOY kind — the workers would search the way the "
        "ladder plays rather than the way the run trains"
    )
    run_calls = _calls(_SRC / "run.py")
    assert "resolve_deploy_search_kind" in run_calls, (
        "run.py does not call `resolve_deploy_search_kind` — the eval pipeline's kind is "
        "being read some other way, and a second reader is a second authority (R1)"
    )
    assert "resolve_search_kind" not in selfplay_calls | run_calls, (
        "the pre-split single resolver is still called — one key would then feed both wires"
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
def test_the_selfplay_resolver_is_what_the_workers_are_configured_with(kind, smoke_run_config):
    """The value the SELF-PLAY resolver returns is the value `SelfPlayHParams` carries to the
    runner — and the deploy key, set to the OTHER kind, does not leak into it.

    Driven off a REAL minted config so the section shape is the one CI gate 7 validates; the
    kind is injected into the DUMPED mapping (the graph record format refuses `gumbel` at
    mint), which is the right seam anyway: this is the resolver-to-hparams wire.
    """
    other = next(k for k in SEARCH_KINDS if k != kind)
    cfg = smoke_run_config("dev_example.yaml").model_dump()
    cfg["selfplay"]["search"]["kind"] = kind
    cfg["deploy"]["search"]["kind"] = other
    cfg["encoding"] = cfg["identity"]["encoding"]
    assert resolve_selfplay_search_kind(cfg) == kind
    assert resolve_deploy_search_kind(cfg) == other
    assert SelfPlayHParams.from_config(cfg).search_kind == kind


def test_the_two_kinds_may_differ_by_construction(smoke_run_config):
    """A Gumbel self-play with a PUCT deploy head VALIDATES (R351(c)'s run7 shape): the
    policy target follows the SELF-PLAY kind, and the deploy kind constrains nothing there."""
    cfg = smoke_run_config("dev_example.yaml").model_dump()
    cfg["selfplay"]["search"]["kind"] = "puct"
    cfg["deploy"]["search"]["kind"] = "gumbel"
    cfg["train"]["policy_target"] = "raw_visit_distribution"
    run = RunConfig.model_validate(cfg)
    assert (run.selfplay.search.kind, run.deploy.search.kind) == ("puct", "gumbel")
    cfg["train"]["policy_target"] = "completed_improved_policy"
    with pytest.raises(ValueError, match="selfplay.search.kind"):
        RunConfig.model_validate(cfg)


def test_a_top_level_search_section_is_refused(smoke_run_config):
    """The pre-split key does not survive as a silent third authority (`extra="forbid"`)."""
    cfg = smoke_run_config("dev_example.yaml").model_dump()
    cfg["search"] = {"kind": "puct"}
    with pytest.raises(ValueError, match="search"):
        RunConfig.model_validate(cfg)


@pytest.mark.parametrize("section", ["selfplay", "deploy"])
def test_each_kind_is_required_with_no_default(section, smoke_run_config):
    cfg = smoke_run_config("dev_example.yaml").model_dump()
    del cfg[section]["search"]
    with pytest.raises(ValueError) as excinfo:
        RunConfig.model_validate(cfg)
    assert f"{section}.search" in str(excinfo.value).replace("\n", " ") or "search" in str(excinfo.value)


def test_an_absent_or_unknown_kind_is_refused_by_either_selector():
    """No fallback, on either shape of input, for either key (R1/LAW-11)."""
    for resolve, section in ((resolve_selfplay_search_kind, "selfplay"),
                             (resolve_deploy_search_kind, "deploy")):
        with pytest.raises(MissingSearchKindError):
            resolve({})
        with pytest.raises(MissingSearchKindError):
            resolve({section: {"search": {}}})
        with pytest.raises(MissingSearchKindError):
            resolve({section: {"search": {"kind": "mctx"}}})
        # the OTHER section's key is not this resolver's authority
        other = "deploy" if section == "selfplay" else "selfplay"
        with pytest.raises(MissingSearchKindError):
            resolve({other: {"search": {"kind": "puct"}}})
