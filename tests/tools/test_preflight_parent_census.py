"""The O-2/O-3 census EXTENSION over the preflight tool's parent half (WPBOX Phase Q).

CARD-PREFLIGHT-SPLIT-PARENT-HALF's oracle-compatibility debt, named in GROUND_PFC §2.2:
the byte-frozen oracle's banned-token census (O-2) and sys.path ban (O-3) sweep ONLY
`preflight_mint.py` plus the four Phase-P test files, so every line that moved to
`preflight_mint_parent.py` silently left the sweep. This file is the "census consciously
extended" arm — the same bans, applied to the sibling, in a NON-frozen test so the split's
own discipline never needs an R43 event to tighten.

Also pinned here, because the split's one-authority constraints are census-shaped:
the sibling never imports the tool (no cycle), never spec-loads anything (the tool is the
only loader), and defines NO `MANIFEST`/`PRODUCTION_CONFIGS`/`EXEMPT_CONFIGS` global —
the audit read path must stay the tool module's, where the frozen ring-2 monkeypatch seam
(`TOOL.MANIFEST = bad -> rc 31`) lives. And the re-export block is drift-pinned: every
top-level name the sibling defines is bound on the tool module path AS THE SAME OBJECT,
so a name added to the sibling without a re-export line goes red here, not in production.
"""
from __future__ import annotations

import ast
import importlib.util
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
PARENT_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint_parent.py"

#: O-2's exact banned set, copied in VALUE from the frozen oracle (tests/tools/
#: test_preflight_mint.py O-2) — copied, not imported, so this file cannot couple to the
#: frozen file's internals; the frozen census is the authority and this is its extension.
BANNED_TOKENS = ("monkeypatch", "unittest.mock", "SimpleNamespace", "MagicMock",
                 "mock.patch", "setattr(", "pytest")


def _code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed — the frozen
    oracle's own census tokenizer, restated with the 3.11-floor guard idiom
    (FSTRING_MIDDLE is 3.12+; on 3.11 f-strings lex as STRING)."""
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(
            tok.string for tok in tokenize.tokenize(handle.readline) if tok.type not in skip
        )


PARENT_SOURCE = PARENT_PATH.read_text()
PARENT_TREE = ast.parse(PARENT_SOURCE)
PARENT_CODE = _code_text(PARENT_PATH)


def test_the_parent_half_carries_no_banned_test_vocabulary() -> None:
    """O-2, extended: the sibling is production surface exactly like the tool."""
    hits = [token for token in BANNED_TOKENS if token in PARENT_CODE]
    assert not hits, (
        f"banned token(s) {hits} in preflight_mint_parent.py code text — the O-2 census "
        "extension; the frozen sweep does not see this file, so this test is where the "
        "discipline lives"
    )


def test_the_parent_half_never_touches_syspath_or_pythonpath() -> None:
    """O-3, extended — the frozen arm's own discipline: mutation tokens over the
    comment-stripped code text, space-collapsed (prose may NAME sys.path; code may not
    touch it)."""
    code = PARENT_CODE.replace(" ", "")
    for token in ("sys.path.append", "sys.path.insert", "sys.path.extend",
                  "sys.path=", "sys.path+=", "PYTHONPATH"):
        assert token not in code, (
            f"preflight_mint_parent.py mutates sys.path ({token!r}) — R5 / LAW-17 admit "
            "ZERO exceptions"
        )


def test_the_parent_half_never_imports_the_tool_and_never_loads_modules() -> None:
    """No cycle and no second loader: the tool loads the sibling, never the reverse."""
    for node in ast.walk(PARENT_TREE):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "preflight_mint" not in name, (
                f"the parent half imports {name!r} — a sibling that imports the tool is a "
                "cycle, and the module-path authority stops being the tool"
            )
    assert "spec_from_file_location" not in PARENT_CODE, (
        "the sibling must not load modules by path — the tool is the ONE loader of the pair"
    )


def test_the_parent_half_defines_no_manifest_global() -> None:
    """The frozen ring-2 monkeypatch seam requires the audit read path to see the TOOL
    module's `MANIFEST` at call time; a sibling-side global would be a second read path."""
    forbidden = {"MANIFEST", "PRODUCTION_CONFIGS", "EXEMPT_CONFIGS"}
    for node in PARENT_TREE.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.AnnAssign,)) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            targets = [alias.asname or alias.name for alias in node.names]
        bad = forbidden & set(targets)
        assert not bad, f"the parent half defines {sorted(bad)} — a second audit read path"


def _load_tool():
    spec = importlib.util.spec_from_file_location("preflight_mint_census", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sibling_top_level_names() -> list[str]:
    names: list[str] = []
    for node in PARENT_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [name for name in names if not name.startswith("__")]


def test_two_trees_in_one_process_each_get_their_own_sibling(tmp_path) -> None:
    """The shim's path-keyed sys.modules guard, given a real producer (LAW-07).

    The byte-copy rig loads a COPIED tool from a scratch tree; if the loader reused the
    real tree's cached sibling (the `getattr(cached, "__file__", …) == str(path)` conjunct
    forced True), every rig perturbation of the copied sibling would silently test the
    WRONG file. Drive both loads in ONE process, in both orders, and pin the binding.
    """
    import shutil

    real = _load_tool()
    scratch = tmp_path / "tools" / "ci_gates"
    scratch.mkdir(parents=True)
    shutil.copy2(TOOL_PATH, scratch / "preflight_mint.py")
    shutil.copy2(PARENT_PATH, scratch / "preflight_mint_parent.py")
    spec = importlib.util.spec_from_file_location("preflight_mint_census_copy",
                                                  scratch / "preflight_mint.py")
    assert spec is not None and spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy._parent_half.__file__ == str(scratch / "preflight_mint_parent.py"), (
        "the copied tool must load the COPIED sibling, never the real tree's cached one"
    )
    real_again = _load_tool()
    assert real_again._parent_half.__file__ == str(PARENT_PATH.resolve()), (
        "…and loading the real tool afterwards must re-bind the REAL sibling — the guard "
        "keys on the resolved path, both directions"
    )


def test_every_sibling_name_is_re_exported_as_the_same_object() -> None:
    """The drift pin: the tool's re-export block must list EVERY name the sibling defines,
    and bind the identical object, so `TOOL.<name>` stays the one authority the oracles
    load. A sibling name added without a re-export line fails HERE, at the census, rather
    than surfacing as a confusing AttributeError inside some later consumer."""
    tool = _load_tool()
    sibling = tool._parent_half
    names = _sibling_top_level_names()
    assert names, "the census found no sibling names — the parser above went blind"
    missing = [name for name in names if not hasattr(tool, name)]
    assert not missing, f"sibling names with NO re-export on the tool module: {missing}"
    diverged = [name for name in names if getattr(tool, name) is not getattr(sibling, name)]
    assert not diverged, (
        f"re-exported name(s) not bound to the sibling's own object: {diverged} — two "
        "objects for one name is the split's one-authority breach"
    )
