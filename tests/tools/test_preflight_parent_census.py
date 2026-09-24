"""The banned-token and sys.path censuses, extended over the preflight tool's parent half.

The byte-frozen oracle's censuses sweep only `preflight_mint.py` and its Phase-P test files,
so every line that moved to `preflight_mint_parent.py` left the sweep. This is the same bans
applied to the sibling, in a non-frozen test.

The split's one-authority constraints are pinned here too: the sibling never imports the tool
(no cycle), never spec-loads anything (the tool is the only loader), and defines no
`MANIFEST`/`EXEMPT_CONFIGS`/census global, so the audit read path stays the
tool module's. The re-export block is drift-pinned: every top-level name the sibling defines
is bound on the tool module as the same object.
"""
from __future__ import annotations

import ast
from pathlib import Path
from _toolpath import load_module_by_path
from _code_text import code_text

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
PARENT_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint_parent.py"

#: The banned set, copied by VALUE from the frozen oracle so this file cannot couple to its
#: internals; the frozen census is the authority and this is its extension.
BANNED_TOKENS = ("monkeypatch", "unittest.mock", "SimpleNamespace", "MagicMock",
                 "mock.patch", "setattr(", "pytest")


PARENT_SOURCE = PARENT_PATH.read_text(encoding="utf-8")
PARENT_TREE = ast.parse(PARENT_SOURCE)
PARENT_CODE = code_text(PARENT_PATH)


def test_the_parent_half_carries_no_banned_test_vocabulary() -> None:
    """Prove the parent half carries no banned test vocabulary; it is production surface."""
    hits = [token for token in BANNED_TOKENS if token in PARENT_CODE]
    assert not hits, (
        f"banned token(s) {hits} in preflight_mint_parent.py code text — the O-2 census "
        "extension; the frozen sweep does not see this file, so this test is where the "
        "discipline lives"
    )


def test_the_parent_half_never_touches_syspath_or_pythonpath() -> None:
    """Prove the parent half never mutates sys.path or PYTHONPATH; prose may name it, code may not."""
    code = PARENT_CODE.replace(" ", "")
    for token in ("sys.path.append", "sys.path.insert", "sys.path.extend",
                  "sys.path=", "sys.path+=", "PYTHONPATH"):
        assert token not in code, (
            f"preflight_mint_parent.py mutates sys.path ({token!r}) — R5 / LAW-17 admit "
            "ZERO exceptions"
        )


def test_the_parent_half_never_imports_the_tool_and_never_loads_modules() -> None:
    """Prove there is no cycle and no second loader: the tool loads the sibling, never the reverse."""
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
    """Prove the parent half defines no manifest global; a sibling-side one is a second read path."""
    forbidden = {"MANIFEST", "EXEMPT_CONFIGS", "production_configs"}
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
    return load_module_by_path("preflight_mint_census", TOOL_PATH)


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
    """Prove the loader's path-keyed guard binds each tree to its own sibling, in both orders.

    Without it, a copied tool would reuse the real tree's cached sibling and every rig
    perturbation would silently test the wrong file.
    """
    import shutil

    real = _load_tool()
    scratch = tmp_path / "tools" / "ci_gates"
    scratch.mkdir(parents=True)
    shutil.copy2(TOOL_PATH, scratch / "preflight_mint.py")
    shutil.copy2(PARENT_PATH, scratch / "preflight_mint_parent.py")
    copy = load_module_by_path("preflight_mint_census_copy", scratch / "preflight_mint.py")
    assert copy._parent_half.__file__ == str(scratch / "preflight_mint_parent.py"), (
        "the copied tool must load the COPIED sibling, never the real tree's cached one"
    )
    real_again = _load_tool()
    assert real_again._parent_half.__file__ == str(PARENT_PATH.resolve()), (
        "…and loading the real tool afterwards must re-bind the REAL sibling — the guard "
        "keys on the resolved path, both directions"
    )


def test_every_sibling_name_is_re_exported_as_the_same_object() -> None:
    """Prove every sibling name is re-exported on the tool module and bound to the same object.

    A sibling name added without a re-export line fails here rather than as a later
    AttributeError inside a consumer.
    """
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
