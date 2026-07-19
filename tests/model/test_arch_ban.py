"""O3 — arch-off-module ban (grep-gate census + mutation self-test).

repo_design §3: reading arch attributes off a live `nn.Module` is banned — arch
travels on the declared dataclasses; the old `model_representation` isinstance sniff
is DELETED and stays deleted. The census scans the model construction-authority
layer (`src/mantis/model/`) for the broadened sniff pattern set (N2/N3) and proves
it bites via a planted-mutation self-test (LAW-07).

Scope note: the attribute-sniff patterns are checked over `src/mantis/model/` (where
build_net + the adapter live and where a sniff would be re-introduced); the deleted
`def model_representation` is checked repo-wide. The pre-existing encoding-layer
`getattr(spec, "representation", "grid")` (WP7 `resolvers.py`) is a SPEC read, not an
nn.Module sniff, and is out of WP9 scope.
"""
from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"
_MODEL = _SRC / "model"

_ARCH_ATTRS = ("in_channels", "filters", "out_features", "board_size")
_ALLOWED_RECEIVERS = {"self", "arch", "spec"}

# (a) isinstance recovering representation from a net class.
_RE_ISINSTANCE = re.compile(r"isinstance\s*\([^)]*\b(GnnNet|HexTacToeNet)\b")
# (b) type(...).__name__ == "GnnNet"|"HexTacToeNet".
_RE_TYPENAME = re.compile(r"type\s*\([^)]*\)\.__name__\s*==\s*[\"'](GnnNet|HexTacToeNet)[\"']")
# (c) attribute read of an arch hyperparam off a live module (receiver not self/arch/spec).
_RE_ATTR = re.compile(r"\b(\w+)\.(in_channels|filters|out_features|board_size)\b")
# (d) hasattr(m, <arch-attr>) probing a module for arch.
_RE_HASATTR = re.compile(
    r"hasattr\s*\([^,]+,\s*[\"'](in_channels|filters|out_features|board_size|representation|"
    r"node_feat_dim|edge_feat_dim|value_head_type)[\"']\s*\)"
)
# (e) dense-default "grid" token (the removed build_net.py:215 / :105 sites).
_RE_DENSE_DEFAULT = re.compile(
    r"getattr\([^)]*,\s*[\"']grid[\"']\s*\)"
    r"|\.get\(\s*[\"']representation[\"']\s*,\s*[\"']grid[\"']"
    r"|,\s*[\"']grid[\"']\s*\)"
)


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py"))


def find_arch_sniffs(root: Path) -> list[str]:
    """Return a list of `file:line: reason` violations for arch-off-module sniffs."""
    violations: list[str] = []
    for path in _py_files(root):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            here = f"{path}:{i}"
            if _RE_ISINSTANCE.search(line):
                violations.append(f"{here}: isinstance on a net class")
            if _RE_TYPENAME.search(line):
                violations.append(f"{here}: type().__name__ net-class compare")
            if _RE_HASATTR.search(line):
                violations.append(f"{here}: hasattr arch-attr probe")
            if _RE_DENSE_DEFAULT.search(line):
                violations.append(f"{here}: dense-by-default 'grid' token")
            for m in _RE_ATTR.finditer(line):
                if m.group(1) not in _ALLOWED_RECEIVERS:
                    violations.append(f"{here}: arch attr read off {m.group(1)!r}.{m.group(2)}")
    return violations


def _has_model_representation_def(root: Path) -> bool:
    return any(
        re.search(r"^\s*def\s+model_representation\b", line, re.M)
        for path in _py_files(root)
        for line in [path.read_text()]
    )


def test_no_arch_sniffs_in_model_layer() -> None:
    assert find_arch_sniffs(_MODEL) == []


def test_model_representation_symbol_is_deleted_repo_wide() -> None:
    assert not _has_model_representation_def(_SRC), "model_representation def must stay deleted"


def test_census_bites_planted_sniff(tmp_path: Path) -> None:
    """Mutation self-test (LAW-07): a planted type-name sniff → census FAILS."""
    planted = tmp_path / "mut.py"
    planted.write_text('def f(m):\n    return type(m).__name__ == "GnnNet"\n')
    assert find_arch_sniffs(tmp_path), "census must bite the planted type-name sniff"


def test_census_bites_planted_attr_and_isinstance(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text('def f(net):\n    return net.filters\n')
    (tmp_path / "b.py").write_text('def g(m):\n    return isinstance(m, GnnNet)\n')
    (tmp_path / "c.py").write_text('def h(cfg):\n    return cfg.get("representation", "grid")\n')
    viols = find_arch_sniffs(tmp_path)
    assert len(viols) >= 3, viols
