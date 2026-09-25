"""The FFI/build law's second half has a check: `target-cpu` appears in no committed build config."""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

#: Committed files that can bind rustc flags or profile settings. `.cargo/config.toml` is scanned
#: when present; its absence is the expected state and is not a finding.
BUILD_CONFIG = (
    "Cargo.toml",
    "pyproject.toml",
    "Makefile",
    "rust-toolchain.toml",
    ".cargo/config.toml",
)

_TARGET_CPU = re.compile(r"target-cpu")
#: The ONE sanctioned site: `make build.native` sets it in an opt-in local build's environment
#: (CLAUDE.md R2). Comments are stripped before the scan so the rule's own prose cannot trip it.
_NATIVE_RECIPE = re.compile(r'^\tRUSTFLAGS="-C target-cpu=native" ')


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def scan_for_target_cpu(root: Path) -> list[str]:
    """`path:line` for every non-comment `target-cpu` outside the Makefile's build.native recipe."""
    files = [root / rel for rel in BUILD_CONFIG] + sorted((root / "crates").glob("*/Cargo.toml")) \
        + sorted((root / "crates").glob("*/pyproject.toml"))
    hits: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not _TARGET_CPU.search(_strip_comment(raw)):
                continue
            if path.name == "Makefile" and _NATIVE_RECIPE.match(raw):
                continue
            hits.append(f"{path.relative_to(root).as_posix()}:{lineno}")
    return hits


def test_the_committed_tree_binds_no_target_cpu() -> None:
    assert scan_for_target_cpu(_REPO) == [], (
        "LAW-13: target-cpu in committed build config makes every artifact host-pinned; the "
        "one sanctioned route is `make build.native` (env-only)"
    )


def test_the_scan_reads_at_least_the_workspace_manifest_and_the_makefile() -> None:
    assert (_REPO / "Cargo.toml").is_file() and (_REPO / "Makefile").is_file()
    assert (_REPO / "Makefile").read_text(encoding="utf-8").count("target-cpu=native") == 1, (
        "the build.native recipe is the ONE env-only site; a second one is a second authority"
    )


def test_a_planted_cargo_rustflags_line_fires(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = []\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("build:\n\techo ok\n", encoding="utf-8")
    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".cargo" / "config.toml").write_text(
        '[build]\nrustflags = ["-C", "target-cpu=native"]\n', encoding="utf-8")
    assert scan_for_target_cpu(tmp_path) == [".cargo/config.toml:2"]


def test_a_comment_naming_the_rule_does_not_fire(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        "# Portable by default: no target-cpu anywhere in committed config\n[workspace]\n",
        encoding="utf-8")
    (tmp_path / "Makefile").write_text(
        'build.native:\n\tRUSTFLAGS="-C target-cpu=native" uv sync\n', encoding="utf-8")
    assert scan_for_target_cpu(tmp_path) == []
