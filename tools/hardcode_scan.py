# >300 justify (R8): the hardcode-literal scanner is one cohesive rule engine — allowlists, strip
# transforms, per-file scanner and the section entry point kept together; loaded dynamically by
# the encoding audit.
"""Hardcode-scan section of `python -m mantis.encoding audit`.

Owns the hardcoded-literal rules, allowlists, strip transforms, per-file scanner and the
`_section_hardcode` entry point. Scans the Rust and Python source trees for bare
encoding-geometry literals outside an allowlist, so a new hardcoded board-size or plane count
surfaces instead of silently drifting from the registry.

It lives in `tools/` because it is a dev-only CI-gate scanner, not part of the shipped package,
and the CLI loads it by file path — no sys.path mutation (LAW-17).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mantis.encoding.audit import AuditReport, Severity


#: The dump path is a PARAMETER. It was a fixed `/tmp/...` name — a world-shared filename written
#: under `except OSError: pass`, so two users on one host raced for it and a failed write was
#: silent. `None` means "do not dump"; the CLI supplies one.
_DEFAULT_HITS_DUMP: Path | None = None


def _registry_targets() -> tuple[str, ...]:
    """The values this scanner looks for, DERIVED from the live registry.

    A frozen dense-era literal list meant the graph-era values were never scanned at all, so the
    one copy-detector in the repo could not see the numbers the graph seam is built from — while
    one of the listed values had quietly acquired a second meaning the list still read as the old
    one.

    Every registry value >= 3 is a target, plus the graph schema constants the registry validates
    against. `< 3` is excluded because `0`/`1`/`2` are arithmetic, not geometry, and scanning them
    would drown the signal — a judgement stated here rather than a silence.
    """
    values: set[int] = set()
    try:
        from mantis.encoding import all_specs
    except ImportError:  # the scanner runs without the extension in some contexts
        return ("19", "8", "6", "11", "362", "3")
    for spec in all_specs():
        for field in (
            "board_size", "trunk_size", "n_planes", "policy_logit_count", "n_source_planes",
            "legal_move_radius", "cluster_window_size", "cluster_threshold", "k_max",
            "n_chain_planes", "node_feat_dim", "edge_feat_dim", "win_length", "graph_radius",
            "win_axes",
        ):
            v = getattr(spec, field, None)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 3:
                values.add(v)
    return tuple(str(v) for v in sorted(values, reverse=True))


_HARDCODE_TARGETS: tuple[str, ...] = _registry_targets()
# Word-boundary number matcher; integer-only, over the DERIVED target set.
_NUM_PATTERN = re.compile(r"\b(" + "|".join(_HARDCODE_TARGETS) + r")\b")
_SKIP_DIRS: frozenset[str] = frozenset(
    {"__pycache__", "target", ".venv", "node_modules", "data", "checkpoints", ".git",
     "vendor", "build", "tests", "benches"}
)
_ALLOW_TOKENS: tuple[str, ...] = (
    "spec.",
    "EncodingSpec",
    "encoding.lookup",
    "registry::lookup",
    "enc-literal-ok",
    "audit: legacy-v6-fallback",   # doc-commented legacy fallback paths
    "#[pyo3(signature",            # PyO3 signature defaults are not geometry violations
    "GroupNorm",                   # GroupNorm group count is NN arch, not encoding geometry
    "GN_GROUPS",
    "gn_group",
)
_TEST_FILE_HINTS: tuple[str, ...] = ("test", "fixtures", "fixture")
# Patterns that look like version tokens or in-string literals; allow.
_VERSION_RE = re.compile(r"v\d+\b")

# ── allowlist constants (rules 1–10) ──────────────────────────────────────────────────

# Rule 5 — tunable hyperparameter tokens: skip any line containing these names.
_TUNABLE_TOKENS: frozenset[str] = frozenset({
    "c_puct", "fpu_reduction", "dirichlet_alpha", "dirichlet_epsilon",
    "temp_min", "eta_min", "timeout", "interval", "poll_interval",
    "figsize", "linewidth", "weight_for",
    "leaf_batch_size",
    "zoi_margin",
    "max_train_burst",
    "hard_abort_grad_norm_steps",
    "backup_count",        # log rotation, infra not geometry
    "batch_size",
    "max_frames",          # display frame count, not geometry
    "fail_gb",             # disk-guard threshold, not geometry
    "gn_groups",           # NN arch knob, not encoding geometry
    "gn_group",
    "epochs",
    "n_workers",
    "budget",
    "divisor",
    "skipped_nonfinite",
    "len(batch)",
    "len(wdr)",            # terminal-UI worker count
    "board.ply",           # game-ply threshold, not geometry
    "human_seeding_max_move",  # corpus seeding param, not encoding
    "max_move",            # opening length limit, not geometry
    "has_player_long_run", # run-length is a game rule, not geometry
    "hex_distance",        # game coordinates, not encoding geometry
    "max_pages",
    "jitter",
    "stride5",             # game-rule constant
    "pages",
})
# Guard: a line carrying these high-risk encoding constants still flags, even on a tunable hit.
_TUNABLE_SKIP_GUARD_TOKENS: frozenset[str] = frozenset({
    "feature_len", "policy_len",
})

# Rule 2 — float tolerances like 1e-5, 1e-8, 2.5e-4.
_FLOAT_TOL_RE = re.compile(r"\d+(?:\.\d+)?[eE]-\d+")

# Rule 2b — decimal fraction literals like 0.5, 1.5, 0.25, 0.8.
_DECIMAL_FRAC_RE = re.compile(r"\d+\.\d+")

# Rule 10 — display / infra context patterns: strip before scanning.
_SLICE_RE     = re.compile(r"\[:\s*\d+\s*\]")
_WITH_CAP_RE  = re.compile(r"\bwith_capacity\s*\(\s*\d+\s*\)")
_DISPLAY_KW_RE = re.compile(
    r"\b(?:fontsize|dpi|alpha|linewidth|markersize|rotation|zorder"
    r"|s=|edgelinewidth)\s*=\s*\d+"
)
_ROUND_PREC_RE = re.compile(r"\bround\s*\([^,]+,\s*\d+\s*\)")
_TOP_N_RE      = re.compile(r"\bmost_common\s*\(\s*\d+\s*\)")
_BYTE_BUF_RE  = re.compile(r"\[\s*0[ui]\d+\s*;\s*\d+\s*\]")
_RUST_STR_CONT_RE = re.compile(r"\\\s*$")
_STR_REPEAT_RE = re.compile(r"\"\s*-*\s*\"\s*\*\s*\d+")
_ROW_IDX_RE = re.compile(r"\b(?:row|p)\[(\d+)\]")
_RUST_MATCH_ARM_RE = re.compile(r"^\s*\d+\s*=>\s*\{")
_SECTION_REF_RE = re.compile(r"§\d+")
_ENTRY_BYTES_RE = re.compile(r"\bpolicy_bytes\s*\+")
_MIXED_COLLECTION_RE = re.compile(
    r"[\(\[]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*[\)\]]"
)

# Rule 4 — range bounds like `0..5`, `0..=8`, `0..19`.
_RANGE_BOUND_RE = re.compile(r"\b\d+\s*\.\.\s*=?\s*\d+\b")

# Rule 9 — whole-file allowlist: these ARE the sources of truth (the Python constants module and
# the crate that owns the canonical registry/spec definitions). registry.toml is TOML, not scanned.
_FULL_FILE_ALLOWLIST: frozenset[str] = frozenset({
    "src/mantis/util/constants.py",
})

# Rule 3 — coordinate-call patterns; strip the match before scanning.
_COORD_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bapply_move\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)"),
    re.compile(r"\bcells\.insert\s*\(\s*\("),
    re.compile(r"\bset_(?:row|col|diag)_\w+\s*\("),
    re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\.into\(\)"),
)

# Rule 8 — canonical-define lines are the source of truth; skip them.
_CANONICAL_DEFINE_RE = re.compile(
    r"^\s*(?:pub\s+(?:const\s+)?|const\s+|static\s+)?"
    r"(BOARD_SIZE|NUM_CELLS|BUFFER_CHANNELS|N_ACTIONS|MARGIN_M|HISTORY_LEN"
    r"|N_PLANES|N_CHAIN_PLANES|BOARD_H|BOARD_W|TOTAL_CELLS"
    r"|BOARD_H_V6W25|BOARD_W_V6W25|N_CELLS_V6W25|N_PLANES_V6W25"
    r"|N_ACTIONS_V6W25|STATE_STRIDE_V6W25|CHAIN_STRIDE_V6W25"
    r"|POLICY_STRIDE_V6W25|AUX_STRIDE_V6W25"
    r"|CLUSTER_THRESHOLD_V6W25|LEGAL_MOVE_RADIUS_V6W25"
    r"|DEFAULT_LEGAL_MOVE_RADIUS|DEFAULT_CLUSTER_THRESHOLD"
    r"|WIRE_CHANNELS|_STRIDE5_STEP"
    r"|JITTER_RADII|MAX_PAGES"
    r"|OPP_STONE_PLANE|MOVES_REMAINING_PLANE|PLY_PARITY_PLANE"
    r"|NODE_FEAT_DIM|EDGE_FEAT_DIM|WIN_LENGTH|GRAPH_RADIUS|WIN_AXES"
    r"|MAX_RETRIES"
    r")\s*[:=\[]"
)

# Rule 7 — trailing comment patterns.
_RUST_LINE_COMMENT_RE = re.compile(r"//.*$")
_PYTHON_LINE_COMMENT_RE = re.compile(r"#.*$")


def _is_test_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if any(h in name for h in _TEST_FILE_HINTS):
        return True
    return any(h in parts for h in _TEST_FILE_HINTS)


def _looks_like_comment(line: str, suffix: str) -> bool:
    stripped = line.lstrip()
    if suffix == ".py" and stripped.startswith("#"):
        return True
    if suffix == ".rs" and stripped.startswith("//"):
        return True
    return False


def _line_is_allowlisted(line: str, suffix: str) -> bool:
    if any(tok in line for tok in _ALLOW_TOKENS):
        return True
    if _looks_like_comment(line, suffix):
        return True
    return False


def _hits_outside_strings(line: str) -> list[str]:
    """Find target literals in `line` that are NOT inside string-quoted spans and not preceded by
    `v` (version tokens like v6, v8)."""
    cleaned = re.sub(r"\"[^\"]*\"|'[^']*'", "", line)
    cleaned = _VERSION_RE.sub("", cleaned)
    return _NUM_PATTERN.findall(cleaned)


# ── rule 1: test-range helpers ────────────────────────────────────────────────────────


def _test_ranges_rust(lines: list[str]) -> frozenset[int]:
    """Return 1-based line numbers inside #[cfg(test)] mod tests { ... } blocks."""
    in_test = False
    brace_depth = 0
    skip: set[int] = set()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_test:
            if stripped == "#[cfg(test)]" or "mod tests" in stripped:
                in_test = True
                brace_depth = stripped.count("{") - stripped.count("}")
                skip.add(i)
                continue
        if in_test:
            skip.add(i)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                in_test = False
                brace_depth = 0
    return frozenset(skip)


def _test_and_docstring_ranges_python(lines: list[str]) -> frozenset[int]:
    """Return 1-based line numbers inside class Test*/def test_* bodies and docstrings."""
    skip: set[int] = set()
    in_docstring = False
    in_test_block = False
    test_indent: int | None = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not in_docstring:
            if '"""' in stripped:
                count = stripped.count('"""')
                if count >= 2:
                    skip.add(i)
                else:
                    in_docstring = True
                    skip.add(i)
            elif "'''" in stripped:
                count = stripped.count("'''")
                if count >= 2:
                    skip.add(i)
                else:
                    in_docstring = True
                    skip.add(i)
        else:
            skip.add(i)
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue

        if stripped.startswith("class Test") or stripped.startswith("def test_"):
            leading = len(line) - len(line.lstrip())
            in_test_block = True
            test_indent = leading
            skip.add(i)
        elif in_test_block:
            current_indent = len(line) - len(line.lstrip()) if stripped else test_indent + 1  # type: ignore[operator]
            if stripped == "" or current_indent > test_indent:  # type: ignore[operator]
                skip.add(i)
            else:
                in_test_block = False
                test_indent = None

    return frozenset(skip)


# ── line-level transform helpers (rules 2, 3, 4, 7) ───────────────────────────────────


def _strip_trailing_comment_rust(line: str) -> str:
    """Remove `// ...` comment from a Rust line (not inside strings)."""
    in_str = False
    str_char = ""
    for idx in range(len(line) - 1):
        ch = line[idx]
        if not in_str:
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
            elif ch == "/" and line[idx + 1] == "/":
                return line[:idx]
        else:
            if ch == str_char and (idx == 0 or line[idx - 1] != "\\"):
                in_str = False
    return line


def _strip_trailing_comment_python(line: str) -> str:
    """Remove `# ...` comment from a Python line (not inside strings)."""
    in_str = False
    str_char = ""
    for idx, ch in enumerate(line):
        if not in_str:
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
            elif ch == "#":
                return line[:idx]
        else:
            if ch == str_char and (idx == 0 or line[idx - 1] != "\\"):
                in_str = False
    return line


def _line_has_coord_pattern(line: str) -> bool:
    """Return True if line contains any coordinate-call pattern (rule 3)."""
    return any(pat.search(line) for pat in _COORD_PATTERNS)


def _apply_line_transforms(line: str, suffix: str) -> str:
    """Apply strip transforms to reduce false positives before scanning."""
    if suffix == ".rs":
        line = _strip_trailing_comment_rust(line)
    elif suffix == ".py":
        line = _strip_trailing_comment_python(line)
    line = _FLOAT_TOL_RE.sub("", line)          # rule 2
    line = _DECIMAL_FRAC_RE.sub("", line)       # rule 2b
    line = _RANGE_BOUND_RE.sub("", line)        # rule 4
    line = _SLICE_RE.sub("[]", line)
    line = _WITH_CAP_RE.sub("", line)
    line = _DISPLAY_KW_RE.sub("", line)
    line = _ROUND_PREC_RE.sub("", line)
    line = _TOP_N_RE.sub("", line)
    line = _BYTE_BUF_RE.sub("", line)
    if _RUST_STR_CONT_RE.search(line):
        return ""
    line = _STR_REPEAT_RE.sub("", line)
    line = _ROW_IDX_RE.sub("", line)
    if _RUST_MATCH_ARM_RE.match(line):
        return ""
    line = _SECTION_REF_RE.sub("", line)
    if _ENTRY_BYTES_RE.search(line):
        return ""
    if _MIXED_COLLECTION_RE.search(line):
        return ""
    return line


def _is_the_owning_file(path: Path, line: str) -> bool:
    """True when `line` defines a canonical name IN the file that owns it.

    The previous rule exempted ANY line defining a name in the canonical set, anywhere, so a COPY
    that reused the canonical spelling was exempt BY NAME — the exact inverse of a copy detector.
    The owner is derived from the name rather than listed per file: a canonical constant is owned
    by the module or crate whose own name the definition sits under.
    """
    stem = path.stem
    m = _CANONICAL_DEFINE_RE.match(line)
    if m is None:
        return False
    name = m.group(1)
    owners = _CANONICAL_OWNERS.get(name)
    if owners is None:
        # A name with no declared owner keeps the old exemption — narrowing is done by adding an
        # owner row, never by guessing one here.
        return True
    return stem in owners


#: Which file owns each canonical constant; a definition anywhere else is a COPY and is scanned.
#: Keyed by stem so a crate move does not silently widen the exemption.
_CANONICAL_OWNERS: dict[str, tuple[str, ...]] = {
    "NODE_FEAT_DIM": ("lib",), "EDGE_FEAT_DIM": ("lib",), "WIN_AXES": ("lib",),
    "WIN_LENGTH": ("moves",), "BOARD_SIZE": ("core",), "TOTAL_CELLS": ("core",),
    "HISTORY_LEN": ("constants",),
    "DEFAULT_LEGAL_MOVE_RADIUS": ("moves",), "DEFAULT_CLUSTER_THRESHOLD": ("moves",),
    "N_CHAIN_PLANES": ("sym", "game_state"),
    "OPP_STONE_PLANE": ("mod",), "MOVES_REMAINING_PLANE": ("mod",), "PLY_PARITY_PLANE": ("mod",),
}


def _scan_file(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return [(lineno, line, hits)] for unjustified hits in `path`."""
    suffix = path.suffix
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()

    if suffix == ".rs":
        skip_linenos = _test_ranges_rust(lines)
    elif suffix == ".py":
        skip_linenos = _test_and_docstring_ranges_python(lines)
    else:
        skip_linenos = frozenset()

    out: list[tuple[int, str, list[str]]] = []
    for i, line in enumerate(lines, start=1):
        if i in skip_linenos:
            continue
        if _line_is_allowlisted(line, suffix):
            continue
        if (any(tok in line for tok in _TUNABLE_TOKENS)
                and not any(g in line for g in _TUNABLE_SKIP_GUARD_TOKENS)):
            continue
        if suffix in (".py", ".rs") and _CANONICAL_DEFINE_RE.match(line) \
                and _is_the_owning_file(path, line):
            continue
        if _line_has_coord_pattern(line):
            continue
        transformed = _apply_line_transforms(line, suffix)
        hits = _hits_outside_strings(transformed)
        if hits:
            out.append((i, line.rstrip(), hits))
    return out


def _section_hardcode(report: AuditReport, repo_root: Path, *, collect_raw: bool = False,
                      hits_dump: Path | None = _DEFAULT_HITS_DUMP) -> None:
    from mantis.encoding.audit import AuditSection

    sect = AuditSection(
        title="§5 Hardcoded literals",
        headers=("file", "hits", "lines (top 3)"),
    )
    targets = [
        repo_root / "crates",
        repo_root / "src",
    ]
    file_hits: dict[Path, list[tuple[int, str, list[str]]]] = {}
    total_hits = 0
    for root in targets:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in (".py", ".rs"):
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if _is_test_path(p):
                continue
            rel_str = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
            if rel_str.replace("\\", "/") in _FULL_FILE_ALLOWLIST:
                continue
            hits = _scan_file(p)
            if hits:
                file_hits[p] = hits
                total_hits += sum(len(h[2]) for h in hits)

    if not file_hits:
        sect.notes.append("no unjustified hardcode hits found")
        report.add_finding("info", "§5", "no unjustified hits")
        report.sections["§5"] = sect
        return

    severity: Severity = "error" if report.strict else "warn"

    if collect_raw:
        for p in sorted(file_hits):
            rel = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
            for lineno, line, hits in file_hits[p]:
                report._raw_hardcode_hits.append(
                    {"file": rel, "line": lineno, "content": line, "hits": hits}
                )
    elif hits_dump is not None:
        # NO `except OSError: pass` — a dump the operator asked for and did not get is a fact,
        # not a silence, and this used to swallow it on a fixed world-shared `/tmp` name.
        with Path(hits_dump).open("w", encoding="utf-8") as fh:
            for p in sorted(file_hits):
                fh.write(f"# {p}\n")
                for lineno, line, hits in file_hits[p]:
                    fh.write(f"  L{lineno}  hits={hits}  | {line}\n")
                fh.write("\n")

    for p in sorted(file_hits):
        hits = file_hits[p]
        rel = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
        n = sum(len(h[2]) for h in hits)
        preview = "; ".join(f"L{ln}={hh}" for ln, _, hh in hits[:3])
        sect.rows.append([rel, str(n), preview])

    if collect_raw:
        report.add_finding(
            severity, "§5",
            f"{total_hits} unjustified literal hit(s) across {len(file_hits)} file(s); "
            f"full dump in JSON hardcode_hits",
        )
        sect.notes.append(f"strict={report.strict} → severity={severity}")
        sect.notes.append("full dump: included in JSON hardcode_hits field")
    else:
        report.add_finding(
            severity, "§5",
            f"{total_hits} unjustified literal hit(s) across {len(file_hits)} file(s); "
            f"full dump → {hits_dump}" if hits_dump else "no dump requested",
        )
        sect.notes.append(f"strict={report.strict} → severity={severity}")
        sect.notes.append(f"full dump: {hits_dump}" if hits_dump else "full dump: not requested")
    report.sections["§5"] = sect
