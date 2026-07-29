# >300 lines: the §5 hardcode-literal scanner is one cohesive rule engine —
# allowlists, strip transforms, per-file scanner, and the _section_hardcode entry
# point kept together; loaded dynamically by mantis/encoding/audit.py (§5).
"""Hardcode-scan section of `python -m mantis.encoding audit` (§5).

Owns §5 (Hardcoded literals) — the rules, allowlists, strip transforms, per-file
scanner, and the `_section_hardcode` entry point. Scans the Rust (`crates/`) and
Python (`src/`) source trees for bare encoding-geometry literals (`19`, `25`,
`361`, `5`, `8`) outside an allowlist, so a new hardcoded board-size / plane
count surfaces instead of silently drifting from the registry.

Relocated from the encoding package to `tools/` (it is a dev-only CI-gate
scanner, not part of the shipped package). The CLI loads it by file path
(no sys.path mutation, LAW-17).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mantis.encoding.audit import AuditReport, Severity


_HARDCODE_HITS_DUMP = Path("/tmp/encoding_audit_hardcode_hits.txt")


# Word-boundary number matcher; integer-only. Captures `19`, `25`, `361`, `5`, `8`.
_NUM_PATTERN = re.compile(r"\b(19|25|361|5|8)\b")
_HARDCODE_TARGETS: tuple[str, ...] = ("19", "25", "361", "5", "8")
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

# ---------------------------------------------------------------------------
# Allowlist constants (rules 1–10)
# ---------------------------------------------------------------------------

# Rule 5 — tunable hyperparameter tokens: skip any line containing these names.
_TUNABLE_TOKENS: frozenset[str] = frozenset({
    "c_puct", "fpu_reduction", "dirichlet_alpha", "dirichlet_epsilon",
    "temp_min", "eta_min", "timeout", "interval", "poll_interval",
    "figsize", "linewidth", "weight_for",
    "leaf_batch_size",     # MCTS batch size (tuned per host via sweep)
    "zoi_margin",          # zone-of-interest search margin (tunable)
    "max_train_burst",     # training throughput cap (tunable)
    "hard_abort_grad_norm_steps",  # gradient-norm abort window (tunable)
    "backup_count",        # log-rotation file count (infra, not geometry)
    "batch_size",          # generic inference batch (tunable unless guarded)
    "max_frames",          # animation/display frame count (not geometry)
    "fail_gb",             # disk-guard threshold (not geometry)
    "gn_groups",           # GroupNorm groups (NN arch knob, not encoding)
    "gn_group",
    "epochs",              # training epoch count (tunable CLI arg)
    "n_workers",           # worker pool count (tunable, used in CPU budget)
    "budget",              # CPU thread budget expression
    "divisor",             # CPU budget divisor expression
    "skipped_nonfinite",   # error counter (training loop)
    "len(batch)",          # batch accumulator size check
    "len(wdr)",            # display worker count for terminal UI
    "board.ply",           # game-ply threshold (not encoding geometry)
    "human_seeding_max_move",  # corpus opening-move seeding param (not encoding)
    "max_move",            # opening game length limit (not encoding geometry)
    "has_player_long_run", # game-rule threat probe (run-length != encoding geometry)
    "hex_distance",        # game-coordinate distance function (not encoding geometry)
    "max_pages",           # scraper page limit (infra, not geometry)
    "jitter",              # MCTS jitter radii (tunable)
    "stride5",             # stride-5 detector step (game-rule constant)
    "pages",               # CLI page argument (infra)
})
# Guard: if a line also contains these tokens, do NOT suppress even if a
# _TUNABLE_TOKEN hit — high-risk encoding constants that must still flag.
_TUNABLE_SKIP_GUARD_TOKENS: frozenset[str] = frozenset({
    "feature_len", "policy_len",
})

# Rule 2 — float tolerances like 1e-5, 1e-8, 2.5e-4.
_FLOAT_TOL_RE = re.compile(r"\d+(?:\.\d+)?[eE]-\d+")

# Rule 2b — decimal fraction literals like 0.5, 1.5, 0.25, 0.8.
_DECIMAL_FRAC_RE = re.compile(r"\d+\.\d+")

# Rule 10 — display / infra context patterns: strip before scanning.
_SLICE_RE     = re.compile(r"\[:\s*\d+\s*\]")                    # 10a
_WITH_CAP_RE  = re.compile(r"\bwith_capacity\s*\(\s*\d+\s*\)")   # 10b
_DISPLAY_KW_RE = re.compile(                                       # 10c
    r"\b(?:fontsize|dpi|alpha|linewidth|markersize|rotation|zorder"
    r"|s=|edgelinewidth)\s*=\s*\d+"
)
_ROUND_PREC_RE = re.compile(r"\bround\s*\([^,]+,\s*\d+\s*\)")    # 10d
_TOP_N_RE      = re.compile(r"\bmost_common\s*\(\s*\d+\s*\)")     # 10e
_BYTE_BUF_RE  = re.compile(r"\[\s*0[ui]\d+\s*;\s*\d+\s*\]")      # 10f
_RUST_STR_CONT_RE = re.compile(r"\\\s*$")                          # 10g
_STR_REPEAT_RE = re.compile(r"\"\s*-*\s*\"\s*\*\s*\d+")           # 10h
_ROW_IDX_RE = re.compile(r"\b(?:row|p)\[(\d+)\]")                 # 10i
_RUST_MATCH_ARM_RE = re.compile(r"^\s*\d+\s*=>\s*\{")             # 10j
_SECTION_REF_RE = re.compile(r"§\d+")                              # 10k
_ENTRY_BYTES_RE = re.compile(r"\bpolicy_bytes\s*\+")               # 10l
_MIXED_COLLECTION_RE = re.compile(                                 # 10m
    r"[\(\[]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*[\)\]]"
)

# Rule 4 — range bounds like `0..5`, `0..=8`, `0..19`.
_RANGE_BOUND_RE = re.compile(r"\b\d+\s*\.\.\s*=?\s*\d+\b")

# Rule 9 — whole-file allowlist (these ARE the sources of truth). Re-anchored to
# the new tree: the Python constants module + the crate that owns the canonical
# registry/spec definitions. registry.toml itself is TOML (not scanned).
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
    """Find target literals in `line` that are NOT inside string-quoted spans
    and not preceded by `v` (version tokens like v6, v8 → skip)."""
    cleaned = re.sub(r"\"[^\"]*\"|'[^']*'", "", line)
    cleaned = _VERSION_RE.sub("", cleaned)
    return _NUM_PATTERN.findall(cleaned)


# ---------------------------------------------------------------------------
# Rule 1 — test-range helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Line-level transform helpers (rules 2, 3, 4, 7)
# ---------------------------------------------------------------------------


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
    line = _SLICE_RE.sub("[]", line)            # 10a
    line = _WITH_CAP_RE.sub("", line)           # 10b
    line = _DISPLAY_KW_RE.sub("", line)         # 10c
    line = _ROUND_PREC_RE.sub("", line)         # 10d
    line = _TOP_N_RE.sub("", line)              # 10e
    line = _BYTE_BUF_RE.sub("", line)           # 10f
    if _RUST_STR_CONT_RE.search(line):          # 10g
        return ""
    line = _STR_REPEAT_RE.sub("", line)         # 10h
    line = _ROW_IDX_RE.sub("", line)            # 10i
    if _RUST_MATCH_ARM_RE.match(line):          # 10j
        return ""
    line = _SECTION_REF_RE.sub("", line)        # 10k
    if _ENTRY_BYTES_RE.search(line):            # 10l
        return ""
    if _MIXED_COLLECTION_RE.search(line):       # 10m
        return ""
    return line


def _scan_file(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return [(lineno, line, hits)] for unjustified hits in `path`."""
    suffix = path.suffix
    try:
        text = path.read_text(errors="replace")
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
        if suffix in (".py", ".rs") and _CANONICAL_DEFINE_RE.match(line):
            continue
        if _line_has_coord_pattern(line):
            continue
        transformed = _apply_line_transforms(line, suffix)
        hits = _hits_outside_strings(transformed)
        if hits:
            out.append((i, line.rstrip(), hits))
    return out


def _section_hardcode(report: AuditReport, repo_root: Path, *, collect_raw: bool = False) -> None:
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
    else:
        try:
            with _HARDCODE_HITS_DUMP.open("w") as fh:
                for p in sorted(file_hits):
                    fh.write(f"# {p}\n")
                    for lineno, line, hits in file_hits[p]:
                        fh.write(f"  L{lineno}  hits={hits}  | {line}\n")
                    fh.write("\n")
        except OSError:
            pass

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
            f"full dump → {_HARDCODE_HITS_DUMP}",
        )
        sect.notes.append(f"strict={report.strict} → severity={severity}")
        sect.notes.append(f"full dump: {_HARDCODE_HITS_DUMP}")
    report.sections["§5"] = sect
