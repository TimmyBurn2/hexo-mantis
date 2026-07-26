#!/usr/bin/env python3
# >300 justify: the pattern table, the tamper-evident known-debt register and the logical-
# line normaliser are one gate's single authority; splitting them would create a second
# place where "what counts as a silent fallback" is decided (LAW-03).
"""CI gate 11 (R45): no silent encoding-fallback arms.

An encoding must never be *defaulted* into existence. LAW-11 and LAW-05 say an absent or
unspecified encoding is an ERROR; a resolver that quietly substitutes a registered name
turns a configuration mistake into a run that trains or infers the wrong thing and reports
success.

KNOWN ARMS OF THIS CLASS (R45 asks that they be enumerated here):

  1. src/mantis/encoding/resolvers.py -- `normalize_encoding_name(None)` -> "v6"  (R28)
  2. src/mantis/encoding/resolvers.py -- `normalize_encoding_name()` on a mapping with
     neither `name` nor `version` -> "v6"  (WPSC 9a75c59, "the fifth arm")
  3. src/mantis/encoding/resolvers.py -- `resolve_from_config()` on None / no `encoding`
     key / mapping with no `version` key -> "v6"  (R28)
  4. crates/mantis-bridge/src/board.rs -- `Board::to_tensor()` encoding-less dense
     fallback  (R28; now a PanicException)
  5. src/mantis/train/pretrain/validate.py -- `_config_encoding()` via
     `enc.get("version", "v6")` and a terminal bare `return "v6"`  (R45, HANDOFF-10)
  6. src/mantis/train/pretrain/cli.py -- `_resolve_encoding_name()` terminal
     `return "v6"`  (R45 / ADJ-03). Live on the pretrain TRAINING path.
  7. src/mantis/train/pretrain/dataset.py -- `make_augmented_collate(..., encoding="v6")`
     signature default  (R45 / ADJ-05). Found by REVIEW-impl, not by this gate's first
     draft, which had no signature-default pattern.
  8. src/mantis/selfplay/inference_local.py -- `LocalInferenceEngine` ternary fallback
     `encoding_spec if ... else lookup("v6")`  (R45 / ADJ-05). **STILL OPEN** -- see
     KNOWN_DEBT below. Owned by WP12-R.
  9. crates/mantis-bridge/src/buffer.rs -- `#[pyo3(signature = (capacity, encoding =
     "v6"))]` on `ReplayBuffer.__new__`, plus both `_engine.pyi` twins  (R45 / ADJ-05).
 10. crates/mantis-bridge/src/hexg.rs -- the same on `HexgBuffer.__new__`, defaulting to
     `"gnn_axis_v1"`  (R45 / ADJ-05). Latent rather than academic: it silently picks v1
     the moment a second graph schema exists, and WP-AXIS2 adds `gnn_axis_v2`.

9 and 10 were invisible to this gate until its Rust comment handling was fixed -- an
earlier draft treated `#[pyo3(...)]` attributes as comments and blanked them.

1-7, 9 and 10 are CLOSED. 8 is open, registered, and reported on every run.

SCOPE: production code only -- `src/` and `crates/`. Tests, benches and fixtures may
legitimately name an encoding as data.

WHAT THIS GATE DOES NOT CATCH: affirmative dispatch. `return lookup("gnn_axis_v1")` guarded
by `if _GNN_GRAPH_MARKER_KEY in state` is a decision made on evidence, not a fallback. A
gate that fires on correct code trains reviewers to ignore it.

ESCAPE HATCH -- for a site that is NOT a fallback at all:

    return "v6"  # silent-encoding-gate: ok -- <why this is not a fallback>

The reason text is mandatory. The hatch must NEVER be used to silence a real arm; a real
arm that cannot be closed yet goes in KNOWN_DEBT, which is loud, owned and tamper-evident.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "crates")
SUFFIXES = {".py", ".rs", ".pyi"}
SKIP_DIR_PARTS = {"tests", "benches", "target", "__pycache__", "fixtures"}

# A checkout can live anywhere -- e.g. a git worktree under a directory literally named
# `target`. Skip decisions are therefore made on the REPO-RELATIVE path only; matching the
# absolute path made the whole gate vacuously green depending on where the repo sat.
MIN_SCANNED_FILES = 80  # a floor, so "scanned nothing, found nothing" can never pass


# ── the registered set (registry.toml, pruned to four at WP3) ────────────────────────
# Longest-first so the alternation cannot match "v6" inside "v6w25".
ENCODINGS = ("v6_live2_ls", "gnn_axis_v1", "v6w25", "v6")
_ENC = "|".join(ENCODINGS)
_Q = f"['\"](?:{_ENC})['\"]"

# Every shape below puts an encoding literal in a DEFAULT position -- the value used when
# nobody said. Derived from arms 1-8 plus the evasion set REVIEW-impl constructed against
# the first draft, which 27 of 31 probes defeated.
PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"\.get\(\s*[^)]*?,\s*{_Q}", "dict.get() with an encoding-name default"),
    (rf"\.setdefault\(\s*[^)]*?,\s*{_Q}", "dict.setdefault() with an encoding-name default"),
    (rf"\.pop\(\s*[^)]*?,\s*{_Q}", "dict.pop() with an encoding-name default"),
    (rf"\bgetattr\(\s*[^)]*?,\s*{_Q}", "getattr() with an encoding-name default"),
    # `or X` / `else X` are fallback positions BY CONSTRUCTION, so the literal may be
    # wrapped in a call there (`else lookup("v6")`) and it is still a fallback. A bare
    # `return lookup(...)` is NOT given the same treatment: inside an affirmative guard it
    # is correct dispatch (resolvers.py:487), and a gate that fires on correct code trains
    # reviewers to ignore it.
    (rf"\bor\s+(?:[\w.]+\(\s*)?{_Q}", "`or <encoding>` fallback"),
    (rf"\belse\s+(?:[\w.]+\(\s*)?{_Q}", "ternary/else fallback to an encoding"),
    (rf"\breturn\s+{_Q}\s*(?:[;)]|$)", "terminal `return <encoding>` fallback"),
    # `(?<![=!<>])=` so comparisons (`== "v6"`, `!= "v6"`) are not mistaken for defaults.
    (rf"(?<![=!<>])=\s*{_Q}\s*(?:[,)\]]|$)", "assignment / signature default / keyword default"),
    (rf"\bunwrap_or\(\s*{_Q}", "Rust unwrap_or() with an encoding-name default"),
    (rf"\bunwrap_or_else\(\s*\|\|\s*{_Q}", "Rust unwrap_or_else() with an encoding default"),
    (rf"\bmap_or\(\s*{_Q}", "Rust map_or() with an encoding-name default"),
    (rf"=>\s*{_Q}", "Rust match arm defaulting to an encoding"),
)

# ── known, owned, still-open arms ────────────────────────────────────────────────────
# NOT an escape hatch. Each entry asserts "this IS a real arm, it is tracked, and it has an
# owner". Matched on the exact source text, so editing the line invalidates the entry and
# fails the gate -- a silent rewrite cannot inherit the exemption.
KNOWN_DEBT: tuple[tuple[str, str, str], ...] = (
    (
        "src/mantis/selfplay/inference_local.py",
        'encoding_spec if encoding_spec is not None else lookup("v6")',
        "ADJ-05 / owner WP12-R (eval-worker encoding_spec threading; WP11-A handoff, "
        "run5-mint blocker). src/mantis/eval/worker.py:78,193 construct "
        "LocalInferenceEngine positionally with no spec and depend on this default. "
        "Closing it inside R45 would change eval-worker behaviour that WP11-A already "
        "reports failing loud (eval_broken), which is WP12-R's decision, not R45's.",
    ),
)

ESCAPE = re.compile(r"silent-encoding-gate:\s*ok\s*--\s*\S")
_COMPILED = tuple((re.compile(p), why) for p, why in PATTERNS)

# Comment syntax is per-language, and getting this wrong is not cosmetic: an earlier draft
# treated any line starting with `#` as a comment, which silently blanked every Rust
# `#[pyo3(...)]` ATTRIBUTE — exactly where the pyo3 signature defaults live. The gate
# reported green over two real arms because of it.
_PY_COMMENT_ONLY = re.compile(r"^\s*#(?!\[)")
_RS_COMMENT_ONLY = re.compile(r"^\s*//")
_PY_STRIP = re.compile(r"(?<!['\"])\s+#(?!\[).*$")
_RS_STRIP = re.compile(r"(?<!['\"])\s+//.*$")


def _comment_res(suffix: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    if suffix == ".rs":
        return _RS_COMMENT_ONLY, _RS_STRIP
    return _PY_COMMENT_ONLY, _PY_STRIP


def _is_justified(lines: list[str], idx: int, suffix: str = ".py") -> bool:
    """True if line `idx` carries an escape, or the comment block directly above it does.

    The whole contiguous comment block is searched: a justification worth writing usually
    needs a sentence, and a marker that only works on one-line comments quietly punishes
    the explanations that are actually useful.
    """
    comment_only, _ = _comment_res(suffix)
    if ESCAPE.search(lines[idx]):
        return True
    j = idx - 1
    while j >= 0 and comment_only.match(lines[j]):
        if ESCAPE.search(lines[j]):
            return True
        j -= 1
    return False


def _logical_lines(lines: list[str], suffix: str = ".py") -> list[tuple[int, str]]:
    """Join bracket-continued physical lines, and strip trailing comments.

    Two evasions REVIEW-impl found in the first draft die here: `return "v6"  # comment`
    (the pattern anchored on end-of-line, so a comment walked straight through) and a
    `.get(` call split across lines (never matched at all). Reported line number is the
    line the construct STARTS on.
    """
    comment_only, strip = _comment_res(suffix)
    out: list[tuple[int, str]] = []
    buf, start, depth = "", 0, 0
    for i, raw in enumerate(lines):
        code = "" if comment_only.match(raw) else strip.sub("", raw)
        if not buf:
            start = i
        buf = f"{buf} {code.strip()}" if buf else code.strip()
        depth += code.count("(") - code.count(")")
        depth += code.count("[") - code.count("]")
        if depth <= 0:
            if buf:
                out.append((start, buf))
            buf, depth = "", 0
    if buf:
        out.append((start, buf))
    return out


def _iter_files():
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            if SKIP_DIR_PARTS & set(path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def scan() -> tuple[list[str], list[str], int, set[int]]:
    """Return (violations, debt_hits, files_scanned, indices_of_matched_debt_entries)."""
    violations: list[str] = []
    debt_hits: list[str] = []
    matched_debt: set[int] = set()
    files_scanned = 0

    for path in _iter_files():
        files_scanned += 1
        rel = str(path.relative_to(REPO_ROOT))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, logical in _logical_lines(lines, path.suffix):
            for rx, why in _COMPILED:
                if not rx.search(logical):
                    continue
                if _is_justified(lines, idx, path.suffix):
                    break
                debt_i = next(
                    (
                        k
                        for k, (dpath, dtext, _r) in enumerate(KNOWN_DEBT)
                        if dpath == rel and dtext in logical
                    ),
                    None,
                )
                if debt_i is not None:
                    matched_debt.add(debt_i)
                    debt_hits.append(f"{rel}:{idx + 1}: {KNOWN_DEBT[debt_i][2]}")
                    break
                violations.append(f"{rel}:{idx + 1}: {why}\n    {logical.strip()}")
                break

    return violations, debt_hits, files_scanned, matched_debt


def find_violations() -> list[str]:
    """Unjustified, unregistered silent-fallback sites."""
    return scan()[0]


def main() -> int:
    violations, debt_hits, files_scanned, matched_debt = scan()
    rc = 0

    if files_scanned < MIN_SCANNED_FILES:
        print(
            f"gate 11 FAIL -- scanned only {files_scanned} files (floor {MIN_SCANNED_FILES}). "
            "A gate that scans nothing finds nothing; refusing to report green."
        )
        return 1

    stale = [KNOWN_DEBT[k][0] for k in range(len(KNOWN_DEBT)) if k not in matched_debt]
    if stale:
        print(
            "gate 11 FAIL -- KNOWN_DEBT entries matched nothing (the code changed under "
            f"them; re-adjudicate rather than editing the register): {stale}"
        )
        rc = 1

    if debt_hits:
        # Loud on every run, by design: registered debt that stops being visible stops
        # being debt and starts being the status quo.
        print(f"gate 11: {len(debt_hits)} REGISTERED-OPEN arm(s), owned, not yet closed:")
        for hit in debt_hits:
            print(f"  {hit}")

    if violations:
        print("\ngate 11 FAIL -- silent encoding-fallback arm(s) (R45, LAW-11/LAW-05):\n")
        print("\n".join(violations))
        print(
            "\nAn absent encoding must RAISE, never default. Use the established named-error "
            "convention: MissingEncodingError in Python, panic/PanicException across the FFI. "
            "If this is genuinely NOT a fallback, justify it in place:\n"
            "    # silent-encoding-gate: ok -- <why>\n"
            "If it IS a real arm that cannot be closed yet, it goes in KNOWN_DEBT with an "
            "owner and an adjudication id -- never in the escape hatch."
        )
        rc = 1

    if rc == 0:
        print(f"gate 11: no unregistered silent encoding-fallback arms ({files_scanned} files)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
