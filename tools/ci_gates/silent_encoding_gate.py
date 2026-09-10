#!/usr/bin/env python3
# >300 justify: the pattern table, the tamper-evident known-debt register and the logical-line
# normaliser are one gate's single authority; splitting them would create a second place where
# "what counts as a silent fallback" is decided.
"""CI gate 11: no silent encoding-fallback arms.

An encoding must never be *defaulted* into existence. An absent or unspecified encoding is an
ERROR; a resolver that quietly substitutes a registered name turns a configuration mistake into a
run that trains or infers the wrong thing and reports success.

The known arms of this class, enumerated here as the ruling requires — all CLOSED: three
`resolvers.py` paths that returned "v6" for a `None`, a keyless mapping or a keyless config; the
encoding-less dense fallback in `Board::to_tensor()`, now a PanicException; two pretrain
resolvers with terminal `return "v6"` arms, one of them on the TRAINING path; a
`make_augmented_collate(..., encoding="v6")` signature default; `LocalInferenceEngine`'s ternary,
whose `encoding_spec` is now a REQUIRED keyword-only parameter; and the pyo3 signature defaults
on `ReplayBuffer.__new__` and `HexgBuffer.__new__` plus their `_engine.pyi` twins. The last two
were invisible until the Rust comment handling was fixed, an earlier draft having treated
`#[pyo3(...)]` attributes as comments and blanked them.

`KNOWN_DEBT` is empty, and the register machinery is retained for the next owned arm: an entry
that stops matching FAILS the gate, so an exemption can never be silently inherited by whatever
replaced the line it named.

SCOPE: production code only, `src/` and `crates/`. Affirmative dispatch is not caught — a
`lookup(...)` guarded by a marker check is a decision made on evidence, and a gate that fires on
correct code trains reviewers to ignore it. The escape hatch for a site that is not a fallback at
all is a trailing `# silent-encoding-gate: ok -- <why>` with mandatory reason text; a real arm
that cannot be closed yet goes in KNOWN_DEBT, which is loud, owned and tamper-evident.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "crates")
SUFFIXES = {".py", ".rs", ".pyi"}
SKIP_DIR_PARTS = {"tests", "benches", "target", "__pycache__", "fixtures"}

# A checkout can live anywhere — e.g. a worktree under a directory literally named `target` — so
# skip decisions are made on the REPO-RELATIVE path only; matching the absolute path made the
# whole gate vacuously green depending on where the repo sat.
MIN_SCANNED_FILES = 80  # a floor, so "scanned nothing, found nothing" can never pass


# The registered set, longest-first so the alternation cannot match "v6" inside "v6w25".
ENCODINGS = ("v6_live2_ls", "gnn_axis_v1", "gnn_axis_r8", "v6w25", "v6")
_ENC = "|".join(ENCODINGS)
# An optional `f`/`r`/`b` prefix: `return f"v6"` is the same arm with a redundant prefix.
_Q = f"(?:[frb]{{0,2}})['\"](?:{_ENC})['\"]"
# In a FALLBACK position (`or`, `else`, a match arm) the literal may be wrapped in any call,
# because the position itself is what makes it a fallback.
_CALL = r"(?:[\w.]+\(\s*)?"
# In an ASSIGNMENT position the wrapper must be an explicit default-DECLARATION helper: a general
# `[\w.]+\(` would flag `spec = lookup("v6")`, affirmative dispatch and the opposite of a default,
# so recall is traded for precision deliberately.
_DECL_CALL = r"(?:(?:Field|field|Argument|Query|Body|Option|Some)\(\s*)?"
# Terminators that can follow a default value; without `;` every Rust `const ... = "v6";` walked
# through.
_END = r"(?:[,;)\]}]|$)"

# Every shape below puts an encoding literal in a DEFAULT position — the value used when nobody
# said. Derived from the known arms plus an evasion set that defeated 27 of 31 first-draft probes.
PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"\.get\(\s*[^)]*?,\s*{_Q}", "dict.get() with an encoding-name default"),
    (rf"\.setdefault\(\s*[^)]*?,\s*{_Q}", "dict.setdefault() with an encoding-name default"),
    (rf"\.pop\(\s*[^)]*?,\s*{_Q}", "dict.pop() with an encoding-name default"),
    (rf"\bgetattr\(\s*[^)]*?,\s*{_Q}", "getattr() with an encoding-name default"),
    # `or X` / `else X` are fallback positions BY CONSTRUCTION, so a wrapped literal there is
    # still a fallback; a bare `return lookup(...)` inside an affirmative guard is not.
    (rf"\bor\s+{_CALL}{_Q}", "`or <encoding>` fallback"),
    (rf"\belse\s+{_CALL}{_Q}", "ternary/else fallback to an encoding"),
    (rf"\breturn\s+\(?\s*{_Q}\s*\)?\s*{_END}", "terminal `return <encoding>` fallback"),
    (rf"\breturn\s+{_Q}\s+if\b", "conditional `return <encoding> if ...` fallback"),
    (rf"\bdefault(?:_factory)?\s*=\s*{_DECL_CALL}{_Q}",
     "explicit `default=<encoding>` (argparse / dataclass / pydantic)"),
    (rf"[{{,]\s*['\"][^'\"]*['\"]\s*:\s*{_Q}\s*{_END}",
     "dict-literal value defaulting to an encoding"),
    (rf"\bunwrap_or\(\s*{_Q}", "Rust unwrap_or() with an encoding-name default"),
    (rf"\bunwrap_or_else\(\s*\|\|\s*\{{?\s*{_Q}", "Rust unwrap_or_else() with an encoding default"),
    (rf"\bmap_or(?:_else)?\(\s*{_CALL}{_Q}",
     "Rust map_or()/map_or_else() with an encoding default"),
    (rf"=>\s*{_CALL}{_Q}", "Rust match arm defaulting to an encoding"),
    # The REPRESENTATION default is the same defect one level up, re-introducing dense-by-default
    # at a site the schema's refusal cannot reach. It sat at SIX places in `src/` while every
    # layer above them said absent-is-an-error.
    (r"getattr\(\s*[^)]*?,\s*[\"']representation[\"']\s*,\s*[\"'](?:grid|graph|dense)[\"']",
     "getattr() defaulting a representation"),
    (r"\.(?:get|setdefault|pop)\(\s*[\"']representation[\"']\s*,\s*[\"'](?:grid|graph|dense)[\"']",
     "dict read defaulting a representation"),
)

# Known, owned, still-open arms — NOT an escape hatch, matched on exact source text so a silent
# rewrite cannot inherit the exemption. EMPTY: a returning ternary is now a hard VIOLATION rather
# than a printed debt row.
KNOWN_DEBT: tuple[tuple[str, str, str], ...] = ()

ESCAPE = re.compile(r"silent-encoding-gate:\s*ok\s*--\s*\S")
_COMPILED = tuple((re.compile(p), why) for p, why in PATTERNS)

# Comment syntax is per-language, and getting it wrong is not cosmetic: treating any line starting
# with `#` as a comment silently blanked every Rust `#[pyo3(...)]` ATTRIBUTE — exactly where the
# pyo3 signature defaults live — and the gate reported green over two real arms.
_PY_COMMENT_ONLY = re.compile(r"^\s*#(?!\[)")
_RS_COMMENT_ONLY = re.compile(r"^\s*//")
_PY_STRIP = re.compile(r"(?<!['\"])\s+#(?!\[).*$")
_RS_STRIP = re.compile(r"(?<!['\"])\s+//.*$")


def _comment_res(suffix: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    if suffix == ".rs":
        return _RS_COMMENT_ONLY, _RS_STRIP
    return _PY_COMMENT_ONLY, _PY_STRIP


def _is_justified(lines: list[str], idx: int, suffix: str = ".py") -> bool:
    """True if line `idx` carries an escape, or the comment block directly above it does. The
    whole contiguous block is searched, because a marker that only works on one-line comments
    quietly punishes the explanations that are actually useful."""
    comment_only, _ = _comment_res(suffix)
    if ESCAPE.search(lines[idx]):
        return True
    j = idx - 1
    while j >= 0 and comment_only.match(lines[j]):
        if ESCAPE.search(lines[j]):
            return True
        j -= 1
    return False


_TRIPLE = re.compile(r'"""|\'\'\'')
_STR_SPAN = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""")
_MAX_JOIN = 600  # safety valve: no real construct is this long


def _bracket_delta(code: str) -> int:
    """Return the net bracket depth of *code*, ignoring brackets inside string literals: counting
    brackets in raw text let a `(` inside a docstring unbalance the joiner, which then swallowed
    a whole module into one logical line."""
    bare = _STR_SPAN.sub("''", code)
    return (bare.count("(") - bare.count(")")) + (bare.count("[") - bare.count("]"))


def _logical_lines(lines: list[str], suffix: str = ".py") -> list[tuple[int, str]]:
    """Join bracket-continued physical lines, and strip comments and docstrings.

    Two evasions die here: `return "v6"  # comment`, which walked through a pattern anchored on
    end-of-line, and a `.get(` call split across lines, which never matched. Triple-quoted blocks
    are skipped entirely, and the reported line number is where the construct STARTS.
    """
    comment_only, strip = _comment_res(suffix)
    out: list[tuple[int, str]] = []
    buf, start, depth = "", 0, 0
    in_doc = False

    for i, raw in enumerate(lines):
        if suffix != ".rs":
            marks = len(_TRIPLE.findall(raw))
            if in_doc:
                if marks % 2:
                    in_doc = False
                continue
            if marks % 2:
                in_doc = True
                continue

        code = "" if comment_only.match(raw) else strip.sub("", raw)
        if not buf:
            start = i
        buf = f"{buf} {code.strip()}" if buf else code.strip()
        depth += _bracket_delta(code)
        if depth <= 0 or len(buf) > _MAX_JOIN:
            if buf:
                out.append((start, buf))
            buf, depth = "", 0
    if buf:
        out.append((start, buf))
    return out


_DEF_CTX = re.compile(r"^\s*(?:async\s+)?(?:def\b|class\b)|#\[pyo3\(signature|\bfn\s+\w+\s*\(")
_ASSIGN = re.compile(rf"(?<![=!<>])=\s*{_DECL_CALL}{_Q}")


def _assignment_default_hit(logical: str) -> bool:
    """True if *logical* binds an encoding literal as a DEFAULT rather than passing one.

    `def f(encoding="v6")` declares a default while `f(encoding="v6")` is a caller stating it
    explicitly, and the two are nearly identical as text; the discriminator is context — a
    signature, or a binding at bracket depth 0.
    """
    m = _ASSIGN.search(logical)
    if m is None:
        return False
    if _DEF_CTX.search(logical):
        return True
    return _bracket_delta(logical[: m.start()]) <= 0


def line_hit(logical: str) -> str | None:
    """THE decision: why this logical line is a silent fallback, or None. Both `scan()` and the
    corpus tests go through this one function — an earlier oracle replayed `_COMPILED` itself and
    so could not see the context-aware assignment check."""
    for rx, why in _COMPILED:
        if rx.search(logical):
            return why
    if _assignment_default_hit(logical):
        return "assignment / signature default binding an encoding"
    return None


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
            why = line_hit(logical)
            if why is not None:
                if _is_justified(lines, idx, path.suffix):
                    continue
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
        # Loud on every run: registered debt that stops being visible stops being debt.
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
