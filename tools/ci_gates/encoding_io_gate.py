#!/usr/bin/env python3
# >300 justify: the per-function positional table, the binary-mode discriminator, the
# self-expiring exemption register and the two scoped rules are one gate's single authority;
# splitting them would create a second place where "what counts as encoding-less I/O" is
# decided, which is the drift this gate exists to remove (LAW-03).
"""CI gate 16 (P0-05): no encoding-less text I/O where it can break a run.

`open()`, `Path.read_text()` and `Path.write_text()` default to `locale.getpreferredencoding()`.
On Linux CI that is UTF-8, so this class is invisible there. On a Windows checkout it is the ANSI
codepage (cp1252 here), and reading any UTF-8 file with a non-ASCII byte raises UnicodeDecodeError.

THIS REPO HAS BEEN BITTEN THREE TIMES BY IT:

  1. tools/check_import_dag.py:107        -- CI gate 9, crashed on any non-ASCII source
  2. tools/ci_gates/check_tracked_refs.py -- CI gate 10, same
  3. tests/train/test_anchor.py:28 and tests/train/test_warmstart.py:22 -- at MODULE scope, so
     pytest died with `Interrupted: 2 errors during collection` and the ENTIRE default tier
     did not run on Windows (S-19)

The trigger is the repo's own house style: section dividers like `# == ... ==` are U+2550, which
is undefined in cp1252. `src/mantis/train/anchor.py` carries 621 non-ASCII bytes.

TWO RULES, each with a measured-clean baseline (R98: no gate over a dirty baseline):

  * `tools/`  -- ZERO encoding-less text I/O anywhere. These are the gates themselves; they must
    not fail on the platform they are meant to protect. Baseline measured at 0 after P0-05.
  * `tests/`  -- ZERO at MODULE SCOPE. A module-scope failure is COLLECTION-fatal: it takes down
    the whole tier, not one test. Baseline measured at 0 (one registered exemption, below).

NOT a rule: function-scope calls in `tests/`. 232 of them exist. They are a real backlog, but
`tests/` is deliberately edit-averse (pyproject.toml:77-81, frozen-oracle discipline), a 232-file
mechanical diff would bury the 6 real fixes it contains, and each one fails at most its own test
rather than the run. Registered in tmp/plan/FOUND.md, not silently forgotten.

NOT scanned: `src/`. 31 sites exist there and they are production paths, but they are a separate
adjudication with a different risk profile (a run that dies mid-training, not a gate that cannot
start) and P0-05's mandate is the collection blocker. Named here so the omission is deliberate
and visible rather than an oversight.

WHAT COUNTS AS SAFE, and why each matters:
  * binary mode (`"rb"`, `mode="wb"`) -- takes no `encoding` at all; demanding one would be
    wrong, and a gate that fires on correct code gets disabled within a week
  * `encoding=` as a keyword
  * `encoding` passed POSITIONALLY -- the index differs per function, hence POSITIONAL_ENCODING
  * `**kwargs` forwarded -- absence cannot be proven statically, so it is not claimed

ESCAPE HATCH -- for a call that genuinely is not file text I/O (`zipfile.ZipFile.open`, a mock,
a custom `.open()`), or where the platform encoding is deliberately wanted:

    with p.open("w") as fh:  # encoding-gate: ok -- <why this needs no encoding>

The reason text is mandatory.

KNOWN OVER-APPROXIMATION: any `.open()` is treated as file text I/O, because the receiver's type
is not reliably decidable statically. Measured at P0-05: of 13 `.open()` attribute calls across
`tools/`, `tests/` and `src/`, every one inside the GATED scope has a Path-like receiver, so this
costs zero false positives today. The one true counter-example is `os.open(path, flags)` in
`src/` (fd-level, takes no encoding) -- out of scope, and it would need the hatch if `src/` is
ever added. Narrowing this would buy precision with false NEGATIVES, which is the wrong trade for
a class that fails silently. Pinned by tests/tools/test_encoding_io_gate.py.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (function, is_method) -> index at which `encoding` appears POSITIONALLY, or None.
#: The builtin and the Path method DIFFER BY ONE because `Path.open` has no `file` parameter:
#:   open(file, mode, buffering, encoding, ...)      -> encoding 3, mode 1
#:   Path.open(mode, buffering, encoding, ...)       -> encoding 2, mode 0
#:   Path.read_text(encoding, errors, newline)       -> encoding 0, no mode
#:   Path.write_text(data, encoding, errors, ...)    -> encoding 1, no mode
#: Collapsing these into one table (the first draft did) makes `p.open("rb")` look like text
#: mode and flags correct binary code -- the false-positive class that gets a gate disabled --
#: and makes `p.open("r", -1, "utf-8")` look encoding-less. Both directions are wrong, so the
#: builtin and the method are keyed separately.
POSITIONAL_ENCODING: dict[tuple[str, bool], int | None] = {
    ("open", False): 3,
    ("open", True): 2,
    ("read_text", True): 0,
    ("write_text", True): 1,
}
POSITIONAL_MODE: dict[tuple[str, bool], int | None] = {
    ("open", False): 1,
    ("open", True): 0,
    ("read_text", True): None,
    ("write_text", True): None,
}

ESCAPE = "encoding-gate: ok --"

#: Registered, owned exemptions. NOT an escape hatch: each asserts "this IS a real site, it is
#: tracked, and it cannot be fixed here". Matched on exact source text, so an entry that stops
#: matching FAILS the gate -- an exemption can never be silently inherited by whatever replaced
#: the line it named. (Shape borrowed from gate 11's KNOWN_DEBT, which earned it.)
EXEMPT: tuple[tuple[str, str, str], ...] = (
    (
        "tests/tools/test_preflight_mint.py",
        "TOOL_SOURCE = TOOL_PATH.read_text()",
        "byte-frozen oracle (tests/tools/conftest.py:3,15 -- 'editing it is an R43 event'). "
        "The read targets tools/ci_gates/preflight_mint.py, which is currently cp1252-decodable, "
        "so it does not fail today; it is one non-ASCII byte away from collection-fatal. "
        "Fix belongs to whoever lifts the freeze.",
    ),
)

#: Non-vacuity floors. A gate that scans nothing finds nothing. Set below the measured counts
#: (tools/ 13, tests/ 254 at P0-05) with headroom for deletions, but high enough that a broken
#: glob or a wrong REPO_ROOT cannot pass silently. The first draft used 15 for tools/ and the
#: gate correctly refused itself -- which is the behaviour these floors exist to produce.
MIN_FILES = {"tools": 10, "tests": 200}


def _call_key(node: ast.Call) -> tuple[str, bool] | None:
    """(name, is_method) for a call we care about, else None."""
    if isinstance(node.func, ast.Name):
        return (node.func.id, False)
    if isinstance(node.func, ast.Attribute):
        return (node.func.attr, True)
    return None


def _call_name(node: ast.Call) -> str | None:
    key = _call_key(node)
    return key[0] if key else None


def _const_str(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def is_binary_mode(node: ast.Call, key: tuple[str, bool]) -> bool:
    """True if this call opens in binary mode, which correctly takes no `encoding`."""
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = _const_str(kw.value)
            return mode is not None and "b" in mode
    idx = POSITIONAL_MODE.get(key)
    if idx is not None and len(node.args) > idx:
        mode = _const_str(node.args[idx])
        return mode is not None and "b" in mode
    return False


def has_encoding(node: ast.Call, key: tuple[str, bool]) -> bool:
    """True if `encoding` is supplied by keyword, positionally, or possibly via **kwargs."""
    for kw in node.keywords:
        if kw.arg == "encoding":
            return True
        if kw.arg is None:  # **kwargs: absence is not provable, so do not claim it
            return True
    idx = POSITIONAL_ENCODING.get(key)
    return idx is not None and len(node.args) > idx


def is_unsafe(node: ast.Call) -> bool:
    """THE decision. Both the scan and its producer test go through this one function."""
    key = _call_key(node)
    if key is None or key not in POSITIONAL_ENCODING:
        return False
    if is_binary_mode(node, key):
        return False
    return not has_encoding(node, key)


def _module_scope_lines(tree: ast.Module) -> set[int]:
    """Lines that execute at IMPORT time, i.e. not nested in any def/class."""
    nested: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                lineno = getattr(sub, "lineno", None)
                if lineno is not None:
                    nested.add(lineno)
    return nested


def _justified(lines: list[str], lineno: int) -> bool:
    """Escape on the line itself, or anywhere in the comment block directly above it."""
    if ESCAPE in lines[lineno - 1]:
        return True
    j = lineno - 2
    while j >= 0 and lines[j].lstrip().startswith("#"):
        if ESCAPE in lines[j]:
            return True
        j -= 1
    return False


def scan() -> tuple[list[str], dict[str, int], set[int]]:
    """Return (violations, files_scanned_per_root, indices_of_matched_exemptions)."""
    violations: list[str] = []
    scanned = {root: 0 for root in MIN_FILES}
    matched_exempt: set[int] = set()

    for root, module_scope_only in (("tools", False), ("tests", True)):
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            scanned[root] += 1
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                violations.append(f"{rel}: unparseable ({exc})")
                continue
            lines = source.splitlines()
            nested = _module_scope_lines(tree)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not is_unsafe(node):
                    continue
                if module_scope_only and node.lineno in nested:
                    continue  # function-scope in tests/: registered backlog, not a violation
                if _justified(lines, node.lineno):
                    continue
                text = lines[node.lineno - 1].strip()
                exempt_i = next(
                    (k for k, (p, snippet, _r) in enumerate(EXEMPT) if p == rel and snippet in text),
                    None,
                )
                if exempt_i is not None:
                    matched_exempt.add(exempt_i)
                    continue
                where = "module scope" if module_scope_only else "tools/"
                violations.append(
                    f"{rel}:{node.lineno}: encoding-less {_call_name(node)}() at {where}\n"
                    f"    {text}"
                )

    return violations, scanned, matched_exempt


def main() -> int:
    violations, scanned, matched_exempt = scan()
    rc = 0

    for root, floor in MIN_FILES.items():
        if scanned[root] < floor:
            print(
                f"gate 16 FAIL -- scanned only {scanned[root]} file(s) under {root}/ "
                f"(floor {floor}). A gate that scans nothing finds nothing; refusing to "
                "report green."
            )
            rc = 1

    stale = [EXEMPT[k][0] for k in range(len(EXEMPT)) if k not in matched_exempt]
    if stale:
        print(
            "gate 16 FAIL -- EXEMPT entries matched nothing (the code moved under them; "
            f"re-adjudicate rather than editing the register): {stale}"
        )
        rc = 1

    if violations:
        print("\ngate 16 FAIL -- encoding-less text I/O (S-19; breaks on any non-UTF-8 locale):\n")
        print("\n".join(violations))
        print(
            "\nPass `encoding=\"utf-8\"` explicitly. Every one of this repo's 639 tracked text "
            "files is UTF-8, so utf-8 is always the right answer here.\n"
            "If the call is genuinely not file text I/O (zipfile, a mock, a custom .open), say so "
            "in place:\n"
            '    # encoding-gate: ok -- <why>\n'
            "If it is a real site that cannot be fixed yet, it goes in EXEMPT with grounds -- "
            "never in the escape hatch."
        )
        rc = 1

    if rc == 0:
        total = sum(scanned.values())
        print(
            f"gate 16: no encoding-less text I/O in tools/, none at module scope in tests/ "
            f"({total} files; {len(EXEMPT)} registered exemption(s))"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
