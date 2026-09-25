#!/usr/bin/env python3
"""CI gate 16: no encoding-less text I/O anywhere in the tree.

`open()`, `read_text()` and `write_text()` default to the platform codepage, so a non-ASCII UTF-8
file raises UnicodeDecodeError off a UTF-8 locale — invisible on Linux CI. ZERO over every
tracked `.py` file, module scope and function scope alike. Safe: binary mode, `encoding=` by
keyword or position, a forwarded `**kwargs`, `os.open` (an fd call whose second argument is
flags, not mode, and which takes no encoding), or `# encoding-gate: ok -- <why>` whose reason
text is MANDATORY. Any `.open()` counts, since the receiver's type is not statically decidable.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (function, is_method) -> the POSITIONAL index of `encoding`. `Path.open` has no `file` parameter, so
#: builtin and method differ by one: open(file, mode, buffering, encoding), Path.open(mode, buffering, encoding).
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

#: Compiled, not a substring: the trailing `\S` is what makes the mandatory reason text
#: mandatory, since a bare `# encoding-gate: ok --` would otherwise silence a site.
ESCAPE_TOKEN = "encoding-gate: ok --"
ESCAPE = re.compile(re.escape("encoding-gate: ok") + r"\s*--\s*\S")

#: Registered exemptions, each a real tracked site that cannot be fixed here. Matched on exact
#: source text, so an entry that stops matching FAILS the gate rather than being inherited.
EXEMPT: tuple[tuple[str, str, str], ...] = (
    (
        "tests/tools/test_preflight_mint.py",
        "TOOL_SOURCE = TOOL_PATH.read_text()",
        "byte-frozen oracle over tools/ci_gates/preflight_mint.py, which is currently "
        "cp1252-decodable, so it does not fail today; it is one non-ASCII byte away from red. "
        "Fix belongs to whoever lifts the freeze.",
    ),
)

#: Non-vacuity floor over the whole tracked tree: high enough that a broken glob or a wrong
#: REPO_ROOT cannot pass silently, with headroom for deletions.
MIN_FILES = {"tree": 700}


def _call_key(node: ast.Call) -> tuple[str, bool] | None:
    """Return `(name, is_method)` for a call we care about, else None."""
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
    """Report whether this call opens in binary mode, which correctly takes no `encoding`."""
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
    """Report whether `encoding` is supplied by keyword, positionally, or via `**kwargs`."""
    for kw in node.keywords:
        if kw.arg == "encoding":
            return True
        if kw.arg is None:  # **kwargs: absence is not provable, so do not claim it
            return True
    idx = POSITIONAL_ENCODING.get(key)
    return idx is not None and len(node.args) > idx


def is_unsafe(node: ast.Call) -> bool:
    """Decide whether a call is encoding-less text I/O; the scan and its producer test share it."""
    key = _call_key(node)
    if key is None or key not in POSITIONAL_ENCODING:
        return False
    if is_binary_mode(node, key):
        return False
    return not has_encoding(node, key)


def _is_os_open(node: ast.Call) -> bool:
    """`os.open` takes flags, not mode, and no encoding — flagging it would demand a TypeError."""
    return (isinstance(node.func, ast.Attribute) and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "os")


def _justified(lines: list[str], lineno: int) -> bool:
    """Report whether an escape sits on the line or in the comment block directly above it."""
    if ESCAPE.search(lines[lineno - 1]):
        return True
    j = lineno - 2
    while j >= 0 and lines[j].lstrip().startswith("#"):
        if ESCAPE.search(lines[j]):
            return True
        j -= 1
    return False


def scan() -> tuple[list[str], dict[str, int], set[int]]:
    """Return the violations, the files scanned, and the matched exemption indices."""
    violations: list[str] = []
    scanned = {root: 0 for root in MIN_FILES}
    matched_exempt: set[int] = set()

    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO_ROOT, capture_output=True, check=True)
    for raw in listed.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        path = REPO_ROOT / raw
        rel = raw.replace("\\", "/")
        scanned["tree"] += 1
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(f"{rel}: unparseable ({exc})")
            continue
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not is_unsafe(node):
                continue
            if _is_os_open(node):
                continue
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
            violations.append(
                f"{rel}:{node.lineno}: encoding-less {_call_name(node)}()\n"
                f"    {text}"
            )

    return violations, scanned, matched_exempt


def main() -> int:
    violations, scanned, matched_exempt = scan()
    rc = 0

    for root, floor in MIN_FILES.items():
        if scanned[root] < floor:
            print(
                f"gate 16 FAIL -- scanned only {scanned[root]} tracked .py file(s) {root}-wide "
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
            "\nPass `encoding=\"utf-8\"` explicitly. Every tracked text file in this repo is UTF-8,\n"
            "so utf-8 is always the right answer here.\n"
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
            f"gate 16: no encoding-less text I/O anywhere in the tree "
            f"({total} tracked .py files; {len(EXEMPT)} registered exemption(s))"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
