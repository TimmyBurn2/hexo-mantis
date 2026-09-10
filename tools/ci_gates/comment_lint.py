#!/usr/bin/env python3
# >300 justify: the three measures, the language scanners that feed them and the verdict they
# are compared against are one authority. Split them and "what counts as a comment line" gets
# decided in two places, which is the drift the ratchet exists to remove. The self-test corpus
# stays in-file so the arms and the predicate they prove move together.
"""Comment-length lint: the measures may fall and may never rise.

The comment rule is a ratchet, not a cap. A block longer than two lines is allowed when it
states a non-obvious invariant, so a hard cap would either red on legitimate text or need an
exemption list nobody maintains. What is enforced instead is direction: this tree's measures
may not exceed the committed floor, and the floor itself may not be raised.

MEASURES, all over tracked ``.py``/``.rs`` under ``src/``, ``tools/``, ``crates/``, ``tests/``:

  * ``comment_excess_lines``   lines beyond two in every run of own-line comments.
  * ``banner_comment_lines``   comment lines carrying a rule of eight or more repeated
    box-drawing or punctuation characters.
  * ``docstring_excess_lines`` lines beyond the first in every module/class/function docstring.

``ruling_cite_comment_lines`` is measured and PRINTED but never gated: the R8 justification
headers gate 15 requires carry the token ``R8``, so gating that count would set two gates
against each other.

Raises:
    SystemExit: rc 1 on a violation or a failed self-test, rc 2 on a usage or input refusal.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

CAP = 2
SCOPES = ("src/", "tools/", "crates/", "tests/")
FLOOR_FILE = "tools/ci_gates/comment_length_floor.txt"
MAIN_BRANCH = "dev"
GATED = ("comment_excess_lines", "banner_comment_lines", "docstring_excess_lines")

_BANNER = re.compile(r"([─-╿=#*~_+.<>-])\1{7,}")
_RULING = re.compile(
    r"\b(?:R\d{1,3}\([a-z]\)|LAW-\d\d|F-\d{2,3}|F-816-\d+|ADJ-[A-Z0-9-]+|RQ-\d+"
    r"|AUDIT-\d|WP[A-Z0-9]{2,}|CARD-[A-Z0-9-]+)"
)


@dataclass(frozen=True)
class Measures:
    """The four counts this lint derives from a tree."""

    comment_excess_lines: int = 0
    banner_comment_lines: int = 0
    docstring_excess_lines: int = 0
    ruling_cite_comment_lines: int = 0

    def __add__(self, other: Measures) -> Measures:
        return Measures(*(getattr(self, f) + getattr(other, f) for f in _FIELDS))


_FIELDS = (
    "comment_excess_lines", "banner_comment_lines",
    "docstring_excess_lines", "ruling_cite_comment_lines",
)


def rust_comment_spans(src: str) -> list[tuple[int, int, int, str]]:
    """Return (start line, start column, end line, text) per Rust comment; lines are 1-based."""
    out: list[tuple[int, int, int, str]] = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append((line, i - src.rfind("\n", 0, i) - 1, line, src[i:j]))
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            depth, j, start = 0, i, line
            while j < n:
                if src[j] == "\n":
                    line += 1
                    j += 1
                elif src[j] == "/" and j + 1 < n and src[j + 1] == "*":
                    depth += 1
                    j += 2
                elif src[j] == "*" and j + 1 < n and src[j + 1] == "/":
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            out.append((start, i - src.rfind("\n", 0, i) - 1, line, src[i:j]))
            i = j
            continue
        if c in "rb" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            j = i + 1 if c == "b" and i + 1 < n and src[i + 1] == "r" else i
            if src[j] == "r":
                k, hashes = j + 1, 0
                while k < n and src[k] == "#":
                    hashes, k = hashes + 1, k + 1
                if k < n and src[k] == '"':
                    term = '"' + "#" * hashes
                    e = src.find(term, k + 1)
                    e = n if e < 0 else e + len(term)
                    line += src.count("\n", i, e)
                    i = e
                    continue
        if c == '"' or (c == "b" and i + 1 < n and src[i + 1] == '"'):
            j = i + (2 if c == "b" else 1)
            while j < n:
                if src[j] == "\\":
                    j += 2
                elif src[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            line += src.count("\n", i, j)
            i = j
            continue
        if c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                j = i + 2
                while j < n and src[j] != "'":
                    j += 1
                i = j + 1
                continue
            if i + 2 < n and src[i + 2] == "'":
                i += 3
                continue
        i += 1
    return out


def _own_line(lines: list[str], row: int, col: int) -> bool:
    return lines[row - 1][:col].strip() == ""


def _excess(flags: list[bool]) -> int:
    total, run = 0, 0
    for f in [*flags, False]:
        if f:
            run += 1
        else:
            total += max(0, run - CAP)
            run = 0
    return total


def measure_source(rel: str, src: str) -> Measures:
    """Measure one file's comment and docstring load. Unparseable input measures as zero."""
    lines = src.split("\n")
    flags = [False] * (len(lines) + 1)
    texts: list[str] = []
    if rel.endswith(".py"):
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type != tokenize.COMMENT:
                    continue
                row, col = tok.start
                texts.append(tok.string)
                if _own_line(lines, row, col):
                    flags[row - 1] = True
        except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
            return Measures()
    else:
        for start, col, end, text in rust_comment_spans(src):
            texts.append(text)
            if _own_line(lines, start, col):
                for row in range(start, end + 1):
                    flags[row - 1] = True
    banner = sum(1 for t in texts for ln in t.split("\n") if _BANNER.search(ln))
    cites = sum(1 for t in texts for ln in t.split("\n") if _RULING.search(ln))
    docs = _docstring_excess(src) if rel.endswith(".py") else 0
    return Measures(_excess(flags), banner, docs, cites)


def _docstring_excess(src: str) -> int:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return 0
    total = 0
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        head = body[0]
        if isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant) \
                and isinstance(head.value.value, str):
            total += max(0, len(head.value.value.strip().split("\n")) - 1)
    return total


def in_scope(rel: str) -> bool:
    """True when a path is a lintable source file inside the four scoped directories."""
    return rel.endswith((".py", ".rs")) and rel.startswith(SCOPES)


def measure_tree(root: Path) -> tuple[Measures, int]:
    """Measure every tracked, in-scope file. Returns the totals and the file count."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    total, seen = Measures(), 0
    for rel in listed.stdout.decode("utf-8").split("\0"):
        if not in_scope(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        total = total + measure_source(rel, path.read_text(encoding="utf-8"))
        seen += 1
    return total, seen


def parse_floor(text: str) -> dict[str, int]:
    """Parse a floor file. Raises ValueError on a malformed or incomplete record."""
    out: dict[str, int] = {}
    for raw in text.split("\n"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in _FIELDS:
            raise ValueError(f"not a `<measure> <count>` record: {raw!r}")
        out[parts[0]] = int(parts[1])
    missing = [f for f in GATED if f not in out]
    if missing:
        raise ValueError(f"floor omits gated measure(s): {', '.join(missing)}")
    return out


def verdict(now: dict[str, int], tree: dict[str, int], ref: dict[str, int] | None,
            label: str) -> tuple[int, list[str]]:
    """Compare a measurement against the tree floor and the ref floor. rc 0 = clean."""
    msgs, rc = [], 0
    for name in GATED:
        if now[name] > tree[name]:
            rc = 1
            msgs.append(
                f"FAIL (grew): {name} is {now[name]}, floor is {tree[name]}. Comments grew. "
                f"Raising {FLOOR_FILE} is not the fix.")
        if ref is not None and tree[name] > ref[name]:
            rc = 1
            msgs.append(
                f"FAIL (ratchet): {name} floor is {tree[name]} here but {ref[name]} at "
                f"{label}. The floor may only fall.")
        if now[name] < tree[name]:
            msgs.append(
                f"note: {name} is {now[name]}, below the floor of {tree[name]} — "
                f"ratchet the floor down in this commit.")
    return rc, msgs


_PY_ARM = '''"""One
two
three"""
# a
# b
# c
# d
# ===========
x = 1  # trailing does not start a block
'''

_RS_ARM = """// a
// b
// c
//! ─────────────
fn f() { let s = \"// not a comment\"; }
"""


def self_test() -> int:
    """Prove both the measurement and the verdict can fire. rc 0 = the trigger is live."""
    bad = []
    m = measure_source("a.py", _PY_ARM)
    if m.comment_excess_lines != 3:
        bad.append(f"python block run: got {m.comment_excess_lines}, want 3")
    if m.banner_comment_lines != 1:
        bad.append(f"python banner: got {m.banner_comment_lines}, want 1")
    if m.docstring_excess_lines != 2:
        bad.append(f"python docstring: got {m.docstring_excess_lines}, want 2")
    r = measure_source("a.rs", _RS_ARM)
    if r.comment_excess_lines != 2:
        bad.append(f"rust block run: got {r.comment_excess_lines}, want 2")
    if r.banner_comment_lines != 1:
        bad.append(f"rust banner: got {r.banner_comment_lines}, want 1")
    if measure_source("a.py", "x = (\n").comment_excess_lines != 0:
        bad.append("unparseable python did not measure as zero")
    if measure_source("a.py", "# one\n# two\n").comment_excess_lines != 0:
        bad.append("a two-line block was counted as excess")
    if not in_scope("src/mantis/run.py") or in_scope("docs/x.py") or in_scope("src/a.md"):
        bad.append("scope predicate is wrong")

    flat = dict.fromkeys(_FIELDS, 10)
    for label, now, tree, ref, want in (
        ("clean equal", 10, 10, 10, 0),
        ("clean below", 5, 10, 10, 0),
        ("grew", 11, 10, 10, 1),
        ("floor raised", 10, 12, 10, 1),
        ("floor lowered", 8, 8, 10, 0),
    ):
        rc, _ = verdict({**flat, **dict.fromkeys(GATED, now)},
                        {**flat, **dict.fromkeys(GATED, tree)},
                        {**flat, **dict.fromkeys(GATED, ref)}, "self-test")
        if rc != want:
            bad.append(f"verdict arm {label!r}: rc {rc}, want {want}")
    try:
        parse_floor("comment_excess_lines 3\n")
    except ValueError:
        pass
    else:
        bad.append("an incomplete floor parsed clean")

    if bad:
        print("comment_lint SELF-TEST FAIL — the trigger cannot be trusted:", file=sys.stderr)
        for b in bad:
            print(f"    {b}", file=sys.stderr)
        return 1
    print("comment_lint self-test: every measurement and verdict arm fires")
    return 0


def resolve_ref(root: Path) -> str | None:
    """Return a git revision carrying the reference floor, or None on a bootstrap tree."""
    for rev in (f"refs/remotes/origin/{MAIN_BRANCH}", f"refs/heads/{MAIN_BRANCH}"):
        probe = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{rev}^{{commit}}"],
                               cwd=root, capture_output=True)
        if probe.returncode == 0:
            return rev
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the self-test, measure the tree, and issue the ratchet verdict."""
    ap = argparse.ArgumentParser(description="comment-length lint with a down-only ratchet")
    ap.add_argument("--self-test", action="store_true", help="run the arms and exit")
    ap.add_argument("--measure", action="store_true",
                    help="print the measurement in floor-file form and exit 0")
    ap.add_argument("--root", default=None, help="tree to measure (default: this repo)")
    args = ap.parse_args(argv)

    if self_test() != 0:
        return 1
    if args.self_test:
        return 0

    root = Path(args.root).resolve() if args.root else Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent,
                       capture_output=True, text=True, check=True).stdout.strip())
    now, seen = measure_tree(root)
    if seen == 0:
        print("comment_lint: REFUSING — no in-scope file was measured. A zero over an empty "
              "scope is a green over nothing.", file=sys.stderr)
        return 2
    measured = {f: getattr(now, f) for f in _FIELDS}
    if args.measure:
        for name in _FIELDS:
            print(f"{name} {measured[name]}")
        return 0

    floor_path = root / FLOOR_FILE
    if not floor_path.is_file():
        print(f"comment_lint: REFUSING — {FLOOR_FILE} is missing.", file=sys.stderr)
        return 2
    try:
        tree_floor = parse_floor(floor_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"comment_lint: REFUSING — {FLOOR_FILE}: {exc}", file=sys.stderr)
        return 2

    ref = resolve_ref(root)
    ref_floor = None
    if ref is not None:
        shown = subprocess.run(["git", "show", f"{ref}:{FLOOR_FILE}"], cwd=root,
                               capture_output=True, text=True)
        if shown.returncode == 0:
            try:
                ref_floor = parse_floor(shown.stdout)
            except ValueError:
                ref_floor = None
    if ref_floor is None:
        print(f"comment_lint: BOOTSTRAP — no floor at {ref or 'any ref'}; the ratchet half is "
              "NOT enforced this run.", file=sys.stderr)

    print(f"comment_lint: {seen} file(s) — " + "  ".join(
        f"{n}={measured[n]}/{tree_floor[n]}" for n in GATED))
    print(f"comment_lint: ruling-cite comment lines {measured['ruling_cite_comment_lines']} "
          "(measured, not gated)")
    rc, msgs = verdict(measured, tree_floor, ref_floor, ref or "<bootstrap>")
    for line in msgs:
        print(f"comment_lint: {line}", file=sys.stderr if rc else sys.stdout)
    if rc == 0:
        print("comment_lint: GREEN")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
