#!/usr/bin/env python3
# >300 justify: the pattern register, the reserved-domain carve-outs, the self-expiring
# exemption table and the two scan modes are one gate's single authority for "what counts as
# host content". Splitting them would create a second place where that question is answered,
# which is the drift this gate exists to remove; the register and the carve-outs are only
# reviewable side by side, because every carve-out exists to keep one named pattern honest.
"""CI gate 17 (R281(c)): no host content in the tracked tree (rule 7).

Rule 7 says box specifics live in the migration workspace, never in this repo. It was enforced
by hand -- a grep the dispatcher remembered to run -- until the R280(c) scan ran it against
`origin/dev` itself and found the rule already broken:

  tests/fixtures/bf16_nulldist/measurement_raw_R181_NULLDIST.json carried 101 absolute box
  paths across 3 directories, written by the capture tool at measurement time and committed
  with the fixture. It was the ONLY file in the tree that matched, it had been public since
  the fixture landed, and nothing in CI could see it.

A rule with no gate is a rule that holds only while someone remembers it. This is the gate.
The blob above was sanitized in the commit BEFORE this one, so this gate adopts over a
genuinely clean tree (R98: no gate over a dirty baseline) and its EXEMPT register ships EMPTY.

WHAT IS AND IS NOT IN THE PATTERN REGISTER, and why the line is drawn there.

IN: structural shapes (absolute home paths, ssh invocation and config keywords, `user@host`,
IPv4) and PUBLIC provider names. None of these identifies the operator; all of them identify a
machine, an account or a vendor, which is what rule 7 is about.

OUT, deliberately: the operator's own name, handle, email and host aliases. Writing those into
a TRACKED gate would leak exactly what the gate exists to keep out -- the gate file becomes the
disclosure. This is the WP0 precedent, where the equivalent term list lived in an UNTRACKED
`coupling_terms.txt` for the same reason. The hook for that is LOCAL_TERMS below: an untracked
newline-delimited regex file, read when present, absent in CI. The tracked half of this gate is
therefore a FLOOR, not a ceiling, and that is a deliberate, disclosed limit rather than an
oversight.

RESERVED DOMAINS ARE NOT HOSTS. `gate@test.invalid` and `gate3c@example.invalid` are live in
tests/tools/; RFC 2606 / RFC 6761 reserve `.invalid`, `.test`, `.example` and `.localhost`
precisely so they can never resolve. A `user@host` pattern that fires on them is reporting a
non-host, and a gate that fires on correct code gets disabled within a week -- so the carve-out
is in the PATTERN, not in EXEMPT. Loopback and unspecified IPv4 are carved out for the same
reason: they name no machine.

ESCAPE HATCH -- for a site that must name a pattern without being one (this file's own register,
a provenance note recording what was normalized):

    <comment> rule7-gate: ok -- <why this is not host content>

The reason text is mandatory. It works on the line itself or in the comment block above it.

TWO MODES:
  --base <ref>   files ADDED or MODIFIED relative to <ref>. The PR mode: it is what keeps new
                 host content out without asking anyone to re-clean history.
  --full-tree    every tracked text file. The adoption mode, and the one that catches a blob
                 that predates the gate. Run it at adoption and whenever the register changes.

Binary files are skipped (no text to leak, and decoding them produces noise). Pinned by
tests/tools/test_rule7_gate.py; the LAW-07 self-test below runs on EVERY invocation.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ESCAPE = "rule7-gate: ok --"

#: Untracked local supplement: one regex per line, `#` comments ignored. Operator-identifying
#: terms (name, handle, email, host aliases) belong HERE and never in this file. Absent in CI
#: by construction -- `.gitignore`d -- so the tracked gate must stand on its own.
LOCAL_TERMS = REPO_ROOT / "tools" / "ci_gates" / "rule7_local_terms.txt"

#: THE REGISTER. name -> (regex, what a hit means).
#: Every entry below names a shape, never a person. See the module docstring on where the line
#: is drawn and why the operator-identifying half is deliberately absent.
PATTERNS: dict[str, tuple[str, str]] = {
    # rule7-gate: ok -- the four entries below DEFINE the patterns; they are the register, not host content
    "abs-root-path": (
        r"/root/",
        "absolute superuser home path from a box capture",
    ),
    "abs-home-path": (
        r"/home/[A-Za-z_][A-Za-z0-9_.-]*",
        "absolute user home path -- names an account on a machine",
    ),
    "detached-run": (
        r"\bnohup\b",
        "detached box-run invocation",
    ),
    "box-outdir": (
        r"\bshakedown_out\b",
        "box run output directory",
    ),
    "ssh-userhost": (
        # Reserved-by-RFC domains carved out IN THE PATTERN: they can never resolve, so a hit
        # on one is a false positive by construction. See the docstring.
        # The reserved label must be the FINAL one: anchoring it with `\b` instead let
        # `example-provider.net` read as reserved, because `e`->`-` IS a word boundary.
        # The self-test caught exactly that; `(?![a-z0-9-])` is what makes it a TLD test.
        r"\b[a-z_][a-z0-9_-]*@"
        r"(?!(?:[a-z0-9-]+\.)*(?:invalid|test|example|localhost)(?![a-z0-9-]))"
        r"(?:[a-z0-9-]+\.)+[a-z]{2,}\b",
        "user@host -- an account on a named machine",
    ),
    "ssh-invocation": (
        r"\bssh\s+-i\b|\bscp\s+-\w+\s|\bssh\s+[a-z0-9_-]+@",
        "ssh/scp invocation against a box",
    ),
    "ssh-config": (
        r"\bIdentityFile\b|\bknown_hosts\b|\bPermitRootLogin\b|\bStrictHostKeyChecking\b",
        "ssh client/server configuration",
    ),
    "provider": (
        r"\bvast\.ai\b|\bamazonaws\b|\bngrok\b|\brunpod\b|\blambdalabs\b|\bpaperspace\b"
        r"|\bcoreweave\b",
        "compute-provider name -- says where the box was rented",
    ),
    "ipv4": (
        # Loopback / unspecified / broadcast name no machine, so they are carved out here
        # rather than in EXEMPT. Octets are range-checked so version strings do not match.
        r"\b(?!0\.0\.0\.0\b)(?!127\.0\.0\.1\b)(?!255\.255\.255\.255\b)"
        r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
        "IPv4 address",
    ),
}

#: Registered, owned exemptions: (path, matched-substring, blob sha256, grounds).
#: NOT an escape hatch -- each asserts "this IS host content, it is tracked, and it cannot be
#: removed here". SELF-EXPIRING TWO WAYS: an entry whose substring stops matching FAILS the
#: gate, and so does one whose recorded sha no longer matches the file, so an exemption can
#: never be inherited by whatever replaced the content it named. (Shape from gate 16's EXEMPT,
#: which earned it; the sha half is R281(c)'s addition.)
#:
#: SHIPS EMPTY, and that is the point: the one known blob (the R181 nulldist fixture) was
#: sanitized in the preceding commit rather than exempted here, so this gate has never been
#: green over a dirty tree.
EXEMPT: tuple[tuple[str, str, str, str], ...] = ()

#: Non-vacuity floor for --full-tree. A gate that scans nothing finds nothing. Set well below
#: the measured tracked-text count with headroom for deletion, but high enough that a broken
#: `git ls-files` or a wrong REPO_ROOT cannot pass silently.
MIN_FULL_TREE_FILES = 400


_COMPILED: list[tuple[str, re.Pattern[str], str]] | None = None


def _compiled() -> list[tuple[str, re.Pattern[str], str]]:
    """The register plus any untracked local supplement, compiled ONCE per process."""
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED
    out = [(name, re.compile(pat), why) for name, (pat, why) in PATTERNS.items()]
    if LOCAL_TERMS.is_file():
        for i, line in enumerate(LOCAL_TERMS.read_text(encoding="utf-8").splitlines(), 1):
            term = line.strip()
            if term and not term.startswith("#"):
                out.append((f"local:{LOCAL_TERMS.name}:{i}", re.compile(term), "local term"))
    _COMPILED = out
    return out


def _justified(lines: list[str], lineno: int) -> bool:
    """Escape on the line itself, or anywhere in the comment block directly above it."""
    if ESCAPE in lines[lineno - 1]:
        return True
    j = lineno - 2
    while j >= 0 and lines[j].lstrip()[:1] in ("#", "/", ";", "-") and lines[j].strip():
        if ESCAPE in lines[j]:
            return True
        j -= 1
    return False


def scan_text(rel: str, text: str) -> list[tuple[str, int, str, str, str]]:
    """THE DECISION. Both the scan and the LAW-07 self-test go through this one function.

    Returns (rel, lineno, pattern_name, matched_text, why) per unjustified hit.
    """
    lines = text.splitlines()
    hits: list[tuple[str, int, str, str, str]] = []
    for name, rx, why in _compiled():
        for lineno, line in enumerate(lines, 1):
            for m in rx.finditer(line):
                if _justified(lines, lineno):
                    continue
                hits.append((rel, lineno, name, m.group(0), why))
    return hits


def _read_text(path: Path) -> str | None:
    """Decoded text, or None for binary/undecodable (nothing to leak, decoding is noise)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def target_files(base: str | None) -> list[str]:
    """Tracked text files to scan: the whole tree, or those added/modified vs `base`."""
    if base is None:
        return [p for p in _git("ls-files").splitlines() if p]
    out = _git("diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD").splitlines()
    tracked = set(_git("ls-files").splitlines())
    return [p for p in out if p and p in tracked]


def self_test() -> bool:
    """LAW-07: the gate must be able to FIRE. Runs on EVERY invocation (lint_gate's posture).

    Plants one violation per pattern class in a temp file and drives the REAL decision
    function. A gate whose trigger cannot fire is a phantom input -- LAW-07's own class.
    """
    # rule7-gate: ok -- planted fixtures for the self-test; they name no real machine
    planted = [
        ("abs-root-path", "repo_path = /root/hexo-mantis"),
        ("abs-home-path", "cache at /home/operator/.cache"),
        ("detached-run", "nohup python -m mantis.run &"),
        ("box-outdir", "outdir = shakedown_out/run5"),
        ("ssh-userhost", "target: boxuser@rented-gpu.example-provider.net"),
        ("ssh-invocation", "scp -r results/ boxhost:/tmp/out"),
        ("ssh-config", "IdentityFile ~/.ssh/id_ed25519"),
        ("provider", "rented from vast.ai for the burn"),
        ("ipv4", "endpoint 203.0.113.7 responded"),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for name, line in planted:
            f = Path(td) / "planted.txt"
            f.write_text(line + "\n", encoding="utf-8")
            hits = scan_text("planted.txt", f.read_text(encoding="utf-8"))
            if not any(h[2] == name for h in hits):
                print(f"gate 17 SELF-TEST FAIL: pattern {name!r} did not fire on {line!r}")
                ok = False
        # Negative controls: the gate must NOT fire on these, or it gets disabled within a week.
        for line in ("contact gate@test.invalid", "bind 127.0.0.1", "listen on 0.0.0.0",
                     "torch 2.11.0+cu128"):
            f = Path(td) / "control.txt"
            f.write_text(line + "\n", encoding="utf-8")
            if scan_text("control.txt", f.read_text(encoding="utf-8")):
                print(f"gate 17 SELF-TEST FAIL: false positive on control {line!r}")
                ok = False
        # The hatch must actually suppress, or every register line below would red the gate.
        f = Path(td) / "hatched.txt"
        f.write_text("path = /root/x  # rule7-gate: ok -- fixture\n", encoding="utf-8")
        if scan_text("hatched.txt", f.read_text(encoding="utf-8")):
            print("gate 17 SELF-TEST FAIL: escape hatch did not suppress")
            ok = False
    # DERIVED, never transcribed: every registered class must have a planted proof, or the
    # self-test silently stops covering whatever was added to the register (LAW-07's own class).
    uncovered = set(PATTERNS) - {name for name, _line in planted}
    if uncovered:
        print(f"gate 17 SELF-TEST FAIL: registered pattern(s) with no planted proof: "
              f"{sorted(uncovered)}")
        ok = False
    if ok:
        print(f"gate 17 self-test: all {len(PATTERNS)} registered pattern classes fire, "
              f"5 controls clean")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="CI gate 17 -- no host content in the tree (rule 7)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="scan files added/modified relative to this ref")
    g.add_argument("--full-tree", action="store_true", help="scan every tracked text file")
    args = ap.parse_args()

    if not self_test():
        return 2

    files = target_files(None if args.full_tree else args.base)
    violations: list[tuple[str, int, str, str, str]] = []
    scanned = 0
    matched_exempt: set[int] = set()

    for rel in files:
        text = _read_text(REPO_ROOT / rel)
        if text is None:
            continue
        scanned += 1
        file_hits = scan_text(rel, text)
        # One `hash-object` per FILE, and only when the register is non-empty.
        blob = _git("hash-object", rel).strip() if (EXEMPT and file_hits) else ""
        for hit in file_hits:
            _, _lineno, _name, matched, _why = hit
            i = next(
                (k for k, (p, sub, sha, _r) in enumerate(EXEMPT)
                 if p == rel and sub in matched and sha == blob),
                None,
            )
            if i is not None:
                matched_exempt.add(i)
                continue
            violations.append(hit)

    rc = 0
    if args.full_tree and scanned < MIN_FULL_TREE_FILES:
        print(
            f"gate 17 FAIL -- scanned only {scanned} tracked text file(s) (floor "
            f"{MIN_FULL_TREE_FILES}). A gate that scans nothing finds nothing; refusing "
            "to report green."
        )
        rc = 1

    stale = [EXEMPT[k][0] for k in range(len(EXEMPT)) if k not in matched_exempt]
    if stale:
        print(
            "gate 17 FAIL -- EXEMPT entries matched nothing (the content moved under them; "
            f"re-adjudicate rather than editing the register): {stale}"
        )
        rc = 1

    if violations:
        print("\ngate 17 FAIL -- host content in the tracked tree (rule 7):\n")
        for rel, lineno, name, matched, why in violations:
            print(f"  {rel}:{lineno}: [{name}] {matched!r}\n      {why}")
        print(
            "\nBox specifics live in the migration workspace, never in this repo. If a capture "
            "tool wrote the path, fix the TOOL to normalize at write time (the convention) and "
            "sanitize the artifact.\nIf the site names a pattern without being one, say so in "
            f"place:\n    <comment> {ESCAPE} <why>\n"
            "If it is real host content that cannot be removed here, it goes in EXEMPT with "
            "grounds and the blob sha -- never in the escape hatch."
        )
        rc = 1

    if rc == 0:
        where = "tracked tree" if args.full_tree else f"files changed vs {args.base}"
        print(
            f"gate 17: no host content in the {where} ({scanned} text file(s); "
            f"{len(PATTERNS)} pattern class(es); {len(EXEMPT)} registered exemption(s))"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
