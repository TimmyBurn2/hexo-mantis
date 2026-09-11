#!/usr/bin/env python3
# >300 justify: the pattern register, the reserved-domain carve-outs, the self-expiring exemption
# table and the two scan modes are one gate's single authority for "what counts as host content".
# The register and the carve-outs are only reviewable side by side, because every carve-out exists
# to keep one named pattern honest.
"""CI gate 17: no host content in the tracked tree (rule 7).

rule7-gate: file-ok -- THIS FILE IS THE PATTERN REGISTER. Every host-shaped literal below is a
regex or a self-test fixture, never a real machine; the set of files allowed to say this is pinned
by tests/tools/test_rule7_gate.py.

Rule 7 was enforced by hand until a scan against `origin/dev` found it already broken: one
committed fixture carried 101 absolute box paths, public since the fixture landed, invisible to
CI. That blob was sanitized in the preceding commit, so this gate adopts over a clean tree and its
EXEMPT register ships EMPTY.

IN the register: structural shapes (absolute home paths, ssh invocation and config keywords,
`user@host`, IPv4) and PUBLIC provider names. OUT, deliberately: the operator's own name, handle,
email and host aliases, since writing those into a TRACKED gate would leak what the gate keeps
out — their hook is the untracked LOCAL_TERMS file, so the tracked half is a FLOOR, not a ceiling.
Reserved domains and loopback/unspecified IPv4 name no machine and are carved out IN THE PATTERN,
because a gate that fires on correct code gets disabled within a week.

The escape hatch is `<comment> rule7-gate: ok -- <why>`, reason text mandatory, on the line or in
the comment block above it. `--base <ref>` scans files added or modified relative to <ref> and
`--full-tree` scans every tracked text file; binary files are skipped, and the LAW-07 self-test
runs on EVERY invocation.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The escape token, COMPILED rather than a substring: `ESCAPE in line` accepted a bare
#: `# rule7-gate: ok --` with nothing after it, while the reason text is MANDATORY. The `\S` after
#: the dashes is what makes that sentence true.
ESCAPE_TOKEN = "rule7-gate: ok --"
ESCAPE = re.compile(re.escape("rule7-gate: ok") + r"\s*--\s*\S")

#: FILE-LEVEL hatch, for a file that is definitionally made of patterns, declared in the first
#: `FILE_ESCAPE_SCAN_LINES` lines. It hides a real leak in the file that carries it, so the gate
#: REPORTS the count in its green line and the producer test asserts the EXACT set of files
#: allowed to carry one.
FILE_ESCAPE = "rule7-gate: file-ok --"
FILE_ESCAPE_SCAN_LINES = 40

#: Untracked local supplement: one regex per line, `#` comments ignored. Operator-identifying
#: terms belong HERE and never in this file, and it is absent in CI by construction, so the
#: tracked gate must stand on its own.
LOCAL_TERMS = REPO_ROOT / "tools" / "ci_gates" / "rule7_local_terms.txt"

#: THE REGISTER. name -> (regex, what a hit means). Every entry names a shape, never a person.
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
        # Reserved-by-RFC domains carved out IN THE PATTERN: they can never resolve. The reserved
        # label must be the FINAL one — anchoring with `\b` let `example-provider.net` read as
        # reserved, because `e`->`-` IS a word boundary.
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
        # Loopback / unspecified / broadcast name no machine, so they are carved out here rather
        # than in EXEMPT. Octets are range-checked so version strings do not match.
        r"\b(?!0\.0\.0\.0\b)(?!127\.0\.0\.1\b)(?!255\.255\.255\.255\b)"
        r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
        "IPv4 address",
    ),
}

#: Registered, owned exemptions: (path, matched-substring, blob sha256, grounds). NOT an escape
#: hatch — each asserts "this IS host content, it is tracked, and it cannot be removed here".
#: SELF-EXPIRING TWO WAYS: an entry whose substring stops matching fails the gate, and so does one
#: whose recorded sha no longer matches the file, so an exemption can never be inherited by
#: whatever replaced the content it named. SHIPS EMPTY: the one known blob was sanitized rather
#: than exempted, so this gate has never been green over a dirty tree.
EXEMPT: tuple[tuple[str, str, str, str], ...] = ()

#: Generated lock files carry four-part package versions (`12.8.4.1` is a CUDA library, not a
#: machine) that the octet ranges cannot tell from an address; in a lock the only host position
#: is a URL host, so `ipv4` counts there only immediately after `://`.
LOCK_FILES = frozenset({"uv.lock"})

#: Non-vacuity floor for --full-tree: a gate that scans nothing finds nothing. Set well below the
#: measured tracked-text count, but high enough that a broken `git ls-files` or a wrong REPO_ROOT
#: cannot pass silently.
MIN_FULL_TREE_FILES = 400


_COMPILED: list[tuple[str, re.Pattern[str], str]] | None = None


def load_local_terms(path: Path) -> list[tuple[str, re.Pattern[str], str]]:
    """Compile the untracked supplement at `path`. Absent file = no terms, never an error.

    Takes the path as an ARGUMENT so the self-test can drive this loader against a planted
    supplement: the operator's real one is absent in CI by design, and an unexercised loader rots.
    """
    if not path.is_file():
        return []
    out: list[tuple[str, re.Pattern[str], str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        term = line.strip()
        if term and not term.startswith("#"):
            try:
                rx = re.compile(term)
            except re.error as exc:
                # NAMED, and by LINE NUMBER ONLY: echoing the term would print the
                # operator-identifying string this file exists to keep out of the output.
                raise SystemExit(
                    f"gate 17: {path.name} line {i} is not a valid regex ({exc.msg}). "
                    f"The term itself is not echoed -- open the file at that line."
                ) from None
            out.append((f"local:{path.name}:{i}", rx, "local term"))
    return out


def _compiled() -> list[tuple[str, re.Pattern[str], str]]:
    """The register plus any untracked local supplement, compiled ONCE per process."""
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED
    out = [(name, re.compile(pat), why) for name, (pat, why) in PATTERNS.items()]
    out.extend(load_local_terms(LOCAL_TERMS))
    _COMPILED = out
    return out


def local_class_count() -> int:
    """How many local terms are live. A COUNT, never the terms -- printing one would leak it.

    The green line used to print `len(PATTERNS)` alone, so a run with the supplement loaded and a
    run without it printed the same sentence, and the reader could not tell whether the scan had
    been run at its declared strength.
    """
    return len(_compiled()) - len(PATTERNS)


def operator_arm_banner() -> str | None:
    """The loud line for a scan run WITHOUT the operator-term arm, or None when it is present.

    `LOCAL_TERMS` is untracked by design, so it lives in ONE working directory: any run from a
    worktree, or on CI, or a fresh clone, silently scanned at the tracked floor while printing
    `0 local term(s) live` — a sentence that reads identically to "the file is there and empty".
    Distinguishes ABSENT from PRESENT-BUT-EMPTY: only the first is a missing arm.
    """
    if LOCAL_TERMS.is_file():
        return None
    return (
        "gate 17: *** OPERATOR-TERM ARM SKIPPED *** "
        f"{LOCAL_TERMS.name} is ABSENT, so this scan ran at the TRACKED FLOOR only. "
        "The supplement is untracked by design and exists in one working directory -- "
        "a worktree, a fresh clone or CI does not have it. A PASS here is weaker than a "
        "pass in the main tree and does NOT clear operator-identifying terms."
    )


def _justified(lines: list[str], lineno: int) -> bool:
    """Escape on the line itself, or anywhere in the comment block directly above it."""
    if ESCAPE.search(lines[lineno - 1]):
        return True
    j = lineno - 2
    while j >= 0 and lines[j].lstrip()[:1] in ("#", "/", ";", "-") and lines[j].strip():
        if ESCAPE.search(lines[j]):
            return True
        j -= 1
    return False


def has_file_escape(text: str) -> bool:
    """True if the file declares the file-level hatch in its opening lines."""
    return any(FILE_ESCAPE in ln for ln in text.splitlines()[:FILE_ESCAPE_SCAN_LINES])


def scan_text(rel: str, text: str) -> list[tuple[str, int, str, str, str]]:
    """THE DECISION. Both the scan and the LAW-07 self-test go through this one function.

    Returns (rel, lineno, pattern_name, matched_text, why) per unjustified hit.
    """
    if has_file_escape(text):
        return []
    lines = text.splitlines()
    hits: list[tuple[str, int, str, str, str]] = []
    lock = Path(rel).name in LOCK_FILES
    for name, rx, why in _compiled():
        for lineno, line in enumerate(lines, 1):
            for m in rx.finditer(line):
                if _justified(lines, lineno):
                    continue
                if lock and name == "ipv4" and not line[:m.start()].endswith("://"):
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


def resolve_base(candidate: str | None) -> str | None:
    """A usable base ref, or None meaning: scan the FULL TREE.

    The first push of a new branch hands CI the all-zeros sha as `github.event.before`, which
    reached `git diff` verbatim and crashed the gate. For a LEAK gate the safe fallback direction
    is WIDE, so an empty / all-zeros / unresolvable base degrades to the full-tree scan:
    over-scanning costs seconds, under-scanning ships a host path.
    """
    if not candidate or set(candidate) == {"0"}:
        return None
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return candidate if probe.returncode == 0 else None


def target_files(base: str | None) -> list[str]:
    """Tracked text files to scan: the whole tree, or those added/modified vs `base`.

    RENAMES: the filter was `AM`, so a moved-and-edited file arrived as `R0xx` and was DROPPED —
    `git mv` a fixture and append a box path and the gate scanned nothing. `ACMR` with
    `--name-status` taking the NEW path is the fix.

    AN EMPTY DIFF printed a green line, and a diff that legitimately touches no text file is the
    same observable as a `--base` that resolved to the wrong thing. The caller is told the scope
    was empty so it can DEGRADE WIDE.
    """
    if base is None:
        return [p for p in _git("ls-files").splitlines() if p]
    tracked = set(_git("ls-files").splitlines())
    # `--name-status -z`: a rename row is `R0xx\0<old>\0<new>`, so the paths cannot be read off
    # `--name-only` without ambiguity, and the NEW side is the one whose bytes are in the tree now.
    fields = _git("diff", "--name-status", "-z", "--diff-filter=ACMR",
                  f"{base}...HEAD").split("\0")
    out: list[str] = []
    i = 0
    while i < len(fields):
        status = fields[i]
        if not status:
            i += 1
            continue
        if status[0] in {"R", "C"}:
            # old at i+1, NEW at i+2
            if i + 2 < len(fields):
                out.append(fields[i + 2])
            i += 3
        else:
            if i + 1 < len(fields):
                out.append(fields[i + 1])
            i += 2
    return [p for p in out if p and p in tracked]


def self_test() -> bool:
    """LAW-07: the gate must be able to FIRE, on EVERY invocation.

    Plants one violation per pattern class in a temp file and drives the REAL decision function.
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
        # The lock carve-out both ways: a version string is not a hit, a URL host still is.
        f = Path(td) / "uv.lock"
        f.write_text('version = "12.8.4.1"\n', encoding="utf-8")
        if scan_text("uv.lock", f.read_text(encoding="utf-8")):
            print("gate 17 SELF-TEST FAIL: a lock file's four-part version read as an address")
            ok = False
        f.write_text('url = "https://203.0.113.7/simple"\n', encoding="utf-8")
        if not any(h[2] == "ipv4" for h in scan_text("uv.lock", f.read_text(encoding="utf-8"))):
            print("gate 17 SELF-TEST FAIL: an address in a lock file's URL host did not fire")
            ok = False
        # The hatch must actually suppress, or every register line below would red the gate.
        f = Path(td) / "hatched.txt"
        f.write_text("path = /root/x  # rule7-gate: ok -- fixture\n", encoding="utf-8")
        if scan_text("hatched.txt", f.read_text(encoding="utf-8")):
            print("gate 17 SELF-TEST FAIL: escape hatch did not suppress")
            ok = False
    # DERIVED, never transcribed: every registered class must have a planted proof, or the
    # self-test silently stops covering whatever was added to the register.
    uncovered = set(PATTERNS) - {name for name, _line in planted}
    if uncovered:
        print(f"gate 17 SELF-TEST FAIL: registered pattern(s) with no planted proof: "
              f"{sorted(uncovered)}")
        ok = False
    # The base-resolver arm: every degraded shape must resolve to full-tree (None) and a real ref
    # must survive. A resolver that narrows instead of widening re-opens the crash as an
    # under-scan, which is worse.
    for degraded in (None, "", "0" * 40, "no-such-ref-xyzzy"):
        if resolve_base(degraded) is not None:
            print(f"gate 17 SELF-TEST FAIL: degraded base {degraded!r} did not widen to full-tree")
            ok = False
    if resolve_base("HEAD") != "HEAD":
        print("gate 17 SELF-TEST FAIL: a resolvable ref must be scanned as itself, not widened")
        ok = False

    if not _local_arm_fires():
        ok = False
    if not _operator_arm_banner_fires():
        ok = False
    if ok:
        print(f"gate 17 self-test: all {len(PATTERNS)} registered pattern classes fire, "
              f"6 controls clean, local-supplement arm wired "
              f"({local_class_count()} local term(s) live)")
        banner = operator_arm_banner()
        if banner:
            print(banner)
    return ok


def _local_arm_fires() -> bool:
    """LAW-07 for the UNTRACKED half: prove the supplement is WIRED, not merely parseable.

    The operator's real supplement is `.gitignore`d, so in CI there is nothing to exercise and the
    loader would be dead code that still reports a count. This plants a synthetic supplement,
    points the module at it and drives the REAL decision function, restoring module state in
    `finally` and printing counts and fixture names only, never a local term's text.
    """
    global _COMPILED, LOCAL_TERMS
    probe, saved_path, saved_cache = "zzplantedlocaltermzz", LOCAL_TERMS, _COMPILED
    ok = True
    with tempfile.TemporaryDirectory() as td:
        planted_file = Path(td) / "rule7_local_terms.txt"
        # A comment and a blank line ride along: both must be ignored, or a `#` note in the
        # operator's file would compile into a regex that matches nearly everything.
        planted_file.write_text(f"# note\n\n{probe}\n", encoding="utf-8")
        try:
            LOCAL_TERMS, _COMPILED = planted_file, None
            if local_class_count() != 1:
                print("gate 17 SELF-TEST FAIL: local supplement did not compile to exactly "
                      f"one term (comments/blank lines leaked?): got {local_class_count()}")
                ok = False
            hits = scan_text("planted.txt", f"value = {probe}\n")
            if not any(h[2].startswith("local:") for h in hits):
                print("gate 17 SELF-TEST FAIL: a local supplement term did not fire through "
                      "scan_text -- the untracked half is NOT wired into the decision")
                ok = False
            if scan_text("planted.txt", f"value = {probe}  {ESCAPE_TOKEN} fixture\n"):
                print("gate 17 SELF-TEST FAIL: escape hatch did not suppress a local term")
                ok = False
        finally:
            LOCAL_TERMS, _COMPILED = saved_path, saved_cache
    return ok


def _operator_arm_banner_fires() -> bool:
    """LAW-07 for the absent-supplement banner: prove it fires ABSENT and stays silent PRESENT.

    Without this control the banner could be wired to a condition that never holds, and every
    worktree run would keep reporting a floor-strength scan in a full-strength sentence.
    """
    global LOCAL_TERMS
    saved, ok = LOCAL_TERMS, True
    with tempfile.TemporaryDirectory() as td:
        present, absent = Path(td) / "rule7_local_terms.txt", Path(td) / "not_here.txt"
        present.write_text("zzplantedlocaltermzz\n", encoding="utf-8")
        try:
            LOCAL_TERMS = absent
            banner = operator_arm_banner()
            if banner is None or "OPERATOR-TERM ARM SKIPPED" not in banner:
                print("gate 17 SELF-TEST FAIL: an ABSENT local-terms file did not raise the "
                      "OPERATOR-TERM ARM SKIPPED banner -- a worktree scan would report a "
                      "floor-strength pass in a full-strength sentence (R312(e))")
                ok = False
            LOCAL_TERMS = present
            if operator_arm_banner() is not None:
                print("gate 17 SELF-TEST FAIL: the banner fired with the supplement PRESENT -- "
                      "a banner that always prints is noise and stops being read")
                ok = False
        finally:
            LOCAL_TERMS = saved
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="CI gate 17 -- no host content in the tree (rule 7)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="scan files added/modified relative to this ref")
    g.add_argument("--full-tree", action="store_true", help="scan every tracked text file")
    args = ap.parse_args()

    if not self_test():
        return 2

    base = None if args.full_tree else resolve_base(args.base)
    if not args.full_tree and base is None:
        print(f"gate 17: base {args.base!r} is absent or unresolvable -- "
              "degrading WIDE to the full-tree scan (leak gates fail toward over-scanning)")
        args.full_tree = True
    files = target_files(base)
    # A diff-scoped run over ZERO files printed a green line, and "the diff touched no text file"
    # is the same observable as "--base resolved to something with no delta". Same posture as the
    # unresolvable-base arm above: a LEAK gate degrades WIDE.
    if not args.full_tree and not files:
        print(f"gate 17: the diff vs {base!r} names NO tracked text file -- degrading WIDE to "
              "the full-tree scan rather than reporting green over an empty scope")
        args.full_tree = True
        base = None
        files = target_files(None)
    violations: list[tuple[str, int, str, str, str]] = []
    scanned = 0
    file_hatched: list[str] = []
    matched_exempt: set[int] = set()

    for rel in files:
        text = _read_text(REPO_ROOT / rel)
        if text is None:
            continue
        scanned += 1
        if has_file_escape(text):
            file_hatched.append(rel)
            continue
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
            f"place:\n    <comment> {ESCAPE_TOKEN} <why>\n"
            "If it is real host content that cannot be removed here, it goes in EXEMPT with "
            "grounds and the blob sha -- never in the escape hatch."
        )
        rc = 1

    banner = operator_arm_banner()
    if banner:
        print(banner, file=sys.stderr)

    if rc == 0:
        where = "tracked tree" if args.full_tree else f"files changed vs {args.base}"
        print(
            f"gate 17: no host content in the {where} ({scanned} text file(s); "
            f"{len(PATTERNS)} tracked + {local_class_count()} local pattern class(es); "
            f"{len(EXEMPT)} registered exemption(s); "
            f"{len(file_hatched)} file-level hatch(es): {sorted(file_hatched)})"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
