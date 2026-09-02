#!/usr/bin/env python3
# >300 justify (R8): the marker table, the cap-token stripper, the count detector and the two
# scoped rules are one gate's single authority; splitting them would create a second place
# where "what counts as a stated tally" is decided, which is the drift this gate exists to
# remove (LAW-03). The self-test corpus stays in-file for the same reason gate 11 keeps its
# own: the arms and the predicate they prove must move together or the proof rots quietly.
# NOTE: the docstring below quotes banned header forms as EXAMPLES. They sit in later
# paragraphs, outside this justification block, which is why the gate passes on itself. Keep
# them there, and keep this paragraph free of figures.
"""CI gate 15 (P1-01): every oversized file carries an R8 justification, and none states a count.

R8 asks a file over the 300-line soft cap to say WHY it is one unit. It never asked for a
tally. The repo ratified that distinction as **G-DFIX-4 / R192(e) "derive-or-delete"** and
applied it to four files, then stalled and never wrote the rule down. This gate is the rule
written down.

TWO HALVES, and the second is the load-bearing one:

  * PRESENCE -- a `.py`/`.rs` file over CAP lines under `src/`, `tools/`, `crates/`, `tests/`
    must carry a justification marker. Measured at the adoption commit: 135 files were over the
    cap and exactly 2 carried nothing (`tests/monitor/test_supervisor.py`,
    `tests/train/test_graph_microbatch_bound.py`).
  * NO COUNT -- a justification may not state a line count. This is what stops the convention
    re-accreting. Measured at the adoption commit: 47 headers stated one, at least 8 were
    already wrong, and `src/mantis/run.py` claimed 867 against 1024. A stale count is not
    noise: it is misinformation a future reader trusts, which is SF-7's own ruling ("a
    justification which is not true is worse than none") applied to the number, not the prose.

WHY THE NO-COUNT HALF IS NOT MERELY STYLE. The alternative gate -- "re-derive the stated count
and require it to match" -- was considered and rejected. It automates a transcription instead of
removing it: every edit to any file over the cap then also edits its own header, forever, to
maintain a number that `wc -l` already answers for free. Prefer removing the bookkeeping over
automating it.

WHAT IS AND IS NOT A COUNT (measured against every header in the tree, not invented):

  fires    "MEASURED size of 488 lines" / "(697, re-measured by `wc -l`" / "(MEASURED 310 lines"
           "R8 >300 justify (402," / "(R8, by 8 lines)" / "pushed this file from 292 to 303 lines"
  silent   ">300 justify" and "300-line soft cap" -- the cap token itself, stripped before the scan
           "the 18 named contract fields" / "O-T1..O-T7" -- digits with no line unit
           "combines three old modules (`training/warmstart_launch.py` 190, ...)" -- the size of a
           DELETED predecessor. It cannot go stale, because the file it counts is frozen. The rule
           bans a number that must be re-edited; this one never must.

KNOWN BLIND SPOT, stated rather than discovered later: a bare number with no unit word ("...
`warmstart_launch.py` 190, ...") is invisible to the detector by design, and so is a
self-referential one written that way. The rule catches the forms the corpus actually uses;
narrowing further would buy nothing and widening would flag the register ids this repo is made
of.

The marker window is 80 lines, not 15. Measured: the deepest legitimate marker in the tree is
`tests/tools/test_preflight_mint.py:75`, where the R8 clause is a paragraph inside a long module
docstring -- the house style for the oracle suites. A 15-line window would have failed 14 files
that carry a real justification. `test_the_deepest_real_marker_in_the_tree_is_inside_the_window`
fails if that ever stops being true, so the window is a measured claim and not a guess.

Self-test (LAW-07 -- a gate whose trigger cannot fire is a phantom input): `--self-test` runs
both halves over a corpus of synthetic headers, each of which MUST or MUST NOT fire. It runs on
every invocation of the real gate too, so the trigger is proven live at the same moment the
verdict is issued. The full producer test is tests/tools/test_r8_header_gate.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CAP = 300
ROOTS = ("src", "tools", "crates", "tests")
EXTS = (".py", ".rs")
SKIP_DIRS = frozenset({"target", "__pycache__", ".venv", ".git", "node_modules"})

#: How far in the marker may sit. See the module docstring: measured, not guessed.
MARKER_WINDOW = 80
#: How far a justification block runs from its marker. Terminated by a blank line first.
BLOCK_MAX = 40

#: Every phrasing that opens a justification in this tree. Deliberately tolerant: the corpus
#: has six house styles (`# >300 justify`, `// Exceeds the 300-line soft cap`, `//! R8-justify`,
#: `//! R8: >300 LOC by design`, `# >300 lines:`, `(R8 justification: >300 LOC ...)`) and
#: forcing one canonical spelling would be a 150-file rewrite that buys nothing.
#: The `R8 justif` arm requires a following `:` or `(` -- without it, prose that merely NAMES
#: the rule ("the R8 justification detector must bite", in this gate's own producer test) reads
#: as a header and silently satisfies the presence check. That was measured here, not imagined.
MARKER_RE = re.compile(
    r"(?:[>\u2265]=?\s*300"
    r"|R8[\s-]*justif\w*\s*[:(]"
    r"|300[- ]line soft cap"
    r"|R8\b[^\n]{0,40}?300"
    r"|300[^\n]{0,40}?\bR8\b)",
    re.IGNORECASE,
)

#: The cap threshold itself, in every spelling. Stripped BEFORE the count scan, so ">300 lines"
#: and "300-line soft cap" cannot be mistaken for a tally. Without this the marker would fail
#: the rule it announces.
CAP_TOKEN_RE = re.compile(r"[>\u2265]=?\s*300|\b300\s*\+|\b300(?=[-\s]line)", re.IGNORECASE)

#: A number welded to a line unit is a tally. `-\s*` catches "a ~120-line harness"; the comma
#: class catches "1,024 lines"; `\s*` spans a newline because these headers wrap mid-clause;
#: `L` catches the terse "the old `training/anchor.py`, 659 L".
#: The lookbehind is load-bearing and was measured, not guessed: without it "a stale size in an
#: R8 line" and "the r153 line-dispersal rule" both read as tallies. A digit glued to a letter
#: is an identifier, never a count.
COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9_])\d[\d,]*\s*(?:-\s*)?(?:lines?|LOC|L)\b", re.IGNORECASE
)
#: The two idioms this repo uses to announce "I transcribed a measurement here", plus the
#: marker parenthetical. They appear in the corpus ONLY inside count clauses, so they are banned
#: outright: a header that cites its own `wc -l` is stating a count even in the sentences where
#: it omits the digits, and `justify (697,` is a size with the unit word left off -- the one
#: shape a `<number> <unit>` rule cannot see. Case-SENSITIVE on `MEASURED` by design: the
#: shouted form is the transcription idiom ("stated at this file's MEASURED size"), while
#: lower-case "measured" is ordinary prose ("the measured 8% ceiling", `tactics/search.rs:13`)
#: and firing on it would flag nine correct headers.
IDIOM_RE = re.compile(
    r"\bwc\s*-\s*l\b|\bre-?measured\b|\bMEASURED\b|\bjustif\w*[^\n]{0,24}?\(\s*\d{2,5}\b"
)

#: Non-vacuity floors, PER ROOT. A gate that scans nothing finds nothing -- but a single global
#: floor is not enough: with one number (measured at 100 against 137 over-cap files) a typo that
#: dropped `src/` entirely still reported green, because the other three roots cleared it on
#: their own. That was mutation-tested here and it is why this is a dict. Set below the measured
#: file counts (src 148, tools 14, crates 134, tests 256) with room for deletion, high enough
#: that losing any ONE root is fatal.
MIN_FILES = {"src": 120, "tools": 10, "crates": 110, "tests": 210}
#: The corpus-wide floors stay too, as a second net: they catch a scope that shrank without any
#: root vanishing (measured: 137 over the cap, 151 justifications).
MIN_OVER_CAP = 120
MIN_HEADERS = 130


def source_files(root: Path = REPO_ROOT) -> list[Path]:
    out: list[Path] = []
    for name in ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in EXTS or not path.is_file():
                continue
            if SKIP_DIRS.intersection(path.parts):
                continue
            out.append(path)
    return out


def find_marker(lines: list[str]) -> int | None:
    """Index of the justification marker within the window, or None."""
    for i, line in enumerate(lines[:MARKER_WINDOW]):
        if MARKER_RE.search(line):
            return i
    return None


#: Leading comment punctuation, stripped before a line is judged blank. A doc-comment paragraph
#: break is `//!` or `#` alone, not an empty line, so without this a Rust module doc runs on for
#: the whole BLOCK_MAX and drags unrelated prose into the justification.
COMMENT_LEAD_RE = re.compile(r"^\s*(?:#+|/{2,}!?|\*)\s*")


def _is_break(line: str) -> bool:
    """True at a paragraph boundary -- blank, comment-only, or a docstring terminator."""
    stripped = line.strip()
    if stripped in ('"""', "'''"):
        return True
    return not COMMENT_LEAD_RE.sub("", line).strip()


#: The minimum number of WORDS a justification must carry, after the marker token itself and
#: the cap spellings are stripped. AUDIT-1 F-25: `check_file` treated ANY `MARKER_RE` match as
#: a justification, so `# >300` and `# R8 justification:` — both empty of reason — passed, and
#: R8's "say WHY the file is one unit" was enforced as "the digits 300 appear near the top".
#: FOUR, and the number is chosen against BOTH ends. It refuses every probe input AUDIT-1 F-25
#: found accepted — `>300` (0 words), `R8 justification:` (0), `R8 justify: one unit` (2), a
#: bare `if n >= 300:` in code (1) — and it accepts the TERSEST real header in the tree's own
#: house styles, e.g. `//! R8: >300 LOC by design -- one indivisible format unit.` A higher
#: floor would make the gate demand verbosity, which is not the rule R8 states.
MIN_REASON_WORDS = 4

#: A word, for the count above: letters or digits, so punctuation and bare symbols do not pad
#: an empty justification up to the floor.
WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def reason_words(block: list[str]) -> list[str]:
    """The words a justification block carries once its own MARKER is removed.

    The marker token is stripped, not merely the cap digits: `R8 justification:` is four
    tokens of pure announcement and counting them would let a header satisfy the floor by
    saying its own name.
    """
    text = "\n".join(COMMENT_LEAD_RE.sub("", line) for line in block)
    text = MARKER_RE.sub(" ", text)
    text = CAP_TOKEN_RE.sub(" ", text)
    return WORD_RE.findall(text)


def block_at(lines: list[str], start: int) -> list[str]:
    """The justification block: the marker line plus the rest of its paragraph."""
    block = [lines[start]]
    for line in lines[start + 1 : start + BLOCK_MAX]:
        if _is_break(line):
            break
        block.append(line)
    return block


def counts_in(block: list[str]) -> list[str]:
    """Every tally in a justification block. THE decision -- gate and self-test share it.

    The comment lead is stripped from each row before the join. These headers wrap mid-clause,
    so a count routinely lands as `... from 292 to 303` / `# lines; WPCLEAN ...`; leaving the
    `#` in place puts a non-space between the digits and their unit and the tally walks free.
    """
    text = "\n".join(COMMENT_LEAD_RE.sub("", line) for line in block)
    text = CAP_TOKEN_RE.sub(" ", text)
    hits = [m.group(0) for m in COUNT_RE.finditer(text)]
    hits += [m.group(0) for m in IDIOM_RE.finditer(text)]
    return [" ".join(h.split()) for h in hits]


def check_file(rel: str, lines: list[str]) -> tuple[list[str], bool, bool]:
    """THE per-file decision: (violations, is_over_cap, carries_a_marker).

    Gate, self-test and producer test all go through this one function. An oracle that
    re-implemented the decision could drift from the thing it certifies -- the defect class
    gate 11's own docstring names.
    """
    violations: list[str] = []
    marker = find_marker(lines)
    over_cap = len(lines) > CAP

    if over_cap and marker is None:
        violations.append(
            f"{rel}: {len(lines)} lines, over the {CAP}-line soft cap, with no R8 "
            f"justification in the first {MARKER_WINDOW} lines.\n"
            "    Say WHY the file is one unit. Do not state its size."
        )
        return violations, over_cap, False
    if marker is None:
        return violations, over_cap, False

    block = block_at(lines, marker)
    words = reason_words(block)
    # SCOPED TO FILES OVER THE CAP. R8's "say WHY" duty exists only for a file that exceeds
    # the cap; an UNDER-cap file that merely mentions R8 in passing ("R8: keeps every file
    # under the 300-line soft cap") owes no justification and must not be asked for one. The
    # no-count rule below stays universal — a stated tally is misinformation wherever it sits.
    if over_cap and len(words) < MIN_REASON_WORDS:
        violations.append(
            f"{rel}:{marker + 1}: the R8 justification states no REASON "
            f"({len(words)} word(s) after the marker; {MIN_REASON_WORDS} is the floor)\n"
            "    R8 asks WHY the file is one unit, not that the marker be present.\n"
            "    A bare `>300` or `R8 justification:` announces the rule and says nothing."
        )
    found = counts_in(block)
    if found:
        violations.append(
            f"{rel}:{marker + 1}: R8 justification states a line count: {found}\n"
            "    Delete the number, keep the reason (G-DFIX-4 / R192(e) derive-or-delete).\n"
            "    A tally must be re-edited on every edit, will eventually be wrong, and is "
            "then read as evidence."
        )
    return violations, over_cap, True


def scan(root: Path = REPO_ROOT) -> tuple[list[str], int, int, dict[str, int]]:
    """Return (violations, files_over_cap, files_carrying_a_marker, files_scanned_per_root)."""
    violations: list[str] = []
    over_cap = 0
    headers = 0
    scanned: dict[str, int] = {name: 0 for name in ROOTS}

    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        scanned[rel.split("/")[0]] += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            violations.append(f"{rel}: not UTF-8 ({exc}); every tracked text file in this repo is")
            continue
        file_violations, is_over, has_marker = check_file(rel, lines)
        violations += file_violations
        over_cap += int(is_over)
        headers += int(has_marker)

    return violations, over_cap, headers, scanned


#: (name, synthetic header, must_fire) -- the trigger's own proof. Each arm is the exact defect
#: the corresponding half exists to catch, written the way the corpus writes it.
SELF_TEST: tuple[tuple[str, list[str], bool], ...] = (
    ("plain reason", ["# >300 justify (R8): one seam, one set of fakes."], False),
    ("stated size", ["# >300 justify (R8), at this file's MEASURED size of 488 lines."], True),
    ("bare parenthetical", ["# R8 >300 justify (697): the manifest rows are data."], True),
    ("bare parenthetical + chronicle", ["# R8 >300 justify (697, re-measured at Phase O)."], True),
    ("overage clause", ["# >300 justify (R8, by 8 lines): two instruments on ONE config family."], True),
    ("growth chronicle", ["# >300 justify: SC-A2 pushed this file from 292 to 303 lines."], True),
    ("wc -l with no digits", ["# >300 justify, stated at the file's size per `wc -l`."], True),
    ("cap token only", ["// Exceeds the 300-line soft cap (R8): the full PyBoard surface."], False),
    ("LOC form of the cap", ["//! R8: >300 LOC by design -- the save path and the load path."], False),
    ("digits that are not a tally", ["# >300 justify: the 18 named contract fields are ONE contract."], False),
    ("a deleted file's size", ["# >300 justify: combines `warmstart_launch.py` 190, `gnn_warmstart.py` 144."], False),
    ("a register id, not a tally", ["# >300 justify: the r153 line-dispersal rule drives every row."], False),
    ("the rule's own name", ["# >300 justify: a stale size in an R8 line is a false statement."], False),
)


def self_test() -> int:
    """Prove both halves can fire before trusting either verdict."""
    failures: list[str] = []
    for name, header, must_fire in SELF_TEST:
        fired = bool(counts_in(header))
        if fired != must_fire:
            verb = "did not fire" if must_fire else "fired"
            failures.append(f"    no-count arm {name!r}: {verb} on {header[0]!r}")

    # Presence arm: a synthetic over-cap file with no marker must be seen as unjustified.
    if find_marker(["x = 1"] * 400) is not None:
        failures.append("    presence arm: find_marker() claimed a marker in a file with none")
    if find_marker(["# >300 justify (R8): one unit."] + ["x = 1"] * 400) != 0:
        failures.append("    presence arm: find_marker() missed a marker on line 1")

    if failures:
        print("gate 15 SELF-TEST FAIL -- the trigger cannot be trusted:")
        print("\n".join(failures))
        return 1
    return 0


def main(argv: list[str]) -> int:
    if self_test() != 0:
        return 1
    if "--self-test" in argv:
        print(f"gate 15 self-test: {len(SELF_TEST)} no-count arms + 2 presence arms, all correct")
        return 0

    violations, over_cap, headers, scanned = scan()
    rc = 0

    for name, floor in MIN_FILES.items():
        # `.get` and not `[...]`: if a root is dropped from ROOTS the key is absent, and a
        # KeyError traceback is a worse verdict than the sentence this floor exists to print.
        found = scanned.get(name, 0)
        if found < floor:
            print(
                f"gate 15 FAIL -- scanned only {found} file(s) under {name}/ "
                f"(floor {floor}). A gate that scans nothing finds nothing; refusing to "
                "report green."
            )
            rc = 1
    if over_cap < MIN_OVER_CAP or headers < MIN_HEADERS:
        print(
            f"gate 15 FAIL -- found {over_cap} file(s) over the cap and {headers} header(s) "
            f"(floors {MIN_OVER_CAP}/{MIN_HEADERS}). The corpus shrank; refusing to report green."
        )
        rc = 1

    if violations:
        print(f"\ngate 15 FAIL -- R8 justification defects ({len(violations)}):\n")
        print("\n\n".join(violations))
        print(
            "\nR8 wants a reason, not a tally. Line counts are derived by `wc -l`, never asserted "
            "(G-DFIX-4 / R192(e)).\nStyle reference: tests/train/test_periodic_checkpoint.py:1-5."
        )
        rc = 1

    if rc == 0:
        print(
            f"gate 15: {over_cap} file(s) over the {CAP}-line cap, all justified; "
            f"{headers} justification(s), none stating a count"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
