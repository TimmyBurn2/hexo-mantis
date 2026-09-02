"""CI gate 6: artifact rejection.

CLI: artifact_gate.py [--base REF]; base defaults to $ARTIFACT_GATE_BASE, else HEAD~1;
an all-zeros/invalid base falls back to HEAD~1. Violations over merge-base(BASE, HEAD)
.. HEAD:
  (1) any changed path under reports/, checkpoints/, logs/, benchmarks/;
  (2) any ADDED file whose blob size is > 1,000,000 bytes outside tests/fixtures/
      (CLAUDE.md R7: >1 MB oracle banks live under tests/fixtures/ — same carve-out as (3));
  (3) any ADDED file under tests/fixtures/ whose blob size is > 10,000,000 bytes — the
      carve-out in (2) is a raised ceiling, NOT an exemption (R8);
  (4) any ADDED *.jsonl outside tests/fixtures/.
Prints one `VIOLATION <reason>: <path>` line each; exit 1 if any, 0 clean, 2 on git error.
"""
import argparse
import os
import subprocess
import sys

ARTIFACT_DIRS = ("reports/", "checkpoints/", "logs/", "benchmarks/")
FIXTURES_PREFIX = "tests/fixtures/"
MAX_ADDED_BYTES = 1_000_000
MAX_FIXTURE_BYTES = 10_000_000


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


#: The widest base the fallback may take. AUDIT-1 F-26: `_resolve_base` returned `"HEAD~1"`
#: for an empty, all-zeros or unresolvable candidate and PRINTED NOTHING, so a first push or a
#: force-push — exactly when the candidate is `000…0` — inspected the LAST COMMIT ONLY while
#: the line above it said the gate had run. `origin/dev` is tried first now, and whichever
#: base is used is named on stdout.
_WIDE_FALLBACKS: tuple[str, ...] = ("origin/dev", "dev", "HEAD~1")


def _resolves(rev: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        capture_output=True, text=True, check=False,
    ).returncode == 0


def _resolve_base(candidate: str) -> tuple[str, str]:
    """`(base, why)` — the revision to diff against, and how it was chosen.

    The `why` is RETURNED rather than logged here so the caller prints it on the green path
    too. A fallback nobody can see is the same as no fallback: F-26 measured this arm silently
    narrowing a whole-branch scan to one commit.
    """
    if candidate and set(candidate) != {"0"} and _resolves(candidate):
        return candidate, "given"
    reason = ("no --base given" if not candidate
              else "all-zeros base (a first push or a branch delete)" if set(candidate) == {"0"}
              else f"base {candidate!r} does not resolve")
    for fallback in _WIDE_FALLBACKS:
        if _resolves(fallback):
            return fallback, f"{reason}; widened to {fallback}"
    return "HEAD~1", f"{reason}; NO fallback resolved — inspecting the last commit ONLY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("ARTIFACT_GATE_BASE", ""))
    args = parser.parse_args(argv)

    try:
        base, why = _resolve_base(args.base)
        print(f"gate 6: base={base} ({why})")
        mb = _git("merge-base", base, "HEAD").strip()
        raw = _git("diff", "--name-status", "-z", mb, "HEAD")
    except subprocess.CalledProcessError as exc:
        print(f"git error: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    fields = raw.split("\0")
    # (status, path, arrives) — `arrives` marks a path whose CONTENT enters the tree at
    # HEAD: an add, or the NEW side of a rename/copy. WP0 RED-TEAM row A measured that
    # gating the size/jsonl checks on `status == "A"` alone let an R-status move carry an
    # oversize or *.jsonl file OUT of tests/fixtures/ unexamined (WPCLEAN Phase RES).
    changed: list[tuple[str, str, bool]] = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status[0] in ("R", "C"):
            old, new = fields[i + 1], fields[i + 2]
            changed.append((status[0], old, False))
            changed.append((status[0], new, True))
            i += 3
        else:
            changed.append((status[0], fields[i + 1], status[0] == "A"))
            i += 2

    violations = 0
    for _status, path, arrives in changed:
        if path.startswith(ARTIFACT_DIRS):
            print(f"VIOLATION artifact-dir: {path}")
            violations += 1
        if arrives:
            try:
                size = int(_git("cat-file", "-s", f"HEAD:{path}").strip())
            except subprocess.CalledProcessError as exc:
                print(f"git error: {exc.stderr.strip()}", file=sys.stderr)
                return 2
            in_fixtures = path.startswith(FIXTURES_PREFIX)
            if in_fixtures:
                if size > MAX_FIXTURE_BYTES:
                    print(f"VIOLATION oversize-fixture: {path}")
                    violations += 1
            elif size > MAX_ADDED_BYTES:
                print(f"VIOLATION large-file: {path}")
                violations += 1
            # Case-folded: `.JSONL` is the same artifact class (WP0 RED-TEAM row A).
            if path.lower().endswith(".jsonl") and not in_fixtures:
                print(f"VIOLATION jsonl-outside-fixtures: {path}")
                violations += 1
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
