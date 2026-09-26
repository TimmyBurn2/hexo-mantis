"""CI gate 19: every commit since the base is one subject line, with an empty body and no trailer.

CLI: commit_convention_gate.py [--base REF] (default origin/dev). Scope: the commits in
merge-base(REF, HEAD)..HEAD. Prints the count, then one `VIOLATION <sha> <subject>` line per commit
longer than one line (a body, a trailer or a wrapped subject); exit 1 on any, 0 clean, 2 when the base does not resolve.
"""
import argparse
import subprocess
import sys

_SEP = "\x1e"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def offending_commits(base: str) -> tuple[int, list[str]]:
    """The commit count since merge-base(base, HEAD), and `sha subject` for each longer than one line.

    Raises:
        subprocess.CalledProcessError: the base does not resolve.
    """
    merge_base = _git("merge-base", base, "HEAD").strip()
    log = _git("log", f"--format=%h%x1f%s%x1f%B{_SEP}", f"{merge_base}..HEAD")
    entries = [e.strip("\n") for e in log.split(_SEP) if e.strip("\n")]
    bad = []
    for entry in entries:
        sha, subject, raw = entry.split("\x1f", 2)
        if "\n" in raw.strip():
            bad.append(f"{sha} {subject}")
    return len(entries), bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/dev")
    base = parser.parse_args().base
    try:
        count, bad = offending_commits(base)
    except subprocess.CalledProcessError as exc:
        print(f"gate 19: base {base!r} does not resolve: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    noun = "commit" if count == 1 else "commits"
    print(f"gate 19: {count} {noun} since merge-base with {base}")
    for line in bad:
        print(f"VIOLATION body-or-trailer: {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
