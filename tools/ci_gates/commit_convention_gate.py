"""CI gate 19: every commit since the base is one line: no body, no trailer, no wrapped subject."""
import argparse
import subprocess
import sys

_SEP = "\x1e"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _offending_commits(base: str) -> tuple[int, list[str]]:
    """The commit count since merge-base(base, HEAD), and `sha subject` for each longer than one line."""
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
    parser = argparse.ArgumentParser(description="gate 19: every commit since base is one line")
    parser.add_argument("--base", default="origin/dev")
    base = parser.parse_args().base
    # An empty or all-zeros base (a first push) widens to origin/dev.
    base = "origin/dev" if not base or set(base) == {"0"} else base
    try:
        count, bad = _offending_commits(base)
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
