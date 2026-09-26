"""CI gate 18: every Rust file touched since the base is rustfmt-clean; nothing else is swept.

CLI: rustfmt_touched_gate.py [--base REF] (default origin/dev). Scope: `.rs` files added, copied,
modified or renamed between merge-base(REF, HEAD) and the WORKING TREE, so an uncommitted edit
counts. Exit 1 on any unformatted file, 0 clean (the scope size is printed either way), 2 when the
base does not resolve or rustfmt cannot run.
"""
import argparse
import subprocess
import sys


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def touched_rust_files(base: str) -> list[str]:
    """The existing `.rs` files that differ between merge-base(base, HEAD) and the working tree.

    Raises:
        subprocess.CalledProcessError: the base does not resolve.
    """
    merge_base = _git("merge-base", base, "HEAD").strip()
    names = _git("diff", "--name-only", "--diff-filter=ACMR", merge_base, "--", "*.rs")
    return sorted(line for line in names.splitlines() if line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/dev")
    base = parser.parse_args().base
    try:
        files = touched_rust_files(base)
    except subprocess.CalledProcessError as exc:
        print(f"gate 18: base {base!r} does not resolve: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    noun = "file" if len(files) == 1 else "files"
    print(f"gate 18: {len(files)} touched Rust {noun} since merge-base with {base}")
    if not files:
        return 0
    proc = subprocess.run(["rustfmt", "--edition", "2021", "--check", "-l", *files],
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        print(f"gate 18: rustfmt could not run (rc {proc.returncode}): {proc.stderr.strip()}",
              file=sys.stderr)
        return 2
    unformatted = [line for line in proc.stdout.splitlines() if line.strip().endswith(".rs")]
    for path in unformatted:
        print(f"UNFORMATTED {path} (run `rustfmt --edition 2021 {path}`)")
    return 1 if unformatted else 0


if __name__ == "__main__":
    sys.exit(main())
