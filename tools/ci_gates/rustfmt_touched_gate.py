"""CI gate 18: every Rust file touched since the base is rustfmt-clean, and no other file is read."""
import argparse
import subprocess
import sys
from pathlib import Path

#: Every `.rs` path from the repository top, whatever directory the gate runs in.
_RUST = ":(top,glob)**/*.rs"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _widen(base: str) -> str:
    """An empty or all-zeros base (a first push) widens to origin/dev; any other base is kept."""
    return "origin/dev" if not base or set(base) == {"0"} else base


def _touched_rust_files(top: Path, base: str) -> list[str]:
    """Top-relative `.rs` paths changed from merge-base(base, HEAD) to the working tree, untracked ones included."""
    merge_base = _git("-C", str(top), "merge-base", base, "HEAD").strip()
    changed = _git("-C", str(top), "diff", "--name-only", "--diff-filter=ACMR", merge_base, "--", _RUST)
    untracked = _git("-C", str(top), "ls-files", "--others", "--exclude-standard", "--", _RUST)
    return sorted({line for line in (changed + untracked).splitlines() if line})


def _unformatted(top: Path, path: str) -> bool:
    """Whether `path` differs from its rustfmt form, formatted through stdin so no `mod` child is visited."""
    source = (top / path).read_text(encoding="utf-8")
    proc = subprocess.run(["rustfmt", "--edition", "2021", "--emit", "stdout"], cwd=top,
                          capture_output=True, text=True, input=source)
    if proc.returncode != 0 or proc.stderr.strip():
        raise RuntimeError(f"{path}: rustfmt rc {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout != source


def main() -> int:
    parser = argparse.ArgumentParser(description="gate 18: rustfmt on the Rust files touched since base")
    parser.add_argument("--base", default="origin/dev")
    base = _widen(parser.parse_args().base)
    try:
        top = Path(_git("rev-parse", "--show-toplevel").strip())
        files = _touched_rust_files(top, base)
    except subprocess.CalledProcessError as exc:
        print(f"gate 18: base {base!r} does not resolve: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    noun = "file" if len(files) == 1 else "files"
    print(f"gate 18: {len(files)} touched Rust {noun} since merge-base with {base}")
    try:
        bad = [path for path in files if _unformatted(top, path)]
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"gate 18: rustfmt gave no verdict: {exc}", file=sys.stderr)
        return 2
    for path in bad:
        print(f"UNFORMATTED {path} (rustfmt --edition 2021 --emit stdout < {path} > {path}.fmt, then move it back)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
