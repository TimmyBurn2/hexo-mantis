"""CI gate 10: no Makefile/doc reference to a path absent from `git ls-files`.

Scope: Makefile, README.md, CLAUDE.md, docs/contracts/*.md, docs/registers/*.md.
docs/design/ is exempt BY DESIGN: design docs legitimately name future layout.
A token passes if it is a tracked file, a directory prefix of a tracked file, or starts
with a GENERATED whitelist entry (vendor/external, target/, dist/). Prints
`file:line: token` per failure; exit 1 on any, else 0.
"""
import re
import subprocess
import sys
from pathlib import Path

TOKEN_RE = re.compile(
    r"(?<![\w/.-])(?:src|tests|tools|configs|crates|docs|vendor)/[A-Za-z0-9_./-]+"
)
GENERATED_WHITELIST = ("vendor/external", "target/", "dist/")
SCOPE = ["Makefile", "README.md", "CLAUDE.md"]


def _scope_files() -> list[Path]:
    files = [Path(p) for p in SCOPE]
    files += sorted(Path("docs/contracts").glob("*.md"))
    files += sorted(Path("docs/registers").glob("*.md"))
    return [f for f in files if f.is_file()]


def main() -> int:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )

    def token_ok(token: str) -> bool:
        if token in tracked:
            return True
        prefix = token.rstrip("/") + "/"
        if any(t.startswith(prefix) for t in tracked):
            return True
        return token.startswith(GENERATED_WHITELIST)

    failures = 0
    for path in _scope_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in TOKEN_RE.finditer(line):
                token = match.group(0).rstrip(".,;:!?")
                if not token_ok(token):
                    print(f"{path}:{lineno}: {token}")
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
