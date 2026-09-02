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


#: The three named files are NOT optional: they exist in every checkout of this repo, and a
#: missing one means the scan is running somewhere it should not be. AUDIT-1 F-26: `_scope_files`
#: filtered on `is_file()`, so a renamed directory or a wrong CWD shrank the scope to nothing
#: and the gate printed nothing and exited 0.
MIN_GLOB_FILES = 5


def _scope_files() -> list[Path]:
    """The files this gate reads. Raises rather than silently narrowing.

    Raises:
        FileNotFoundError: a named scope file is missing, or a glob directory yielded fewer
            than `MIN_GLOB_FILES` — both mean the scan is not looking where it thinks it is,
            and a gate that inspects nothing must never report clean (LAW-07).
    """
    files = [Path(p) for p in SCOPE]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"gate 10's named scope is incomplete: {missing} not found from {Path.cwd()}. "
            "These files exist in every checkout; their absence means the gate is running "
            "from the wrong directory or against a renamed path, and scanning what is left "
            "would report a clean tree over a scope nobody chose."
        )
    globbed = sorted(Path("docs/contracts").glob("*.md")) + \
        sorted(Path("docs/registers").glob("*.md"))
    if len(globbed) < MIN_GLOB_FILES:
        raise FileNotFoundError(
            f"gate 10's glob scope yielded {len(globbed)} file(s), below the floor of "
            f"{MIN_GLOB_FILES}. `docs/contracts/` and `docs/registers/` are not optional; a "
            "renamed directory used to shrink this scan to nothing at rc 0."
        )
    return files + globbed


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
    scope = _scope_files()
    print(f"gate 10: scanning {len(scope)} file(s) for references to untracked paths")
    for path in scope:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in TOKEN_RE.finditer(line):
                token = match.group(0).rstrip(".,;:!?")
                if not token_ok(token):
                    print(f"{path}:{lineno}: {token}")
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
