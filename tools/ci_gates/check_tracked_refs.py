"""CI gate 10: no Makefile/doc reference to a path absent from `git ls-files`.

Scope: Makefile, README.md, CLAUDE.md, and every `*.md` in the directories named by
`GLOB_SCOPE`. docs/design/ is exempt BY DESIGN: design docs legitimately name future
layout. A token passes if it is a tracked file, a directory prefix of a tracked file, or
starts with a GENERATED whitelist entry (vendor/external, target/, dist/), or names a
DISSOLVED path. Prints `file:line: token` per failure; exit 1 on any, else 0.
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

#: Glob directory -> its OWN floor. A combined floor let a dissolved directory contribute zero
#: while another directory's files carried the total over the bar, so the gate reported green
#: over a scope nobody chose — the AUDIT-1 F-26 class, one level up. R346(e) dissolved
#: `docs/registers/` and this gate globbed it for a whole era without raising.
GLOB_SCOPE: dict[str, int] = {"docs/contracts": 5, "docs/governance": 4}

#: Scanned-file exemptions, by declaration and with grounds, the way `docs/design/` is exempt.
SCAN_EXEMPT: dict[str, str] = {
    "docs/governance/RULINGS.md": (
        "canonical register: it corrects only by annotation (R9), its own header declares its "
        "coordinates stale by construction, and ANNOTATION 7 deliberately preserves a wrong path "
        "string as evidence — a gate over it would demand the silent edit R9 forbids"
    ),
}

#: Paths this repository REMOVED. A doc that records the removal names the old path correctly, so
#: the token is not drift. Self-expiring: if a dissolved path is tracked again the gate raises,
#: because the whitelist would then be hiding live references.
DISSOLVED_PATHS: dict[str, str] = {
    "docs/registers/": "dissolved by R346(e); governance moved to docs/governance/",
}


def _scope_files() -> list[Path]:
    """The files this gate reads. Raises rather than silently narrowing.

    Raises:
        FileNotFoundError: a named scope file is missing, an exempt file is missing, or a glob
            directory yielded fewer than its own floor — each means the scan is not looking where
            it thinks it is, and a gate that inspects nothing must never report clean (LAW-07).
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
    globbed: list[Path] = []
    for directory, floor in GLOB_SCOPE.items():
        found = sorted(Path(directory).glob("*.md"))
        if len(found) < floor:
            raise FileNotFoundError(
                f"gate 10's glob scope for `{directory}/` yielded {len(found)} file(s), below its "
                f"own floor of {floor}. Each directory carries its own floor precisely so a "
                "dissolved one cannot be absorbed by a full one at rc 0."
            )
        globbed.extend(p for p in found if str(p) not in SCAN_EXEMPT)
    absent_exempt = [p for p in SCAN_EXEMPT if not Path(p).is_file()]
    if absent_exempt:
        raise FileNotFoundError(
            f"gate 10 exempts {absent_exempt}, which is not in the tree. An exemption for a path "
            "that no longer exists is a dead carve-out, and the file it was written for may have "
            "been renamed into the scan unnoticed."
        )
    return files + globbed


def _check_dissolved(tracked: set[str]) -> None:
    """Refuse the dissolved-path whitelist once a dissolved path is tracked again.

    Raises:
        FileNotFoundError: a `DISSOLVED_PATHS` entry has files in `git ls-files`, so the
            whitelist has stopped recording a removal and started hiding live references.
    """
    revived = [p for p in DISSOLVED_PATHS if any(t.startswith(p) for t in tracked)]
    if revived:
        raise FileNotFoundError(
            f"gate 10's dissolved-path whitelist names {revived}, which is tracked again. "
            "Remove the entry: a path that exists must be checked, not whitelisted."
        )


def main() -> int:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )
    _check_dissolved(tracked)

    def token_ok(token: str) -> bool:
        if token in tracked:
            return True
        prefix = token.rstrip("/") + "/"
        if any(t.startswith(prefix) for t in tracked):
            return True
        if token.startswith(GENERATED_WHITELIST):
            return True
        return any(token.startswith(d) for d in DISSOLVED_PATHS)

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
