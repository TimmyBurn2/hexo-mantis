#!/usr/bin/env bash
# `make vendor`: reads vendor/pins.toml; clones each pin into vendor/external/<name>
# (gitignored) at its exact sha and applies the optional tracked patch. Empty pin table
# is honest empty behavior (exit 0), not a gate. IDEMPOTENT (R326(e)): a warm, correct
# tree is a no-op, not a failure.
set -euo pipefail
python3 - <<'PYEOF'
import subprocess
import sys
import tomllib
from pathlib import Path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def apply_patch(dest: Path, patch: Path, name: str) -> None:
    """Apply `patch` to `dest`, or report it already applied.

    WHY THIS IS NOT A BARE `git apply` (R326(e)). It used to be, and `make vendor` therefore
    FAILED on a warm box holding a correct tree — the second run re-applied a patch that was
    already in the working tree and `git apply` refused. That turned "verify the vendor state"
    into "delete vendor/external and start over", and a re-fetch was the only way to find out
    whether the tree was right. The reverse-check is what `git apply --check --reverse` is for:
    it succeeds exactly when the patch is ALREADY present, which is the state a re-run should
    treat as done.

    THE THIRD OUTCOME IS THE ONE THAT MATTERS. Neither forward nor reverse applying means the
    tree is neither patched nor clean — a partial application, a hand-edit, a different patch.
    That is NOT idempotency and it is not silently ignored: it raises with both refusals
    printed, because a vendor tree in an unknown state compiles into an engine that plays a
    different game at a depth receipt no downstream oracle can distinguish from the right one.
    """
    reverse = _run(["git", "-C", str(dest), "apply", "--check", "--reverse", str(patch)])
    if reverse.returncode == 0:
        print(f"vendor: {name} patch already applied; tree left as it is")
        return
    forward = _run(["git", "-C", str(dest), "apply", str(patch)])
    if forward.returncode == 0:
        print(f"vendor: {name} patch applied")
        return
    raise SystemExit(
        f"vendor: {name} tree is in an UNKNOWN state — {patch} applies neither forward nor in "
        f"reverse, so it is neither patched nor clean.\n"
        f"  forward: {forward.stderr.strip()}\n"
        f"  reverse: {reverse.stderr.strip()}\n"
        f"Remove {dest} and re-run `make vendor` to rebuild it from the pin."
    )


pins = tomllib.loads(Path("vendor/pins.toml").read_text(encoding="utf-8")).get("pins", {})
if not pins:
    print("vendor: no pins declared; nothing to fetch")
    sys.exit(0)
for name, spec in pins.items():
    url, sha = spec["url"], spec["sha"]
    dest = Path("vendor/external") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    # A warm tree already at the pinned sha needs no network: `fetch` is what makes a re-run
    # need connectivity it does not need, and the box re-runs this to VERIFY, not to update.
    head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"]).stdout.strip()
    if head != sha:
        subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", sha], check=True)
    patch = spec.get("patch")
    if patch:
        apply_patch(dest, Path(patch).resolve(), name)
    print(f"vendor: {name} @ {sha[:12]} ready")
PYEOF
