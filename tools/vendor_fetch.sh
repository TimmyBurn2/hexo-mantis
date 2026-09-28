#!/usr/bin/env bash
# `make vendor`: clones each vendor/pins.toml pin into vendor/external/<name> (gitignored) at
# its sha and applies the optional tracked patch; `vendor_fetch.sh <pin>...` does only the named
# pins and ALSO fetches their release assets, each verified by sha256 on arrival. An empty pin
# table is honest empty behavior (exit 0). IDEMPOTENT: a warm, correct tree is a no-op.
set -euo pipefail
python3 - "$@" <<'PYEOF'
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path

EXTERNAL = Path("vendor/external")


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


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, what: str) -> None:
    got = sha256_of(path)
    if got != expected:
        raise SystemExit(f"vendor: sha256 mismatch for {what} ({path}): got {got}, the pin says "
                         f"{expected}. Remove it and re-run to fetch it from the pin.")


def download(url: str, path: Path, expected: str) -> None:
    """Stream `url` to a `.part` beside `path`; it becomes `path` only once its sha256 is the pin's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as src, part.open("wb") as out:
            for block in iter(lambda: src.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected:
            raise SystemExit(f"vendor: sha256 mismatch for {path.name} from {url}: got "
                             f"{digest.hexdigest()}, the pin says {expected}; nothing was kept")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def fetch_assets(pin: str, assets: dict) -> None:
    """Every asset of `pin`: downloads first (an `unpack` archive extracted), then each `from` member verified."""
    for name, spec in assets.items():
        if "from" in spec:
            continue
        path = EXTERNAL / spec["path"]
        if path.is_file():
            verify(path, spec["sha256"], f"{pin}.{name}")
            print(f"vendor: {pin}.{name} already verified")
        else:
            download(spec["url"], path, spec["sha256"])
            print(f"vendor: {pin}.{name} fetched and verified")
        if "unpack" in spec:
            target = EXTERNAL / spec["unpack"]
            home = target.parent.resolve()
            if home != (EXTERNAL / spec["path"]).parent.resolve() or not home.is_relative_to(EXTERNAL.resolve()):
                raise SystemExit(f"vendor: {pin}.{name} unpacks to {target}, outside vendor/external or "
                                 "away from its archive's directory")
            members = [m for m in assets.values() if m.get("from") == name]
            # Extracted beside the target and swapped in whole: a leftover part or old tree is never trusted.
            part, old = target.with_name(target.name + ".part"), target.with_name(target.name + ".old")
            shutil.rmtree(part, ignore_errors=True)
            shutil.rmtree(old, ignore_errors=True)
            if not target.is_dir() or not all((EXTERNAL / m["path"]).is_file() for m in members):
                part.mkdir(parents=True)
                with tarfile.open(path) as tar:
                    tar.extractall(part, filter="data")
                if target.exists():
                    target.rename(old)
                part.rename(target)
                shutil.rmtree(old, ignore_errors=True)
                print(f"vendor: {pin}.{name} unpacked")
    for name, spec in assets.items():
        if "from" not in spec:
            continue
        source = assets.get(spec["from"], {})
        if "unpack" not in source or source.get("url") != spec.get("url"):
            raise SystemExit(f"vendor: {pin}.{name} names from = {spec['from']!r}, which is not an "
                             "unpack asset with the same url")
        verify(EXTERNAL / spec["path"], spec["sha256"], f"{pin}.{name} ({spec['path']})")
        print(f"vendor: {pin}.{name} verified")


pins = tomllib.loads(Path("vendor/pins.toml").read_text(encoding="utf-8")).get("pins", {})
named = sys.argv[1:]
for name in named:
    if name not in pins:
        raise SystemExit(f"vendor: vendor/pins.toml declares no [pins.{name}]")
if not pins:
    print("vendor: no pins declared; nothing to fetch")
    sys.exit(0)
for name, spec in pins.items():
    if named and name not in named:
        continue
    url, sha = spec["url"], spec["sha"]
    dest = EXTERNAL / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    elif not (dest / ".git").exists():
        # `git -C` on a plain directory resolves to the ENCLOSING repository and would fetch and
        # check out there.
        raise SystemExit(f"vendor: {dest} exists but is not a clone; move it aside and re-run")
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
    if named:
        fetch_assets(name, spec.get("assets", {}))
PYEOF
