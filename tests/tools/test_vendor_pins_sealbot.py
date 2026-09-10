"""The vendor pin is a commit sha, never a branch name.

`tools/vendor_fetch.sh` reads only `url`/`sha`/`patch`, so `branch` and `as_of` are carried as
data that can never become the thing fetched. One defect per row:

- a pin table that only looks pinned — `sha = "master"` is valid TOML and fetches a moving
  target, and the non-emptiness conjunct stops the regex passing over zero rows;
- a pin that drifted from the recorded sha, which arm 1 cannot see and which changes which
  engine plays;
- a patch declared and not tracked — the fetcher's one-argument `.get` skips it SILENTLY and
  the build quietly regains `-march=native`;
- the fetcher learning to read a branch, which nothing else in the repo would notice.

The upstream `git ls-remote` re-check is deliberately not a pytest row: it needs the network,
and a network-conditional oracle degrades to a pass on a box without one.
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PINS = _REPO / "vendor" / "pins.toml"
_FETCHER = _REPO / "tools" / "vendor_fetch.sh"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The recorded sha, verified upstream by `git ls-remote`. Written in full: an elided tail
#: cannot be compared.
_SEALBOT_SHA = "c94749c21c16c3b072fff6da49762dd5f92f3986"

#: Keys a fetcher may never read; `branch`/`as_of` live in the pin as data only.
_FORBIDDEN_FETCH_KEYS = ("branch", "ref", "tag")


def _pins() -> dict:
    """Return the pin table; an absent `[pins]` header is a KeyError, never an empty default."""
    return tomllib.loads(_PINS.read_text())["pins"]


def test_every_pin_is_a_forty_hex_commit_sha_and_the_table_is_not_empty() -> None:
    """Prove every pin is a forty-hex commit sha and the table is not empty."""
    assert _SHA_RE.match(_SEALBOT_SHA) is not None, "the detector itself must fire"
    assert _SHA_RE.match("master") is None, "the detector must REJECT a branch name"

    pins = _pins()
    assert pins != {}, (
        "the pin table is empty: Phase A's `[pins.sealbot]` row has not landed, so arm 1 "
        "would otherwise pass over zero rows and certify nothing (R81/R86)"
    )
    offenders = [name for name, spec in pins.items() if _SHA_RE.match(spec["sha"]) is None]
    assert offenders == [], (
        f"R139: a pin is a COMMIT SHA, never a branch name. Non-sha pins: {offenders}"
    )


def test_the_sealbot_pin_carries_r145s_exact_sha_and_a_public_url() -> None:
    """Prove the sealbot pin carries the recorded sha and a public URL.

    The URL is asserted by shape, not value: provider names stay out of `tests/`.
    """
    spec = _pins()["sealbot"]
    assert spec["sha"] == _SEALBOT_SHA, (
        "the sealbot pin does not carry R145's recorded sha. A different sha means upstream "
        "moved or the pin was hand-edited; either way R145 must be re-ruled, never silently "
        "re-pinned (PREREG_A §8 abort 3)."
    )
    assert spec["url"].startswith("https://"), spec["url"]
    assert spec["url"].endswith(".git"), spec["url"]


def test_the_sealbot_pin_declares_a_tracked_patch_file() -> None:
    """Prove the declared patch is tracked: the fetcher skips a missing one silently."""
    spec = _pins()["sealbot"]
    patch = spec["patch"]

    tracked = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "--error-unmatch", patch],
        capture_output=True, text=True, check=False,
    )
    assert tracked.returncode == 0, (
        f"the declared patch {patch!r} is not tracked by git; an untracked patch is applied "
        f"on the author's box and nowhere else. stderr={tracked.stderr.strip()}"
    )
    assert (_REPO / patch).is_file(), f"{patch!r} is declared and tracked but not present"
    assert "-march=native" in (_REPO / patch).read_text(), (
        "the patch must name the flag it removes; a patch that does not mention "
        "`-march=native` is not the LAW-15 insurance DESIGN_A §2.6 argues for"
    )


def test_the_fetcher_reads_no_branch_ref_or_tag_key() -> None:
    """Prove the fetcher reads no branch, ref or tag key from the pin table."""
    source = _FETCHER.read_text()
    probe = 'url, sha = spec["url"], spec["branch"]'
    assert any(f'"{k}"' in probe for k in _FORBIDDEN_FETCH_KEYS), "the detector must fire"

    offenders = [k for k in _FORBIDDEN_FETCH_KEYS if f'"{k}"' in source or f"'{k}'" in source]
    assert offenders == [], (
        f"tools/vendor_fetch.sh reads {offenders} from the pin table. R139: the checkout is "
        f"sha-driven, and a branch name must never become the thing that is fetched."
    )
