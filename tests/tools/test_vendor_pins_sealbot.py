"""⊕ WP12-R Phase A / O-A5 (DESIGN_A §2.3, PREREG_A §1) — the pin, in R139's format.

R139's rider is *a pin is a COMMIT SHA, never a branch name*, and R145 records the exact
sha. `tools/vendor_fetch.sh` reads only `url`/`sha`/`patch` (`:17`, `:24`), so `branch` and
`as_of` can be carried as DATA without ever becoming the thing that is fetched — that is
the rider satisfied structurally rather than by convention, and these rows are what make it
falsifiable.

The defect each row is the ONLY witness to:

- **arm 1** — a pin table that looks pinned. The regex is not decorative and M-A10b is what
  proves it: `sha = "master"` is a perfectly valid TOML string and would fetch a moving
  target. The non-emptiness conjunct is what stops the row passing vacuously over today's
  empty table — a regex over zero rows is `assert True` wearing a loop.
- **arm 2** — a pin that drifted from the ruling. R145 recorded the sha with grounds; a
  single changed hex digit (M-A10a) is invisible to arm 1 and changes which engine plays.
- **arm 3 (patch)** — a patch declared and not tracked. `vendor_fetch.sh:24` reads it with
  a one-argument `.get`, so an untracked or renamed patch is skipped SILENTLY and the build
  quietly regains `-march=native` — the FP-contraction surface DESIGN_A §2.6 removes on
  LAW-15 grounds.
- **arm 4** — the fetcher learning to read a branch. This is the row that keeps `branch`
  and `as_of` data: the moment the fetcher reads either, R139's rider is dead and no other
  row in this repo notices.

**Arm 3 of DESIGN_A §5 (the upstream `git ls-remote` re-check) is deliberately NOT a pytest
row.** It needs the network; a network-conditional oracle degrades to a pass on a box
without one, and "verification that degrades to a pass is worse than none" (PREREG_A §8
abort 3). It is an IMPL-stage manual command whose raw output is pasted into
`IMPL_NOTES_A.md`; unavailable ⇒ `not_run` + HALT.
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

#: R145's recorded sha, verified upstream at DESIGN_A §0.3 by `git ls-remote`. Written in
#: full: an elided tail cannot be compared, and DESIGN_A rev-2 shipped one that was not even
#: the real tail (C-13).
_SEALBOT_SHA = "c94749c21c16c3b072fff6da49762dd5f92f3986"

#: Keys a FETCHER may never read. `branch`/`as_of` live in the pin as R139 data.
_FORBIDDEN_FETCH_KEYS = ("branch", "ref", "tag")


def _pins() -> dict:
    """The pin table. A missing `[pins]` header is a KeyError here, never an empty default —
    the file's own contract puts `[pins]` at `:10` and an absent table is a malformed pin
    file, not an empty one."""
    return tomllib.loads(_PINS.read_text())["pins"]


def test_every_pin_is_a_forty_hex_commit_sha_and_the_table_is_not_empty() -> None:
    """O-A5 arm 1."""
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
    """O-A5 arm 2. The URL is asserted by SHAPE, not by value: Rule 7 keeps provider names
    out of `tests/`, and `vendor/pins.toml` is the ONE place R139 puts the string."""
    spec = _pins()["sealbot"]
    assert spec["sha"] == _SEALBOT_SHA, (
        "the sealbot pin does not carry R145's recorded sha. A different sha means upstream "
        "moved or the pin was hand-edited; either way R145 must be re-ruled, never silently "
        "re-pinned (PREREG_A §8 abort 3)."
    )
    assert spec["url"].startswith("https://"), spec["url"]
    assert spec["url"].endswith(".git"), spec["url"]


def test_the_sealbot_pin_declares_a_tracked_patch_file() -> None:
    """O-A5 arm 4 (PREREG_A numbering). The patch removes `-march=native` from the vendored
    build (DESIGN_A §2.6); `vendor_fetch.sh:24` skips a missing one SILENTLY."""
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
    """O-A5 arm 4 (DESIGN_A §5 numbering) — R139's rider made structural. `branch`/`as_of`
    are carried as data precisely BECAUSE the fetcher cannot reach them."""
    source = _FETCHER.read_text()
    probe = 'url, sha = spec["url"], spec["branch"]'
    assert any(f'"{k}"' in probe for k in _FORBIDDEN_FETCH_KEYS), "the detector must fire"

    offenders = [k for k in _FORBIDDEN_FETCH_KEYS if f'"{k}"' in source or f"'{k}'" in source]
    assert offenders == [], (
        f"tools/vendor_fetch.sh reads {offenders} from the pin table. R139: the checkout is "
        f"sha-driven, and a branch name must never become the thing that is fetched."
    )
