"""R324(c) — the sealbot BUILD step is reproducible FROM THE TREE, not from a record.

WHAT THIS FILE IS THE ONLY WITNESS TO. `make vendor` fetches and patches; nothing built the
extension, so `sealbot` refused at `BUILD_ABSENT` and the only account of how to repair it
lived in a leg record. SITTING4-PREP-1 measured the consequence directly: the repair is
per-checkout state (`vendor/external/` is gitignored), so every checkout that wants the rung
must re-run it, and there was no tracked command to re-run.

WHY THE REFUSAL ROWS ARE THE LOAD-BEARING ONES. A build script that builds is easy; a build
script that refuses to build over the WRONG SOURCE is the point. A `.so` compiled from an
unpinned tree, or from a tree whose patch never applied, loads and plays — it just plays a
different engine, at a depth receipt no downstream oracle can distinguish from the right one.
So each precondition gets its own row and its own exit code, driven against a synthetic repo
root rather than mocked.

WHY THE SCRIPT IS DRIVEN AGAINST A COPY. The real `vendor/external/sealbot` is present and
BUILT on the box and in the main tree, so a refusal row driven at the repo root would either
not refuse or would need the real tree moved out of the way. The script resolves the repo
root from its own location (`dirname $0/..`) and uses only relative paths, so a synthetic
root holding `tools/` + `vendor/` exercises the same code with no real state touched.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "tools" / "vendor_build_sealbot.sh"
_PINS = _REPO / "vendor" / "pins.toml"

#: Any 40-hex literal. The script must carry NONE: a second sha authority beside
#: `vendor/pins.toml` is exactly the drift R145 pins the pin against.
_SHA_LITERAL = re.compile(r"\b[0-9a-f]{40}\b")


def _synthetic_root(tmp_path: Path) -> Path:
    """A repo root carrying only what the script reads: the script and the real pins file."""
    root = tmp_path / "root"
    (root / "tools").mkdir(parents=True)
    (root / "vendor").mkdir(parents=True)
    shutil.copy2(_SCRIPT, root / "tools" / _SCRIPT.name)
    shutil.copy2(_PINS, root / "vendor" / "pins.toml")
    return root


def _fake_pin(root: Path, *, sha: str, march_native: bool, win_threshold: bool) -> None:
    """A fetched-looking sealbot pin at `sha`, with the patch's two EFFECTS switchable."""
    dest = root / "vendor" / "external" / "sealbot"
    current = dest / "current"
    current.mkdir(parents=True)
    (current / "setup.py").write_text(
        "extra = ['-O3'" + (", '-march=native'" if march_native else "") + "]\n",
        encoding="utf-8",
    )
    (current / "minimax_bot.cpp").write_text(
        ("constexpr int WIN_THRESHOLD = 6;\n" if win_threshold else "int unrelated = 0;\n"),
        encoding="utf-8",
    )
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}

    def run(*args: str) -> None:
        subprocess.run(args, cwd=dest, check=True, capture_output=True, env=env)

    run("git", "init", "-q", "-b", "main")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "pinned")
    # Rewrite the pin to whatever sha this synthetic tree actually landed at, unless the row
    # wants DRIFT — in which case the caller passes the drifted value.
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    pins = (root / "vendor" / "pins.toml").read_text(encoding="utf-8")
    real = _SHA_LITERAL.search(pins)
    assert real is not None, "the tracked pins.toml no longer carries a 40-hex sha"
    (root / "vendor" / "pins.toml").write_text(
        pins.replace(real.group(0), head if sha == "MATCH" else sha), encoding="utf-8"
    )


def _drive(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(root / "tools" / _SCRIPT.name)], capture_output=True,
                          text=True)


def test_the_build_script_is_tracked_and_executable() -> None:
    """A repair that is not in a commit is a repair that lives in a record (R324(c))."""
    assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"
    tracked = subprocess.run(["git", "-C", str(_REPO), "ls-files", "--error-unmatch",
                              str(_SCRIPT.relative_to(_REPO))], capture_output=True, text=True)
    assert tracked.returncode == 0, f"{_SCRIPT} is not tracked: {tracked.stderr}"
    assert _SCRIPT.stat().st_mode & 0o111, f"{_SCRIPT} is not executable"


def test_the_script_carries_no_sha_literal_and_reads_the_pin_instead() -> None:
    """ONE sha authority. A hardcoded sha builds the right source until the pin moves, and
    then builds the wrong one silently — the class R145 exists to close."""
    body = _SCRIPT.read_text(encoding="utf-8")
    found = _SHA_LITERAL.findall(body)
    assert not found, f"{_SCRIPT.name} carries its own sha literal(s) {found}; read pins.toml"
    assert "vendor/pins.toml" in body, (
        f"{_SCRIPT.name} names no pin file, so it cannot be reading the pinned sha at all"
    )


def test_the_refusal_reason_names_this_script_and_this_script_exists() -> None:
    """The reason and the tracked step must not drift apart: the reason is what an operator
    reads at the refusal, and a named path that does not exist is worse than none."""
    from mantis.bots.sealbot import BUILD_ABSENT_REASON, BUILD_SCRIPT

    assert BUILD_SCRIPT in BUILD_ABSENT_REASON, BUILD_ABSENT_REASON
    assert (_REPO / BUILD_SCRIPT).is_file(), f"the reason names {BUILD_SCRIPT}, which is absent"
    assert "make vendor" not in BUILD_ABSENT_REASON, (
        "the build reason must name its OWN missing step only (O-A1's reason SHAPE arm)"
    )


def test_it_refuses_when_no_pin_has_been_fetched(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    proc = _drive(root)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "make vendor" in proc.stderr, proc.stderr


def test_it_refuses_when_the_fetched_tree_drifted_from_the_pin(tmp_path: Path) -> None:
    """The row that stops a build over an unpinned engine. `git checkout <sha>` is
    content-addressed, so a drifted tree means someone moved it after the fetch."""
    root = _synthetic_root(tmp_path)
    _fake_pin(root, sha="0" * 40, march_native=False, win_threshold=True)
    proc = _drive(root)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "pins.toml says" in proc.stderr, proc.stderr


@pytest.mark.parametrize(
    ("march_native", "win_threshold", "marker"),
    [(True, True, "march=native"), (False, False, "WIN_THRESHOLD")],
)
def test_it_refuses_when_the_pinned_patch_is_not_applied(
    tmp_path: Path, march_native: bool, win_threshold: bool, marker: str,
) -> None:
    """BOTH hunks, separately. `vendor_fetch.sh` reads `patch` with a one-argument `.get`, so
    a renamed or untracked patch is skipped SILENTLY — the exact hole arm 3 of
    `test_vendor_pins_sealbot.py` names, here caught one step later at the build."""
    root = _synthetic_root(tmp_path)
    _fake_pin(root, sha="MATCH", march_native=march_native, win_threshold=win_threshold)
    proc = _drive(root)
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert marker in proc.stderr, proc.stderr
    assert "NOT applied" in proc.stderr, proc.stderr
