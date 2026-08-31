"""R326(e) — `make vendor` is IDEMPOTENT: a warm, correct tree is a no-op, not a failure.

WHAT THIS FILE IS THE ONLY WITNESS TO. `tools/vendor_fetch.sh` applied its tracked patch
UNCONDITIONALLY, so the second `make vendor` on a box that already held a correct vendor tree
FAILED — `git apply` refuses a patch that is already in the working tree. RECAL-SITTING-4
recorded the consequence directly (`RECAL_SITTING4_EXIT.md` §7.3): the one command that exists
to VERIFY vendor state could not be run to verify it, so the sitting fell back to the BUILD
script's four preconditions and the fetch step was certified by something else entirely.

WHY THE UNKNOWN-STATE ROW IS THE LOAD-BEARING ONE. Idempotency is easy to fake: swallow the
failure and print success. That would be strictly worse than the bug, because a vendor tree
that is neither patched nor clean compiles into an engine that loads and plays at a depth
receipt no downstream oracle can distinguish from the right one (the same argument
`test_vendor_build_sealbot.py` makes for its refusal rows). So the third outcome — applies
neither forward nor in reverse — must REFUSE, and it has its own row.

WHY THE SCRIPT IS DRIVEN AGAINST A SYNTHETIC ROOT AND A LOCAL SOURCE REPO. The real
`vendor/external/sealbot` is present and patched in the main tree, so a row driven at the repo
root would exercise the warm path only and could never see the cold one. The script resolves
`vendor/pins.toml` and `vendor/external/` relative to its CWD, so a temp root carrying only a
pins file exercises the same code with no real state touched — and a `git init`ed source repo
means every row runs OFFLINE. A test that needed the network would be skipped on the box, which
is the one place this behaviour matters.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "tools" / "vendor_fetch.sh"

#: A git identity for the synthetic source repo. The box and CI both run with a repo-local
#: identity, but a temp repo has none, and `git commit` without one fails before the row under
#: test ever runs.
_GIT_ID = ("-c", "user.email=test@example.invalid", "-c", "user.name=vendor fetch test")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    out = subprocess.run(["git", *_GIT_ID, *args], cwd=cwd, check=True,
                         capture_output=True, text=True)
    return out


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    """A one-commit git repo to serve as the pin's `url`, and its sha."""
    src = tmp_path / "source"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "engine.c").write_text("int WIN_THRESHOLD = 5;\n", encoding="utf-8")
    _git(src, "add", "engine.c")
    _git(src, "commit", "-qm", "initial")
    return src, _git(src, "rev-parse", "HEAD").stdout.strip()


def _patch_file(tmp_path: Path, src: Path) -> Path:
    """A tracked-patch equivalent: the diff that flips the source's one constant."""
    (src / "engine.c").write_text("int WIN_THRESHOLD = 6;\n", encoding="utf-8")
    diff = _git(src, "diff").stdout
    _git(src, "checkout", "--", "engine.c")
    patch = tmp_path / "engine.patch"
    patch.write_text(diff, encoding="utf-8")
    return patch


def _root(tmp_path: Path, *, url: str, sha: str, patch: Path | None) -> Path:
    root = tmp_path / "root"
    (root / "vendor").mkdir(parents=True)
    body = "[pins]\n"
    if url:
        body += f'[pins.engine]\nurl = "{url}"\nsha = "{sha}"\n'
        if patch is not None:
            body += f'patch = "{patch}"\n'
    (root / "vendor" / "pins.toml").write_text(body, encoding="utf-8")
    return root


def _fetch(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(_SCRIPT)], cwd=root, check=False,
                          capture_output=True, text=True)


@pytest.fixture
def warm(tmp_path: Path) -> Path:
    src, sha = _source_repo(tmp_path)
    patch = _patch_file(tmp_path, src)
    return _root(tmp_path, url=str(src), sha=sha, patch=patch)


def test_the_first_run_clones_and_patches(warm: Path) -> None:
    first = _fetch(warm)
    assert first.returncode == 0, first.stderr
    assert "patch applied" in first.stdout, first.stdout
    engine = warm / "vendor" / "external" / "engine" / "engine.c"
    assert engine.read_text(encoding="utf-8").strip().endswith("= 6;"), (
        "the patch's EFFECT must be in the tree, not merely reported"
    )


def test_the_second_run_is_a_no_op_and_not_a_failure(warm: Path) -> None:
    """The bug, driven. Pre-R326(e) this second call returned 1."""
    assert _fetch(warm).returncode == 0
    second = _fetch(warm)
    assert second.returncode == 0, (
        f"a warm, correct vendor tree must be a no-op; got rc {second.returncode}\n"
        f"{second.stderr}"
    )
    assert "already applied" in second.stdout, second.stdout
    engine = warm / "vendor" / "external" / "engine" / "engine.c"
    assert engine.read_text(encoding="utf-8").strip().endswith("= 6;"), (
        "the no-op must leave the patched tree patched, not revert it"
    )


def test_a_tree_that_is_NEITHER_patched_NOR_clean_is_REFUSED(warm: Path) -> None:
    """Idempotency must not be bought by swallowing the failure. A tree in an unknown state is
    the one outcome that has to stay loud.

    MUTATION THAT REDS IT: `apply_patch` returning quietly on the third branch instead of
    raising — the "fix" that makes the other two rows pass while publishing a vendor tree
    nobody has verified."""
    assert _fetch(warm).returncode == 0
    engine = warm / "vendor" / "external" / "engine" / "engine.c"
    engine.write_text("int WIN_THRESHOLD = 7;  /* hand-edited */\n", encoding="utf-8")
    broken = _fetch(warm)
    assert broken.returncode != 0, "an unknown vendor tree must REFUSE, not report ready"
    assert "UNKNOWN state" in broken.stderr, broken.stderr
    assert "applies neither forward nor in reverse" in broken.stderr, broken.stderr


def test_an_empty_pin_table_is_honest_empty_behaviour(tmp_path: Path) -> None:
    """Unchanged by R326(e), and pinned so the idempotency rewrite cannot have eaten it."""
    root = _root(tmp_path, url="", sha="", patch=None)
    out = _fetch(root)
    assert out.returncode == 0, out.stderr
    assert "no pins declared" in out.stdout


def test_the_script_carries_no_sha_and_no_url_of_its_own() -> None:
    """`vendor/pins.toml` is the ONE authority for what gets fetched and from where. A literal
    in the script would be a second authority, which is R79's exact prohibition and the same
    row `test_vendor_build_sealbot.py` holds over the build half."""
    import re
    body = _SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{40}\b", body), "the script carries a sha literal"
    assert "github.com" not in body and "https://" not in body, (
        "the script carries a URL literal; the pin file is the only authority"
    )
