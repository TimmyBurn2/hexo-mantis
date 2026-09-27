"""`vendor_fetch.sh <pin>` fetches the pin's release assets verified by sha256; offline, over `file://` assets and a local repo."""
from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "tools" / "vendor_fetch.sh"
_GIT_ID = ("-c", "user.email=test@example.invalid", "-c", "user.name=vendor fetch test")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *_GIT_ID, *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "source"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "README").write_text("engine source\n", encoding="utf-8")
    _git(src, "add", "README")
    _git(src, "commit", "-qm", "initial")
    return src, _git(src, "rev-parse", "HEAD")


def _archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "release.tar.gz"
    with tarfile.open(path, "w:gz") as tar:  # encoding-gate: ok -- a gzip archive, not text
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
    return path


class _World:
    """A root whose `vendor/pins.toml` pins a local repo, an archive with two members and a loose file."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.src, self.sha = _source_repo(tmp_path)
        self.engine = b"#!/bin/sh\necho engine\n"
        self.inner_net = b"net-inside-the-release"
        self.archive = _archive(tmp_path, {"Rel/engine/bin": self.engine,
                                           "Rel/nets/big.onnx": self.inner_net})
        self.loose = tmp_path / "gen-0030.onnx"
        self.loose.write_bytes(b"a small network")
        self.root = tmp_path / "root"
        (self.root / "vendor").mkdir(parents=True)

    def write_pins(self, *, loose_sha: str | None = None, member_sha: str | None = None) -> None:
        archive_url = self.archive.as_uri()
        body = (
            "[pins]\n"
            f'[pins.eng]\nurl = "{self.src}"\nsha = "{self.sha}"\n'
            f'[pins.eng.assets.release]\nurl = "{archive_url}"\n'
            f'sha256 = "{_sha(self.archive.read_bytes())}"\n'
            'path = "eng-assets/release.tar.gz"\nunpack = "eng-assets/release"\n'
            f'[pins.eng.assets.engine]\nurl = "{archive_url}"\nfrom = "release"\n'
            f'sha256 = "{member_sha or _sha(self.engine)}"\npath = "eng-assets/release/Rel/engine/bin"\n'
            f'[pins.eng.assets.big]\nurl = "{archive_url}"\nfrom = "release"\n'
            f'sha256 = "{_sha(self.inner_net)}"\npath = "eng-assets/release/Rel/nets/big.onnx"\n'
            f'[pins.eng.assets.small]\nurl = "{self.loose.as_uri()}"\n'
            f'sha256 = "{loose_sha or _sha(self.loose.read_bytes())}"\npath = "eng-assets/gen-0030.onnx"\n'
        )
        (self.root / "vendor" / "pins.toml").write_text(body, encoding="utf-8")

    def fetch(self, *pins: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(_SCRIPT), *pins], cwd=self.root, check=False,
                              capture_output=True, text=True)

    def external(self, rel: str) -> Path:
        return self.root / "vendor" / "external" / rel


@pytest.fixture
def world(tmp_path: Path) -> _World:
    w = _World(tmp_path)
    w.write_pins()
    return w


def test_a_named_pin_clones_and_fetches_every_asset_verified(world: _World) -> None:
    out = world.fetch("eng")
    assert out.returncode == 0, out.stderr
    assert world.external("eng/README").is_file(), "the named pin is still cloned at its sha"
    assert world.external("eng-assets/gen-0030.onnx").read_bytes() == b"a small network"
    assert world.external("eng-assets/release/Rel/engine/bin").read_bytes() == world.engine
    assert world.external("eng-assets/release/Rel/nets/big.onnx").read_bytes() == world.inner_net


def test_a_bare_run_clones_and_fetches_no_asset(world: _World) -> None:
    out = world.fetch()
    assert out.returncode == 0, out.stderr
    assert world.external("eng/README").is_file()
    assert not world.external("eng-assets").exists(), "assets are the named pin's step, never `make vendor`'s"


def test_a_planted_wrong_sha256_REFUSES_and_leaves_no_file(world: _World) -> None:
    """MUTATION THAT REDS IT: writing the download to its path before the digest is compared."""
    world.write_pins(loose_sha="0" * 64)
    out = world.fetch("eng")
    assert out.returncode != 0, "a download whose sha256 is not the pin's must refuse"
    assert "sha256 mismatch" in out.stderr and "gen-0030.onnx" in out.stderr, out.stderr
    assert not world.external("eng-assets/gen-0030.onnx").exists()
    assert not list(world.external("eng-assets").glob("*.part")), "no partial download survives"


def test_a_member_whose_sha256_is_not_the_pins_REFUSES(world: _World) -> None:
    world.write_pins(member_sha="f" * 64)
    out = world.fetch("eng")
    assert out.returncode != 0
    assert "sha256 mismatch" in out.stderr and "Rel/engine/bin" in out.stderr, out.stderr


def test_a_warm_verified_tree_is_a_no_op_that_needs_no_network(world: _World) -> None:
    assert world.fetch("eng").returncode == 0
    world.archive.unlink()
    world.loose.unlink()
    second = world.fetch("eng")
    assert second.returncode == 0, second.stderr
    assert "already verified" in second.stdout, second.stdout


def test_a_warm_file_that_no_longer_verifies_is_REFUSED_not_refetched(world: _World) -> None:
    assert world.fetch("eng").returncode == 0
    world.external("eng-assets/release/Rel/engine/bin").write_bytes(b"swapped")
    out = world.fetch("eng")
    assert out.returncode != 0 and "sha256 mismatch" in out.stderr, out.stderr


def test_an_unknown_pin_name_is_REFUSED(world: _World) -> None:
    out = world.fetch("nope")
    assert out.returncode != 0 and "no [pins.nope]" in out.stderr, out.stderr


def test_a_member_must_name_an_unpacked_asset_with_its_own_url(world: _World) -> None:
    body = (world.root / "vendor" / "pins.toml").read_text(encoding="utf-8")
    body = body.replace('from = "release"\nsha256 = "' + _sha(world.engine),
                        'from = "small"\nsha256 = "' + _sha(world.engine), 1)
    (world.root / "vendor" / "pins.toml").write_text(body, encoding="utf-8")
    out = world.fetch("eng")
    assert out.returncode != 0 and "engine" in out.stderr and "unpack" in out.stderr, out.stderr


def test_a_directory_at_the_clone_path_that_is_not_a_clone_is_REFUSED(world: _World) -> None:
    """Without the check, `git -C` resolves to the ENCLOSING repo and fetches and checks out there."""
    world.external("eng").mkdir(parents=True)
    (world.external("eng") / "stray").write_text("scratch", encoding="utf-8")
    out = world.fetch()
    assert out.returncode != 0 and "not a clone" in out.stderr, out.stderr
